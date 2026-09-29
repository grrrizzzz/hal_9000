"""Streamed, metered answer generation (plan §7.1 ``generate``, §7.4 limits, §13).

:func:`stream_answer` runs one streamed chat completion
(:func:`hal9000.llm.stream_response`, or a fake with the same signature)
inside ``stage("generate")``:

- **Metering.** Through :meth:`Meter.stream` with the pipeline's agent
  (``knowledge``, or ``knowledge-eval`` for evaluation), so the monthly cap is
  checked before the call and the ledger row is written when the final
  ``Usage`` arrives. The same usage feeds ``kb_tokens_total`` /
  ``kb_cost_usd_total{purpose="answer"}`` and the bound trace.
- **Time to first token** is marked on the trace.
- **Limits.** ``max_tokens`` (``answer.max_output_tokens``) and the question's
  wall-clock deadline: the remaining time is the request timeout, and the
  deadline is checked between deltas.
- **Cancellation.** When ``cancel`` is set (the client went away) the stream
  is closed, which cancels the upstream request. The provider never sends
  the final usage then, so the tokens are counted locally and recorded as an
  *estimated* usage: the cost still reaches the ledger (approved decision).
  The same happens when the deadline passes or the stream fails part-way.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from hal9000.core.prompt import Prompt
from hal9000.knowledge.config import AGENT
from hal9000.knowledge.obs.timing import Stage, stage
from hal9000.knowledge.obs.traces import mark_first_token
from hal9000.llm import Usage

__all__ = [
    "Cancelled",
    "call_within",
    "DeadlineExceeded",
    "Generation",
    "StreamFn",
    "metered_generate",
    "record",
    "stream_answer",
]

StreamFn = Callable[..., Iterator[Any]]


class Cancelled(BaseException):  # noqa: N818 - a cancellation, like asyncio.CancelledError
    """The client went away; generation stopped (recorded as outcome ``cancelled``)."""


class DeadlineExceeded(TimeoutError):
    """The question's wall-clock limit (plan §7.4, 90 s) passed."""


@dataclass
class Generation:
    text: str
    usage: Usage | None = None
    estimated: bool = False


def record(usage: Usage, purpose: str, *, meter: Any | None, agent: str) -> float:
    """Metrics and trace for one call's usage (and the ledger when ``meter`` is given)."""
    from hal9000.knowledge.obs.metrics import record_usage

    fake = str(usage.get("model", "")).startswith("fake/")
    cost = record_usage(usage, purpose, cost_usd=0.0 if fake else None)
    if meter is not None and not fake:
        meter.record(usage, agent)
    return cost


def _model_for(tier: str) -> str:
    from hal9000 import config

    try:
        return config.model_for(tier)
    except Exception:  # no provider configured: the estimate still needs a label
        return f"unknown/{tier}"


def _estimate(prompt: Prompt, text: str, tier: str, model: str | None) -> Usage:
    from hal9000.knowledge.tokens import count_tokens, encoding_for_model
    from hal9000.llm import _messages_text

    name = model or _model_for(tier)
    encoding = encoding_for_model(name)
    return Usage(
        model=name,
        prompt=count_tokens(_messages_text(prompt.messages), encoding),
        cached=0,
        completion=count_tokens(text, encoding),
        reasoning=0,
        estimated=True,
    )


def stream_answer(
    prompt: Prompt,
    *,
    tier: str,
    stream_fn: StreamFn,
    on_delta: Callable[[str], None],
    meter: Any | None = None,
    agent: str = AGENT,
    cancel: threading.Event | None = None,
    deadline: float | None = None,
    max_tokens: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Generation:
    """Stream one answer, calling ``on_delta`` per text delta; returns the full text and usage.

    Raises :class:`Cancelled`, :class:`DeadlineExceeded`, the meter's
    ``BudgetExceeded`` (before any call), or whatever the provider raised.
    """
    got: dict[str, Usage] = {}

    def sink(usage: Usage) -> None:
        got["usage"] = usage
        record(usage, "answer", meter=None, agent=agent)

    kw: dict[str, Any] = {}
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if deadline is not None:
        remaining = deadline - clock()
        if remaining <= 0:
            raise DeadlineExceeded("the question's time limit passed before generation")
        kw["timeout"] = remaining

    parts: list[str] = []
    with stage(Stage.GENERATE):
        if meter is not None:
            meter.precheck(agent)

            def ledger_and_metrics(prompt, tier, *, on_usage=None, **extra):
                def both(usage: Usage) -> None:
                    if on_usage is not None and not str(usage.get("model", "")).startswith("fake/"):
                        on_usage(usage)  # the meter's ledger row
                    sink(usage)

                return stream_fn(prompt, tier, on_usage=both, **extra)

            stream = meter.stream(ledger_and_metrics, prompt, tier, agent=agent, **kw)
        else:
            stream = stream_fn(prompt, tier, on_usage=sink, **kw)
        try:
            for item in stream:
                if isinstance(item, str):
                    if item:
                        parts.append(item)
                        mark_first_token()
                        on_delta(item)
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if deadline is not None and clock() > deadline:
                    raise DeadlineExceeded("the question's time limit passed while answering")
        except BaseException:
            close = getattr(stream, "close", None)
            if close is not None:
                close()  # cancels the upstream request
            if "usage" not in got:
                model = getattr(stream_fn, "model", None)  # fakes name theirs
                usage = _estimate(prompt, "".join(parts), tier, model)
                record(usage, "answer", meter=meter, agent=agent)
                got["usage"] = usage
            raise
    usage = got.get("usage")
    return Generation(
        text="".join(parts), usage=usage, estimated=bool(usage and usage.get("estimated"))
    )


def call_within(
    deadline: float | None,
    clock: Callable[[], float],
    fn: Callable[..., Any],
    *args: Any,
    **kw: Any,
) -> Any:
    """``fn(*args, **kw)``, but raise :class:`DeadlineExceeded` if it isn't done by ``deadline``.

    For the question's paid calls that take no timeout of their own (condensing,
    the query embedding): the call runs on a helper thread in a copy of this
    context, so its usage still reaches the ledger and the trace if it finishes
    after the question gave up on it.
    """
    if deadline is None:
        return fn(*args, **kw)
    remaining = deadline - clock()
    if remaining <= 0:
        raise DeadlineExceeded("the question's time limit passed")
    import contextvars

    box: dict[str, Any] = {}
    context = contextvars.copy_context()

    def run() -> None:
        try:
            box["value"] = context.run(fn, *args, **kw)
        except BaseException as error:  # handed to the caller below
            box["error"] = error

    worker = threading.Thread(target=run, name="kb-bounded-call", daemon=True)
    worker.start()
    worker.join(remaining)
    if worker.is_alive():
        raise DeadlineExceeded("the question's time limit passed")
    if "error" in box:
        raise box["error"]
    return box["value"]


def metered_generate(
    generate_fn: Callable[..., str],
    *,
    meter: Any | None,
    agent: str,
    purpose: str,
    default_tier: str,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[..., str]:
    """``(prompt, tier=None) -> str`` over ``generate_fn``, metered and counted as ``purpose``.

    For the retriever's condense (and multi-query) calls: the cap is checked
    first, each call's usage reaches the ledger, Prometheus and the trace, and
    with a ``deadline`` the call is abandoned when the question's time is up.
    """

    import inspect

    try:
        accepts_usage = "on_usage" in inspect.signature(generate_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        accepts_usage = True

    def generate(prompt: Prompt, tier: str | None = None) -> str:
        if meter is not None:
            meter.precheck(agent)
        if not accepts_usage:  # a scripted stand-in: nothing to record
            return call_within(deadline, clock, generate_fn, prompt, tier or default_tier)
        return call_within(
            deadline,
            clock,
            generate_fn,
            prompt,
            tier or default_tier,
            on_usage=lambda usage: record(usage, purpose, meter=meter, agent=agent),
        )

    return generate
