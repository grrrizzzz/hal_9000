"""What a gold generation will cost and take, and how far along it is.

- :class:`UsageBasis` / :func:`ledger_basis`: completion tokens per call
  (visible output **plus reasoning**, which reasoning models bill as output)
  and seconds per call, from earlier ``knowledge-eval`` calls to the same
  model in the cost ledger when there are enough of them, else the config's
  allowances (``gold.estimate`` in ``config/knowledge_eval.yaml``).
- :class:`Progress`: one flushed stderr line per LLM call with the phase,
  running spend, the measured seconds per call and an ETA. Lines never carry
  question or document text (document titles, shortened, only).
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:  # pragma: no cover
    from hal9000.knowledge.eval.config import GoldEstimate
    from hal9000.storage.db import Database

__all__ = ["GOLD_RUN_PREFIX", "Progress", "UsageBasis", "format_duration", "ledger_basis"]

#: ledger run IDs of ``kb gold generate`` (see :func:`hal9000.knowledge.eval.spend.open_spend`)
GOLD_RUN_PREFIX = "kb-eval-gold-"
#: consecutive ledger rows further apart than this are not one call's duration
MAX_GAP_SECONDS = 900


def format_duration(seconds: float) -> str:
    """``"<1 min"``, ``"45 min"``, ``"1 h 28 min"``."""
    minutes = round(seconds / 60)
    if minutes < 1:
        return "<1 min"
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


@dataclass(frozen=True)
class UsageBasis:
    """Per-call completion tokens (incl. reasoning) and seconds, and where they come from."""

    completion_tokens: float
    seconds: float
    #: ledger calls the token figure comes from (0: the config allowance)
    calls: int = 0
    #: call-to-call intervals the time figure comes from (0: the config default)
    timed: int = 0
    what: str = "earlier calls"

    def describe(self) -> str:
        if self.calls:
            tokens = (
                f"~{self.completion_tokens:,.0f} completion tokens per call incl. reasoning, "
                f"the mean of {self.calls} {self.what}"
            )
        else:
            tokens = (
                f"default allowance of {self.completion_tokens:,.0f} completion tokens per call "
                "incl. reasoning (no earlier calls to this model in the ledger)"
            )
        if self.timed:
            timing = f"~{self.seconds:.0f} s per call from their ledger timestamps"
        else:
            timing = f"{self.seconds:.0f} s per call (default)"
        return f"{tokens}; {timing}"


def _seconds(at: str) -> float | None:
    try:
        return datetime.fromisoformat(at).timestamp()
    except (TypeError, ValueError):
        return None


def ledger_basis(db: Database | None, model: str, tier: str, allowance: GoldEstimate) -> UsageBasis:
    """Per-call usage from the ledger's earlier ``knowledge-eval`` calls to ``model``.

    Gold-generation runs are preferred (same purpose); with fewer than
    ``allowance.min_ledger_calls`` of them, every evaluation call to the model
    counts; with fewer still, the config's allowance for ``tier`` is used.
    Per-call time is the mean interval between consecutive rows of one run.
    """
    from hal9000.knowledge.config import EVAL_AGENT

    rows = db.ledger_calls(agent=EVAL_AGENT, model=model) if db is not None else []
    gold = [r for r in rows if str(r["run_id"]).startswith(GOLD_RUN_PREFIX)]
    enough = allowance.min_ledger_calls
    if len(gold) >= enough:
        chosen, what = gold, f"earlier gold-generation calls to {model}"
    elif len(rows) >= enough:
        chosen, what = rows, f"earlier evaluation calls to {model}"
    else:
        chosen, what = [], ""
    default_tokens = float(allowance.completion(tier))
    tokens = (
        sum(int(r["completion_tokens"] or 0) for r in chosen) / len(chosen)
        if chosen
        else default_tokens
    )
    by_run: dict[str, list[float]] = defaultdict(list)
    for row in chosen:
        stamp = _seconds(row["at"])
        if stamp is not None:
            by_run[row["run_id"]].append(stamp)
    gaps = [
        b - a
        for stamps in by_run.values()
        for a, b in zip(stamps, stamps[1:], strict=False)
        if 0 < b - a <= MAX_GAP_SECONDS
    ]
    timed = len(gaps) >= max(1, enough - 1)
    seconds = sum(gaps) / len(gaps) if timed else float(allowance.seconds_per_call)
    return UsageBasis(tokens, seconds, len(chosen), len(gaps) if timed else 0, what)


class Progress:
    """One line per LLM call on stderr (flushed), e.g.

    ``[12/71] window 9/29 (Home inspection report, pp. 20–22): 34 items so far;
    $0.43 spent; ~70 s/call; ~45 min left``
    """

    def __init__(
        self,
        *,
        spent: Callable[[], float] = lambda: 0.0,
        out: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.spent = spent
        self.out = out
        self.clock = clock
        self.done = 0
        self.started: float | None = None
        self.lines: list[str] = []

    def start(self) -> None:
        if self.started is None:
            self.started = self.clock()

    def tick(self, label: str, *, items: int, remaining: int) -> str:
        """Record one finished call; ``remaining`` is the calls still expected after it."""
        self.start()
        assert self.started is not None
        self.done += 1
        per_call = (self.clock() - self.started) / self.done
        left = format_duration(per_call * remaining)
        line = (
            f"[{self.done}/{self.done + remaining}] {label}: {items} items so far; "
            f"${self.spent():.2f} spent; ~{per_call:.0f} s/call; "
            f"{left if left.startswith('<') else '~' + left} left"
        )
        self.lines.append(line)
        print(line, file=self.out or sys.stderr, flush=True)
        return line
