"""The shared stage timer and the closed stage enum (plan §9.1–9.2, rule K7).

Every step of answering (plan §7.1) and ingesting (plan §5.1) is wrapped in
:func:`stage`, using exactly the names in :class:`Stage`. One ``with`` block
feeds two outputs:

1. the per-request :class:`~hal9000.knowledge.contracts.Trace` bound with
   :func:`start_trace` (a ``contextvars`` variable, so concurrent requests and
   threads don't mix), and
2. the Prometheus histogram ``kb_stage_seconds{stage}`` in the knowledge
   base's own :data:`REGISTRY` (served at ``/metrics`` by KWP8).

Changing :class:`Stage` needs the orchestrator's approval (rule K4).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Histogram

if TYPE_CHECKING:  # pragma: no cover
    from hal9000.knowledge.contracts import Trace

__all__ = [
    "INGEST_STAGES",
    "QUESTION_STAGES",
    "REGISTRY",
    "STAGE_SECONDS",
    "Stage",
    "current_trace",
    "recorded_by_trace",
    "stage",
    "start_trace",
]


class Stage(StrEnum):
    """The closed set of stage names (plan §9.2)."""

    # questions (plan §7.1)
    CONDENSE = "condense"
    EMBED_QUERY = "embed_query"
    VECTOR = "vector"
    BM25 = "bm25"
    FUSE = "fuse"
    RERANK = "rerank"
    SELECT = "select"
    ABSTAIN = "abstain"
    GENERATE = "generate"
    VERIFY = "verify"
    # ingestion (plan §5.1)
    SCREEN = "screen"
    STORE = "store"
    PARSE = "parse"
    CLEAN = "clean"
    SEGMENT = "segment"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    REGRESSION = "regression"


QUESTION_STAGES = tuple(Stage)[:10]
INGEST_STAGES = tuple(Stage)[10:]

#: The knowledge base's own registry, kept apart from the process default so
#: tests can read it back and ``/metrics`` exposes only ``kb_*`` series.
REGISTRY = CollectorRegistry(auto_describe=True)

STAGE_SECONDS = Histogram(
    "kb_stage_seconds",
    "Wall-clock time per pipeline stage.",
    ["stage"],
    registry=REGISTRY,
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
)

_trace: ContextVar[Trace | None] = ContextVar("kb_trace", default=None)
_open: ContextVar[tuple[str, ...]] = ContextVar("kb_open_stages", default=())


def _coerce(name: str | Stage) -> Stage:
    try:
        return Stage(name)
    except ValueError:
        raise ValueError(
            f"Unknown stage {name!r}. Stage names are the closed set in plan §9.2: "
            f"{', '.join(s.value for s in Stage)}."
        ) from None


def _mark(exc: BaseException, current: Stage) -> None:
    """Note on the exception that the bound trace recorded it failing ``current``.

    ``metrics.record_error`` skips exactly this exception object for this
    stage (the trace counts it once, when it finishes).
    """
    trace = _trace.get()
    if trace is None:
        return
    try:
        marked = exc.__dict__.setdefault("_kb_trace_stages", (trace, set()))
    except (AttributeError, TypeError):  # pragma: no cover - an exception without a __dict__
        return
    if marked[0] is trace:
        marked[1].add(current.value)
    else:  # re-raised inside another request's trace
        exc.__dict__["_kb_trace_stages"] = (trace, {current.value})


def recorded_by_trace(exc: object, stage_name: str) -> bool:
    """Whether the bound trace already recorded ``exc`` failing stage ``stage_name``."""
    trace = _trace.get()
    marked = getattr(exc, "__dict__", {}).get("_kb_trace_stages") if trace is not None else None
    return marked is not None and marked[0] is trace and stage_name in marked[1]


def current_trace() -> Trace | None:
    """The trace bound to this request, if any."""
    return _trace.get()


@contextmanager
def start_trace(trace: Trace | None = None, **fields) -> Iterator[Trace]:
    """Bind ``trace`` (or a new one built from ``fields``) for the enclosed block."""
    from hal9000.knowledge.contracts import Trace

    bound = trace if trace is not None else Trace(**fields)
    token = _trace.set(bound)
    open_token = _open.set(())
    bound._t0 = time.perf_counter()
    try:
        yield bound
    finally:
        _open.reset(open_token)
        _trace.reset(token)


@contextmanager
def stage(name: str | Stage) -> Iterator[None]:
    """Time the enclosed block as stage ``name``.

    Raises :class:`ValueError` at once for a name outside :class:`Stage`. Nested
    stages record their depth and parent. Exceptions propagate; the timing is
    still recorded, with the exception's type as ``error``.
    """
    from hal9000.knowledge.contracts import StageTiming

    current = _coerce(name)
    parents = _open.get()
    token = _open.set(parents + (current.value,))
    start = time.perf_counter()
    error: str | None = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        _mark(exc, current)
        raise
    finally:
        elapsed = time.perf_counter() - start
        _open.reset(token)
        trace = _trace.get()
        # the live question latencies are questions' own: retrievals inside an
        # ingestion (the post-ingest regression) only go in that trace's row
        if trace is None or trace.kind == "question" or current not in QUESTION_STAGES:
            STAGE_SECONDS.labels(stage=current.value).observe(elapsed)
        if trace is not None:
            t0 = trace._t0 or start
            trace.stages.append(
                StageTiming(
                    stage=current,
                    started_ms=round((start - t0) * 1000, 3),
                    duration_ms=round(elapsed * 1000, 3),
                    depth=len(parents),
                    parent=parents[-1] if parents else None,
                    error=error,
                )
            )
