"""Trace rows in ``knowledge.db`` and the one call that finishes a request (plan §9.1).

A request is observed like this (KWP6 answers, KWP3 ingests)::

    with observe(conn, question=text, profile=active.name) as trace:
        ...                       # stage("condense"), stage("vector"), ...
        mark_first_token()        # when the first answer token streams out
        trace.hits = hits_from_retrieval(result)
        trace.context_tokens = result.context_tokens
        trace.outcome = "answered"
        trace.grounding = "verified"

On exit :func:`finish_trace` fills ``total_ms``, counts the question's
Prometheus metrics, persists the row, exports OpenTelemetry spans when enabled
and, at most once a day per process, deletes rows older than
:data:`RETENTION_DAYS`. An exception inside the block marks the trace
``error`` and still records it. Observability never breaks the request: a
failure to persist or export is logged, not raised.

Errors are counted from the trace (``kb_errors_total``): each failing stage
once, at its innermost level, including failures the pipeline handled (an
embedding call that fell back to BM25-only), and ``stage="request"`` for a
failure outside any stage. So don't also call ``record_error`` for exceptions
that left a ``stage()`` block of an observed request. Tokens and cost come from
``metrics.record_usage`` (call it from the usage sink), which also fills
``trace.tokens_by_model`` and ``trace.cost_usd``.

The ``traces`` columns other than ``body`` are what Grafana's SQLite panels
read; none of them holds question text. ``body`` is the full trace as JSON
(question and standalone query included) for the Query Inspector.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from hal9000.knowledge.contracts import RetrievalResult, RetrievedChunk, Trace, TraceHit
from hal9000.knowledge.obs import metrics, otel
from hal9000.knowledge.obs.timing import current_trace, start_trace

__all__ = [
    "RETENTION_DAYS",
    "RetentionJob",
    "finish_trace",
    "hits_from_retrieval",
    "load_trace",
    "mark_first_token",
    "observe",
    "purge_traces",
    "retrieval_seconds",
    "top_score",
    "trace_row",
    "write_trace",
]

log = logging.getLogger(__name__)

#: plan §12: traces are kept 180 days.
RETENTION_DAYS = 180


def _ts(moment: datetime) -> str:
    """UTC, millisecond ISO-8601 (one fixed width, so text order is time order)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


# ------------------------------------------------------------- persistence
def trace_row(trace: Trace) -> dict[str, object]:
    """The ``traces`` row for ``trace``: scalar columns for Grafana plus the JSON body."""
    return {
        "id": trace.id,
        "created_at": _ts(trace.created_at),
        "kind": trace.kind,
        "outcome": trace.outcome,
        "profile": trace.profile,
        "total_ms": trace.total_ms,
        "ttft_ms": trace.ttft_ms,
        "cost_usd": trace.cost_usd,
        "body": trace.model_dump_json(),
    }


def write_trace(conn: sqlite3.Connection, trace: Trace) -> None:
    """Insert or replace ``trace``'s row."""
    row = trace_row(trace)
    columns = ", ".join(row)
    marks = ", ".join(f":{key}" for key in row)
    with conn:
        conn.execute(f"INSERT OR REPLACE INTO traces ({columns}) VALUES ({marks})", row)


def load_trace(conn: sqlite3.Connection, trace_id: str) -> Trace | None:
    row = conn.execute("SELECT body FROM traces WHERE id = ?", (trace_id,)).fetchone()
    return Trace.model_validate_json(row[0]) if row else None


def purge_traces(
    conn: sqlite3.Connection, *, days: int = RETENTION_DAYS, now: datetime | None = None
) -> int:
    """Delete trace rows older than ``days``; returns how many went."""
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days!r}")
    cutoff = _ts((now or datetime.now(UTC)) - timedelta(days=days))
    with conn:
        cursor = conn.execute(
            "DELETE FROM traces WHERE julianday(created_at) < julianday(?)", (cutoff,)
        )
    return cursor.rowcount


class RetentionJob:
    """Runs :func:`purge_traces` at most once per ``interval_s`` in this process."""

    def __init__(self, *, days: int = RETENTION_DAYS, interval_s: float = 86_400.0) -> None:
        self.days = days
        self.interval_s = interval_s
        self._last: float | None = None
        self._lock = threading.Lock()

    def maybe_run(self, conn: sqlite3.Connection, *, now: datetime | None = None) -> int | None:
        """Purge if the interval has passed; returns rows deleted, or ``None`` if skipped."""
        clock = time.monotonic()
        with self._lock:
            if self._last is not None and clock - self._last < self.interval_s:
                return None
            self._last = clock
        deleted = purge_traces(conn, days=self.days, now=now)
        if deleted:
            log.info("deleted %d trace rows older than %d days", deleted, self.days)
        return deleted


_retention = RetentionJob()


# ------------------------------------------------------ building a trace
def mark_first_token(trace: Trace | None = None) -> None:
    """Record time to first token on ``trace`` (default: the bound one), once."""
    trace = trace if trace is not None else current_trace()
    if trace is not None and trace.ttft_ms is None and trace._t0:
        trace.ttft_ms = round((time.perf_counter() - trace._t0) * 1000, 3)


def hits_from_retrieval(result: RetrievalResult) -> list[TraceHit]:
    """Candidates (fused / reranked order) as trace hits, flagging those selected."""
    selected = {chunk.chunk_id for chunk in result.selected}
    seen: set[int] = set()
    hits: list[TraceHit] = []
    chunks: Iterable[RetrievedChunk] = [*result.candidates, *result.selected]
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        hits.append(
            TraceHit(
                chunk_id=chunk.chunk_id,
                rank=chunk.rank,
                score=chunk.score,
                dense_score=chunk.dense_score,
                bm25_score=chunk.bm25_score,
                rerank_score=chunk.rerank_score,
                selected=chunk.chunk_id in selected,
            )
        )
    return hits


def top_score(trace: Trace) -> float | None:
    """The abstain score (plan §7.1): rank 1's rerank score if reranked, else its dense score.

    Never the fused (RRF) score. ``None`` when there is no dense or rerank score.
    """
    if not trace.hits:
        return None
    first = min(trace.hits, key=lambda hit: hit.rank)
    if first.rerank_score is not None:
        return first.rerank_score
    return first.dense_score


def retrieval_seconds(trace: Trace) -> float | None:
    """Local retrieval time: the outermost vector/bm25/fuse/rerank/select stages."""
    wanted = metrics.RETRIEVAL_STAGES
    durations = [
        t.duration_ms
        for t in trace.stages
        if t.stage.value in wanted and (t.parent is None or t.parent.value not in wanted)
    ]
    return sum(durations) / 1000 if durations else None


def _count_errors(trace: Trace) -> None:
    failing = [t for t in trace.stages if t.error]
    for timing in failing:
        # an exception leaving a nested stage fails its parents too: count it once, innermost
        inner = any(
            other.parent == timing.stage and other.depth == timing.depth + 1 and other.error
            for other in failing
        )
        if not inner:
            metrics.record_error(timing.stage, timing.error)
    if trace.error and not failing:
        metrics.record_error("request", trace.error)


def finish_trace(
    trace: Trace,
    conn: sqlite3.Connection | None = None,
    *,
    top_score_value: float | None = None,
    retention: bool = True,
) -> None:
    """Complete ``trace``: metrics, the ``traces`` row (if ``conn``), spans, retention.

    ``top_score_value`` overrides the abstain score derived from ``trace.hits``.
    """
    if trace.total_ms is None and trace._t0:
        trace.total_ms = round((time.perf_counter() - trace._t0) * 1000, 3)
    if trace.kind == "question":
        if trace.outcome is None:
            trace.outcome = "error" if trace.error else "answered"
        score = top_score_value if top_score_value is not None else top_score(trace)
        metrics.record_question(
            trace.outcome,
            ttft_s=trace.ttft_ms / 1000 if trace.ttft_ms is not None else None,
            answer_s=trace.total_ms / 1000 if trace.total_ms is not None else None,
            retrieval_s=retrieval_seconds(trace),
            top_score=score,
            context_tokens=trace.context_tokens if trace.outcome == "answered" else None,
        )
    _count_errors(trace)
    if conn is not None:
        try:
            write_trace(conn, trace)
            if retention:
                _retention.maybe_run(conn)
        except sqlite3.Error:
            log.exception("could not write trace %s", trace.id)
    otel.export_trace(trace)


@contextmanager
def observe(
    conn: sqlite3.Connection | None = None, *, trace: Trace | None = None, **fields
) -> Iterator[Trace]:
    """Bind a trace for the block (see :func:`start_trace`) and finish it on exit."""
    with start_trace(trace, **fields) as bound:
        try:
            yield bound
        except Exception as exc:
            bound.error = bound.error or type(exc).__name__
            if bound.kind == "question":
                bound.outcome = "error"
            raise
        finally:
            try:
                finish_trace(bound, conn)
            except Exception:  # pragma: no cover - never mask the request's own result
                log.exception("could not finish trace %s", bound.id)
