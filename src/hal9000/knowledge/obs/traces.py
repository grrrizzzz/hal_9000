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
``error`` and still records it. A ``BaseException`` that is not an
``Exception`` (``asyncio.CancelledError``, or ``GeneratorExit`` when a client
closes a stream) is recorded as outcome ``cancelled`` (the metric and the row's
``outcome`` column; the contracts' ``Outcome`` has no such value, so the JSON
body keeps ``outcome`` null) and re-raised. Observability never breaks the
request: a failure to persist (counted as
``kb_errors_total{stage="request",kind="TraceWriteError"}``) or to export is
logged, not raised.

``trace.error`` (and the OTel ``kb.error`` attribute) holds only
:func:`describe_error`'s ``"TypeName: message"`` with the message cut to
:data:`ERROR_MESSAGE_CHARS` characters. KWP6 and the pipeline must not raise
exceptions whose message carries question or document text; set
``trace.error`` yourself with :func:`describe_error` on a sanitized exception
if you need to.

**Transactions.** :func:`write_trace` and :func:`purge_traces` never commit or
roll back a transaction they didn't open. With a transaction open on ``conn``
they join it through a savepoint and leave the commit to the caller; with none
open they commit their own write. A dedicated connection for traces is simplest.

Errors are counted from the trace (``kb_errors_total``): each failing stage
once, at its innermost level, including failures the pipeline handled (an
embedding call that fell back to BM25-only), and ``stage="request"`` for a
failure outside any stage. ``metrics.record_error`` for an exception that
already left a ``stage()`` block of the observed request is ignored (the trace
counts it), so reporting a handled failure again never counts it twice.
Tokens and cost come from
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
from pathlib import Path

from hal9000.knowledge.contracts import RetrievalResult, RetrievedChunk, Trace, TraceHit
from hal9000.knowledge.obs import metrics, otel
from hal9000.knowledge.obs.timing import current_trace, start_trace

__all__ = [
    "ERROR_MESSAGE_CHARS",
    "RETENTION_DAYS",
    "RetentionJob",
    "configure_retention",
    "configured_retention_days",
    "describe_error",
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

#: plan §12: traces are kept 180 days (``knowledge.yaml`` ``trace_retention_days``).
RETENTION_DAYS = 180
#: ``trace.error`` keeps at most this much of an exception's message.
ERROR_MESSAGE_CHARS = 200


def describe_error(exc: BaseException) -> str:
    """``"TypeName: message"``, the message cut to :data:`ERROR_MESSAGE_CHARS` characters."""
    name = type(exc).__name__
    message = " ".join(str(exc).split())[:ERROR_MESSAGE_CHARS]
    return f"{name}: {message}" if message else name


def _error_kind(error: str) -> str:
    return error.split(":", 1)[0]


def _ts(moment: datetime) -> str:
    """UTC, millisecond ISO-8601 (one fixed width, so text order is time order)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


# ------------------------------------------------------------- persistence
@contextmanager
def _atomic(conn: sqlite3.Connection) -> Iterator[None]:
    """Our own write: a savepoint inside the caller's transaction, else our own commit."""
    if conn.in_transaction:
        conn.execute("SAVEPOINT kb_obs")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK TO kb_obs")
            conn.execute("RELEASE kb_obs")
            raise
        conn.execute("RELEASE kb_obs")
        return
    try:
        yield
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    if conn.in_transaction:
        conn.commit()


def trace_row(trace: Trace, *, outcome: str | None = None) -> dict[str, object]:
    """The ``traces`` row for ``trace``: scalar columns for Grafana plus the JSON body.

    ``outcome`` overrides the column (``cancelled``, which the contract lacks).
    """
    return {
        "id": trace.id,
        "created_at": _ts(trace.created_at),
        "kind": trace.kind,
        "outcome": outcome if outcome is not None else trace.outcome,
        "profile": trace.profile,
        "total_ms": trace.total_ms,
        "ttft_ms": trace.ttft_ms,
        "cost_usd": trace.cost_usd,
        "body": trace.model_dump_json(),
    }


def write_trace(conn: sqlite3.Connection, trace: Trace, *, outcome: str | None = None) -> None:
    """Insert or replace ``trace``'s row (see the module notes on transactions)."""
    row = trace_row(trace, outcome=outcome)
    columns = ", ".join(row)
    marks = ", ".join(f":{key}" for key in row)
    with _atomic(conn):
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
    with _atomic(conn):
        cursor = conn.execute(
            "DELETE FROM traces WHERE julianday(created_at) < julianday(?)", (cutoff,)
        )
    return cursor.rowcount


def configured_retention_days(yaml_path: Path | str | None = None) -> int:
    """``trace_retention_days`` from ``config/knowledge.yaml`` (default :data:`RETENTION_DAYS`).

    An unreadable file or a value under one day falls back to the default, with
    a warning: retention must never break a request.
    """
    from hal9000.knowledge.config import load_knowledge_file

    try:
        days = int(load_knowledge_file(yaml_path).trace_retention_days)
    except Exception as error:
        log.warning("trace retention: cannot read knowledge.yaml (%s); keeping %d days",
                    error, RETENTION_DAYS)  # fmt: skip
        return RETENTION_DAYS
    if days < 1:
        log.warning(
            "trace_retention_days must be >= 1, got %d; keeping %d days", days, RETENTION_DAYS
        )
        return RETENTION_DAYS
    return days


class RetentionJob:
    """Runs :func:`purge_traces` at most once per ``interval_s`` in this process.

    ``days`` fixes the retention; without it, ``trace_retention_days`` is read
    from ``yaml_path`` (default: the configured ``config/knowledge.yaml``) each
    time the job runs, so an edit applies from the next daily purge.
    """

    def __init__(
        self,
        *,
        days: int | None = None,
        interval_s: float = 86_400.0,
        yaml_path: Path | str | None = None,
    ) -> None:
        self._days = days
        self.yaml_path = yaml_path
        self.interval_s = interval_s
        self._last: float | None = None
        self._lock = threading.Lock()

    @property
    def days(self) -> int:
        return self._days if self._days is not None else configured_retention_days(self.yaml_path)

    def maybe_run(self, conn: sqlite3.Connection, *, now: datetime | None = None) -> int | None:
        """Purge if the interval has passed; returns rows deleted, or ``None`` if skipped."""
        clock = time.monotonic()
        with self._lock:
            if self._last is not None and clock - self._last < self.interval_s:
                return None
            self._last = clock
        days = self.days
        deleted = purge_traces(conn, days=days, now=now)
        if deleted:
            log.info("deleted %d trace rows older than %d days", deleted, days)
        return deleted


_retention = RetentionJob()


def configure_retention(yaml_path: Path | str | None) -> None:
    """Read ``trace_retention_days`` from this ``knowledge.yaml`` (the service's own)."""
    global _retention
    _retention = RetentionJob(yaml_path=yaml_path)


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


def _count_errors(trace: Trace, *, ignore: str | None = None) -> None:
    failing = [t for t in trace.stages if t.error and t.error != ignore]
    for timing in failing:
        # an exception leaving a nested stage fails its parents too: count it once, innermost
        inner = any(
            other.parent == timing.stage and other.depth == timing.depth + 1 and other.error
            for other in failing
        )
        if not inner:
            metrics.count_trace_error(timing.stage, timing.error)
    if trace.error and not failing and _error_kind(trace.error) != ignore:
        metrics.count_trace_error("request", _error_kind(trace.error))


def finish_trace(
    trace: Trace,
    conn: sqlite3.Connection | None = None,
    *,
    top_score_value: float | None = None,
    retention: bool = True,
    cancelled: bool = False,
) -> None:
    """Complete ``trace``: metrics, the ``traces`` row (if ``conn``), spans, retention.

    ``top_score_value`` overrides the abstain score derived from ``trace.hits``.
    ``cancelled`` records outcome ``cancelled`` (the body's ``outcome`` is null)
    and doesn't count the cancellation itself as an error.
    """
    if trace.total_ms is None and trace._t0:
        trace.total_ms = round((time.perf_counter() - trace._t0) * 1000, 3)
    outcome: str | None = None
    if cancelled:
        outcome = metrics.CANCELLED
        trace.outcome = None
    if trace.kind == "question":
        if outcome is None:
            if trace.outcome is None:
                trace.outcome = "error" if trace.error else "answered"
            outcome = trace.outcome
        score = top_score_value if top_score_value is not None else top_score(trace)
        metrics.record_question(
            outcome,
            ttft_s=trace.ttft_ms / 1000 if trace.ttft_ms is not None else None,
            answer_s=trace.total_ms / 1000 if trace.total_ms is not None else None,
            retrieval_s=retrieval_seconds(trace),
            top_score=score,
            context_tokens=trace.context_tokens if trace.outcome == "answered" else None,
        )
    _count_errors(trace, ignore=_error_kind(trace.error) if cancelled and trace.error else None)
    if conn is not None:
        try:
            write_trace(conn, trace, outcome=outcome)
            if retention:
                _retention.maybe_run(conn)
        except sqlite3.Error:
            metrics.record_error("request", "TraceWriteError")
            log.exception("could not write trace %s", trace.id)
    otel.export_trace(trace, outcome=outcome)


@contextmanager
def observe(
    conn: sqlite3.Connection | None = None, *, trace: Trace | None = None, **fields
) -> Iterator[Trace]:
    """Bind a trace for the block (see :func:`start_trace`) and finish it on exit.

    An ``Exception`` ends the question as ``error``; any other ``BaseException``
    (cancellation, a closed stream) as ``cancelled``. Both are re-raised.
    """
    cancelled = False
    with start_trace(trace, **fields) as bound:
        try:
            yield bound
        except Exception as exc:
            bound.error = bound.error or describe_error(exc)
            if bound.kind == "question":
                bound.outcome = "error"
            raise
        except BaseException as exc:
            cancelled = True
            bound.error = describe_error(exc)
            raise
        finally:
            try:
                finish_trace(bound, conn, cancelled=cancelled)
            except Exception:  # pragma: no cover - never mask the request's own result
                log.exception("could not finish trace %s", bound.id)
