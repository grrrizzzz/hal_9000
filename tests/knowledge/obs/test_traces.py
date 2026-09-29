"""Trace rows, their metrics and the 180-day retention (plan §9.1, §12)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from hal9000.knowledge.contracts import RetrievalResult, RetrievedChunk, Trace, TraceHit
from hal9000.knowledge.obs import metrics, traces
from hal9000.knowledge.obs.metrics import REGISTRY
from hal9000.knowledge.obs.timing import stage
from hal9000.knowledge.obs.traces import (
    ERROR_MESSAGE_CHARS,
    RETENTION_DAYS,
    RetentionJob,
    describe_error,
    finish_trace,
    hits_from_retrieval,
    load_trace,
    mark_first_token,
    observe,
    purge_traces,
    retrieval_seconds,
    top_score,
    write_trace,
)

QUESTION = "When was the synthetic water heater replaced?"


def _get(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def _chunk(chunk_id: int, rank: int, **scores) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        text="synthetic",
        page_start=1,
        page_end=1,
        tokens=10,
        rank=rank,
        score=scores.pop("score", 1 / (60 + rank)),
        **scores,
    )


def test_an_observed_question_is_written_with_every_field(kb_db):
    answered = _get("kb_questions_total", {"outcome": "answered"})
    ttft = _get("kb_ttft_seconds_count")
    retrieval = _get("kb_retrieval_seconds_count")
    result = RetrievalResult(
        query=QUESTION,
        standalone_query=QUESTION,
        profile="fixed-512",
        candidates=[_chunk(7, 1, dense_score=0.71), _chunk(9, 2, dense_score=0.52)],
        selected=[_chunk(7, 1, dense_score=0.71)],
        context_tokens=640,
    )
    with observe(kb_db, question=QUESTION, standalone_query=QUESTION, profile="fixed-512") as trace:
        with stage("embed_query"):
            pass
        with stage("select"):
            with stage("vector"):
                time.sleep(0.002)
            with stage("bm25"):
                pass
        with stage("generate"):
            mark_first_token()
            time.sleep(0.001)
            mark_first_token()  # only the first call counts
        trace.hits = hits_from_retrieval(result)
        trace.context_tokens = result.context_tokens
        trace.tokens_by_model = {"fake-chat": {"prompt": 700, "cached": 0, "completion": 60}}
        trace.cost_usd = 0.0004
        trace.outcome = "answered"
        trace.grounding = "verified"

    row = kb_db.execute("SELECT * FROM traces WHERE id = ?", (trace.id,)).fetchone()
    assert row["kind"] == "question" and row["outcome"] == "answered"
    assert row["profile"] == "fixed-512"
    assert row["total_ms"] >= row["ttft_ms"] > 0
    assert row["cost_usd"] == pytest.approx(0.0004)
    assert datetime.fromisoformat(row["created_at"]).tzinfo is not None
    # no scalar column holds question text; only the body (for the Query Inspector) does
    assert QUESTION not in json.dumps([row[k] for k in row.keys() if k != "body"])

    stored = load_trace(kb_db, trace.id)
    assert stored.model_dump() == trace.model_dump()
    assert [(h.chunk_id, h.rank, h.selected) for h in stored.hits] == [(7, 1, True), (9, 2, False)]
    assert stored.hits[0].dense_score == 0.71
    assert [t.stage.value for t in stored.stages] == [
        "embed_query",
        "vector",
        "bm25",
        "select",
        "generate",
    ]
    assert stored.grounding == "verified" and stored.context_tokens == 640
    assert stored.tokens_by_model["fake-chat"]["completion"] == 60

    assert _get("kb_questions_total", {"outcome": "answered"}) == answered + 1
    assert _get("kb_ttft_seconds_count") == ttft + 1
    assert _get("kb_retrieval_seconds_count") == retrieval + 1


def test_retrieval_seconds_counts_nested_stages_once():
    trace = Trace()
    with traces.start_trace(trace):
        with stage("select"):
            with stage("vector"):
                time.sleep(0.003)
        with stage("fuse"):
            pass
        with stage("generate"):
            time.sleep(0.002)
    select = next(t for t in trace.stages if t.stage.value == "select")
    fuse = next(t for t in trace.stages if t.stage.value == "fuse")
    assert retrieval_seconds(trace) == pytest.approx((select.duration_ms + fuse.duration_ms) / 1000)
    assert retrieval_seconds(Trace()) is None


def test_an_exception_marks_the_trace_and_counts_the_innermost_stage_once(kb_db):
    errors = _get("kb_errors_total", {"stage": "vector", "kind": "RuntimeError"})
    outer = _get("kb_errors_total", {"stage": "select", "kind": "RuntimeError"})
    failed = _get("kb_questions_total", {"outcome": "error"})
    with pytest.raises(RuntimeError):
        with observe(kb_db, question=QUESTION) as trace:
            with stage("select"):
                with stage("vector"):
                    raise RuntimeError("vec0 exploded")
    assert trace.outcome == "error" and trace.error == "RuntimeError: vec0 exploded"
    assert (
        kb_db.execute("SELECT outcome FROM traces WHERE id = ?", (trace.id,)).fetchone()[0]
        == "error"
    )
    assert _get("kb_errors_total", {"stage": "vector", "kind": "RuntimeError"}) == errors + 1
    assert _get("kb_errors_total", {"stage": "select", "kind": "RuntimeError"}) == outer
    assert _get("kb_questions_total", {"outcome": "error"}) == failed + 1


def test_an_error_outside_any_stage_is_counted_as_request():
    before = _get("kb_errors_total", {"stage": "request", "kind": "KeyError"})
    with pytest.raises(KeyError):
        with observe(question=QUESTION):
            raise KeyError("x")
    assert _get("kb_errors_total", {"stage": "request", "kind": "KeyError"}) == before + 1


def test_a_handled_stage_failure_is_counted_but_the_answer_stands(kb_db):
    """Embedding fails, the pipeline falls back to BM25-only (plan §7.4)."""
    before = _get("kb_errors_total", {"stage": "embed_query", "kind": "ConnectionError"})
    with observe(kb_db, question=QUESTION) as trace:
        try:
            with stage("embed_query"):
                raise ConnectionError
        except ConnectionError:
            pass
        with stage("bm25"):
            pass
        trace.outcome = "answered"
    assert (
        _get("kb_errors_total", {"stage": "embed_query", "kind": "ConnectionError"}) == before + 1
    )
    assert load_trace(kb_db, trace.id).outcome == "answered"


def test_a_handled_failure_reported_again_is_counted_once(kb_db):
    """The retriever catches the embedding error outside the stage and reports it too."""
    labels = {"stage": "embed_query", "kind": "ConnectionError"}
    before = _get("kb_errors_total", labels)
    with observe(kb_db, question=QUESTION) as trace:
        try:
            with stage("embed_query"):
                raise ConnectionError
        except ConnectionError as error:
            metrics.record_error("embed_query", error)  # already in the trace: not again
            metrics.record_error("vector", error)  # a stage that didn't fail: counted
        trace.outcome = "answered"
    assert _get("kb_errors_total", labels) == before + 1
    # outside an observed request every report counts
    metrics.record_error("embed_query", ConnectionError())
    assert _get("kb_errors_total", labels) == before + 2


def test_ingest_traces_do_not_count_as_questions(kb_db):
    before = sum(_get("kb_questions_total", {"outcome": o}) for o in ("answered", "error"))
    with observe(kb_db, kind="ingest") as trace:
        with stage("parse"):
            pass
    assert trace.outcome is None
    assert sum(_get("kb_questions_total", {"outcome": o}) for o in ("answered", "error")) == before
    assert (
        kb_db.execute("SELECT kind FROM traces WHERE id = ?", (trace.id,)).fetchone()[0] == "ingest"
    )


def test_top_score_is_the_abstain_score_never_rrf():
    assert top_score(Trace()) is None
    dense = Trace(
        hits=[
            TraceHit(chunk_id=2, rank=2, score=0.03, dense_score=0.9),
            TraceHit(chunk_id=1, rank=1, score=0.016, dense_score=0.6),
        ]
    )
    assert top_score(dense) == 0.6
    reranked = Trace(
        hits=[TraceHit(chunk_id=1, rank=1, score=0.016, dense_score=0.6, rerank_score=3.2)]
    )
    assert top_score(reranked) == 3.2
    bm25_only = Trace(hits=[TraceHit(chunk_id=1, rank=1, score=0.016, bm25_score=11.0)])
    assert top_score(bm25_only) is None


def test_finish_trace_observes_top_score_and_context_only_when_known():
    scores, contexts = _get("kb_top_score_count"), _get("kb_context_tokens_count")
    finish_trace(
        Trace(outcome="abstained", hits=[TraceHit(chunk_id=1, rank=1, score=0.01, dense_score=0.2)])
    )
    assert _get("kb_top_score_count") == scores + 1
    assert _get("kb_context_tokens_count") == contexts  # nothing was sent to the model
    finish_trace(Trace(outcome="answered", context_tokens=500), top_score_value=0.8)
    assert _get("kb_top_score_count") == scores + 2
    assert _get("kb_context_tokens_count") == contexts + 1
    finish_trace(Trace(error="Boom"))  # no outcome given: an error


def test_a_failed_write_is_logged_counted_and_not_raised(tmp_path, caplog):
    labels = {"stage": "request", "kind": "TraceWriteError"}
    before = _get("kb_errors_total", labels)
    conn = sqlite3.connect(tmp_path / "nothing.db")  # no traces table
    finish_trace(Trace(outcome="answered"), conn)
    assert "could not write trace" in caplog.text
    assert _get("kb_errors_total", labels) == before + 1


def _insert(conn, trace_id: str, created: datetime) -> None:
    write_trace(conn, Trace(id=trace_id, created_at=created, outcome="answered"))


def test_purge_deletes_rows_older_than_180_days(kb_db):
    now = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    _insert(kb_db, "old", now - timedelta(days=RETENTION_DAYS, seconds=1))
    _insert(kb_db, "edge", now - timedelta(days=RETENTION_DAYS) + timedelta(seconds=1))
    _insert(kb_db, "new", now - timedelta(days=1))
    # a row written in another offset still compares by instant, not by text
    kb_db.execute(
        "INSERT INTO traces (id, created_at, body) VALUES ('offset', ?, '{}')",
        ((now - timedelta(days=200)).astimezone(timezone(timedelta(hours=-7))).isoformat(),),
    )
    kb_db.commit()
    assert purge_traces(kb_db, now=now) == 2
    left = {r[0] for r in kb_db.execute("SELECT id FROM traces")}
    assert left == {"edge", "new"}
    with pytest.raises(ValueError):
        purge_traces(kb_db, days=0)


def test_the_retention_job_runs_at_most_once_per_interval(kb_db, monkeypatch):
    now = datetime.now(UTC)
    _insert(kb_db, "ancient", now - timedelta(days=400))
    job = RetentionJob(interval_s=3600)
    assert job.maybe_run(kb_db) == 1
    _insert(kb_db, "ancient-2", now - timedelta(days=400))
    assert job.maybe_run(kb_db) is None  # within the interval
    clock = time.monotonic() + 7200
    monkeypatch.setattr(traces.time, "monotonic", lambda: clock)
    assert job.maybe_run(kb_db) == 1


def test_writing_a_trace_runs_retention(kb_db, monkeypatch):
    _insert(kb_db, "ancient", datetime.now(UTC) - timedelta(days=365))
    monkeypatch.setattr(traces, "_retention", RetentionJob())
    finish_trace(Trace(outcome="answered"), kb_db)
    assert kb_db.execute("SELECT COUNT(*) FROM traces WHERE id = 'ancient'").fetchone()[0] == 0


def test_naive_timestamps_are_treated_as_utc(kb_db):
    write_trace(kb_db, Trace(id="naive", created_at=datetime(2026, 1, 1, 12, 0)))
    assert (
        kb_db.execute("SELECT created_at FROM traces WHERE id='naive'").fetchone()[0]
        == "2026-01-01T12:00:00.000+00:00"
    )
    assert load_trace(kb_db, "missing") is None


def test_mark_first_token_needs_a_started_trace():
    unstarted = Trace()
    mark_first_token(unstarted)
    assert unstarted.ttft_ms is None
    mark_first_token()  # nothing bound: a no-op


# ------------------------------------------------------------ retention pin
def test_retention_is_180_days(kb_db):
    """plan §12, with dates written out rather than derived from the constant."""
    assert RETENTION_DAYS == 180
    now = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    _insert(kb_db, "day-181", datetime(2026, 3, 31, 12, 0, tzinfo=UTC))  # 181 days before
    _insert(kb_db, "day-179", datetime(2026, 4, 2, 12, 0, tzinfo=UTC))  # 179 days before
    assert purge_traces(kb_db, now=now) == 1
    assert {r[0] for r in kb_db.execute("SELECT id FROM traces")} == {"day-179"}
    assert RetentionJob().days == 180


def test_retention_reads_trace_retention_days_from_knowledge_yaml(kb_db, tmp_path, caplog):
    yaml_path = tmp_path / "knowledge.yaml"
    job = RetentionJob(yaml_path=yaml_path, interval_s=0)
    assert job.days == RETENTION_DAYS  # no file: the default
    yaml_path.write_text("trace_retention_days: 30\n")
    assert job.days == 30  # read when the job runs, so an edit applies at the next purge
    now = datetime.now(UTC)
    _insert(kb_db, "day-31", now - timedelta(days=31))
    _insert(kb_db, "day-29", now - timedelta(days=29))
    assert job.maybe_run(kb_db) == 1
    assert {r[0] for r in kb_db.execute("SELECT id FROM traces")} == {"day-29"}
    for bad in ("trace_retention_days: 0\n", "trace_retention_days: [\n", "no_such_key: 1\n"):
        yaml_path.write_text(bad)
        assert job.days == RETENTION_DAYS  # never breaks a request
    assert "keeping 180 days" in caplog.text
    assert RetentionJob(days=7, yaml_path=yaml_path).days == 7


def test_the_service_points_retention_at_its_own_knowledge_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(traces, "_retention", traces._retention)  # restored afterwards
    traces.configure_retention(tmp_path / "knowledge.yaml")
    (tmp_path / "knowledge.yaml").write_text("trace_retention_days: 90\n")
    assert traces._retention.days == 90


# ------------------------------------------------------------ cancellation
def test_a_closed_stream_is_cancelled_not_answered(kb_db):
    cancelled = _get("kb_questions_total", {"outcome": "cancelled"})
    answered = _get("kb_questions_total", {"outcome": "answered"})
    exit_errors = _get("kb_errors_total", {"stage": "generate", "kind": "GeneratorExit"})
    seen = {}

    def stream():
        with observe(kb_db, question=QUESTION) as trace:
            seen["trace"] = trace
            with stage("generate"):
                trace.outcome = "answered"  # set early, as a streaming handler might
                yield "first"
                yield "second"  # pragma: no cover - the client is gone

    tokens = stream()
    assert next(tokens) == "first"
    tokens.close()  # the client disconnected: GeneratorExit at the yield

    trace = seen["trace"]
    assert _get("kb_questions_total", {"outcome": "cancelled"}) == cancelled + 1
    assert _get("kb_questions_total", {"outcome": "answered"}) == answered
    assert _get("kb_errors_total", {"stage": "generate", "kind": "GeneratorExit"}) == exit_errors
    row = kb_db.execute("SELECT outcome, body FROM traces WHERE id = ?", (trace.id,)).fetchone()
    assert row["outcome"] == "cancelled"
    body = json.loads(row["body"])
    assert body["outcome"] is None and body["error"] == "GeneratorExit"
    assert load_trace(kb_db, trace.id).outcome is None  # the body still validates


def test_task_cancellation_is_recorded_and_re_raised(kb_db):
    cancelled = _get("kb_questions_total", {"outcome": "cancelled"})
    ids = []

    async def answer():
        with observe(kb_db, question=QUESTION) as trace:
            ids.append(trace.id)
            with stage("generate"):
                await asyncio.sleep(10)

    async def main():
        task = asyncio.create_task(answer())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert _get("kb_questions_total", {"outcome": "cancelled"}) == cancelled + 1
    assert (
        kb_db.execute("SELECT outcome FROM traces WHERE id = ?", (ids[0],)).fetchone()[0]
        == "cancelled"
    )


def test_a_real_error_before_cancellation_is_still_counted():
    before = _get("kb_errors_total", {"stage": "embed_query", "kind": "ConnectionError"})
    with pytest.raises(KeyboardInterrupt):
        with observe(question=QUESTION):
            try:
                with stage("embed_query"):
                    raise ConnectionError
            except ConnectionError:
                pass
            raise KeyboardInterrupt
    assert (
        _get("kb_errors_total", {"stage": "embed_query", "kind": "ConnectionError"}) == before + 1
    )


def test_cancelled_ingest_traces_are_not_questions(kb_db):
    before = _get("kb_questions_total", {"outcome": "cancelled"})
    with pytest.raises(KeyboardInterrupt):
        with observe(kb_db, kind="ingest") as trace:
            raise KeyboardInterrupt
    assert _get("kb_questions_total", {"outcome": "cancelled"}) == before
    assert (
        kb_db.execute("SELECT outcome FROM traces WHERE id = ?", (trace.id,)).fetchone()[0]
        == "cancelled"
    )


# ------------------------------------------------------------ error text
def test_describe_error_keeps_the_type_and_200_characters():
    assert describe_error(ValueError()) == "ValueError"
    assert describe_error(ValueError("bad\n  pdf")) == "ValueError: bad pdf"
    long = describe_error(RuntimeError("x" * 500))
    assert long == "RuntimeError: " + "x" * ERROR_MESSAGE_CHARS
    assert ERROR_MESSAGE_CHARS == 200


def test_trace_error_is_bounded_and_the_metric_kind_is_the_type():
    before = _get("kb_errors_total", {"stage": "request", "kind": "ValueError"})
    with pytest.raises(ValueError):
        with observe(question=QUESTION) as trace:
            raise ValueError("y" * 1000)
    assert len(trace.error) == len("ValueError: ") + 200
    assert _get("kb_errors_total", {"stage": "request", "kind": "ValueError"}) == before + 1


# ------------------------------------------------------------ transactions
def _other(db_path):
    other = sqlite3.connect(db_path)
    try:
        return {r[0] for r in other.execute("SELECT id FROM traces")}
    finally:
        other.close()


def test_write_trace_joins_the_callers_transaction_without_committing_it(kb_db, tmp_path):
    db_path = tmp_path / "knowledge.db"
    kb_db.execute(
        "INSERT INTO traces (id, created_at, body) VALUES ('callers', '2026-01-01', '{}')"
    )
    assert kb_db.in_transaction
    write_trace(kb_db, Trace(id="mine", outcome="answered"))
    assert kb_db.in_transaction  # still the caller's transaction
    assert _other(db_path) == set()  # nothing committed
    kb_db.rollback()
    assert {r[0] for r in kb_db.execute("SELECT id FROM traces")} == set()


def test_a_failed_write_rolls_back_only_itself(kb_db, monkeypatch):
    kb_db.execute(
        "INSERT INTO traces (id, created_at, body) VALUES ('callers', '2026-01-01', '{}')"
    )
    monkeypatch.setattr(
        traces, "trace_row", lambda trace, outcome=None: {"id": "x", "no_such_column": 1}
    )
    with pytest.raises(sqlite3.OperationalError):
        write_trace(kb_db, Trace())
    assert kb_db.in_transaction
    assert {r[0] for r in kb_db.execute("SELECT id FROM traces")} == {"callers"}
    kb_db.commit()


def test_without_a_transaction_the_write_is_committed(kb_db, tmp_path, monkeypatch):
    write_trace(kb_db, Trace(id="solo"))
    assert not kb_db.in_transaction
    assert _other(tmp_path / "knowledge.db") == {"solo"}
    monkeypatch.setattr(
        traces, "trace_row", lambda trace, outcome=None: {"id": "x", "no_such_column": 1}
    )
    with pytest.raises(sqlite3.OperationalError):
        write_trace(kb_db, Trace())
    assert not kb_db.in_transaction


def test_purge_inside_the_callers_transaction(kb_db, tmp_path):
    write_trace(kb_db, Trace(id="ancient", created_at=datetime(2020, 1, 1, tzinfo=UTC)))
    kb_db.execute(
        "INSERT INTO traces (id, created_at, body) VALUES ('callers', ?, '{}')",
        (datetime.now(UTC).isoformat(),),
    )
    assert purge_traces(kb_db) == 1
    assert kb_db.in_transaction
    assert _other(tmp_path / "knowledge.db") == {"ancient"}  # the delete isn't committed either
    kb_db.commit()
    assert _other(tmp_path / "knowledge.db") == {"callers"}
