"""Trace rows, their metrics and the 180-day retention (plan §9.1, §12)."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from hal9000.knowledge.contracts import RetrievalResult, RetrievedChunk, Trace, TraceHit
from hal9000.knowledge.obs import traces
from hal9000.knowledge.obs.metrics import REGISTRY
from hal9000.knowledge.obs.timing import stage
from hal9000.knowledge.obs.traces import (
    RETENTION_DAYS,
    RetentionJob,
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
    assert trace.outcome == "error" and trace.error == "RuntimeError"
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


def test_a_failed_write_is_logged_not_raised(tmp_path, caplog):
    conn = sqlite3.connect(tmp_path / "nothing.db")  # no traces table
    finish_trace(Trace(outcome="answered"), conn)
    assert "could not write trace" in caplog.text


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
