"""The shared ``stage()`` timer and the closed stage enum (plan §9.1–9.2, rule K7)."""

from __future__ import annotations

import threading
import time

import pytest

from hal9000.knowledge.contracts import Trace
from hal9000.knowledge.obs import timing
from hal9000.knowledge.obs.timing import REGISTRY, Stage, current_trace, stage, start_trace

#: Plan §9.2, written out independently of the enum.
PLAN_STAGES = [
    "condense",
    "embed_query",
    "vector",
    "bm25",
    "fuse",
    "rerank",
    "select",
    "abstain",
    "generate",
    "verify",
    "screen",
    "store",
    "parse",
    "clean",
    "segment",
    "chunk",
    "embed",
    "index",
    "regression",
]


def _count(name: str) -> float:
    return REGISTRY.get_sample_value("kb_stage_seconds_count", {"stage": name}) or 0.0


def _sum(name: str) -> float:
    return REGISTRY.get_sample_value("kb_stage_seconds_sum", {"stage": name}) or 0.0


def test_the_enum_is_exactly_the_plan_set():
    assert [s.value for s in Stage] == PLAN_STAGES
    assert [s.value for s in timing.QUESTION_STAGES][-1] == "verify"
    assert [s.value for s in timing.INGEST_STAGES][0] == "screen"


def test_nested_stages_record_into_the_trace():
    with start_trace(question="How old is the furnace?") as trace:
        assert current_trace() is trace
        with stage("select"):
            with stage(Stage.VECTOR):
                time.sleep(0.002)
            with stage("bm25"):
                pass
    assert current_trace() is None

    names = [(t.stage.value, t.depth, t.parent) for t in trace.stages]
    # inner stages finish (and are recorded) before the outer one
    assert names == [("vector", 1, Stage.SELECT), ("bm25", 1, Stage.SELECT), ("select", 0, None)]
    vector, bm25, outer = trace.stages
    assert vector.duration_ms >= 2.0
    assert outer.duration_ms >= vector.duration_ms + bm25.duration_ms
    assert outer.started_ms <= vector.started_ms <= bm25.started_ms
    assert Trace.model_validate_json(trace.model_dump_json()).stages == trace.stages


def test_stage_increments_the_histogram():
    before_count, before_sum = _count("rerank"), _sum("rerank")
    with stage("rerank"):
        time.sleep(0.001)
    assert _count("rerank") == before_count + 1
    assert _sum("rerank") >= before_sum + 0.001
    assert "kb_stage_seconds" in {m.name for m in REGISTRY.collect()}


def test_an_unknown_stage_raises_and_records_nothing():
    with start_trace() as trace:
        with pytest.raises(ValueError, match="Unknown stage 'retrieve'"):
            with stage("retrieve"):
                pass  # pragma: no cover - never entered
    assert trace.stages == []
    assert REGISTRY.get_sample_value("kb_stage_seconds_count", {"stage": "retrieve"}) is None


def test_a_failing_stage_is_still_timed_and_the_error_propagates():
    before = _count("parse")
    with start_trace(kind="ingest") as trace:
        with pytest.raises(RuntimeError):
            with stage("parse"):
                raise RuntimeError("bad pdf")
    assert _count("parse") == before + 1
    assert trace.stages[0].error == "RuntimeError"


def test_stage_without_a_trace_only_observes_the_histogram():
    before = _count("embed")
    with stage("embed"):
        pass
    assert _count("embed") == before + 1


def test_traces_do_not_mix_across_threads():
    results: dict[str, list[str]] = {}
    barrier = threading.Barrier(2)

    def work(name: str, stages: list[str]) -> None:
        with start_trace(question=name) as trace:
            for s in stages:
                barrier.wait()
                with stage(s):
                    pass
        results[name] = [t.stage.value for t in trace.stages]

    a = threading.Thread(target=work, args=("a", ["condense", "vector"]))
    b = threading.Thread(target=work, args=("b", ["chunk", "index"]))
    a.start(), b.start()
    a.join(), b.join()
    assert results == {"a": ["condense", "vector"], "b": ["chunk", "index"]}


def test_start_trace_accepts_an_existing_trace():
    mine = Trace(question="q")
    with start_trace(mine) as bound:
        with stage("generate"):
            pass
    assert bound is mine and mine.stages[0].stage == Stage.GENERATE
