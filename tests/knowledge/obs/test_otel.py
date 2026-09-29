"""Optional OpenTelemetry export: one root span per request, a child per stage (plan §9.1)."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from hal9000.knowledge.contracts import Trace  # noqa: E402
from hal9000.knowledge.obs import otel  # noqa: E402
from hal9000.knowledge.obs.timing import stage  # noqa: E402
from hal9000.knowledge.obs.traces import observe  # noqa: E402

QUESTION = "What did the synthetic roofer quote?"


@pytest.fixture
def exporter():
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    assert otel.configure(provider) is True
    yield memory
    provider.shutdown()


def _chat(conn=None):
    with observe(conn, question=QUESTION, profile="fixed-512") as trace:
        with stage("embed_query"):
            pass
        with stage("select"):
            with stage("vector"):
                time.sleep(0.001)
            with stage("bm25"):
                pass
            with stage("fuse"):
                pass
        with stage("abstain"):
            pass
        with stage("generate"):
            time.sleep(0.001)
        with stage("verify"):
            pass
        trace.outcome = "answered"
    return trace


def test_one_chat_is_one_root_span_with_a_child_per_stage(exporter, kb_db):
    trace = _chat(kb_db)
    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.parent is None]
    assert len(roots) == 1
    root = roots[0]
    assert root.name == "kb.question"
    assert root.attributes["kb.trace_id"] == trace.id
    assert root.attributes["kb.outcome"] == "answered"

    children = [s for s in spans if s is not root]
    assert sorted(s.name for s in children) == sorted(t.stage.value for t in trace.stages)
    assert len(children) == len(trace.stages) == 8
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}
    by_name = {s.name: s for s in children}
    for name in ("embed_query", "select", "abstain", "generate", "verify"):
        assert by_name[name].parent.span_id == root.context.span_id, name
    for name in ("vector", "bm25", "fuse"):
        assert by_name[name].parent.span_id == by_name["select"].context.span_id, name
    for span in children:
        assert root.start_time <= span.start_time <= span.end_time <= root.end_time
    # no question text anywhere in the exported spans
    for span in spans:
        assert QUESTION not in repr(dict(span.attributes)) and QUESTION not in span.name


def test_a_failing_stage_sets_error_status(exporter):
    with pytest.raises(ValueError):
        with observe(question=QUESTION):
            with stage("generate"):
                raise ValueError("bad")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["generate"].status.status_code == StatusCode.ERROR
    assert spans["kb.question"].status.status_code == StatusCode.ERROR
    assert spans["kb.question"].attributes["kb.error"] == "ValueError"


def test_ingest_traces_get_an_ingest_root(exporter):
    with observe(kind="ingest"):
        with stage("parse"):
            pass
    assert {s.name for s in exporter.get_finished_spans()} == {"kb.ingest", "parse"}


def test_a_trace_without_total_uses_its_stages(exporter):
    from hal9000.knowledge.contracts import StageTiming

    trace = Trace(stages=[StageTiming(stage="vector", started_ms=1, duration_ms=5)])
    assert otel.export_trace(trace) is True
    root = next(s for s in exporter.get_finished_spans() if s.parent is None)
    assert root.end_time - root.start_time == 6_000_000


def test_export_is_off_without_an_endpoint(monkeypatch):
    monkeypatch.delenv(otel.ENDPOINT_ENV, raising=False)
    otel.reset()
    assert otel.enabled() is False
    assert otel.export_trace(Trace()) is False


@pytest.mark.parametrize("protocol", ["", "grpc"])
def test_an_endpoint_turns_export_on(monkeypatch, protocol):
    monkeypatch.setenv(otel.ENDPOINT_ENV, "http://127.0.0.1:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", protocol)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "hal-kb-test")
    otel.reset()
    try:
        assert otel.enabled() is True
        assert otel._provider.resource.attributes["service.name"] == "hal-kb-test"
    finally:
        otel._provider.shutdown()  # nothing was exported, so nothing is sent
        otel.reset()


def test_a_missing_sdk_is_a_warning_not_an_error(monkeypatch, caplog):
    import builtins

    real_import = builtins.__import__

    def no_otel(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv(otel.ENDPOINT_ENV, "http://127.0.0.1:4318")
    monkeypatch.setattr(builtins, "__import__", no_otel)
    otel.reset()
    assert otel.enabled() is False
    assert "not installed" in caplog.text
