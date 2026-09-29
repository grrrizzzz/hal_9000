"""Every plan §9.2 metric on /metrics with the right type and labels (handoff KWP8)."""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from hal9000.cost.metering import usage_cost
from hal9000.knowledge.contracts import Trace
from hal9000.knowledge.obs import metrics
from hal9000.knowledge.obs.metrics import REGISTRY
from hal9000.knowledge.obs.timing import stage, start_trace
from hal9000.knowledge.obs.traces import observe
from tests.knowledge.obs.conftest import add_eval_run, seed_corpus

#: Plan §9.2, written out independently of metrics.METRICS:
#: exposition family name -> (type, label names).
PLAN = {
    "kb_stage_seconds": ("histogram", {"stage"}),
    "kb_retrieval_seconds": ("histogram", set()),
    "kb_ttft_seconds": ("histogram", set()),
    "kb_answer_seconds": ("histogram", set()),
    "kb_questions": ("counter", {"outcome"}),
    "kb_errors": ("counter", {"stage", "kind"}),
    "kb_tokens": ("counter", {"model", "purpose"}),
    "kb_cost_usd": ("counter", {"model", "purpose"}),
    "kb_month_spend_usd": ("gauge", set()),
    "kb_month_cap_usd": ("gauge", set()),
    "kb_top_score": ("histogram", set()),
    "kb_context_tokens": ("histogram", set()),
    "kb_grounding_failures": ("counter", {"kind"}),
    "kb_feedback": ("counter", {"rating"}),
    "kb_ingest_files": ("counter", {"status"}),
    "kb_ingest_pages": ("counter", {"text_layer"}),
    "kb_documents": ("gauge", {"profile"}),
    "kb_chunks": ("gauge", {"profile"}),
    "kb_db_bytes": ("gauge", {"profile"}),
    "kb_regression_recall": ("gauge", set()),
    "kb_eval_metric": ("gauge", {"profile", "metric"}),
}

SECRET = "zqx-synthetic-question-marker"


def _get(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def _scrape(client: TestClient) -> dict:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return response.text


def _families(text: str) -> dict:
    return {family.name: family for family in text_string_to_metric_families(text)}


@pytest.fixture
def client(db_path, kb_db):
    app = FastAPI()
    app.include_router(metrics.metrics_router(db_path))
    return TestClient(app)


def _exercise_every_helper(kb_db) -> None:
    seed_corpus(kb_db)
    with kb_db:
        kb_db.execute(
            "INSERT INTO regression_runs (at, trigger, recall_2k, n_questions)"
            " VALUES ('2026-09-01T00:00:00+00:00', 'ingest', 0.9, 40)"
        )
    add_eval_run(
        kb_db,
        "run-1",
        "fixed-512",
        {"n_questions": 40, "mrr": {"value": 0.5, "low": 0.4, "high": 0.6}},
    )
    with observe(kb_db, question=SECRET, standalone_query=SECRET, profile="fixed-512") as trace:
        with stage("select"):
            with stage("vector"):
                pass
        metrics.record_usage(
            {"model": "fake-chat", "prompt": 10, "completion": 5}, "answer", cost_usd=0.001
        )
        trace.context_tokens = 900
        trace.outcome = "answered"
    metrics.record_error("parse", ValueError("boom " + SECRET))
    metrics.record_grounding_failure("numeral")
    metrics.record_feedback(1)
    metrics.record_ingest_file("added")
    metrics.record_ingest_pages(with_text_layer=3, without_text_layer=1)
    metrics.set_month_budget(1.5, 10)


def test_every_plan_metric_is_exposed_with_its_labels(client, kb_db):
    _exercise_every_helper(kb_db)
    families = _families(_scrape(client))
    for name, (kind, labels) in PLAN.items():
        assert name in families, f"{name} missing from /metrics"
        family = families[name]
        assert family.type == kind, name
        assert family.samples, f"{name} has no samples"
        for sample in family.samples:
            assert set(sample.labels) - {"le"} == labels, (name, sample)


def test_the_module_table_matches_the_plan():
    exposed = {
        (name[: -len("_total")] if kind == "counter" else name): (kind, set(labels))
        for name, (kind, labels) in metrics.METRICS.items()
    }
    assert exposed == PLAN


def test_metrics_never_expose_question_text(client, kb_db):
    _exercise_every_helper(kb_db)
    text = _scrape(client)
    assert SECRET not in text
    assert "boom" not in text  # an exception's message never becomes a label
    assert _families(text)["kb_errors"].samples


def test_process_metrics_are_served_but_not_in_the_kb_registry(client):
    text = _scrape(client)
    assert "process_cpu_seconds_total" in text or "python_info" in text
    assert not [m for m in REGISTRY.collect() if not m.name.startswith("kb_")]


def test_closed_label_sets_start_at_zero(client):
    families = _families(_scrape(client))
    outcomes = {
        s.labels["outcome"] for s in families["kb_questions"].samples if s.name.endswith("_total")
    }
    assert outcomes == {"answered", "abstained", "retrieval_only", "error"}
    assert {s.labels["status"] for s in families["kb_ingest_files"].samples} >= {
        "added",
        "duplicate",
        "skipped",
        "failed",
    }


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.record_question("maybe"),
        lambda: metrics.record_question("answered", ttft_s=-1),
        lambda: metrics.record_question("answered", context_tokens=-1),
        lambda: metrics.record_usage({"model": "m", "prompt": 1}, "chat"),
        lambda: metrics.record_usage({"model": "m", "prompt": -1}, "answer", cost_usd=0),
        lambda: metrics.record_error("retrieve", "X"),
        lambda: metrics.record_grounding_failure("date"),
        lambda: metrics.record_feedback(0),
        lambda: metrics.record_feedback("meh"),
        lambda: metrics.record_ingest_file("ok"),
        lambda: metrics.record_ingest_pages(with_text_layer=-1),
        lambda: metrics.set_month_budget(-1, 10),
    ],
)
def test_helpers_reject_values_outside_the_closed_sets(call):
    with pytest.raises(ValueError):
        call()


def test_record_question_observes_each_histogram():
    before = {
        n: _get(f"{n}_count")
        for n in (
            "kb_ttft_seconds",
            "kb_answer_seconds",
            "kb_retrieval_seconds",
            "kb_top_score",
            "kb_context_tokens",
        )
    }
    q_before = _get("kb_questions_total", {"outcome": "abstained"})
    metrics.record_question(
        "abstained", ttft_s=0.5, answer_s=2, retrieval_s=0.1, top_score=0.2, context_tokens=100
    )
    assert _get("kb_questions_total", {"outcome": "abstained"}) == q_before + 1
    for name, count in before.items():
        assert _get(f"{name}_count") == count + 1, name
    # omitted values are not observed
    metrics.record_question("error")
    assert _get("kb_ttft_seconds_count") == before["kb_ttft_seconds"] + 1


def test_record_usage_prices_like_the_ledger_and_fills_the_trace():
    usage = {"model": "openai/gpt-4.1-mini", "prompt": 1200, "cached": 200, "completion": 300}
    labels = {"model": "openai/gpt-4.1-mini", "purpose": "answer"}
    tokens, cost = _get("kb_tokens_total", labels), _get("kb_cost_usd_total", labels)
    with start_trace(question="q") as trace:
        spent = metrics.record_usage(usage, "answer")
        metrics.record_usage(usage, "answer", cost_usd=0.5)
    assert spent == pytest.approx(usage_cost(usage))
    assert _get("kb_tokens_total", labels) == tokens + 3000
    assert _get("kb_cost_usd_total", labels) == pytest.approx(cost + spent + 0.5)
    assert trace.tokens_by_model == {
        "openai/gpt-4.1-mini": {"prompt": 2400, "cached": 400, "completion": 600}
    }
    assert trace.cost_usd == pytest.approx(spent + 0.5)


def test_record_usage_without_a_trace_and_with_a_strange_model_name():
    metrics.record_usage({"model": "we ird\nmodel", "prompt": 1}, "embed", cost_usd=0)
    assert _get("kb_tokens_total", {"model": "weirdmodel", "purpose": "embed"}) >= 1
    metrics.record_usage({"prompt": 1}, "embed", cost_usd=0)
    assert _get("kb_tokens_total", {"model": "unknown", "purpose": "embed"}) >= 1


def test_error_kinds_are_type_names():
    before = _get("kb_errors_total", {"stage": "generate", "kind": "TimeoutError"})
    metrics.record_error("generate", TimeoutError("the question text here"))
    metrics.record_error("generate", TimeoutError)
    metrics.record_error("generate", "TimeoutError")
    assert _get("kb_errors_total", {"stage": "generate", "kind": "TimeoutError"}) == before + 3
    metrics.record_error("request", "!!!")
    assert _get("kb_errors_total", {"stage": "request", "kind": "unknown"}) >= 1


def test_feedback_accepts_the_table_values_and_names():
    up, down = (
        _get("kb_feedback_total", {"rating": "up"}),
        _get("kb_feedback_total", {"rating": "down"}),
    )
    metrics.record_feedback(1)
    metrics.record_feedback("up")
    metrics.record_feedback(-1)
    assert _get("kb_feedback_total", {"rating": "up"}) == up + 2
    assert _get("kb_feedback_total", {"rating": "down"}) == down + 1


def test_month_budget_can_be_read_at_scrape_time():
    spend = {"value": 1.0}
    metrics.bind_month_budget(lambda: spend["value"], 10)
    assert _get("kb_month_spend_usd") == 1.0 and _get("kb_month_cap_usd") == 10
    spend["value"] = 8.5
    assert _get("kb_month_spend_usd") == 8.5
    metrics.bind_month_budget(lambda: 1 / 0, lambda: 12.0)
    assert math.isnan(REGISTRY.get_sample_value("kb_month_spend_usd", {}))
    assert _get("kb_month_cap_usd") == 12
    metrics.set_month_budget(0, 10)  # set() replaces the bound function
    assert _get("kb_month_spend_usd") == 0


def test_install_db_collector_replaces_and_removes(db_path):
    first = metrics.install_db_collector(db_path)
    second = metrics.install_db_collector(db_path)
    assert first is not second
    assert [m.name for m in REGISTRY.collect()].count("kb_regression_recall") == 1
    assert metrics.install_db_collector(None) is None
    assert "kb_regression_recall" not in [m.name for m in REGISTRY.collect()]


def test_render_without_process_metrics():
    assert b"process_" not in metrics.render_metrics(process=False)


def test_the_router_matches_the_api_contract_stub():
    """KWP6 swaps the 501 stub for this router; the OpenAPI operation stays the same."""
    from hal9000.knowledge.api.app import create_app

    stub = create_app().openapi()["paths"]["/metrics"]["get"]
    app = FastAPI()
    app.include_router(metrics.metrics_router())
    mine = app.openapi()["paths"]["/metrics"]["get"]
    stub_responses = {k: v for k, v in stub["responses"].items() if k != "501"}
    assert mine["operationId"].split("_")[0] == stub["operationId"].split("_")[0] == "metrics"
    assert mine["tags"] == stub["tags"]
    assert (
        mine["responses"]["200"]["content"]["text/plain"]
        == stub_responses["200"]["content"]["text/plain"]
    )


def test_a_bound_trace_is_not_required(client):
    assert Trace().kind == "question"
    assert _scrape(client)
