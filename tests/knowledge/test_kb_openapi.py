"""The API contract: every plan §11 route, a valid schema, and no drift from the committed file.

KWP6 replaced the KWP0 stubs; the approved additions (``GET /api/eval/runs/{id}/results``,
``POST /api/jobs/{id}/retry``) are listed separately from the plan's routes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openapi_spec_validator import validate

from hal9000.knowledge.api import ROUTES, create_app

REPO = Path(__file__).resolve().parents[2]
COMMITTED = REPO / "web" / "src" / "api" / "openapi.json"
SHA = "ab" * 32

#: Plan §11, written out independently of api.ROUTES. `[/{id}]` expands to both forms.
PLAN_ROUTES = {
    ("POST", "/api/session"),
    ("DELETE", "/api/session"),
    ("POST", "/api/uploads"),
    ("GET", "/api/jobs"),
    ("GET", "/api/jobs/stream"),
    ("GET", "/api/documents"),
    ("GET", "/api/documents/{id}"),
    ("PATCH", "/api/documents/{id}"),
    ("DELETE", "/api/documents/{id}"),
    ("POST", "/api/documents/{id}/reindex"),
    ("GET", "/api/files/{sha}"),
    ("GET", "/api/files/{sha}/pages/{n}"),
    ("POST", "/api/chat"),
    ("GET", "/api/conversations"),
    ("GET", "/api/conversations/{id}"),
    ("DELETE", "/api/conversations"),
    ("DELETE", "/api/conversations/{id}"),
    ("POST", "/api/feedback"),
    ("GET", "/api/search"),
    ("GET", "/api/eval/gold"),
    ("GET", "/api/eval/gold/{id}"),
    ("PATCH", "/api/eval/gold/{id}"),
    ("GET", "/api/eval/judge-labels"),
    ("GET", "/api/eval/judge-labels/{id}"),
    ("PUT", "/api/eval/judge-labels/{id}"),
    ("GET", "/api/eval/runs"),
    ("GET", "/api/eval/runs/{id}"),
    ("GET", "/api/profile"),
    ("PUT", "/api/profile"),
    ("GET", "/api/traces"),
    ("GET", "/api/traces/{id}"),
    ("GET", "/health"),
    ("GET", "/metrics"),
}
#: Contract additions approved for KWP6 (web/src/api/pending-contract.ts).
APPROVED_ROUTES = {
    ("GET", "/api/eval/runs/{id}/results"),
    ("POST", "/api/jobs/{id}/retry"),
}

RUBRIC = {"correctness": "correct", "faithful": True, "citations_correct": True}
#: A valid request for each route, so the stub (not validation) answers.
REQUESTS = {
    ("POST", "/api/session"): dict(json={"passcode": "x"}),
    ("POST", "/api/uploads"): dict(
        files=[("files", ("a.txt", b"hello", "text/plain"))], data={"relative_path": "A/a.txt"}
    ),
    ("PATCH", "/api/documents/{id}"): dict(json={"title": "New"}),
    ("POST", "/api/chat"): dict(json={"message": "How old is the roof?"}),
    ("POST", "/api/feedback"): dict(json={"message_id": 1, "rating": 1}),
    ("GET", "/api/search"): dict(params={"q": "roof"}),
    ("PATCH", "/api/eval/gold/{id}"): dict(json={"status": "approved"}),
    ("PUT", "/api/eval/judge-labels/{id}"): dict(json={"human": RUBRIC}),
    ("PUT", "/api/profile"): dict(json={"name": "p"}),
}


@pytest.fixture(scope="module")
def schema() -> dict:
    return create_app().openapi()


def _served(schema: dict) -> set[tuple[str, str]]:
    return {(method.upper(), path) for path, ops in schema["paths"].items() for method in ops}


def test_the_schema_lists_every_plan_route_and_nothing_else(schema):
    assert _served(schema) == PLAN_ROUTES | APPROVED_ROUTES == set(ROUTES)


def test_the_committed_schema_is_valid_and_complete():
    committed = json.loads(COMMITTED.read_text())
    validate(committed)
    assert _served(committed) == PLAN_ROUTES | APPROVED_ROUTES


def test_the_committed_file_matches_the_app():
    exporter = _load_exporter()
    assert COMMITTED.read_text() == exporter.render(), (
        "web/src/api/openapi.json is stale: run `uv run python scripts/kb_export_openapi.py`"
    )


def test_the_validator_rejects_a_broken_schema(schema):
    broken = json.loads(json.dumps(schema))
    broken["paths"]["/api/chat"]["post"]["responses"] = "nope"
    with pytest.raises(Exception):  # noqa: B017 - any validation error will do
        validate(broken)


def test_models_from_contracts_are_in_the_schema(schema):
    components = set(schema["components"]["schemas"])
    for name in (
        "Answer",
        "Citation",
        "ChatEventEnvelope",
        "ChatResetEvent",
        "DocumentSummary",
        "GoldItem",
        "EvidenceQuote",
        "Job",
        "Trace",
        "Stage",
        "RetrievalResult",
        "EvalRun",
    ):
        assert name in components, name
    chat = schema["paths"]["/api/chat"]["post"]
    assert chat["responses"]["200"]["content"] == {
        "text/event-stream": {"schema": {"$ref": "#/components/schemas/ChatEventEnvelope"}}
    }
    assert chat["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ChatRequest"
    )
    ids = [op["operationId"] for ops in schema["paths"].values() for op in ops.values()]
    assert len(ids) == len(set(ids)) and "list_documents" in ids


@pytest.mark.parametrize(("method", "path"), sorted(PLAN_ROUTES | APPROVED_ROUTES))
def test_without_services_every_route_but_metrics_is_unavailable(method, path):
    """``create_app()`` alone (the OpenAPI export) serves no data; nothing is a stub any more."""
    client = TestClient(create_app())
    url = path.replace("{id}", "1").replace("{sha}", SHA).replace("{n}", "2")
    response = client.request(method, url, **REQUESTS.get((method, path), {}))
    if path == "/metrics":
        assert response.status_code == 200
        return
    assert response.status_code == 503, response.text
    assert "not configured" in response.json()["detail"]


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "kb_export_openapi", REPO / "scripts" / "kb_export_openapi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_script_writes_and_checks(tmp_path, capsys):
    exporter = _load_exporter()
    out = tmp_path / "openapi.json"
    assert exporter.main(["--check", "--out", str(out)]) == 1  # missing
    assert exporter.main(["--out", str(out)]) == 0
    assert exporter.main(["--check", "--out", str(out)]) == 0
    out.write_text(out.read_text().replace("HAL 9000 house knowledge", "drifted"))
    assert exporter.main(["--check", "--out", str(out)]) == 1
    assert "out of date" in capsys.readouterr().err
