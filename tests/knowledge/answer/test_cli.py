"""``kb ask``, ``kb serve`` and the evaluation answer binding."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pytest

from hal9000.knowledge.answer.eval_binding import binding
from hal9000.knowledge.answer.fake import ScriptedStream, fake_deps
from hal9000.knowledge.answer.pipeline import AnswerDeps, AnswerPipeline
from hal9000.knowledge.cli import ask as ask_cli
from hal9000.knowledge.cli import serve as serve_cli
from hal9000.knowledge.config import ConfigMissing
from hal9000.knowledge.contracts import EvalRun
from hal9000.knowledge.index.fake import enable_fake_mode
from tests.fakes import FakeLLM

PEST_WRONG = "The termite repair cost $1,324.00 [S1]."
PEST_RIGHT = "Invoice 1001 shows the termite repair cost $1,234.00 [S1]."


def test_ask_prints_the_answer_sources_verify_and_cost(settings, conn, embedder):
    stream = ScriptedStream([PEST_WRONG, PEST_RIGHT])
    pipeline = AnswerPipeline(
        settings, AnswerDeps(stream_fn=stream, generate_fn=FakeLLM([]), embed_fn=embedder)
    )
    out = io.StringIO()
    args = argparse.Namespace(question="How much was the termite repair?", trace=True)
    assert ask_cli.run(args, pipeline=pipeline, conn=conn, out=out) == 0
    text = out.getvalue()
    assert "[regenerating: $1,324.00 does not appear in [S1]]" in text
    assert PEST_RIGHT in text
    assert "[S1] Invoice 1001 (invoice, Example Pest Co., 2023-07-14), page 1" in text
    assert "grounding: figures verified" in text and "Cost: $0.0000" in text
    assert "Stages (ms):" in text and "generate" in text and "Retrieved" in text


def test_ask_without_an_active_profile_explains(settings, embedder, tmp_path, capsys):
    from hal9000.knowledge.storage import open_db

    empty = open_db(tmp_path / "empty.db")
    pipeline = AnswerPipeline(settings, AnswerDeps(stream_fn=ScriptedStream([])))
    args = argparse.Namespace(question="Anything?", trace=False)
    assert ask_cli.run(args, pipeline=pipeline, conn=empty, out=io.StringIO()) == 1
    assert "No index profile is active" in capsys.readouterr().err


def test_ask_refuses_without_a_passcode_free_config_error(monkeypatch, capsys):
    from hal9000 import config as hal_config

    def broken():
        raise hal_config.ConfigError("KB_CHAT_TIER='huge' must be one of: fast, powerful.")

    monkeypatch.setattr("hal9000.knowledge.config.settings", broken)
    args = argparse.Namespace(question="Anything?", trace=False)
    assert ask_cli.run(args) == 1
    assert "KB_CHAT_TIER" in capsys.readouterr().err


def test_serve_fake_seeds_the_synthetic_library_and_serves(monkeypatch, tmp_path, capsys):
    served = {}

    def fake_run(app, host, port, log_level):
        from fastapi.testclient import TestClient

        client = TestClient(app)
        assert client.post("/api/session", json={"passcode": "pod-bay-doors"}).status_code == 200
        docs = client.get("/api/documents").json()
        served["titles"] = {d["document"]["title"] for d in docs}
        answer = client.post("/api/chat", json={"message": "How old is the furnace?"})
        served["chat"] = answer.text
        served["where"] = (host, port)

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(serve_cli, "_temp_root", lambda: tmp_path / "fake")
    args = argparse.Namespace(host="127.0.0.1", port=8931, fake=True)
    try:
        assert serve_cli.run(args) == 0
    finally:
        enable_fake_mode(False)
    assert served["where"] == ("127.0.0.1", 8931)
    assert {"Home inspection report", "Invoice 1001 — termite repair"} <= served["titles"]
    done = [
        json.loads(line[5:]) for line in served["chat"].splitlines() if line.startswith("data:")
    ][-1]
    assert done["event"] == "done" and "installed in 2009" in done["answer"]["text"]
    assert done["answer"]["citations"][0]["page"] == 2  # the viewer opens the cited page
    assert done["answer"]["grounding"] == "verified"
    out = capsys.readouterr().out
    assert "fake mode" in out and "pod-bay-doors" in out
    assert not (tmp_path / "fake").exists()  # the temporary library is removed


def test_serve_without_a_passcode_refuses(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not serve"))
    args = argparse.Namespace(host="127.0.0.1", port=None, fake=False)
    assert serve_cli.run(args) == 1
    assert "KB_PASSCODE" in capsys.readouterr().err


# ------------------------------------------------------------------ eval binding
def test_the_eval_binding_answers_with_full_sources_and_no_traces(settings, conn, meter_factory):
    deps = fake_deps(delay_s=0)
    answer = binding(conn, settings, meter=meter_factory(), run=None, deps=deps)
    output = answer("How much was the termite repair?")
    assert "$1,234.00" in output.text and not output.abstained
    assert output.sources[0].marker == "S1" and output.sources[0].title == "Invoice 1001"
    assert "Total due: $1,234.00" in output.sources[0].text
    assert output.citations and output.citations[0].marker == "S1"
    assert conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0] == 0
    assert answer("Is there a swimming pool heater?").abstained


def test_the_eval_binding_meters_as_knowledge_eval(settings, conn, meter_factory):
    meter = meter_factory()
    answer = binding(conn, settings, meter=meter, run=None)
    assert answer.pipeline.deps.agent == "knowledge-eval"
    assert answer.pipeline.deps.meter is meter and answer.pipeline.record_traces is False


def test_the_eval_binding_uses_the_runs_profile(settings, conn, meter_factory):
    name = conn.execute("SELECT name FROM profiles").fetchone()[0]
    run = EvalRun(id="r1", stage="S5", config={"profile": name})
    answer = binding(conn, settings, meter=meter_factory(), run=run, deps=fake_deps(delay_s=0))
    assert answer("How old is the furnace?").text
    missing = EvalRun(id="r2", stage="S5", config={"profile": "pdfium.fixed-99.hdr.fake-64"})
    with pytest.raises(ConfigMissing, match="is not built"):
        binding(conn, settings, meter=meter_factory(), run=missing)


def test_the_harness_finds_the_binding(settings, conn, meter_factory):
    from hal9000.knowledge.eval.wiring import answerer_for

    answer = answerer_for(conn, settings, meter=meter_factory(), run=None)
    assert callable(answer)


def test_fake_settings_copy_the_policy_and_activate_the_fake_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("KB_PASSCODE", raising=False)
    settings = serve_cli.fake_settings(tmp_path)
    assert settings.passcode == serve_cli.FAKE_PASSCODE
    assert settings.policy_path.read_text().startswith("How HAL answers")
    assert serve_cli.FAKE_PROFILE in Path(settings.yaml_path).read_text()
    monkeypatch.setenv("KB_PASSCODE", "a-synthetic-passcode")
    assert serve_cli.fake_settings(tmp_path / "b").passcode == "a-synthetic-passcode"
