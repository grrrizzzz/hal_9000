"""`kb gold ...` and `kb eval ...` end to end, offline: estimates, caps, metering, wiring."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from hal9000.__main__ import main
from hal9000.knowledge.eval import wiring
from hal9000.knowledge.eval.store import insert_gold, list_gold, list_judge_labels, set_gold_status
from hal9000.knowledge.eval.types import AnswerOutput, RetrievalOutput, SelectedItem, Source
from hal9000.knowledge.storage import open_db
from hal9000.llm import Usage
from tests.fakes import tool_call
from tests.knowledge.eval.conftest import store_documents
from tests.knowledge.eval.fakes import GoldLLM
from tests.knowledge.eval.test_export_report import fixture_run
from tests.knowledge.fixtures import corpus

REPO = Path(__file__).resolve().parents[3]

SMALL_CONFIG = """
gold:
  quotas: {factoid: 4, money: 2, date: 2, list: 2, cross_document: 2, unanswerable: 3}
scoring:
  n_boot: 50
judge:
  n_items: 3
"""


@pytest.fixture
def env(tmp_path, monkeypatch, openai_env, fixture_docs):
    data, config = tmp_path / "data", tmp_path / "config"
    config.mkdir()
    (config / "knowledge_eval.yaml").write_text(SMALL_CONFIG)
    monkeypatch.setenv("HAL_DATA_DIR", str(data))
    monkeypatch.setenv("HAL_CONFIG_DIR", str(config))
    conn = open_db(data / "knowledge.db")
    store_documents(conn, fixture_docs)
    yield conn
    conn.close()
    wiring.register_retriever(None)
    wiring.register_answerer(None)


class FakeProvider:
    """Stands in for hal9000.llm.generate_response (reports usage, like the real one)."""

    def __init__(self, llm, usage=(2000, 300)):
        self.llm = llm
        self.usage = usage
        self.calls = 0

    def __call__(self, prompt, tier="powerful", *, on_usage=None):
        if self.calls == 0:
            print("FIRST LLM CALL")
        self.calls += 1
        reply = self.llm(prompt, tier)
        if on_usage:
            on_usage(
                Usage(
                    model="openai/gpt-5",
                    prompt=self.usage[0],
                    cached=0,
                    completion=self.usage[1],
                    reasoning=0,
                )
            )
        return reply


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider(GoldLLM())
    monkeypatch.setattr("hal9000.llm.generate_response", fake)
    return fake


def ledger(tmp_path) -> list[tuple[str, float]]:
    conn = sqlite3.connect(tmp_path / "data" / "hal9000.db")
    try:
        return conn.execute("SELECT agent, usd FROM ledger").fetchall()
    finally:
        conn.close()


# ----------------------------------------------------------------- gold generate
def test_gold_generate_prints_the_estimate_first_and_meters_as_knowledge_eval(
    env, provider, tmp_path, capsys
):
    assert main(["kb", "gold", "generate", "--allow-keyword-check", "--yes"]) == 0
    out = capsys.readouterr().out
    assert out.index("Estimate: about") < out.index("FIRST LLM CALL")
    assert "cap --max-cost $3.00" in out
    assert (
        "keyword search only" in out
    )  # retrieval not wired yet: the lexical fallback checks drafts
    items = list_gold(env)
    assert len(items) == 15 and all(i.status == "draft" for i in items)
    rows = ledger(tmp_path)
    assert len(rows) == provider.calls > 0
    assert {agent for agent, _ in rows} == {"knowledge-eval"}
    assert "Stored 15 draft items" in out


def test_gold_generate_refuses_an_estimate_over_max_cost(env, provider, capsys):
    assert (
        main(["kb", "gold", "generate", "--allow-keyword-check", "--max-cost", "0.0001", "--yes"])
        == 3
    )
    assert "exceeds --max-cost" in capsys.readouterr().err
    assert provider.calls == 0


def test_gold_generate_respects_kb_eval_cap_usd(env, provider, monkeypatch, capsys):
    monkeypatch.setenv("KB_EVAL_CAP_USD", "0.0001")
    assert main(["kb", "gold", "generate", "--allow-keyword-check", "--yes"]) == 3
    assert "KB_EVAL_CAP_USD" in capsys.readouterr().err
    assert provider.calls == 0


def test_gold_generate_needs_yes_when_not_interactive(env, provider, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["kb", "gold", "generate", "--allow-keyword-check"]) == 3
    assert "pass --yes" in capsys.readouterr().out
    assert provider.calls == 0


def test_gold_generate_confirms_interactively(env, provider, monkeypatch, capsys):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Tty("n\n"))
    assert main(["kb", "gold", "generate", "--allow-keyword-check"]) == 3
    assert provider.calls == 0
    monkeypatch.setattr("sys.stdin", Tty("y\n"))
    assert main(["kb", "gold", "generate", "--allow-keyword-check"]) == 0
    assert provider.calls > 0


def test_gold_generate_stops_at_the_hard_cap_and_keeps_partial_results(
    env, monkeypatch, tmp_path, capsys
):
    expensive = FakeProvider(GoldLLM(), usage=(200_000, 20_000))  # ~$0.45 per call on gpt-5
    monkeypatch.setattr("hal9000.llm.generate_response", expensive)
    assert (
        main(["kb", "gold", "generate", "--allow-keyword-check", "--max-cost", "1.00", "--yes"])
        == 4  # stopped by the cap: what was done is stored
    )
    out = capsys.readouterr().out
    assert "STOPPED EARLY" in out
    assert expensive.calls == 3  # the cap is checked before each call
    assert sum(usd for _, usd in ledger(tmp_path)) >= 1.0
    assert 0 < len(list_gold(env)) < 15


def test_gold_generate_without_documents_or_key(tmp_path, monkeypatch, openai_env, capsys):
    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path / "data"))
    assert main(["kb", "gold", "generate", "--allow-keyword-check", "--yes"]) == 1
    assert "run `hal9000 kb ingest` first" in capsys.readouterr().err


def test_gold_generate_without_an_llm_key(env, tmp_path, monkeypatch, capsys):
    # main() loads the .env it finds from the working directory, which would put a real
    # OPENAI_API_KEY back (and make a paid call): run from a folder that has none.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_MODEL_POWERFUL", "openai/gpt-5")
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert main(["kb", "gold", "generate", "--allow-keyword-check", "--yes"]) == 1
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


# --------------------------------------------------------------- export/import
def test_gold_export_and_import_stay_under_data(env, tmp_path, capsys):
    from hal9000.knowledge.contracts import GoldItem

    insert_gold(env, [GoldItem(question="Who painted the fence?", category="unanswerable")])
    inside_repo = REPO / "docs" / "gold-export-test.jsonl"
    assert main(["kb", "gold", "export", str(inside_repo)]) == 1
    assert "never be committed" in capsys.readouterr().err
    assert not inside_repo.exists()
    target = tmp_path / "data" / "knowledge" / "eval" / "gold.jsonl"
    assert main(["kb", "gold", "export", str(target)]) == 0
    assert target.read_text().count("\n") == 1
    assert main(["kb", "gold", "import", str(target)]) == 0
    assert "0 new, 1 updated" in capsys.readouterr().out
    assert main(["kb", "gold", "import", str(target.with_name("missing.jsonl"))]) == 1
    target.write_text('{"question": "x"}\n')
    assert main(["kb", "gold", "import", str(target)]) == 1
    assert "nothing imported" in capsys.readouterr().err


# ---------------------------------------------------------------- eval commands
def test_eval_export_and_report(env, tmp_path, fixture_docs, capsys):
    eval_dir = tmp_path / "data" / "knowledge" / "eval"
    run = fixture_run(env, fixture_docs, eval_dir, config={"profile": "p1", "size": 512})
    assert main(["kb", "eval", "export", run.id]) == 0
    assert (eval_dir / "runs" / run.id / "export.json").exists()
    assert main(["kb", "eval", "report", run.id]) == 0
    report = tmp_path / "docs" / "KNOWLEDGE_EVAL_REPORT.md"
    assert report.exists() and "evaluation report" in report.read_text()
    assert main(["kb", "eval", "export", "nope"]) == 1
    assert main(["kb", "eval", "report", "nope"]) == 1
    err = capsys.readouterr().err
    assert "no evaluation run or sweep 'nope'" in err


def test_eval_export_refuses_a_leak(env, tmp_path, fixture_docs, capsys):
    eval_dir = tmp_path / "data" / "knowledge" / "eval"
    run = fixture_run(
        env, fixture_docs, eval_dir, config={"profile": "p1", "x": fixture_docs[0].pages[5]}
    )
    assert main(["kb", "eval", "export", run.id]) == 1
    assert "refusing" in capsys.readouterr().err


def _approve_facts(conn):
    from hal9000.knowledge.contracts import EvidenceQuote, GoldItem

    items = [
        GoldItem(question=f.question, reference_answer=f.answer, category=f.category,
                 evidence=[EvidenceQuote(quote=f.quote, page=f.page)], split="dev")
        for f in corpus.QA_FACTS[:4]
    ]  # fmt: skip
    stored = insert_gold(conn, items)
    set_gold_status(conn, [g.id for g in stored], "approved")
    return stored


def test_eval_regression_needs_retrieval_then_writes_a_row(env, capsys):
    assert main(["kb", "eval", "regression", "--yes"]) == 1
    assert "no approved answerable questions" in capsys.readouterr().err
    _approve_facts(env)
    assert main(["kb", "eval", "regression", "--yes"]) == 1  # NotWired: no retrieval to run
    err = capsys.readouterr().err
    assert "retrieval is not available yet" in err or "no index to search" in err
    seen = {}

    def factory(conn, settings, *, meter, run):
        seen["meter"] = meter

        def retrieve(question):
            fact = next(f for f in corpus.QA_FACTS if f.question == question)
            return RetrievalOutput([SelectedItem(1, fact.quote, 50)], abstain_score=0.9)

        return wiring.RetrievalBinding(retrieve, None, "pdfium.fixed-512", 512)

    wiring.register_retriever(factory)
    assert main(["kb", "eval", "regression", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "query-embedding tokens" in out and "recall @ 2000 = 1.000" in out
    assert seen["meter"] is not None
    row = env.execute("SELECT trigger, recall_2k FROM regression_runs").fetchone()
    assert tuple(row) == ("manual", 1.0)


def test_eval_judge_prep(env, tmp_path, fixture_docs, monkeypatch, capsys):
    assert main(["kb", "eval", "judge-prep", "nope", "--yes"]) == 1
    assert "no evaluation run or finished S4 sweep 'nope'" in capsys.readouterr().err
    eval_dir = tmp_path / "data" / "knowledge" / "eval"
    run = fixture_run(env, fixture_docs, eval_dir, config={"profile": "p1"}, stage="S4")
    assert main(["kb", "eval", "judge-prep", run.id, "--yes"]) == 1
    assert "no approved dev items" in capsys.readouterr().err
    _approve_facts(env)
    assert main(["kb", "eval", "judge-prep", run.id, "--yes"]) == 1
    # KWP6's answer binding is found; the run's profile has not been built here
    err = capsys.readouterr().err
    assert "index profile 'p1' is not built" in err or "no index to search" in err

    def answerer(conn, settings, *, meter, run):
        assert run.stage == "S4" and meter is not None
        return lambda q: AnswerOutput(f"Per the sources [S1]: {q}", sources=[Source("S1", "text")])

    wiring.register_answerer(answerer)
    verdict = tool_call("record_judgement", correctness="correct", faithful=True,
                        citations_correct=True, abstention_correct=True)  # fmt: skip
    judge = FakeProvider(lambda p, t: verdict)
    monkeypatch.setattr("hal9000.llm.generate_response", judge)
    assert main(["kb", "eval", "judge-prep", run.id, "--yes"]) == 0
    out = capsys.readouterr().out
    assert out.index("Estimate:") < out.index("FIRST LLM CALL")
    labels = list_judge_labels(env, eval_run_id=run.id)
    assert len(labels) == 3 and all(label.judge for label in labels)
    assert {agent for agent, _ in ledger(tmp_path)} == {"knowledge-eval"}
    assert main(["kb", "eval", "judge-prep", run.id, "--yes"]) == 0
    assert "already exist" in capsys.readouterr().err


def test_judge_prep_and_regression_respect_the_eval_cap(
    env, tmp_path, fixture_docs, monkeypatch, capsys
):
    eval_dir = tmp_path / "data" / "knowledge" / "eval"
    run = fixture_run(env, fixture_docs, eval_dir, config={"profile": "p1"}, stage="S4")
    _approve_facts(env)
    monkeypatch.setenv("KB_EVAL_CAP_USD", "0")
    assert main(["kb", "eval", "judge-prep", run.id, "--yes"]) == 3
    assert main(["kb", "eval", "regression", "--yes"]) == 3
    assert capsys.readouterr().err.count("KB_EVAL_CAP_USD") == 2
    monkeypatch.setenv("KB_EVAL_CAP_USD", "10")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["kb", "eval", "judge-prep", run.id]) == 3
    assert main(["kb", "eval", "regression"]) == 3
    assert capsys.readouterr().out.count("pass --yes") == 2


def test_gold_generate_needs_the_real_retriever_unless_allowed(env, provider, capsys):
    assert main(["kb", "gold", "generate", "--yes"]) == 1
    err = capsys.readouterr().err
    assert "--allow-keyword-check" in err  # NotWired: KWP4 absent, or no index built yet
    assert "retrieval is not available yet" in err or "no index to search" in err
    assert provider.calls == 0
    seen = {}

    def factory(conn, settings, *, meter, run):
        seen["meter"] = meter
        return wiring.RetrievalBinding(
            lambda q: RetrievalOutput([SelectedItem(1, "A passage about the yard.", 10)]), 1, "p"
        )

    wiring.register_retriever(factory)
    assert main(["kb", "gold", "generate", "--yes"]) == 0
    assert "keyword search" not in capsys.readouterr().out
    assert seen["meter"] is not None
    notes = [i.reviewer_note or "" for i in list_gold(env) if i.category == "unanswerable"]
    assert notes and not any("keyword search" in n for n in notes)


def test_eval_report_refuses_to_write_a_leak(env, tmp_path, fixture_docs, capsys):
    eval_dir = tmp_path / "data" / "knowledge" / "eval"
    run = fixture_run(
        env, fixture_docs, eval_dir, config={"profile": fixture_docs[0].pages[5][:60]}
    )
    assert main(["kb", "eval", "report", run.id]) == 1
    assert "report not written" in capsys.readouterr().err
    assert not (tmp_path / "docs" / "KNOWLEDGE_EVAL_REPORT.md").exists()


def test_importing_the_harness_does_not_import_rapidfuzz():
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json, sys\n"
        "import hal9000.knowledge.eval.scoring, hal9000.knowledge.eval.gold\n"
        "import hal9000.knowledge.eval.regression, hal9000.knowledge.eval.report\n"
        "print(json.dumps('rapidfuzz' in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) is False
