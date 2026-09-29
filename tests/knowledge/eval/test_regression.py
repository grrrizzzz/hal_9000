"""Post-ingest regression: approved set, retrieval only, a regression_runs row (plan §9.2)."""

from __future__ import annotations

import pytest

from hal9000.knowledge.contracts import EvidenceQuote, GoldItem
from hal9000.knowledge.eval.config import EvalConfig
from hal9000.knowledge.eval.regression import ingest_hook, post_ingest_regression, run_regression
from hal9000.knowledge.eval.store import insert_gold, set_gold_status
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem
from hal9000.knowledge.obs.collector import KnowledgeDbCollector
from hal9000.knowledge.obs.timing import start_trace
from tests.knowledge.fixtures import corpus

CFG = EvalConfig.model_validate({"scoring": {"n_boot": 100}})


def _count(text: str) -> int:
    return len(text.split())


def seed(conn):
    facts = corpus.QA_FACTS[:4]
    items = [
        GoldItem(question=f.question, reference_answer=f.answer, category="factoid",
                 evidence=[EvidenceQuote(quote=f.quote, page=f.page)])
        for f in facts
    ] + [
        GoldItem(question="Who painted the fence?", category="unanswerable"),
        GoldItem(question="A draft never approved?", category="factoid",
                 evidence=[EvidenceQuote(quote="Total: $605.00", page=1)]),
    ]  # fmt: skip
    stored = insert_gold(conn, items)
    set_gold_status(conn, [g.id for g in stored[:5]], "approved")
    return facts


class Retrieve:
    """Finds the evidence for the first ``hits`` questions it is asked."""

    def __init__(self, facts, hits: int, fail_on: str | None = None) -> None:
        self.answers = {f.question: f.quote for f in facts}
        self.hits = hits
        self.fail_on = fail_on
        self.asked: list[str] = []

    def __call__(self, question: str) -> RetrievalOutput:
        self.asked.append(question)
        if question == self.fail_on:
            raise RuntimeError("index unavailable")
        found = len(self.asked) <= self.hits
        text = self.answers[question] if found else "An unrelated passage about the yard."
        return RetrievalOutput([SelectedItem(1, text, 300)], abstain_score=0.8)


def test_regression_scores_the_approved_set_and_writes_a_row(kb_db, tmp_path):
    facts = seed(kb_db)
    retrieve = Retrieve(facts, hits=3)
    with start_trace() as trace:
        result = run_regression(kb_db, retrieve, profile_id=None, config=CFG, count=_count)
    assert [s.stage for s in trace.stages] == ["regression"]
    assert sorted(retrieve.asked) == sorted(f.question for f in facts)  # approved answerable only
    assert result.n_questions == 4 and result.trigger == "manual"
    assert result.recall.value == pytest.approx(0.75)
    assert result.recall.low <= 0.75 <= result.recall.high
    row = kb_db.execute("SELECT * FROM regression_runs").fetchone()
    assert (row["trigger"], row["n_questions"]) == ("manual", 4)
    assert row["recall_2k"] == pytest.approx(0.75) and row["recall_low"] <= row["recall_high"]
    families = {f.name: f for f in KnowledgeDbCollector(tmp_path / "knowledge.db").collect()}
    assert families["kb_regression_recall"].samples[0].value == pytest.approx(0.75)
    assert "0.750" in result.describe()


def test_regression_uses_the_active_profile_and_counts_retrieval_errors(kb_db):
    facts = seed(kb_db)
    with kb_db:
        kb_db.execute(
            "INSERT INTO profiles (id, name, parser, chunker, size, header, embed_model, dimensions, "
            "active, created_at) VALUES (7, 'p', 'pdfium', 'fixed', 512, 1, 'm', 8, 1, 'now')"
        )
    retrieve = Retrieve(facts, hits=4, fail_on=facts[0].question)
    result = run_regression(kb_db, retrieve, config=CFG, count=_count)
    assert result.profile_id == 7 and result.errors == 1
    assert result.recall.value == pytest.approx(0.75)
    assert "1 retrieval errors" in result.describe()


def test_regression_with_chunks_larger_than_the_budget_is_na(kb_db):
    facts = seed(kb_db)
    result = run_regression(kb_db, Retrieve(facts, 4), chunk_size=4000, config=CFG, count=_count)
    assert result.recall is None and "N/A" in result.describe()
    assert kb_db.execute("SELECT recall_2k FROM regression_runs").fetchone()[0] is None


def test_no_approved_questions_writes_nothing(kb_db):
    result = run_regression(kb_db, Retrieve([], 0), config=CFG)
    assert result.n_questions == 0 and result.row_id is None
    assert kb_db.execute("SELECT COUNT(*) FROM regression_runs").fetchone()[0] == 0


def test_the_post_ingest_hook_marks_the_trigger_and_never_raises(kb_db, caplog):
    facts = seed(kb_db)
    result = post_ingest_regression(kb_db, Retrieve(facts, 4), config=CFG, count=_count)
    assert result.trigger == "ingest" and result.recall.value == 1.0
    assert kb_db.execute("SELECT trigger FROM regression_runs").fetchone()[0] == "ingest"

    def broken_count(text):
        raise RuntimeError("tokenizer broke")

    assert post_ingest_regression(kb_db, Retrieve(facts, 4), config=CFG, count=broken_count) is None
    assert "post-ingest regression failed" in caplog.text


# ------------------------------------------------------ the service's ingest hook
def _errors(kind: str) -> float:
    from hal9000.knowledge.obs.metrics import REGISTRY

    return REGISTRY.get_sample_value("kb_errors_total", {"stage": "regression", "kind": kind}) or 0


class _Profile:
    name = "p"


def test_a_budget_stop_writes_no_row_and_is_counted(kb_db, caplog):
    from hal9000.cost.metering import MonthlyCapExceeded

    facts = seed(kb_db)
    before = _errors("MonthlyCapExceeded")

    def out_of_money(question):
        raise MonthlyCapExceeded("knowledge-eval reached its cap")

    with pytest.raises(MonthlyCapExceeded):  # not scored as a recall of zero
        run_regression(kb_db, out_of_money, config=CFG, count=_count)
    assert post_ingest_regression(kb_db, out_of_money, config=CFG, count=_count) is None
    assert kb_db.execute("SELECT COUNT(*) FROM regression_runs").fetchone()[0] == 0
    assert _errors("MonthlyCapExceeded") == before + 1  # the post-ingest one
    assert facts and "post-ingest regression failed" in caplog.text


def test_the_ingest_hook_does_nothing_without_approved_questions(kb_db, monkeypatch):
    from hal9000.knowledge.eval import wiring

    def unexpected(*args, **kwargs):
        raise AssertionError("no binding (and no spend) without approved questions")

    monkeypatch.setattr(wiring, "retriever_for", unexpected)
    hook = ingest_hook(object(), meter=None)
    assert hook(kb_db, _Profile()) is None
    assert kb_db.execute("SELECT COUNT(*) FROM regression_runs").fetchone()[0] == 0


def test_the_ingest_hook_checks_the_eval_cap_and_never_raises(kb_db, caplog):
    from hal9000.cost.metering import MonthlyCapExceeded

    seed(kb_db)
    checked = []

    class CappedMeter:
        def precheck(self, agent):
            checked.append(agent)
            raise MonthlyCapExceeded(f"{agent} reached its cap")

    before = _errors("MonthlyCapExceeded")
    assert ingest_hook(object(), meter=CappedMeter())(kb_db, _Profile()) is None
    assert checked == ["knowledge-eval"]
    assert kb_db.execute("SELECT COUNT(*) FROM regression_runs").fetchone()[0] == 0
    assert _errors("MonthlyCapExceeded") == before + 1


def test_the_ingest_hook_scores_the_profile_it_was_given(kb_db, monkeypatch, tmp_path):
    from hal9000.knowledge.eval import wiring

    facts = seed(kb_db)
    asked = {}

    def retriever_for(conn, settings, *, meter=None, run=None):
        asked["profile"] = run.config["profile"]
        return wiring.RetrievalBinding(
            retrieve=Retrieve(facts, 4), profile_id=None, profile_name="p", chunk_size=None
        )

    monkeypatch.setattr(wiring, "retriever_for", retriever_for)
    (tmp_path / "knowledge_eval.yaml").write_text("scoring:\n  n_boot: 100\n")

    class Settings:
        config_dir = tmp_path

    result = ingest_hook(Settings(), meter=None)(kb_db, _Profile())
    assert asked == {"profile": "p"} and result.trigger == "ingest"
    assert kb_db.execute("SELECT trigger FROM regression_runs").fetchone()[0] == "ingest"
