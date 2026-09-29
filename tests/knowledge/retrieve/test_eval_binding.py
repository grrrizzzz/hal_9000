"""The evaluation binding: KWP5's retrieve protocol, its configuration and its metering."""

from __future__ import annotations

import pytest

from hal9000.knowledge.config import EVAL_AGENT, ConfigMissing, settings
from hal9000.knowledge.contracts import EvalRun, EvidenceQuote, GoldItem
from hal9000.knowledge.eval.scoring import ScoringConfig, ScoringItem, score_question
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem
from hal9000.knowledge.eval.wiring import RetrievalBinding, retriever_for
from hal9000.knowledge.index.fake import FakeEmbedder
from hal9000.knowledge.retrieve import RetrievalConfig, set_threshold
from hal9000.knowledge.retrieve.eval_binding import binding, scored_budget
from tests.fakes import FakeLLM
from tests.knowledge.fixtures import corpus
from tests.knowledge.retrieve.conftest import ACTIVE, PARENT_CHILD, profile_id


def run_with(**config) -> EvalRun:
    return EvalRun(id="r1", stage="S3", config=config)


def gold(fact) -> GoldItem:
    return GoldItem(
        id=1,
        question=fact.question,
        category=fact.category,
        evidence=[EvidenceQuote(quote=fact.quote, page=fact.page)],
    )


def test_wiring_finds_the_binding_and_it_retrieves_the_active_profile(conn, fake):
    bound = retriever_for(conn, settings(), meter=None)
    assert isinstance(bound, RetrievalBinding)
    assert bound.profile_name == ACTIVE and bound.profile_id == profile_id(conn, ACTIVE)
    assert bound.chunk_size == 256
    output = bound.retrieve(corpus.QA_FACTS[3].question)
    assert isinstance(output, RetrievalOutput) and output.selected
    assert all(isinstance(item, SelectedItem) for item in output.selected)
    assert output.abstain_score is not None and output.latency_ms > 0
    # no budget in the run: select with the largest scored budget
    assert bound.retrieve.config.budget_tokens == scored_budget(settings()) == 4000
    assert sum(item.prompt_tokens for item in output.selected) <= 4000


def test_it_is_metered_as_knowledge_eval(conn, monkeypatch):
    bound = binding(conn, settings(), meter=None, embed_fn=FakeEmbedder())
    assert bound.retrieve.retriever.embeddings.agent == EVAL_AGENT

    wrapped = []

    class Meter:
        def wrap(self, fn, agent):
            wrapped.append((fn.__name__, agent))
            return FakeLLM(["a\nb\nc"])

    monkeypatch.setattr("hal9000.config.require_llm_key", lambda: None)
    multi = binding(conn, settings(), meter=Meter(), run=run_with(query_transform="multi"))
    assert wrapped == [("generate_response", EVAL_AGENT)]
    assert multi.retrieve(corpus.QA_FACTS[3].question).selected
    with pytest.raises(ConfigMissing, match="needs a Meter"):
        binding(conn, settings(), meter=None, run=run_with(query_transform="hyde"))


def test_the_run_configuration_is_respected(conn):
    llm = FakeLLM(["rewrite one\nrewrite two\nrewrite three"])
    run = run_with(profile=PARENT_CHILD, retrieval="bm25", budget=1000, query_transform="multi")
    bound = binding(conn, settings(), meter=None, run=run, generate=llm)
    cfg = bound.retrieve.config
    assert (cfg.mode, cfg.budget_tokens, cfg.transform) == ("bm25", 1000, "multi")
    assert bound.profile_name == PARENT_CHILD and bound.chunk_size == 512  # the parent size
    output = bound.retrieve(corpus.QA_FACTS[2].question)
    assert len(llm.calls) == 1
    assert sum(i.prompt_tokens for i in output.selected) <= 1000
    assert output.abstain_score is None and output.abstained is False  # BM25: no abstain

    explicit = RetrievalConfig(mode="dense", budget_tokens=500)
    bound = binding(conn, settings(), meter=None, run=run, config=explicit)
    assert bound.retrieve.config is explicit


def test_the_abstain_decision_reaches_the_harness(conn):
    set_threshold(conn, profile_id(conn, ACTIVE), None, 2.0)
    output = binding(conn, settings(), meter=None).retrieve(corpus.QA_FACTS[0].question)
    assert output.abstained is True and output.selected


def test_kwp5_scoring_accepts_the_output(conn):
    """KWP5's scorer runs on exactly what select returned, at every budget."""
    bound = binding(conn, settings(), meter=None)
    config = ScoringConfig(budgets=(1000, 2000, 4000), hit_ks=(1, 5))
    recalls = []
    for fact in corpus.QA_FACTS:
        output = bound.retrieve(fact.question)
        score = score_question(ScoringItem.from_gold(gold(fact)), output, chunk_size=256,
                               config=config)  # fmt: skip
        assert set(score.recall) == {1000, 2000, 4000}
        assert score.context_tokens[4000] == sum(i.prompt_tokens for i in output.selected)
        recalls.append(score.recall[4000])
    assert sum(r == 1.0 for r in recalls) >= len(recalls) // 2  # the fixture facts are found


def test_no_index_is_reported_as_not_wired(conn, tmp_path):
    from hal9000.knowledge.eval.wiring import NotWired
    from hal9000.knowledge.storage import open_db

    with pytest.raises(NotWired, match="no index profile named"):
        binding(conn, settings(), meter=None, run=run_with(profile="missing"))
    empty = open_db(tmp_path / "empty.db")
    try:
        with pytest.raises(NotWired, match="^no index to search: ingest documents first$"):
            binding(empty, settings(), meter=None)
    finally:
        empty.close()
