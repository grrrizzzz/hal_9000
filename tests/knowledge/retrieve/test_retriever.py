"""The retriever over the ingested fixture corpus (plan §7.1, §7.4; handoff KWP4 acceptance)."""

from __future__ import annotations

import logging
import statistics
import time
from datetime import date

import numpy as np
import pytest

from hal9000.knowledge.config import ConfigMissing
from hal9000.knowledge.contracts import RetrievalResult, SearchFilters
from hal9000.knowledge.eval.scoring import budget_prefix
from hal9000.knowledge.eval.types import SelectedItem
from hal9000.knowledge.index.embedding import EmbeddingService
from hal9000.knowledge.index.fake import FakeEmbedder, FakeModeRequired
from hal9000.knowledge.obs.timing import QUESTION_STAGES, start_trace
from hal9000.knowledge.retrieve import (
    NoActiveProfile,
    QueryVectorCache,
    RetrievalConfig,
    Retriever,
    Turn,
    search_documents,
    set_threshold,
)
from hal9000.knowledge.retrieve.retriever import DEFAULT_QUERY_CACHE
from hal9000.knowledge.retrieve.select import fingerprint, is_near_duplicate, overlap, shingles
from hal9000.knowledge.tokens import count_tokens
from tests.fakes import FakeLLM
from tests.knowledge.fixtures import corpus
from tests.knowledge.retrieve.conftest import ACTIVE, PARENT_CHILD, profile_id

HEATER = "How old is the water heater?"


def stages(trace) -> list[str]:
    return [timing.stage.value for timing in trace.stages]


def run(retriever: Retriever, question: str = HEATER, **kw) -> tuple[RetrievalResult, list[str]]:
    with start_trace() as trace:
        result = retriever.retrieve(question, **kw)
    return result, stages(trace)


def unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def index_text(chunk) -> str:
    return f"{chunk.header}\n{chunk.text}" if chunk.header else chunk.text


def raises(error: BaseException):
    def embed(*args, **kw):
        raise error

    return embed


# ------------------------------------------------------------------ stages
def test_every_stage_is_traced_with_its_plan_name(conn, embeddings):
    llm = FakeLLM(["How old is the water heater in the house?"])

    class Stub:
        name = "stub"

        def score(self, query, passages):
            return [0.5] * len(passages)

    retriever = Retriever(conn, embeddings, generate=llm, reranker=Stub())
    _, plain = run(retriever)
    assert plain == ["embed_query", "vector", "bm25", "fuse", "select", "abstain"]
    history = [Turn("user", "Tell me about the water heater."), Turn("assistant", "It leaks.")]
    cfg = RetrievalConfig(rerank="minilm")
    _, full = run(retriever, "How old is it?", history=history, config=cfg)
    assert full == [
        "condense", "embed_query", "vector", "bm25", "fuse", "rerank", "select", "abstain",
    ]  # fmt: skip
    assert set(full) <= {s.value for s in QUESTION_STAGES}
    _, bm25 = run(retriever, config=RetrievalConfig(mode="bm25"))
    assert bm25 == ["bm25", "fuse", "select", "abstain"]
    _, dense = run(retriever, config=RetrievalConfig(mode="dense"))
    assert dense == ["embed_query", "vector", "fuse", "select", "abstain"]


def test_the_result_describes_the_question(conn, embeddings):
    result, _ = run(Retriever(conn, embeddings))
    assert result.query == result.standalone_query == HEATER
    assert result.profile == ACTIVE and result.mode == "hybrid" and result.degraded is None
    assert result.candidates and result.selected
    assert [c.rank for c in result.candidates] == list(range(1, len(result.candidates) + 1))
    top = result.selected[0]
    assert top.file_sha and top.title and top.collection and top.prompt_tokens > 0
    assert top.fused_score == top.score and top.dense_score is not None
    assert any("13 years old" in c.text for c in result.selected)
    RetrievalResult.model_validate_json(result.model_dump_json())  # round-trips


# ------------------------------------------------------------------ condense
def test_condense_is_called_only_with_history(conn, embeddings):
    llm = FakeLLM([])  # any call would fail the test
    retriever = Retriever(conn, embeddings, generate=llm)
    result, _ = run(retriever)
    assert llm.calls == [] and result.standalone_query == HEATER

    llm.responses.append("Standalone question: How old is the water heater?")
    history = [Turn("user", f"question {i}") for i in range(9)]
    result, trace = run(retriever, "And how old is it?", history=history)
    assert len(llm.calls) == 1 and llm.tiers == ["fast"]
    assert result.query == "And how old is it?"
    assert result.standalone_query == HEATER
    assert trace.count("condense") == 1
    prompt = llm.last_prompt.messages[-1]["content"]
    assert "question 3" in prompt and "question 2" not in prompt  # only the last six turns
    assert "And how old is it?" in prompt


@pytest.mark.parametrize("reply", [RuntimeError("provider down"), "", "  \n  "])
def test_a_failed_condense_falls_back_to_the_question(conn, embeddings, caplog, reply):
    class Broken:
        def __call__(self, prompt, tier):
            if isinstance(reply, Exception):
                raise reply
            return reply

    retriever = Retriever(conn, embeddings, generate=Broken())
    with caplog.at_level(logging.WARNING):
        result, trace = run(retriever, history=[Turn("user", "hi")])
    assert result.standalone_query == HEATER and "condense" in trace
    assert "using the question as asked" in caplog.text
    assert result.selected


def test_condense_without_a_generator_or_turns_keeps_the_question(conn, embeddings, caplog):
    with caplog.at_level(logging.WARNING):
        result, trace = run(Retriever(conn, embeddings), history=[Turn("user", "hi")])
    assert result.standalone_query == HEATER and "no generate function" in caplog.text
    none, trace = run(
        Retriever(conn, embeddings, generate=FakeLLM([])),
        history=[Turn("user", "hi")],
        config=RetrievalConfig(history_turns=0),
    )
    assert none.standalone_query == HEATER and "condense" not in trace


# ------------------------------------------------------------------ filters
def test_filters_restrict_candidates_and_selection(conn, embeddings):
    retriever = Retriever(conn, embeddings)
    invoices, _ = run(retriever, "How much was charged?", filters=SearchFilters(doc_type="invoice"))
    assert invoices.candidates
    assert {c.doc_type for c in invoices.candidates} == {"invoice"}

    house, _ = run(retriever, "total", filters=SearchFilters(collection="House"))
    collections = {c.collection for c in house.candidates}
    assert "House/Bills/" in collections  # a sub-folder is included
    assert collections <= {"House/", "House/Bills/"}

    bills, _ = run(retriever, "total", filters=SearchFilters(collection="House/Bills/"))
    assert {c.collection for c in bills.candidates} == {"House/Bills/"}

    none, _ = run(retriever, "total", filters=SearchFilters(collection="Garden"))
    assert none.candidates == [] and none.selected == [] and none.abstain_score is None
    assert none.abstained is True  # nothing to answer from

    prefix, _ = run(retriever, "total", filters=SearchFilters(collection="Hou"))
    assert prefix.candidates == []  # "Hou" is a folder of its own, not a prefix of House/

    dated, _ = run(retriever, "total", filters=SearchFilters(date_from=date(2000, 1, 1)))
    assert dated.candidates and all(c.doc_date is not None for c in dated.candidates)


# ------------------------------------------------------------------ select
@pytest.mark.parametrize("profile", [ACTIVE, PARENT_CHILD])
@pytest.mark.parametrize("send_headers", [True, False])
def test_the_budget_is_never_exceeded(conn, embeddings, profile, send_headers):
    retriever = Retriever(conn, embeddings, profile=profile)
    for fact in corpus.QA_FACTS[:6]:
        for budget in (300, 1000, 2000, 4000):
            cfg = RetrievalConfig(budget_tokens=budget, send_headers=send_headers)
            result = retriever.retrieve(fact.question, config=cfg)
            used = sum(c.prompt_tokens for c in result.selected)
            assert result.context_tokens == used <= budget
            ids = [c.chunk_id for c in result.selected]
            assert len(ids) == len(set(ids))


def test_prompt_tokens_count_headers_only_when_sent(conn, embeddings):
    retriever = Retriever(conn, embeddings)
    with_headers = retriever.retrieve(HEATER)
    for c in with_headers.candidates:
        assert c.header  # the profile writes contextual headers
        assert c.prompt_tokens == count_tokens(f"{c.header}\n{c.text}") > c.tokens
    without = retriever.retrieve(HEATER, config=RetrievalConfig(send_headers=False))
    assert all(c.prompt_tokens == c.tokens for c in without.candidates)


def test_selection_is_prefix_consistent_with_the_scorer(conn, embeddings):
    """The scorer's budget_prefix of a 4k selection is exactly what a smaller budget selects."""
    for name in (ACTIVE, PARENT_CHILD):
        retriever = Retriever(conn, embeddings, profile=name)
        for fact in corpus.QA_FACTS:
            big = retriever.retrieve(fact.question, config=RetrievalConfig(budget_tokens=4000))
            items = [SelectedItem(c.chunk_id, c.text, c.prompt_tokens) for c in big.selected]
            for budget in (1000, 2000):
                small = retriever.retrieve(
                    fact.question, config=RetrievalConfig(budget_tokens=budget)
                )
                assert [i.chunk_id for i in budget_prefix(items, budget)] == [
                    c.chunk_id for c in small.selected
                ]


def test_parent_child_expands_children_to_parents_once(conn, embeddings):
    retriever = Retriever(conn, embeddings, profile=PARENT_CHILD)
    seen_shared_parent = False
    for fact in corpus.QA_FACTS:
        result = retriever.retrieve(fact.question, config=RetrievalConfig(budget_tokens=4000))
        assert all(c.parent_id is not None for c in result.candidates)  # children are searched
        parents = {c.parent_id for c in result.candidates}
        selected = [c.chunk_id for c in result.selected]
        assert set(selected) <= parents  # every selected item is a parent section
        assert all(c.parent_id is None for c in result.selected)
        assert len(selected) == len(set(selected))  # a parent is counted once
        rows = dict(
            conn.execute(
                f"SELECT id, tokens FROM chunks WHERE id IN ({','.join('?' * len(selected))})",
                selected,
            ).fetchall()
        )
        for item in result.selected:
            assert item.tokens == rows[item.chunk_id]
            assert item.prompt_tokens == count_tokens(f"{item.header}\n{item.text}")
        assert result.context_tokens == sum(c.prompt_tokens for c in result.selected)
        # two children of one selected parent: the parent's tokens are counted once
        children = [c.parent_id for c in result.candidates if c.parent_id in set(selected)]
        seen_shared_parent |= len(children) > len(set(children))
        # the parent carries its best child's rank and scores
        first_child = next(c for c in result.candidates if c.parent_id == selected[0])
        assert result.selected[0].rank == first_child.rank
        assert result.selected[0].dense_score == first_child.dense_score
    assert seen_shared_parent


def test_near_duplicates_are_dropped(conn, embeddings):
    """bundle.pdf repeats inspection.pdf: the same passage is sent once."""
    retriever = Retriever(conn, embeddings)
    dropped = across_files = 0
    for fact in corpus.QA_FACTS[:9]:  # the inspection, estimate and invoice facts
        result = retriever.retrieve(fact.question, config=RetrievalConfig(budget_tokens=4000))
        prints = [fingerprint(c.text) for c in result.selected]
        for i, later in enumerate(prints):
            assert not any(is_near_duplicate(later, earlier) for earlier in prints[:i])
        chosen = {c.chunk_id for c in result.selected}
        last = result.selected[-1].rank
        for c in result.candidates:
            if c.rank < last and c.chunk_id not in chosen:  # skipped, not cut by the budget
                dropped += 1
                twin = max(
                    result.selected, key=lambda s: overlap(shingles(c.text), shingles(s.text))
                )
                assert is_near_duplicate(fingerprint(c.text), fingerprint(twin.text))
                across_files += {c.collection, twin.collection} == {"Scans/", "House/"}
    assert dropped > 0 and across_files > 0  # the report's passage and its scanned copy


# ------------------------------------------------------------------ rerank
class Reverser:
    """A stub cross-encoder that scores the last passage highest."""

    name = "reverser"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, query, passages):
        self.calls.append(len(passages))
        n = len(passages)
        return [(i + 1) / (n + 1) for i in range(n)]


def test_a_reversing_reranker_reverses_the_order(conn, embeddings):
    stub = Reverser()
    retriever = Retriever(conn, embeddings, reranker=stub)
    base = retriever.retrieve(HEATER, config=RetrievalConfig(top_k_each=10))
    cfg = RetrievalConfig(top_k_each=10, rerank="minilm")
    result, trace = run(retriever, config=cfg)
    n = stub.calls[0]
    assert n == 10 and "rerank" in trace
    assert [c.chunk_id for c in result.candidates] == [c.chunk_id for c in base.candidates[:n]][
        ::-1
    ]
    assert [c.rank for c in result.candidates] == list(range(1, n + 1))
    top = result.candidates[0]
    assert top.rerank_score == top.score == pytest.approx(n / (n + 1))
    assert top.fused_score == base.candidates[n - 1].fused_score  # the fused score is kept
    assert result.selected[0].chunk_id == top.chunk_id
    assert result.abstain_score == pytest.approx(n / (n + 1))


def test_the_configured_reranker_is_loaded_when_none_is_given(conn, embeddings, monkeypatch):
    import hal9000.knowledge.retrieve.retriever as module

    stub = Reverser()
    asked = []
    monkeypatch.setattr(module, "reranker_for", lambda name: asked.append(name) or stub)
    Retriever(conn, embeddings).retrieve(HEATER, config=RetrievalConfig(rerank="bge-base"))
    assert asked == ["bge-base"] and stub.calls


def test_a_reranker_returning_the_wrong_count_is_an_error(conn, embeddings):
    class Short:
        name = "short"

        def score(self, query, passages):
            return [0.1]

    with pytest.raises(ValueError, match="scores for"):
        Retriever(conn, embeddings, reranker=Short()).retrieve(
            HEATER, config=RetrievalConfig(rerank="minilm")
        )


# ------------------------------------------------------------------ abstain
def test_the_abstain_score_is_the_top_dense_cosine_not_the_fused_score(conn, embeddings, fake):
    result = Retriever(conn, embeddings).retrieve(HEATER)
    top = result.selected[0]
    want = float(unit(fake.vector(HEATER)) @ unit(fake.vector(index_text(top))))
    assert result.abstain_score == pytest.approx(want, abs=1e-5)
    assert result.abstain_score == pytest.approx(top.dense_score)
    assert result.abstain_score != pytest.approx(top.fused_score)  # RRF is ~1/61
    assert result.threshold is None and result.abstained is False


def test_a_bm25_only_top_chunk_gets_its_cosine_looked_up(conn, embeddings, fake):
    question = "CHI-INV-20250314"  # an identifier: invisible to the fake embedder
    cfg = RetrievalConfig(fusion="weighted", alpha=0.0, top_k_each=2)
    result = Retriever(conn, embeddings).retrieve(question, config=cfg)
    top = result.selected[0]
    assert top.dense_score is None and top.bm25_score is not None
    want = float(unit(fake.vector(question)) @ unit(fake.vector(index_text(top))))
    assert result.abstain_score == pytest.approx(want, abs=1e-5)


def test_below_the_threshold_it_abstains_with_the_sources_attached(conn, embeddings):
    retriever = Retriever(conn, embeddings)
    score = retriever.retrieve(HEATER).abstain_score
    set_threshold(conn, profile_id(conn, ACTIVE), None, score + 0.01)
    result = retriever.retrieve(HEATER)
    assert result.abstained is True and result.threshold == pytest.approx(score + 0.01)
    assert result.selected and result.candidates  # the closest sources are still shown
    set_threshold(conn, profile_id(conn, ACTIVE), "none", score - 0.01)
    assert retriever.retrieve(HEATER).abstained is False


def test_the_threshold_switches_with_profile_and_reranker(conn, embeddings):
    set_threshold(conn, profile_id(conn, ACTIVE), None, 0.11)
    set_threshold(conn, profile_id(conn, PARENT_CHILD), None, 0.22)
    set_threshold(conn, profile_id(conn, ACTIVE), "minilm", 0.99)
    active = Retriever(conn, embeddings).retrieve(HEATER)
    assert active.threshold == pytest.approx(0.11)
    other = Retriever(conn, embeddings, profile=PARENT_CHILD).retrieve(HEATER)
    assert other.threshold == pytest.approx(0.22)
    reranked = Retriever(conn, embeddings, reranker=Reverser()).retrieve(
        HEATER, config=RetrievalConfig(rerank="minilm")
    )
    assert reranked.threshold == pytest.approx(0.99)
    assert reranked.abstained is (reranked.abstain_score < 0.99)
    uncalibrated = Retriever(conn, embeddings, reranker=Reverser()).retrieve(
        HEATER, config=RetrievalConfig(rerank="bge-base")
    )
    assert uncalibrated.threshold is None and uncalibrated.abstained is False


def test_bm25_only_mode_never_abstains(conn, embeddings, fake):
    set_threshold(conn, profile_id(conn, ACTIVE), None, 2.0)  # would refuse everything
    set_threshold(conn, profile_id(conn, ACTIVE), "minilm", 2.0)
    retriever = Retriever(conn, embeddings, reranker=Reverser())
    for cfg in (RetrievalConfig(mode="bm25"), RetrievalConfig(mode="bm25", rerank="minilm")):
        result = retriever.retrieve(HEATER, config=cfg)
        assert result.mode == "bm25" and result.selected
        assert result.abstain_score is None and result.abstained is False
        assert all(c.dense_score is None for c in result.candidates)
    assert fake.calls == 0  # no query embedding at all


# ------------------------------------------------------------------ degraded
def test_an_embedding_failure_degrades_to_bm25(conn, caplog):
    embeddings = EmbeddingService(conn, embed_fn=raises(RuntimeError("provider down")))
    set_threshold(conn, profile_id(conn, ACTIVE), None, 2.0)
    with caplog.at_level(logging.WARNING):
        result, trace = run(Retriever(conn, embeddings))
    assert result.mode == "bm25" and result.degraded == "embedding_unavailable"
    assert result.selected and result.abstain_score is None and result.abstained is False
    assert trace == ["embed_query", "bm25", "fuse", "select", "abstain"]
    assert "BM25 only" in caplog.text


def test_a_configuration_error_is_not_mistaken_for_an_outage(conn, monkeypatch):
    monkeypatch.delenv("KB_FAKE")
    with pytest.raises(FakeModeRequired):
        Retriever(conn, EmbeddingService(conn)).retrieve(HEATER)


def test_a_repeated_question_skips_the_embedding_call(conn, embeddings, fake):
    """Retrievers share one bounded process cache: a per-request retriever still hits it."""
    first = Retriever(conn, embeddings).retrieve(HEATER)
    assert fake.calls == 1 and len(DEFAULT_QUERY_CACHE) == 1
    other = FakeEmbedder()
    second, trace = run(Retriever(conn, EmbeddingService(conn, embed_fn=other)))
    assert other.calls == 0 and DEFAULT_QUERY_CACHE.hits == 1
    assert "embed_query" in trace  # a cache hit is still timed
    assert [c.chunk_id for c in first.selected] == [c.chunk_id for c in second.selected]
    [vector] = DEFAULT_QUERY_CACHE._items.values()
    assert not vector.flags.writeable
    with pytest.raises(ValueError):
        vector[0] = 1.0
    own = QueryVectorCache()
    Retriever(conn, EmbeddingService(conn, embed_fn=other), query_cache=own).retrieve(HEATER)
    assert len(own) == 1 and other.calls == 0  # its own cache missed; embed_cache (DB) hit


# ------------------------------------------------------------------ transforms
def test_multi_query_makes_one_generate_call_and_fuses_four_queries(conn, embeddings, fake):
    llm = FakeLLM(['1. water heater age\n- tank manufactured year\n"relief valve dripping"'])
    retriever = Retriever(conn, embeddings, generate=llm)
    result, trace = run(retriever, config=RetrievalConfig(transform="multi"))
    assert len(llm.calls) == 1 and llm.tiers == ["fast"]
    assert trace.count("embed_query") == 4 and trace.count("vector") == 4
    assert trace.count("bm25") == 4 and trace.count("fuse") == 5  # four lists, then across
    assert fake.texts == 4
    assert result.selected and result.standalone_query == HEATER
    plain = retriever.retrieve(HEATER)
    assert [c.chunk_id for c in result.candidates] != [c.chunk_id for c in plain.candidates]

    llm.responses += ["Standalone: How old is the heater?", "a\nb\nc"]
    run(retriever, "and its age?", history=[Turn("user", "heater")],
        config=RetrievalConfig(transform="multi"))  # fmt: skip
    assert len(llm.calls) == 3  # condense + one multi-query call


def test_hyde_embeds_the_hypothetical_passage(conn, embeddings, fake):
    passage = "The water heater was manufactured in 2012 and is 13 years old."
    llm = FakeLLM([passage])
    result, trace = run(
        Retriever(conn, embeddings, generate=llm), config=RetrievalConfig(transform="hyde")
    )
    assert len(llm.calls) == 1 and trace.count("embed_query") == 2  # the passage, the question
    top = result.selected[0]
    knn = float(unit(fake.vector(passage)) @ unit(fake.vector(index_text(top))))
    assert top.dense_score == pytest.approx(knn, abs=1e-5)  # the KNN used the passage
    asked = float(unit(fake.vector(HEATER)) @ unit(fake.vector(index_text(top))))
    assert result.abstain_score == pytest.approx(asked, abs=1e-5)  # abstain: the question
    assert result.abstain_score != pytest.approx(knn, abs=1e-3)
    assert result.standalone_query == HEATER


def test_transforms_need_a_generator_and_survive_its_failure(conn, embeddings, caplog):
    with pytest.raises(ValueError, match="needs a generate"):
        Retriever(conn, embeddings).retrieve(HEATER, config=RetrievalConfig(transform="hyde"))
    broken = FakeLLM([])  # raises AssertionError when called
    with caplog.at_level(logging.WARNING):
        result = Retriever(conn, embeddings, generate=broken).retrieve(
            HEATER, config=RetrievalConfig(transform="multi")
        )
    assert result.selected and "multi transform failed" in caplog.text
    empty = Retriever(conn, embeddings, generate=FakeLLM([""])).retrieve(
        HEATER, config=RetrievalConfig(transform="hyde")
    )
    assert empty.abstain_score == pytest.approx(
        Retriever(conn, embeddings).retrieve(HEATER).abstain_score
    )


# ------------------------------------------------------------------ profiles
def test_the_profile_is_the_active_one_at_call_time(conn, embeddings):
    from hal9000.knowledge.index.profiles import activate_profile, get_profile

    retriever = Retriever(conn, embeddings)
    assert retriever.retrieve(HEATER).profile == ACTIVE
    activate_profile(conn, PARENT_CHILD)
    assert retriever.retrieve(HEATER).profile == PARENT_CHILD
    stored = Retriever(conn, embeddings, profile=get_profile(conn, ACTIVE))
    assert stored.retrieve(HEATER).profile == ACTIVE
    unsaved = get_profile(conn, ACTIVE).model_copy(update={"id": None})
    assert Retriever(conn, embeddings, profile=unsaved).retrieve(HEATER).profile == ACTIVE


def test_a_missing_profile_is_a_clear_error(conn, embeddings, tmp_path):
    from hal9000.knowledge.storage import open_db

    with pytest.raises(NoActiveProfile, match="no index profile named"):
        Retriever(conn, embeddings, profile="nope").retrieve(HEATER)
    empty = open_db(tmp_path / "empty.db")
    try:
        with pytest.raises(ConfigMissing, match="no active index profile"):
            Retriever(empty, EmbeddingService(empty, embed_fn=FakeEmbedder())).retrieve(HEATER)
    finally:
        empty.close()


# ------------------------------------------------------------------ search
def test_search_documents_returns_retrieval_without_generation(conn, embeddings):
    result = search_documents(
        conn, embeddings, HEATER, k=2, config=RetrievalConfig(transform="multi")
    )
    assert len(result.selected) == 2 and len(result.candidates) > 2
    assert result.context_tokens == sum(c.prompt_tokens for c in result.selected)
    filtered = search_documents(
        conn, embeddings, "total", filters=SearchFilters(doc_type="estimate")
    )
    assert {c.doc_type for c in filtered.candidates} == {"estimate"}
    with pytest.raises(ValueError):
        search_documents(conn, embeddings, HEATER, k=0)


# ------------------------------------------------------------------ speed
def test_retrieval_is_fast_in_fake_mode(conn, embeddings):
    """A question over the fixture corpus: well under 50 ms locally (no generate calls)."""
    retriever = Retriever(conn, embeddings)
    for fact in corpus.QA_FACTS:  # warm the embedding cache and token counts
        retriever.retrieve(fact.question)
    timings = []
    for fact in corpus.QA_FACTS:
        retriever.query_cache = type(retriever.query_cache)()  # embed_query hits the DB cache
        started = time.perf_counter()
        retriever.retrieve(fact.question)
        timings.append((time.perf_counter() - started) * 1000)
    print(f"retrieval ms: median {statistics.median(timings):.1f}, max {max(timings):.1f}")
    assert statistics.median(timings) < 50


def test_small_helpers(conn, embeddings):
    from hal9000.knowledge.index.profiles import get_profile
    from hal9000.knowledge.retrieve import QueryVectorCache
    from hal9000.knowledge.retrieve.retriever import _date

    cache = QueryVectorCache(size=2)
    for i in range(3):
        cache.put(("m", 1, str(i)), np.zeros(1))
    assert cache.get(("m", 1, "0")) is None and cache.get(("m", 1, "2")) is not None
    assert (cache.hits, cache.misses) == (1, 1)
    assert _date("2023-06-22") == date(2023, 6, 22) and _date("June") is None and _date("") is None
    profile = get_profile(conn, ACTIVE)
    assert Retriever(conn, embeddings)._cosine(profile, 10**9, np.ones(64)) is None


# ------------------------------------------------------------------ review fixes
def test_an_empty_selection_abstains_and_an_overflowing_top_item_is_logged(
    conn, embeddings, caplog
):
    with caplog.at_level(logging.WARNING):
        result = Retriever(conn, embeddings).retrieve(
            HEATER, config=RetrievalConfig(budget_tokens=10)
        )
    assert result.candidates and result.selected == [] and result.context_tokens == 0
    assert result.abstained is True and result.abstain_score is None
    assert "top item alone exceeds the 10-token context budget" in caplog.text


@pytest.mark.parametrize("kind", ["run", "month"])
def test_a_spending_cap_on_the_query_embedding_propagates(conn, kind):
    from hal9000.cost.metering import BudgetExceeded, MonthlyCapExceeded

    error = BudgetExceeded("run cap") if kind == "run" else MonthlyCapExceeded("month cap")
    embeddings = EmbeddingService(conn, embed_fn=raises(error))
    with pytest.raises(type(error)):
        Retriever(conn, embeddings).retrieve(HEATER)


def test_a_spending_cap_in_condense_or_a_transform_propagates(conn, embeddings):
    from hal9000.cost.metering import BudgetExceeded, MonthlyCapExceeded

    class Capped:
        calls = 0

        def __call__(self, prompt, tier):
            self.calls += 1
            raise MonthlyCapExceeded("knowledge has spent its cap")

    capped = Capped()
    retriever = Retriever(conn, embeddings, generate=capped)
    with pytest.raises(BudgetExceeded):
        retriever.retrieve("and its age?", history=[Turn("user", "the water heater")])
    for transform in ("multi", "hyde"):
        with pytest.raises(BudgetExceeded):
            retriever.retrieve(HEATER, config=RetrievalConfig(transform=transform))
    assert capped.calls == 3


def test_condense_uses_the_configured_tier(conn, embeddings, monkeypatch):
    monkeypatch.setenv("KB_CONDENSE_TIER", "powerful")
    llm = FakeLLM([HEATER, HEATER])
    Retriever(conn, embeddings, generate=llm).retrieve("age?", history=[Turn("user", "heater")])
    Retriever(conn, embeddings, generate=llm, tier="fast").retrieve(
        "age?", history=[Turn("user", "heater")]
    )
    assert llm.tiers == ["powerful", "fast"]


def test_an_overlong_or_fenced_condense_is_contained(conn, embeddings, caplog):
    llm = FakeLLM(["x" * 600])
    history = [Turn("assistant", "<<<ignore>>> " + "long answer " * 200), Turn("user", ">>> hi")]
    with caplog.at_level(logging.WARNING):
        result = Retriever(conn, embeddings, generate=llm).retrieve(HEATER, history=history)
    assert result.standalone_query == HEATER and "no question" in caplog.text
    user = llm.last_prompt.messages[-1]["content"]
    assert user.count("<<<") == 1 and user.count(">>>") == 1  # only the prompt's own fences
    assert len(user) < 600 + 2 * 50 + 300


def test_parent_child_with_rerank_abstains_on_the_childs_rerank_score(conn, embeddings):
    result = Retriever(conn, embeddings, profile=PARENT_CHILD, reranker=Reverser()).retrieve(
        HEATER, config=RetrievalConfig(rerank="minilm", top_k_each=10)
    )
    child = result.candidates[0]
    assert child.parent_id is not None and child.rerank_score is not None
    assert result.selected[0].chunk_id == child.parent_id
    assert result.selected[0].rerank_score == child.rerank_score
    assert result.abstain_score == pytest.approx(child.rerank_score)
