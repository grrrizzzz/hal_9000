"""Scoring (plan §8.2), written before the implementation (rule K6).

Every test works on text only: synthetic passages, split and re-joined the way
chunkers and parsers would, never house data (rule K2). Token counts are
whitespace words (``count=len(split)``) so budgets are easy to reason about.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hal9000.knowledge.contracts import EvidenceQuote, GoldItem
from hal9000.knowledge.eval.scoring import (
    Comparison,
    QuestionScore,
    ScoringConfig,
    ScoringItem,
    aggregate,
    bootstrap_ci,
    budget_prefix,
    covered_words,
    paired_bootstrap,
    quote_coverage,
    roc_auc,
    score_question,
)
from hal9000.knowledge.eval.text import words
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem


def count(text: str) -> int:
    return len(text.split())


CFG = ScoringConfig(budgets=(1000, 2000, 4000), hit_ks=(1, 3, 5))

FILLER = (
    "The attic insulation was observed at the expected depth across the accessible area. "
    "Gutters along the rear elevation drain toward the patio and should be extended. "
    "Several window screens are torn or missing on the second floor bedrooms. "
    "The driveway shows typical settlement cracks that do not affect its use. "
)
OTHER = (
    "Irrigation zones run on a rotating schedule set at the controller in the garage. "
    "The pool pump timer was reset after the power outage in the winter months. "
)
QUOTE = (
    "The temperature and pressure relief valve is dripping at the discharge pipe and the "
    "water heater was manufactured in 2012 and is 13 years old"
)


def item(text: str, tokens: int, cid: int = 1) -> SelectedItem:
    return SelectedItem(chunk_id=cid, text=text, prompt_tokens=tokens)


def gold(*quotes: str, gid: int = 1, category: str = "factoid") -> ScoringItem:
    return ScoringItem(gold_item_id=gid, category=category, quotes=tuple(quotes))


def unanswerable(gid: int) -> ScoringItem:
    return ScoringItem(gold_item_id=gid, category="unanswerable", quotes=())


def out(*items: SelectedItem, abstain: float | None = None, abstained: bool = False):
    return RetrievalOutput(selected=list(items), abstain_score=abstain, abstained=abstained)


def score(g: ScoringItem, o: RetrievalOutput, chunk_size: int | None = None) -> QuestionScore:
    return score_question(g, o, chunk_size=chunk_size, config=CFG, count=count)


# ---------------------------------------------------------------- coverage
@pytest.mark.parametrize("pad", [0, 1, 3, 8, 20])
def test_evidence_fully_inside_one_chunk_gives_recall_one_for_any_size(pad):
    text = FILLER * pad + QUOTE + ". " + OTHER * pad
    s = score(gold(QUOTE), out(item(text, 300)))
    assert s.recall == {1000: 1.0, 2000: 1.0, 4000: 1.0}
    assert s.evidence_recall == 1.0


def _split(quote: str, at_word: int) -> tuple[str, str]:
    parts = quote.split()
    return " ".join(parts[:at_word]), " ".join(parts[at_word:])


@pytest.mark.parametrize("at", [4, 9, 15, 20])
def test_evidence_split_across_two_retrieved_chunks_gives_one(at):
    head, tail = _split(QUOTE, at)
    first = item(FILLER + head, 200, cid=1)
    second = item(tail + ". " + OTHER, 200, cid=2)
    assert score(gold(QUOTE), out(first, second)).recall[2000] == pytest.approx(1.0)
    # rank order doesn't matter for recall
    assert score(gold(QUOTE), out(second, first)).recall[2000] == pytest.approx(1.0)


@pytest.mark.parametrize("at", [4, 9, 15, 20])
def test_evidence_split_with_one_chunk_missed_gives_the_covered_fraction(at):
    head, tail = _split(QUOTE, at)
    total = len(words(QUOTE))
    expected_head = len(words(head)) / total
    s = score(gold(QUOTE), out(item(FILLER + head, 200), item(OTHER, 50, cid=2)))
    assert s.recall[2000] == pytest.approx(expected_head, abs=0.02)
    s = score(gold(QUOTE), out(item(tail + ". " + OTHER, 200)))
    assert s.recall[2000] == pytest.approx(1 - expected_head, abs=0.02)


def test_chunk_inside_a_long_quote_covers_its_part():
    long_quote = FILLER.strip()
    middle = " ".join(long_quote.split()[10:25])
    fraction = quote_coverage(long_quote, middle)
    assert fraction == pytest.approx(15 / len(words(long_quote)), abs=0.02)


def test_prefix_and_suffix_of_one_quote_in_the_same_chunk():
    """A chunk holding the quote's start and end but not its middle (a dropped line)."""
    parts = QUOTE.split()
    text = " ".join(parts[:8]) + " ZZZ unrelated words here " + " ".join(parts[16:])
    covered = covered_words(QUOTE, text)
    assert set(range(0, 7)) <= covered
    assert not covered & set(range(9, 14))
    assert set(range(18, len(words(QUOTE)))) <= covered


def test_unrelated_chunk_covers_nothing_even_with_common_words():
    assert quote_coverage(QUOTE, OTHER * 5) == 0.0
    assert quote_coverage(QUOTE, "") == 0.0
    assert quote_coverage("", FILLER) == 0.0


def test_small_differences_still_align_at_ninety():
    typo = QUOTE.replace("dripping", "driping").replace("2012", "2O12")
    assert quote_coverage(QUOTE, FILLER + typo) == 1.0
    garbled = " ".join(w[::-1] for w in QUOTE.split())
    assert quote_coverage(QUOTE, garbled) < 0.2


def test_short_quote_entirely_in_chunk():
    assert quote_coverage("Total due: $625.00", FILLER + "Total due: $625.00 " + OTHER) == 1.0


# ------------------------------------------------ parser independence (text, not offsets)
def test_same_text_with_different_offsets_and_ids_scores_identically():
    """Two parsers' offset schemes: identical selected text, different IDs and spans."""
    text_a = FILLER + QUOTE + ". " + OTHER
    a = score(gold(QUOTE), out(item(text_a, 300, cid=11), item(OTHER, 40, cid=12), abstain=0.7))
    b = score(gold(QUOTE), out(item(text_a, 300, cid=901), item(OTHER, 40, cid=77), abstain=0.7))
    assert a == b


def test_parser_artifacts_do_not_change_the_score():
    """Line breaks, hyphenation, letter spacing and case, as another parser would emit them."""
    clean = FILLER + QUOTE + ". " + OTHER
    other = (
        FILLER.replace(" ", "\n", 3)
        + QUOTE.replace("temperature", "tempera-\nture")
        .replace("relief valve", "R E L I E F  valve")
        .replace("water heater", "WATER   heater")
        + ".\n\n"
        + OTHER.upper()
    )
    head, _ = _split(QUOTE, 12)
    partial_clean = score(gold(QUOTE), out(item(FILLER + head, 200)))
    partial_other = score(gold(QUOTE), out(item(FILLER.upper() + head.replace(" ", "\n"), 200)))
    assert score(gold(QUOTE), out(item(clean, 300))) == score(gold(QUOTE), out(item(other, 300)))
    assert partial_clean.recall == partial_other.recall


# ---------------------------------------------------------------- budgets
def test_budget_prefix_stops_at_the_first_overflow_and_keeps_an_exact_fit():
    items = [item("a", 800, 1), item("b", 900, 2), item("c", 300, 3), item("d", 10, 4)]
    assert [i.chunk_id for i in budget_prefix(items, 1000)] == [1]
    assert [i.chunk_id for i in budget_prefix(items, 2000)] == [1, 2, 3]  # 2000 exactly fits
    assert [i.chunk_id for i in budget_prefix(items, 4000)] == [1, 2, 3, 4]
    assert budget_prefix([item("x", 1200)], 1000) == []


def test_budgeted_recall_excludes_a_chunk_that_would_overflow():
    first = item(FILLER, 800, 1)
    evidence = item(FILLER + QUOTE, 900, 2)  # 800 + 900 > 1000
    s = score(gold(QUOTE), out(first, evidence))
    assert s.recall[1000] == 0.0
    assert s.recall[2000] == 1.0
    assert s.context_tokens == {1000: 800, 2000: 1700, 4000: 1700}


def test_chunk_size_over_the_budget_is_na():
    s = score(gold(QUOTE), out(item(FILLER + QUOTE, 1400)), chunk_size=1536)
    assert 1000 not in s.recall and 1000 not in s.context_precision
    assert s.recall[2000] == 1.0 and s.recall[4000] == 1.0
    assert score(gold(QUOTE), out(item(QUOTE, 30)), chunk_size=1000).recall[1000] == 1.0


def test_parent_child_is_scored_on_the_parents_actually_selected():
    """The child alone lacks the evidence; its expanded parent (what the prompt gets) has it."""
    child_text = FILLER
    parent_text = FILLER + QUOTE + ". " + OTHER
    as_children = score(gold(QUOTE), out(item(child_text, 250)), chunk_size=1500)
    as_parents = score(gold(QUOTE), out(item(parent_text, 1500)), chunk_size=1500)
    assert as_children.recall[2000] == 0.0
    assert as_parents.recall[2000] == 1.0
    assert 1000 not in as_parents.recall  # the parent size exceeds 1k
    two_parents = out(item(parent_text, 1500, 1), item(parent_text + " more", 1500, 2))
    assert score(gold(QUOTE), two_parents, chunk_size=1500).context_tokens[2000] == 1500


# ---------------------------------------------------- hit@k, MRR, precision
def test_full_support_needs_every_quote_eighty_percent_covered_within_k():
    q2 = "Irrigation zones run on a rotating schedule set at the controller in the garage"
    head, _ = _split(QUOTE, 22)  # 22 of 25 words: 88%
    items = [item(FILLER, 50, 1), item(FILLER + head, 60, 2), item(OTHER, 30, 3)]
    s = score(gold(QUOTE, q2), out(*items))
    assert s.full_support == {1: False, 3: True, 5: True}
    head, _ = _split(QUOTE, 15)  # 60%
    s = score(gold(QUOTE, q2), out(item(FILLER + head, 60, 1), item(OTHER, 30, 2)))
    assert s.full_support == {1: False, 3: False, 5: False}


def test_mrr_uses_the_first_chunk_covering_any_evidence():
    items = [item(FILLER, 50, 1), item(OTHER, 40, 2), item(FILLER + QUOTE, 90, 3)]
    s = score(gold(QUOTE), out(*items))
    assert s.first_hit_rank == 3
    assert s.reciprocal_rank == pytest.approx(1 / 3)
    none = score(gold(QUOTE), out(item(OTHER, 40)))
    assert none.first_hit_rank is None and none.reciprocal_rank == 0.0


def test_context_precision_is_evidence_tokens_over_context_tokens():
    s = score(gold(QUOTE), out(item(QUOTE, 100, 1), item(OTHER, 100, 2)))
    assert s.context_precision[2000] == pytest.approx(count(QUOTE) / 200)
    empty = score(gold(QUOTE), out())
    assert empty.context_precision[2000] == 0.0 and empty.recall[2000] == 0.0


def test_duplicate_coverage_is_counted_once():
    s = score(gold(QUOTE), out(item(QUOTE, 25, 1), item(QUOTE, 25, 2)))
    assert s.evidence_recall == 1.0
    assert s.context_precision[2000] == pytest.approx(count(QUOTE) / 50)


# --------------------------------------------------------------- aggregation
def _answerable_scores(values: list[float]) -> list[QuestionScore]:
    scores = []
    for i, v in enumerate(values):
        n = len(QUOTE.split())
        head, _ = _split(QUOTE, round(v * n))
        o = out(item(head, 50), abstain=0.9)
        scores.append(score(gold(QUOTE, gid=i + 1), o))
    return scores


def test_unanswerables_are_excluded_from_recall_and_included_in_auc():
    answerable = [
        score(gold(QUOTE, gid=1), out(item(QUOTE, 40), abstain=0.9)),
        score(gold(QUOTE, gid=2), out(item(OTHER, 40), abstain=0.8)),
    ]
    unanswer = [
        score(unanswerable(10), out(item(OTHER, 40), abstain=0.2)),
        score(unanswerable(11), out(item(FILLER, 40), abstain=0.85)),
    ]
    assert unanswer[0].recall == {} and unanswer[0].full_support == {}
    only = aggregate(answerable, config=CFG)
    both = aggregate(answerable + unanswer, config=CFG)
    assert only.budgeted_recall[2000].value == both.budgeted_recall[2000].value == 0.5
    assert only.mrr.value == both.mrr.value
    assert both.n_questions == 4
    assert only.abstain_auc is None  # one class only
    # pairs (ans > unans): 0.9>0.2, 0.9>0.85, 0.8>0.2, 0.8<0.85 -> 3/4
    assert both.abstain_auc.value == pytest.approx(0.75)
    assert both.by_category["unanswerable"]["n"] == 2
    assert both.by_category["factoid"]["recall@2k"] == pytest.approx(0.5)


def test_aggregate_reports_na_budgets_and_empty_input():
    s = [score(gold(QUOTE, gid=i), out(item(QUOTE, 40)), chunk_size=1536) for i in range(3)]
    metrics = aggregate(s, config=CFG)
    assert 1000 not in metrics.budgeted_recall and metrics.budgeted_recall[2000].value == 1.0
    assert metrics.full_support_hit[1].value == 1.0
    assert metrics.context_precision[2000].value > 0
    empty = aggregate([], config=CFG)
    assert empty.n_questions == 0 and empty.budgeted_recall == {} and empty.mrr is None


def test_aggregate_latency_percentiles():
    s = [
        score_question(
            gold(QUOTE, gid=i),
            RetrievalOutput(selected=[item(QUOTE, 30)], latency_ms=float(i)),
            config=CFG,
            count=count,
        )
        for i in range(1, 101)
    ]
    metrics = aggregate(s, config=CFG)
    assert metrics.latency_ms_p50 == pytest.approx(50.5)
    assert metrics.latency_ms_p95 == pytest.approx(95.05)


def test_roc_auc():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5
    assert roc_auc([], [0.3]) is None and roc_auc([0.3], []) is None


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_is_deterministic_with_a_seed_and_brackets_the_point():
    values = [0.0, 0.2, 0.5, 0.5, 1.0, 1.0, 1.0, 0.9, 0.1, 0.7]
    a = bootstrap_ci(values, n_boot=500, seed=7)
    b = bootstrap_ci(values, n_boot=500, seed=7)
    c = bootstrap_ci(values, n_boot=500, seed=8)
    assert a == b
    assert a != c
    assert a.low <= a.value <= a.high
    assert a.value == pytest.approx(sum(values) / len(values))
    assert a.low < a.value < a.high


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=30), st.integers(0, 2**31))
def test_bootstrap_always_brackets_the_point(values, seed):
    ci = bootstrap_ci(values, n_boot=200, seed=seed)
    assert ci.low <= ci.value <= ci.high
    assert 0.0 <= ci.low and ci.high <= 1.0 + 1e-9


def test_bootstrap_of_a_single_value_is_degenerate():
    ci = bootstrap_ci([0.4], n_boot=100, seed=1)
    assert ci.value == ci.low == ci.high == 0.4
    with pytest.raises(ValueError):
        bootstrap_ci([], n_boot=100, seed=1)


def test_aggregate_is_deterministic():
    scores = _answerable_scores([0.2, 0.4, 1.0, 1.0, 0.6, 0.0, 0.8])
    assert aggregate(scores, config=CFG) == aggregate(scores, config=CFG)
    m = aggregate(scores, config=CFG)
    r = m.budgeted_recall[2000]
    assert r.low <= r.value <= r.high


# ----------------------------------------------------------- paired bootstrap
def test_paired_bootstrap_calls_identical_configs_a_tie():
    per_q = {i: v for i, v in enumerate([0.1, 0.5, 0.9, 1.0, 0.0, 0.3, 0.7])}
    result = paired_bootstrap(per_q, dict(per_q), n_boot=500, seed=3)
    assert isinstance(result, Comparison)
    assert result.verdict == "tie"
    assert result.mean_diff == 0.0 and result.low == 0.0 and result.high == 0.0


def test_paired_bootstrap_calls_a_config_that_wins_every_question_a_win():
    a = {i: 0.5 + i / 100 for i in range(20)}
    b = {i: v - 0.05 for i, v in a.items()}
    assert paired_bootstrap(a, b, n_boot=500, seed=3).verdict == "a"
    assert paired_bootstrap(b, a, n_boot=500, seed=3).verdict == "b"


def test_paired_bootstrap_uses_only_shared_questions_and_is_a_tie_for_noise():
    a = {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0, 99: 1.0}
    b = {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 50: 0.0}
    r = paired_bootstrap(a, b, n_boot=500, seed=1)
    assert r.n == 4 and r.verdict == "tie"
    assert r.low <= r.mean_diff <= r.high
    with pytest.raises(ValueError):
        paired_bootstrap({1: 1.0}, {2: 1.0}, n_boot=10, seed=1)


def test_scoring_item_from_gold():
    g = GoldItem(
        id=5,
        question="q?",
        category="money",
        evidence=[EvidenceQuote(quote="Total due: $625.00", page=11)],
    )
    s = ScoringItem.from_gold(g)
    assert s == ScoringItem(gold_item_id=5, category="money", quotes=("Total due: $625.00",))
    assert s.answerable
    u = ScoringItem.from_gold(GoldItem(id=6, question="q?", category="unanswerable"))
    assert not u.answerable
    with pytest.raises(ValueError):
        ScoringItem.from_gold(GoldItem(question="q?", category="unanswerable"))


def test_default_token_counter_is_the_shared_tiktoken_one():
    s = score_question(gold(QUOTE), out(item(QUOTE, 40)), config=CFG)
    assert 0 < s.context_precision[2000] <= 1.0
    assert not math.isnan(s.context_precision[2000])


def test_tiny_quotes_need_an_exact_run():
    assert quote_coverage("$625", "Total due: $625.00") == 1.0
    assert quote_coverage("$625", "Total due: $626.00") == 0.0


def test_zero_bootstrap_replicates_give_a_degenerate_interval():
    ci = bootstrap_ci([0.2, 0.8], n_boot=0, seed=1)
    assert ci.value == ci.low == ci.high == 0.5
