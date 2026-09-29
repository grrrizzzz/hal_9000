"""Scoring that is fair across chunk sizes and parsers (plan §8.2).

Everything is computed on the **exact** output of ``select`` (after parent
expansion and de-duplication), in rank order, with each item's prompt token
count, and on evidence *text*, never offsets:

- **Coverage.** A selected item covers part of an evidence quote where they
  align with ``rapidfuzz`` partial alignment at >= 90 similarity, on
  normalized text (:func:`hal9000.knowledge.eval.text.joined`: letters and
  digits only, so case, whitespace, hyphenation and letter spacing don't
  matter). The whole quote is tried first, then the item inside the quote
  (an item smaller than the quote), then the longest prefix and suffix of the
  quote (a quote straddling a chunk boundary). Within an alignment, the quote
  characters that match (Indel opcodes) are covered; a quote *word* is
  covered when at least half its characters are. A partial alignment must
  span at least ``min_span_words`` words, so a stray common phrase doesn't
  count.
- **Evidence recall** is the share of the question's evidence words covered
  by the union of the selected items.
- **Budgeted recall @ B** fills the context in rank order and stops at the
  first item that would overflow ``B`` (it is excluded). A chunk size larger
  than ``B`` is N/A: the key is absent.
- **Full-support hit @ k**: every quote at least 80% covered within the top k.
- **MRR**: reciprocal rank of the first item covering any evidence.
- **Context precision @ B**: covered evidence tokens / context tokens.
- **Abstention separability**: ROC AUC of the abstain score, answerable
  (positive) vs unanswerable. Unanswerables never enter recall.
- **Uncertainty**: a seeded percentile bootstrap (the interval is widened to
  include the point estimate, so it always brackets it), and a paired
  bootstrap over per-question differences to compare two configurations: a
  difference counts only when its interval excludes zero.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel

from hal9000.knowledge.contracts import EvalMetrics, GoldItem, MetricCI
from hal9000.knowledge.eval.text import joined, joined_text
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem

__all__ = [
    "Comparison",
    "QuestionScore",
    "ScoringConfig",
    "ScoringItem",
    "aggregate",
    "bootstrap_ci",
    "budget_label",
    "budget_prefix",
    "covered_words",
    "paired_bootstrap",
    "quote_coverage",
    "roc_auc",
    "score_question",
]

TokenCounter = Callable[[str], int]

#: An exact run of this many joined characters must be shared before aligning.
_SEED = 5


@dataclass(frozen=True)
class ScoringConfig:
    """Scoring parameters (``scoring`` in ``config/knowledge_eval.yaml``)."""

    budgets: tuple[int, ...] = (1000, 2000, 4000)
    hit_ks: tuple[int, ...] = (1, 3, 5, 10)
    min_similarity: float = 90.0
    full_support: float = 0.8
    min_span_words: int = 3
    word_cover: float = 0.5
    n_boot: int = 1000
    seed: int = 20260928


@dataclass(frozen=True)
class ScoringItem:
    """What scoring needs from a gold item: its evidence quotes and category."""

    gold_item_id: int
    category: str
    quotes: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @classmethod
    def from_gold(cls, item: GoldItem) -> ScoringItem:
        if item.id is None:
            raise ValueError("a gold item must be stored (have an id) before it is scored")
        return cls(item.id, item.category, tuple(e.quote for e in item.evidence))


@dataclass(frozen=True)
class QuestionScore:
    """One question's scores. Budget-keyed dicts omit N/A budgets."""

    gold_item_id: int
    category: str
    answerable: bool
    recall: dict[int, float] = field(default_factory=dict)
    evidence_recall: float | None = None
    full_support: dict[int, bool] = field(default_factory=dict)
    first_hit_rank: int | None = None
    reciprocal_rank: float | None = None
    context_precision: dict[int, float] = field(default_factory=dict)
    context_tokens: dict[int, int] = field(default_factory=dict)
    abstain_score: float | None = None
    abstained: bool = False
    latency_ms: float | None = None


# ------------------------------------------------------------------ coverage
def _matched(needle: str, q0: int, hay: str, hit, out: set[int]) -> None:
    """Mark the quote characters an alignment of ``needle`` (at ``q0`` in the quote) matched.

    ``hit`` aligns ``needle[src_start:src_end]`` with ``hay[dest_start:dest_end]``;
    only characters in equal Indel blocks inside that window count.
    """
    a, b = hit.src_start, hit.src_end
    for op in Indel.opcodes(needle[a:b], hay[hit.dest_start : hit.dest_end]):
        if op.tag == "equal":
            out.update(range(q0 + a + op.src_start, q0 + a + op.src_end))


def _aligned(needle: str, hay: str, cutoff: float):
    return fuzz.partial_ratio_alignment(needle, hay, score_cutoff=cutoff)


def _longest(
    qj: str, ends: Sequence[int], cj: str, cutoff: float, *, prefix: bool
) -> tuple[int, object] | None:
    """Binary search for the longest word-aligned prefix (or suffix) of ``qj`` found in ``cj``.

    ``ends`` are candidate cut points (character offsets in ``qj``) ordered from
    the shortest candidate to the longest.
    """
    lo, hi, best = 0, len(ends) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        cut = ends[mid]
        needle = qj[:cut] if prefix else qj[cut:]
        hit = _aligned(needle, cj, cutoff)
        if hit is not None:
            best = (cut, hit)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _seeded(qj: str, cj: str) -> bool:
    """Cheap filter: do the two share an exact run of :data:`_SEED` characters?"""
    if len(qj) <= _SEED:
        return qj in cj
    return any(qj[i : i + _SEED] in cj for i in range(len(qj) - _SEED + 1))


def covered_words(quote: str, text: str, config: ScoringConfig | None = None) -> set[int]:
    """Indices of the quote's words that ``text`` covers."""
    cfg = config or ScoringConfig()
    q = joined(quote)
    qj = q.text
    cj = joined_text(text)
    if not qj or not cj or not _seeded(qj, cj):
        return set()
    cutoff = cfg.min_similarity
    chars: set[int] = set()
    hit = _aligned(qj, cj, cutoff) if len(qj) <= len(cj) else None
    if hit is not None:
        _matched(qj, 0, cj, hit, chars)
    else:
        inside = _aligned(cj, qj, cutoff) if len(cj) < len(qj) else None
        if inside is not None:
            # the item lies inside the quote: swap the roles back
            for op in Indel.opcodes(
                qj[inside.dest_start : inside.dest_end], cj[inside.src_start : inside.src_end]
            ):
                if op.tag == "equal":
                    at = inside.dest_start
                    chars.update(range(at + op.src_start, at + op.src_end))
        n = q.n_words
        k = min(cfg.min_span_words, n)
        if n > k:
            # prefixes of k..n-1 words, suffixes of k..n-1 words (shortest first)
            prefix_ends = [q.spans[i - 1][1] for i in range(k, n)]
            suffix_starts = [q.spans[n - i][0] for i in range(k, n)]
            found = _longest(qj, prefix_ends, cj, cutoff, prefix=True)
            if found is not None:
                cut, h = found
                _matched(qj[:cut], 0, cj, h, chars)
            found = _longest(qj, suffix_starts, cj, cutoff, prefix=False)
            if found is not None:
                cut, h = found
                _matched(qj[cut:], cut, cj, h, chars)
    return {
        i
        for i, (a, b) in enumerate(q.spans)
        if sum(1 for j in range(a, b) if j in chars) >= cfg.word_cover * (b - a)
    }


def quote_coverage(quote: str, text: str, config: ScoringConfig | None = None) -> float:
    """The fraction of ``quote``'s words that ``text`` covers."""
    n = joined(quote).n_words
    return len(covered_words(quote, text, config)) / n if n else 0.0


# -------------------------------------------------------------------- budget
def budget_prefix(items: Sequence[SelectedItem], budget: int) -> list[SelectedItem]:
    """Items in rank order up to ``budget`` tokens; the first that would overflow stops it."""
    kept: list[SelectedItem] = []
    used = 0
    for item in items:
        if used + item.prompt_tokens > budget:
            break
        kept.append(item)
        used += item.prompt_tokens
    return kept


def budget_label(budget: int) -> str:
    """``2000`` -> ``"2k"`` (``by_category`` keys: ``recall@2k``)."""
    return f"{budget // 1000}k" if budget % 1000 == 0 else str(budget)


def _default_counter() -> TokenCounter:
    from hal9000.knowledge.tokens import count_tokens

    return count_tokens


def score_question(
    item: ScoringItem,
    output: RetrievalOutput,
    *,
    chunk_size: int | None = None,
    config: ScoringConfig | None = None,
    count: TokenCounter | None = None,
) -> QuestionScore:
    """Score one question on ``select``'s output.

    Args:
        chunk_size: the profile's chunk size (the *parent* size for
            parent-child); budgets smaller than it are N/A. None: never N/A.
        count: model-token counter for evidence tokens (context precision);
            defaults to the shared ``cl100k_base`` counter.
    """
    cfg = config or ScoringConfig()
    selected = list(output.selected)
    base = {
        "gold_item_id": item.gold_item_id,
        "category": item.category,
        "answerable": item.answerable,
        "abstain_score": output.abstain_score,
        "abstained": output.abstained,
        "latency_ms": output.latency_ms,
    }
    if not item.answerable:
        return QuestionScore(**base)
    counter = count or _default_counter()
    quote_words = [joined(q).n_words for q in item.quotes]
    quote_tokens = [counter(q) for q in item.quotes]
    total_words = sum(quote_words)
    # coverage[rank][quote] -> covered word indices
    coverage = [[covered_words(q, s.text, cfg) for q in item.quotes] for s in selected]

    def union(upto: int) -> list[set[int]]:
        sets: list[set[int]] = [set() for _ in item.quotes]
        for row in coverage[:upto]:
            for qi, cov in enumerate(row):
                sets[qi] |= cov
        return sets

    def recall_of(sets: list[set[int]]) -> float:
        return sum(len(s) for s in sets) / total_words if total_words else 0.0

    recall: dict[int, float] = {}
    precision: dict[int, float] = {}
    context: dict[int, int] = {}
    for budget in cfg.budgets:
        if chunk_size is not None and chunk_size > budget:
            continue
        kept = budget_prefix(selected, budget)
        sets = union(len(kept))
        tokens = sum(s.prompt_tokens for s in kept)
        evidence = sum((len(s) / n) * t for s, n, t in zip(sets, quote_words, quote_tokens) if n)
        recall[budget] = recall_of(sets)
        context[budget] = tokens
        precision[budget] = min(1.0, evidence / tokens) if tokens else 0.0
    support: dict[int, bool] = {}
    for k in cfg.hit_ks:
        sets = union(k)
        support[k] = all(
            n == 0 or len(s) / n >= cfg.full_support for s, n in zip(sets, quote_words)
        )
    first = next((rank for rank, row in enumerate(coverage, 1) if any(row)), None)
    return QuestionScore(
        **base,
        recall=recall,
        evidence_recall=recall_of(union(len(selected))),
        full_support=support,
        first_hit_rank=first,
        reciprocal_rank=1.0 / first if first else 0.0,
        context_precision=precision,
        context_tokens=context,
    )


# --------------------------------------------------------------- statistics
def roc_auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """P(positive > negative) + half the ties (Mann-Whitney); None when a class is empty."""
    if len(positives) == 0 or len(negatives) == 0:
        return None
    pos = np.asarray(positives, dtype=float)[:, None]
    neg = np.asarray(negatives, dtype=float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def _ci(value: float, samples: np.ndarray) -> MetricCI:
    if samples.size == 0:
        return MetricCI(value=value, low=value, high=value)
    low, high = np.percentile(samples, [2.5, 97.5])
    return MetricCI(value=value, low=min(float(low), value), high=max(float(high), value))


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 1000, seed: int = 0) -> MetricCI:
    """Mean with a seeded percentile-bootstrap 95% interval that brackets it."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("no values to bootstrap")
    value = float(data.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(n_boot, data.size))
    return _ci(value, data[idx].mean(axis=1))


def _auc_ci(scores: Sequence[QuestionScore], n_boot: int, seed: int) -> MetricCI | None:
    pairs = [(s.abstain_score, s.answerable) for s in scores if s.abstain_score is not None]
    pos = [v for v, a in pairs if a]
    neg = [v for v, a in pairs if not a]
    value = roc_auc(pos, neg)
    if value is None:
        return None
    # stratified resampling keeps both classes in every replicate
    rng = np.random.default_rng(seed)
    p, n = np.asarray(pos), np.asarray(neg)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        rp = p[rng.integers(0, p.size, p.size)]
        rn = n[rng.integers(0, n.size, n.size)]
        samples[b] = roc_auc(rp, rn)
    return _ci(value, samples)


@dataclass(frozen=True)
class Comparison:
    """A paired-bootstrap comparison of config A against config B."""

    verdict: Literal["a", "b", "tie"]
    mean_diff: float
    low: float
    high: float
    n: int


def paired_bootstrap(
    a: Mapping[int, float], b: Mapping[int, float], *, n_boot: int = 1000, seed: int = 0
) -> Comparison:
    """Compare two configs on the questions both scored (per-question differences A - B).

    A wins only when the whole 95% interval of the mean difference is above
    zero, B only when it is below; otherwise it's a tie.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two runs share no questions")
    diffs = np.asarray([a[q] - b[q] for q in shared], dtype=float)
    ci = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    verdict: Literal["a", "b", "tie"] = "a" if ci.low > 0 else "b" if ci.high < 0 else "tie"
    return Comparison(verdict, ci.value, ci.low, ci.high, len(shared))


# --------------------------------------------------------------- aggregation
def _mean_ci(values: list[float], cfg: ScoringConfig, salt: int) -> MetricCI | None:
    if not values:
        return None
    return bootstrap_ci(values, n_boot=cfg.n_boot, seed=cfg.seed + salt)


def _category_row(scores: list[QuestionScore], cfg: ScoringConfig) -> dict[str, float]:
    row: dict[str, float] = {"n": float(len(scores))}
    answerable = [s for s in scores if s.answerable]
    for budget in cfg.budgets:
        vals = [s.recall[budget] for s in answerable if budget in s.recall]
        if vals:
            row[f"recall@{budget_label(budget)}"] = float(np.mean(vals))
    if answerable:
        row["mrr"] = float(np.mean([s.reciprocal_rank or 0.0 for s in answerable]))
    row["abstained"] = float(np.mean([s.abstained for s in scores])) if scores else 0.0
    return row


def aggregate(
    scores: Iterable[QuestionScore], *, config: ScoringConfig | None = None
) -> EvalMetrics:
    """Aggregate metrics with bootstrap intervals (unanswerables only enter the AUC)."""
    cfg = config or ScoringConfig()
    scores = list(scores)
    answerable = [s for s in scores if s.answerable]
    budgeted: dict[int, MetricCI] = {}
    precision: dict[int, MetricCI] = {}
    for i, budget in enumerate(cfg.budgets):
        ci = _mean_ci([s.recall[budget] for s in answerable if budget in s.recall], cfg, i)
        if ci is not None:
            budgeted[budget] = ci
        ci = _mean_ci(
            [s.context_precision[budget] for s in answerable if budget in s.context_precision],
            cfg,
            100 + i,
        )
        if ci is not None:
            precision[budget] = ci
    support: dict[int, MetricCI] = {}
    for i, k in enumerate(cfg.hit_ks):
        hits = [float(s.full_support[k]) for s in answerable if k in s.full_support]
        ci = _mean_ci(hits, cfg, 200 + i)
        if ci is not None:
            support[k] = ci
    latencies = [s.latency_ms for s in scores if s.latency_ms is not None]
    categories = sorted({s.category for s in scores})
    return EvalMetrics(
        n_questions=len(scores),
        budgeted_recall=budgeted,
        full_support_hit=support,
        mrr=_mean_ci([s.reciprocal_rank or 0.0 for s in answerable], cfg, 300),
        context_precision=precision,
        abstain_auc=_auc_ci(scores, cfg.n_boot, cfg.seed + 400),
        latency_ms_p50=float(np.percentile(latencies, 50)) if latencies else None,
        latency_ms_p95=float(np.percentile(latencies, 95)) if latencies else None,
        by_category={
            c: _category_row([s for s in scores if s.category == c], cfg) for c in categories
        },
    )
