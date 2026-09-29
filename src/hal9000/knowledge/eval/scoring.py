"""Scoring that is fair across chunk sizes and parsers (plan §8.2).

Everything is computed on the **exact** output of ``select`` (after parent
expansion and de-duplication), in rank order, with each item's prompt token
count, and on evidence *text*, never offsets:

- **Coverage.** A selected item covers part of an evidence quote where they
  align with ``rapidfuzz`` partial alignment at >= 90 similarity, on
  normalized text (:func:`hal9000.knowledge.eval.text.joined`: letters and
  digits only, so case, whitespace, hyphenation and letter spacing don't
  matter). Every alignment must start and end on the chunk's word boundaries,
  and a span shorter than ``min_fuzzy_chars`` must match exactly. The whole
  quote is tried first (or the item inside it, when the item is smaller), then
  the longest prefix and suffix of the quote (a quote straddling a chunk
  boundary), each at least ``min_span_words`` words. Within an alignment, the
  quote characters that match (Indel opcodes) are covered, and a plain word
  is covered when at least half its characters are. A word of a number token
  (amount, date, reference number) is covered only when the aligned part of
  the chunk holds the *identical* token.
- **Evidence recall** is the share of the question's evidence words covered
  by the union of the selected items.
- **Budgeted recall @ B** fills the context in rank order and stops at the
  first item that would overflow ``B`` (it is excluded). A chunk size larger
  than ``B``, or a top item that alone overflows ``B``, is N/A: the key is
  absent (and stays out of the means).
- **Full-support hit @ k**: every quote at least 80% covered within the top k.
- **MRR**: reciprocal rank of the first item covering any evidence.
- **Context precision @ B**: covered evidence tokens / context tokens.
- **Abstention separability**: ROC AUC of the abstain score, answerable
  (positive) vs unanswerable. Unanswerables never enter recall.
- **Uncertainty**: a seeded percentile bootstrap (the interval is widened to
  include the point estimate, so it always brackets it), and a paired
  bootstrap over per-question differences to compare two configurations: a
  difference counts only when its interval excludes zero and at least
  ``min_paired`` questions are shared (with a sign-flip permutation p-value
  alongside).
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np

from hal9000.knowledge.contracts import EvalMetrics, GoldItem, MetricCI
from hal9000.knowledge.eval.text import compound_tokens, joined
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
    "sign_flip_p",
]

TokenCounter = Callable[[str], int]

#: An exact run of this many joined characters must be shared before aligning.
_SEED = 5
#: Windows tried per alignment before giving up (each rejected one is masked).
_RETRIES = 6


@dataclass(frozen=True)
class ScoringConfig:
    """Scoring parameters (``scoring`` in ``config/knowledge_eval.yaml``)."""

    budgets: tuple[int, ...] = (1000, 2000, 4000)
    hit_ks: tuple[int, ...] = (1, 3, 5, 10)
    min_similarity: float = 90.0
    full_support: float = 0.8
    min_span_words: int = 3
    word_cover: float = 0.5
    #: aligned spans shorter than this (joined characters) must match exactly
    min_fuzzy_chars: int = 12
    n_boot: int = 1000
    seed: int = 20260928
    #: fewer shared questions than this: a paired comparison is "insufficient"
    min_paired: int = 10


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
@dataclass(frozen=True)
class _Chunk:
    """A selected item's text prepared for alignment (cached per text)."""

    text: str
    joined: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    source: tuple[int, ...]
    numbers: tuple[tuple[int, int, str], ...]

    def tokens_between(self, a: int, b: int) -> set[str]:
        """Normalized number tokens of the original text overlapping joined ``[a, b)``."""
        lo, hi = self.source[a], self.source[b - 1] + 1
        return {norm for s, e, norm in self.numbers if s < hi and e > lo}


@lru_cache(maxsize=16384)
def _chunk(text: str) -> _Chunk:
    j = joined(text)
    return _Chunk(
        text,
        j.text,
        tuple(a for a, _ in j.spans),
        tuple(b for _, b in j.spans),
        j.source,
        tuple(compound_tokens(text)),
    )


@dataclass(frozen=True)
class _Alignment:
    """Quote ``[qa, qb)`` aligned with chunk ``[ca, cb)`` (joined coordinates)."""

    qa: int
    qb: int
    ca: int
    cb: int


def _neighbours(points: Sequence[int], x: int) -> set[int]:
    """The closest points at or below and at or above ``x``."""
    i = bisect.bisect_left(points, x)
    out = set()
    if i < len(points):
        out.add(points[i])
    if i > 0 and (i == len(points) or points[i] != x):
        out.add(points[i - 1])
    return out


def _find(needle: str, q0: int, chunk: _Chunk, cfg: ScoringConfig) -> _Alignment | None:
    """The best alignment of ``needle`` in the chunk that starts and ends on its word boundaries.

    rapidfuzz proposes the best window; it is snapped to the chunk's word
    boundaries and re-scored. A window that can't be snapped is masked out and
    the next best is tried, so ``yes`` never matches inside ``eyes`` but still
    finds a real ``yes`` later on. A span shorter than ``min_fuzzy_chars`` must
    match exactly.
    """
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Indel

    hay = chunk.joined
    for _attempt in range(_RETRIES):
        hit = fuzz.partial_ratio_alignment(needle, hay, score_cutoff=cfg.min_similarity)
        if hit is None:
            return None
        sub = needle[hit.src_start : hit.src_end]
        best: tuple[float, int, int] | None = None
        for a in _neighbours(chunk.starts, hit.dest_start):
            for b in _neighbours(chunk.ends, hit.dest_end):
                if a >= b:
                    continue
                window = chunk.joined[a:b]
                if len(sub) < cfg.min_fuzzy_chars or len(window) < cfg.min_fuzzy_chars:
                    score = 100.0 if sub == window else 0.0
                else:
                    score = Indel.normalized_similarity(sub, window) * 100
                if score >= cfg.min_similarity and (best is None or score > best[0]):
                    best = (score, a, b)
        if best is not None:
            return _Alignment(q0 + hit.src_start, q0 + hit.src_end, best[1], best[2])
        width = hit.dest_end - hit.dest_start
        hay = hay[: hit.dest_start] + "\0" * max(1, width) + hay[hit.dest_start + max(1, width) :]
    return None


def _longest(
    qj: str, cuts: Sequence[int], chunk: _Chunk, cfg: ScoringConfig, *, prefix: bool
) -> _Alignment | None:
    """Binary search for the longest word-aligned prefix (or suffix) of the quote in the chunk.

    ``cuts`` are candidate cut points in ``qj``, from the shortest candidate to the longest.
    """
    lo, hi, best = 0, len(cuts) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        cut = cuts[mid]
        found = _find(qj[:cut], 0, chunk, cfg) if prefix else _find(qj[cut:], cut, chunk, cfg)
        if found is not None:
            best = found
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _seeded(qj: str, cj: str) -> bool:
    """Cheap filter: do the two share an exact run of :data:`_SEED` characters?"""
    if len(qj) <= _SEED:
        return qj in cj
    return any(qj[i : i + _SEED] in cj for i in range(len(qj) - _SEED + 1))


@lru_cache(maxsize=4096)
def _quote_numbers(quote: str) -> tuple[str | None, ...]:
    """For each quote word, the number token it belongs to (None for plain words)."""
    q = joined(quote)
    numbers = compound_tokens(quote)
    out: list[str | None] = []
    for a, _b in q.spans:
        origin = q.source[a]
        out.append(next((norm for s, e, norm in numbers if s <= origin < e), None))
    return tuple(out)


def covered_words(quote: str, text: str, config: ScoringConfig | None = None) -> set[int]:
    """Indices of the quote's words that ``text`` covers.

    A plain word is covered when at least ``word_cover`` of its characters
    match inside an alignment. A word of a number token (an amount, date or
    reference number) is covered only when the aligned part of the chunk holds
    the identical token: ``$625.00`` is never covered by ``$425.00`` or
    ``$1,625.00``.
    """
    from rapidfuzz.distance import Indel

    cfg = config or ScoringConfig()
    q = joined(quote)
    qj = q.text
    chunk = _chunk(text)
    if not qj or not chunk.joined or not _seeded(qj, chunk.joined):
        return set()
    alignments: list[_Alignment] = []
    whole = _find(qj, 0, chunk, cfg)
    if whole is not None:
        alignments.append(whole)
    if whole is None or whole.qa > 0 or whole.qb < len(qj):
        n = q.n_words
        k = min(cfg.min_span_words, n)
        if n > k:
            prefix_ends = [q.spans[i - 1][1] for i in range(k, n)]
            suffix_starts = [q.spans[n - i][0] for i in range(k, n)]
            for found in (
                _longest(qj, prefix_ends, chunk, cfg, prefix=True),
                _longest(qj, suffix_starts, chunk, cfg, prefix=False),
            ):
                if found is not None:
                    alignments.append(found)
    chars: set[int] = set()
    for al in alignments:
        for op in Indel.opcodes(qj[al.qa : al.qb], chunk.joined[al.ca : al.cb]):
            if op.tag == "equal":
                chars.update(range(al.qa + op.src_start, al.qa + op.src_end))
    numbers = _quote_numbers(quote)
    covered = set()
    for i, (a, b) in enumerate(q.spans):
        token = numbers[i]
        if token is None:
            if sum(1 for j in range(a, b) if j in chars) >= cfg.word_cover * (b - a):
                covered.add(i)
        elif any(
            al.qa <= a and b <= al.qb and token in chunk.tokens_between(al.ca, al.cb)
            for al in alignments
        ):
            covered.add(i)
    return covered


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
            continue  # N/A: this profile's chunks can't fit the budget
        if selected and selected[0].prompt_tokens > budget:
            continue  # N/A: the top item alone overflows, so nothing could be sent
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
    """A paired comparison of config A against config B on their shared questions.

    ``verdict`` is ``"a"`` or ``"b"`` only when the bootstrap interval of the
    mean per-question difference excludes zero; ``"tie"`` otherwise; and
    ``"insufficient"`` (never a win) when fewer than ``min_paired`` questions
    are shared. ``only_a``/``only_b`` count questions just one run scored, so
    a mismatch between the two question sets is visible. ``p_value`` is a
    seeded two-sided sign-flip permutation test, a second opinion.
    """

    verdict: Literal["a", "b", "tie", "insufficient"]
    mean_diff: float
    low: float
    high: float
    n: int
    only_a: int = 0
    only_b: int = 0
    p_value: float = 1.0

    @property
    def mismatched(self) -> bool:
        return bool(self.only_a or self.only_b)


def sign_flip_p(diffs: Sequence[float], *, n_perm: int = 1000, seed: int = 0) -> float:
    """Two-sided p-value of a paired sign-flip permutation test on the mean difference."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0 or not np.any(d):
        return 1.0
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    permuted = np.abs((flips * d).mean(axis=1))
    return float((1 + np.sum(permuted >= observed - 1e-12)) / (n_perm + 1))


def paired_bootstrap(
    a: Mapping[int, float],
    b: Mapping[int, float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    min_paired: int = 10,
) -> Comparison:
    """Compare two configs on the questions both scored (per-question differences A - B)."""
    shared = sorted(set(a) & set(b))
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    if not shared:
        return Comparison("insufficient", 0.0, 0.0, 0.0, 0, only_a, only_b)
    diffs = np.asarray([a[q] - b[q] for q in shared], dtype=float)
    ci = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    p = sign_flip_p(diffs, n_perm=max(n_boot, 1), seed=seed)
    verdict: Literal["a", "b", "tie", "insufficient"]
    if len(shared) < min_paired:
        verdict = "insufficient"
    else:
        verdict = "a" if ci.low > 0 else "b" if ci.high < 0 else "tie"
    return Comparison(verdict, ci.value, ci.low, ci.high, len(shared), only_a, only_b, p)


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
