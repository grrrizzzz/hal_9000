"""Combining the dense and BM25 rankings (plan §7.1 ``fuse``).

- :func:`rrf`: reciprocal rank fusion, ``score(d) = Σ 1 / (k + rank_i(d))`` with
  ranks starting at 1 and ``k = 60``. It only says whether the rankings agree,
  so it is never the abstain score (plan §7.1).
- :func:`weighted`: min-max normalise each list's scores to [0, 1], then
  ``α · dense + (1 − α) · bm25``; a chunk missing from a list scores 0 there.
  With ``α = 1`` the result is the dense order (then the BM25-only chunks, in
  BM25 order), with ``α = 0`` the BM25 order.

Both keep each chunk's component scores and ranks, because KWP4's abstain
check needs the dense cosine and the trace records all of them. Ties are
broken by the better dense rank, then the better BM25 rank, then the chunk ID,
so the output is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from hal9000.knowledge.obs.timing import Stage, stage

__all__ = ["RRF_K", "FusedHit", "Scored", "fuse", "rrf", "weighted"]

RRF_K = 60


@dataclass(frozen=True)
class Scored:
    """One retriever's hit: a chunk and its score (higher is better)."""

    chunk_id: int
    score: float


@dataclass(frozen=True)
class FusedHit:
    chunk_id: int
    score: float
    rank: int
    dense_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None


def _ranks(hits: Sequence[Scored]) -> dict[int, tuple[int, float]]:
    """chunk -> (1-based rank, score); a repeated chunk keeps its first (best) position."""
    out: dict[int, tuple[int, float]] = {}
    for position, hit in enumerate(hits, start=1):
        out.setdefault(hit.chunk_id, (position, hit.score))
    return out


def _combine(dense: Sequence[Scored], bm25: Sequence[Scored], score_of) -> list[FusedHit]:
    d, b = _ranks(dense), _ranks(bm25)
    rows = []
    for chunk_id in {*d, *b}:
        vector_rank, dense_score = d.get(chunk_id, (None, None))
        bm25_rank, bm25_score = b.get(chunk_id, (None, None))
        rows.append((score_of(chunk_id), chunk_id, dense_score, bm25_score, vector_rank, bm25_rank))
    rows.sort(
        key=lambda r: (-r[0], r[4] if r[4] is not None else math.inf,
                       r[5] if r[5] is not None else math.inf, r[1])
    )  # fmt: skip
    return [
        FusedHit(chunk_id, score, rank, ds, bs, vr, br)
        for rank, (score, chunk_id, ds, bs, vr, br) in enumerate(rows, start=1)
    ]


def rrf(dense: Sequence[Scored], bm25: Sequence[Scored], *, k: int = RRF_K) -> list[FusedHit]:
    """Reciprocal rank fusion of the two rankings (each best first)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    d, b = _ranks(dense), _ranks(bm25)

    def score(chunk_id: int) -> float:
        total = 0.0
        for ranks in (d, b):
            if chunk_id in ranks:
                total += 1.0 / (k + ranks[chunk_id][0])
        return total

    return _combine(dense, bm25, score)


def _normalised(hits: Sequence[Scored]) -> dict[int, float]:
    ranks = _ranks(hits)
    if not ranks:
        return {}
    scores = [score for _, score in ranks.values()]
    low, high = min(scores), max(scores)
    if high == low:
        return {chunk_id: 1.0 for chunk_id in ranks}
    return {chunk_id: (score - low) / (high - low) for chunk_id, (_, score) in ranks.items()}


def weighted(dense: Sequence[Scored], bm25: Sequence[Scored], *, alpha: float) -> list[FusedHit]:
    """``α · dense + (1 − α) · bm25`` over min-max normalised scores."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    nd, nb = _normalised(dense), _normalised(bm25)
    return _combine(dense, bm25, lambda c: alpha * nd.get(c, 0.0) + (1.0 - alpha) * nb.get(c, 0.0))


def fuse(
    dense: Sequence[Scored],
    bm25: Sequence[Scored],
    *,
    method: Literal["rrf", "weighted", "dense", "bm25"] = "rrf",
    k: int = RRF_K,
    alpha: float | None = None,
) -> list[FusedHit]:
    """Fuse per the configured method (``retrieval.method``), timed as ``stage("fuse")``.

    ``dense`` alone and ``bm25`` alone are the single-retriever methods (and the
    BM25-only degraded mode), expressed as weighted fusion with α = 1 or 0.
    """
    with stage(Stage.FUSE):
        if method == "rrf":
            return rrf(dense, bm25, k=k)
        if method == "weighted":
            if alpha is None:
                raise ValueError("weighted fusion needs retrieval.weighted_alpha")
            return weighted(dense, bm25, alpha=alpha)
        if method == "dense":
            return weighted(dense, [], alpha=1.0)
        if method == "bm25":
            return weighted([], bm25, alpha=0.0)
        raise ValueError(f"unknown fusion method {method!r}")
