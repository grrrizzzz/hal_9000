"""Fusion (plan §7.1 fuse): RRF matches hand calculation; weighted α = 1 is the dense order."""

from __future__ import annotations

import pytest

from hal9000.knowledge.index.fusion import Scored, fuse, rrf, weighted
from hal9000.knowledge.obs.timing import start_trace


def ranking(*ids: int) -> list[Scored]:
    # scores fall with rank; the values don't matter to RRF
    return [Scored(chunk_id, 1.0 - 0.1 * position) for position, chunk_id in enumerate(ids)]


def test_rrf_matches_a_hand_calculation():
    dense = ranking(10, 20, 30)
    bm25 = ranking(30, 40, 10)
    fused = rrf(dense, bm25, k=60)
    # 10: 1/61 + 1/63; 30: 1/63 + 1/61; 20: 1/62; 40: 1/62
    expected = {10: 1 / 61 + 1 / 63, 30: 1 / 63 + 1 / 61, 20: 1 / 62, 40: 1 / 62}
    assert {hit.chunk_id: hit.score for hit in fused} == pytest.approx(expected)
    # ties: 10 and 30 tie exactly, 10 has the better dense rank; 20 and 40 tie, 20 has a dense rank
    assert [hit.chunk_id for hit in fused] == [10, 30, 20, 40]
    assert [hit.rank for hit in fused] == [1, 2, 3, 4]
    top = fused[0]
    assert (top.vector_rank, top.bm25_rank, top.dense_score, top.bm25_score) == (1, 3, 1.0, 0.8)
    assert fused[3].vector_rank is None and fused[3].dense_score is None


def test_rrf_with_one_list_and_k_zero():
    fused = rrf(ranking(1, 2), [], k=0)
    assert [(h.chunk_id, h.score) for h in fused] == [(1, 1.0), (2, 0.5)]
    with pytest.raises(ValueError):
        rrf([], [], k=-1)


def test_weighted_alpha_one_is_the_dense_order():
    dense = [Scored(5, 0.91), Scored(3, 0.90), Scored(8, 0.40), Scored(1, 0.40)]
    bm25 = [Scored(1, 12.0), Scored(9, 11.0), Scored(5, 1.0)]
    fused = weighted(dense, bm25, alpha=1.0)
    assert [hit.chunk_id for hit in fused][: len(dense)] == [5, 3, 8, 1]
    assert [hit.chunk_id for hit in fused] == [5, 3, 8, 1, 9]


def test_weighted_alpha_zero_is_the_bm25_order_and_mixes_in_between():
    dense = [Scored(1, 0.9), Scored(2, 0.5), Scored(3, 0.1)]
    bm25 = [Scored(3, 10.0), Scored(2, 5.0), Scored(1, 0.0)]
    assert [h.chunk_id for h in weighted(dense, bm25, alpha=0.0)] == [3, 2, 1]
    mixed = weighted(dense, bm25, alpha=0.5)
    # normalised: 1 -> (1, 0), 2 -> (0.5, 0.5), 3 -> (0, 1): all 0.5; dense rank breaks ties
    assert [h.score for h in mixed] == pytest.approx([0.5, 0.5, 0.5])
    assert [h.chunk_id for h in mixed] == [1, 2, 3]
    with pytest.raises(ValueError):
        weighted(dense, bm25, alpha=1.5)


def test_fuse_dispatches_on_the_method_and_is_timed():
    dense, bm25 = ranking(1, 2), ranking(2, 3)
    with start_trace() as trace:
        assert [h.chunk_id for h in fuse(dense, bm25, method="rrf")] == [2, 1, 3]
    assert [s.stage for s in trace.stages] == ["fuse"]
    assert [h.chunk_id for h in fuse(dense, bm25, method="dense")] == [1, 2]
    assert [h.chunk_id for h in fuse(dense, bm25, method="bm25")] == [2, 3]
    assert [h.chunk_id for h in fuse(dense, bm25, method="weighted", alpha=1.0)] == [1, 2, 3]
    with pytest.raises(ValueError, match="weighted_alpha"):
        fuse(dense, bm25, method="weighted")
    with pytest.raises(ValueError, match="unknown"):
        fuse(dense, bm25, method="nope")  # type: ignore[arg-type]
