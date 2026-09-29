"""The retrieval primitives KWP4 composes (plan §7.1): vector search, BM25 search, fusion.

Each public function is timed with its plan stage (``vector``, ``bm25``;
``fuse`` lives in :mod:`.fusion`), so callers must not wrap them again. Every
result keeps its score: the dense cosine similarity is the abstain score
(never the RRF score).

    hits_v = vector_search(conn, profile, query_vector, k=30, filters=filters)
    hits_k = keyword_search(conn, profile, question, k=30, filters=filters)
    fused = fuse(as_scored(hits_v), as_scored(hits_k), method="rrf")
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np

from hal9000.knowledge.contracts import IndexProfile, SearchFilters
from hal9000.knowledge.index.fusion import FusedHit, Scored, fuse, rrf, weighted
from hal9000.knowledge.index.keyword import FtsIndex, KeywordHit
from hal9000.knowledge.index.vector import SqliteVecIndex, VectorHit
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = [
    "FusedHit",
    "KeywordHit",
    "Scored",
    "VectorHit",
    "as_scored",
    "fuse",
    "keyword_search",
    "rrf",
    "vector_search",
    "weighted",
]


def _id(profile: IndexProfile) -> int:
    if profile.id is None:
        raise ValueError(f"profile {profile.name!r} is not stored")
    return profile.id


def vector_search(
    conn: sqlite3.Connection,
    profile: IndexProfile,
    query_vector: Sequence[float] | np.ndarray,
    *,
    k: int = 30,
    filters: SearchFilters | None = None,
) -> list[VectorHit]:
    """Cosine KNN over the profile's ``vec0`` table, filters inside the KNN (``stage("vector")``)."""
    with stage(Stage.VECTOR):
        return SqliteVecIndex(conn, _id(profile), profile.dimensions).search(
            query_vector, k, filters
        )


def keyword_search(
    conn: sqlite3.Connection,
    profile: IndexProfile,
    query: str,
    *,
    k: int = 30,
    filters: SearchFilters | None = None,
) -> list[KeywordHit]:
    """BM25 over the profile's FTS5 table for a user's text (``stage("bm25")``)."""
    with stage(Stage.BM25):
        return FtsIndex(conn, _id(profile)).search(query, k, filters)


def as_scored(hits: Sequence[VectorHit | KeywordHit]) -> list[Scored]:
    """Hits in rank order as :class:`Scored` pairs, the input of the fusion functions."""
    return [Scored(hit.chunk_id, hit.score) for hit in sorted(hits, key=lambda h: h.rank)]
