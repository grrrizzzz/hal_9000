"""The retriever the answer pipeline uses: KWP4's :mod:`hal9000.knowledge.retrieve`.

One :class:`Retriever` is built per request (it holds the request's
connection and must not be shared across threads; the process-wide
query-vector cache keeps that cheap). ``GET /api/search`` uses
:func:`search_documents`. An embedding failure comes back as
``result.degraded == "embedding_unavailable"`` (BM25 only), and a spending cap
propagates as ``BudgetExceeded``.
"""

from __future__ import annotations

from typing import Any

from hal9000.knowledge.retrieve import (
    NoActiveProfile,
    QueryVectorCache,
    RetrievalConfig,
    Retriever,
    Turn,
    search_documents,
)

__all__ = [
    "NoActiveProfile",
    "QueryVectorCache",
    "RetrievalConfig",
    "Retriever",
    "Turn",
    "embedding_unavailable",
    "search_documents",
]


def embedding_unavailable(result: Any) -> bool:
    """Whether retrieval fell back to BM25 because the query embedding failed (plan §7.4)."""
    return getattr(result, "degraded", None) == "embedding_unavailable"
