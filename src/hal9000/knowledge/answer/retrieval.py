"""The retriever the answer pipeline uses: KWP4's when it is installed, else the interim stand-in.

``hal9000.knowledge.retrieve`` (KWP4) implements the orchestrator-fixed
interface in ``RETRIEVER_INTERFACE.md``. It is built on a parallel branch;
until it merges, :mod:`._interim_retriever` provides the same names. Nothing
else in the answer pipeline or the API imports either module directly.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - which branch is taken depends on whether KWP4 has merged
    from hal9000.knowledge.retrieve import RetrievalConfig, Retriever, Turn

    INTERIM = False
except ModuleNotFoundError as error:  # pragma: no cover
    if error.name != "hal9000.knowledge.retrieve":
        raise
    from hal9000.knowledge.answer._interim_retriever import RetrievalConfig, Retriever, Turn

    INTERIM = True

__all__ = ["INTERIM", "RetrievalConfig", "Retriever", "Turn", "embedding_unavailable"]


def embedding_unavailable(result: Any, requested_mode: str) -> bool:
    """Whether retrieval fell back to BM25 because the query embedding failed (plan §7.4).

    KWP4 marks it with ``result.degraded == "embedding_unavailable"``; a
    result without that field shows it as a BM25-only result for a
    configuration that asked for dense search.
    """
    marker = getattr(result, "degraded", None)
    if marker is not None:
        return marker == "embedding_unavailable"
    return requested_mode != "bm25" and getattr(result, "mode", None) == "bm25"
