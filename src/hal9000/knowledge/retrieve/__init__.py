"""Query processing, reranking and abstention: the retriever (plan §7.1, §7.4, §8.5; KWP4).

The interface is fixed by the orchestrator (``RETRIEVER_INTERFACE.md``)::

    from hal9000.knowledge.retrieve import RetrievalConfig, Retriever, Turn

    retriever = Retriever(conn, embeddings, generate=fast_llm)
    result = retriever.retrieve("How old is the water heater?", history, filters)
    result.selected        # exactly what the prompt receives, within the budget
    result.abstained       # top dense cosine (or reranker score) below the threshold

- :mod:`.config`: :class:`RetrievalConfig`, :class:`Turn`.
- :mod:`.retriever`: :class:`Retriever` and :func:`search_documents` (retrieval
  only: ``GET /api/search``, HAL's knowledge tool).
- :mod:`.select`: the context budget, parent expansion and near-duplicate removal.
- :mod:`.rerank`: :class:`Reranker`, :class:`NoRerank`, :class:`OnnxCrossEncoder`.
- :mod:`.transforms`: the condense, multi-query and HyDE prompts.
- :mod:`.thresholds`: ``abstain_thresholds`` per (profile, reranker).
- :mod:`.eval_binding`: the evaluation harness's retrieval (KWP5).

Importing this package loads neither fastembed nor ONNX Runtime.
"""

from hal9000.knowledge.retrieve.config import RERANKERS, RetrievalConfig, Turn, reranker_key
from hal9000.knowledge.retrieve.rerank import NoRerank, OnnxCrossEncoder, Reranker, reranker_for
from hal9000.knowledge.retrieve.retriever import (
    NoActiveProfile,
    QueryVectorCache,
    Retriever,
    search_documents,
)
from hal9000.knowledge.retrieve.select import NEAR_DUPLICATE, Selection, select
from hal9000.knowledge.retrieve.thresholds import get_threshold, set_threshold

__all__ = [
    "NEAR_DUPLICATE",
    "RERANKERS",
    "NoActiveProfile",
    "NoRerank",
    "OnnxCrossEncoder",
    "QueryVectorCache",
    "Reranker",
    "RetrievalConfig",
    "Retriever",
    "Selection",
    "Turn",
    "get_threshold",
    "reranker_for",
    "reranker_key",
    "search_documents",
    "select",
    "set_threshold",
]
