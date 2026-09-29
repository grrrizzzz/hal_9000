"""Cross-encoder reranking (plan §7.1 ``rerank``, §8.3 T8).

A :class:`Reranker` scores (query, passage) pairs. Scores are **normalised to
[0, 1]** so a threshold can be stored for them (plan §8.5): the ONNX
cross-encoders output raw logits, which are passed through the logistic
sigmoid ``1 / (1 + e^-x)``. The sigmoid is monotonic, so it never changes the
order; it only puts every model's scores on a stable, bounded scale (a logit of
0 is 0.5). The threshold is still calibrated per (profile, reranker), because
the two models' score distributions differ.

- :class:`NoRerank`: the ``none`` choice; the retriever skips the stage.
- :class:`OnnxCrossEncoder`: fastembed's ``TextCrossEncoder`` over ONNX
  Runtime (the ``knowledge-rerank`` uv group, imported lazily; the model is
  downloaded from Hugging Face on first use, so its real test is
  ``models``-marked). ``minilm`` is ``Xenova/ms-marco-MiniLM-L-6-v2`` (22 M
  parameters), ``bge-base`` is ``BAAI/bge-reranker-base`` (278 M).

**Truncation.** Both models read at most 512 tokens per pair (query and
passage together, in their own WordPiece/SentencePiece tokens), so for a
large chunk they judge only its beginning. That is inherent to the models and
is measured by the evaluation, not worked around here (plan §8.3 T8).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "MODELS",
    "NoRerank",
    "OnnxCrossEncoder",
    "Reranker",
    "reranker_for",
    "sigmoid",
]

#: rerank choice -> fastembed model name
MODELS: dict[str, str] = {
    "minilm": "Xenova/ms-marco-MiniLM-L-6-v2",
    "bge-base": "BAAI/bge-reranker-base",
}


def sigmoid(x: float) -> float:
    """The logistic function, numerically safe for large magnitudes."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@runtime_checkable
class Reranker(Protocol):
    """Scores passages for a query; higher is more relevant, normalised to [0, 1]."""

    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...  # pragma: no cover


class NoRerank:
    """The ``none`` choice. The retriever never calls it; it scores everything 0.5."""

    name = "none"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [0.5] * len(passages)


class OnnxCrossEncoder:
    """A fastembed ``TextCrossEncoder``; the model loads on the first :meth:`score`.

    Args:
        model: ``minilm``, ``bge-base`` or a fastembed cross-encoder model name.
        cache_dir: where fastembed keeps downloaded models (its default otherwise).
        threads: ONNX Runtime intra-op threads (its default otherwise).
        batch_size: pairs per ONNX run.
    """

    def __init__(
        self,
        model: str = "minilm",
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self.name = model
        self.model_name = MODELS.get(model, model)
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(
                    model_name=self.model_name, cache_dir=self.cache_dir, threads=self.threads
                )
            return self._model

    def logits(self, query: str, passages: Sequence[str]) -> list[float]:
        """The model's raw scores (logits)."""
        if not passages:
            return []
        model = self._load()
        return [float(x) for x in model.rerank(query, list(passages), batch_size=self.batch_size)]

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """``sigmoid(logit)`` per passage: the normalised, thresholdable score."""
        return [sigmoid(x) for x in self.logits(query, passages)]


_cache: dict[str, Reranker] = {}
_cache_lock = threading.Lock()


def reranker_for(rerank: str) -> Reranker:
    """The reranker for a configured choice; one loaded model per choice per process."""
    if rerank in ("none", "", None):
        return NoRerank()
    if rerank not in MODELS:
        raise ValueError(f"unknown reranker {rerank!r} (none, {', '.join(MODELS)})")
    with _cache_lock:
        if rerank not in _cache:
            _cache[rerank] = OnnxCrossEncoder(rerank)
        return _cache[rerank]
