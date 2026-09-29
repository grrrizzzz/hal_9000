"""Retrieval configuration and conversation turns (the fixed retrieval interface).

:class:`RetrievalConfig` is the per-question configuration. The chat reads it
from ``config/knowledge.yaml`` (:meth:`RetrievalConfig.from_settings`); the
evaluation sweep builds one per run to vary T7-T10 and the budget (plan §8.3).

The yaml's ``retrieval.method`` maps onto ``mode`` and ``fusion``:

======== ========== ==========
method   mode       fusion
======== ========== ==========
dense    dense      (unused)
bm25     bm25       (unused)
rrf      hybrid     rrf
weighted hybrid     weighted (``weighted_alpha`` is required: rule K8)
======== ========== ==========

``rrf_k`` and ``history_turns`` are additive fields (defaults 60 and 6, plan
§7.1 and §7.3) so the yaml's values reach the retriever.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from hal9000.knowledge.config import ConfigMissing, KnowledgeFile, load_knowledge_file

__all__ = ["RERANKERS", "RetrievalConfig", "Turn", "reranker_key"]

Mode = Literal["hybrid", "dense", "bm25"]
Fusion = Literal["rrf", "weighted"]
Rerank = Literal["none", "minilm", "bge-base"]
Transform = Literal["none", "multi", "hyde"]

#: The rerank choices (plan §8.3 T8).
RERANKERS: tuple[str, ...] = ("none", "minilm", "bge-base")
_MODES = ("hybrid", "dense", "bm25")
_FUSIONS = ("rrf", "weighted")
_TRANSFORMS = ("none", "multi", "hyde")


def reranker_key(rerank: str | None) -> str:
    """The ``abstain_thresholds.reranker`` key: ``''`` for no reranker, else its name."""
    return "" if rerank in (None, "", "none") else str(rerank)


@dataclass(frozen=True)
class Turn:
    """One earlier message of the conversation (used only to condense a follow-up)."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class RetrievalConfig:
    mode: Mode = "hybrid"
    fusion: Fusion = "rrf"
    alpha: float = 0.5  # weighted fusion: dense weight
    top_k_each: int = 30  # candidates from each retriever
    rerank: Rerank = "none"
    budget_tokens: int = 2000  # context budget for `select`
    send_headers: bool = True  # prompt tokens count header tokens when True
    transform: Transform = "none"
    # additive: the yaml's other retrieval knobs
    rrf_k: int = 60
    history_turns: int = 6

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, not {self.mode!r}")
        if self.fusion not in _FUSIONS:
            raise ValueError(f"fusion must be one of {_FUSIONS}, not {self.fusion!r}")
        if self.rerank not in RERANKERS:
            raise ValueError(f"rerank must be one of {RERANKERS}, not {self.rerank!r}")
        if self.transform not in _TRANSFORMS:
            raise ValueError(f"transform must be one of {_TRANSFORMS}, not {self.transform!r}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.top_k_each < 1 or self.budget_tokens < 1 or self.rrf_k < 0:
            raise ValueError("top_k_each and budget_tokens must be positive, rrf_k >= 0")
        if self.history_turns < 0:
            raise ValueError("history_turns must be >= 0")

    # ----------------------------------------------------------------- loading
    @staticmethod
    def method_fields(method: str, alpha: float | None = None) -> dict[str, Any]:
        """``retrieval.method`` (dense, bm25, rrf, weighted) as ``mode``/``fusion``/``alpha``."""
        if method in ("dense", "bm25"):
            return {"mode": method}
        if method == "rrf":
            return {"mode": "hybrid", "fusion": "rrf"}
        if method == "weighted":
            if alpha is None:
                raise ConfigMissing(
                    "retrieval.method is weighted but retrieval.weighted_alpha is not set "
                    "(it is chosen by the setup evaluation, plan §8.3 T7)"
                )
            return {"mode": "hybrid", "fusion": "weighted", "alpha": float(alpha)}
        raise ValueError(f"unknown retrieval method {method!r} (dense, bm25, rrf or weighted)")

    @classmethod
    def from_file(cls, knowledge: KnowledgeFile) -> RetrievalConfig:
        """The configuration a parsed ``config/knowledge.yaml`` describes."""
        r = knowledge.retrieval
        values: dict[str, Any] = cls.method_fields(r.method, r.weighted_alpha)
        values.update(
            top_k_each=max(r.top_k_vector, r.top_k_bm25),
            budget_tokens=r.context_budget_tokens,
            rrf_k=r.rrf_k,
            history_turns=r.history_turns,
            rerank=knowledge.reranker or "none",
            transform=r.transform,
        )
        return cls(**values)

    @classmethod
    def from_settings(cls, settings: Any = None) -> RetrievalConfig:
        """Read the ``retrieval:`` block (and ``reranker``) of ``config/knowledge.yaml``.

        ``settings`` is a :class:`~hal9000.knowledge.config.KnowledgeSettings`
        (its ``yaml_path`` is read), a parsed :class:`KnowledgeFile`, or None
        for the current settings. ``top_k_each`` is the larger of
        ``top_k_vector`` and ``top_k_bm25``.
        """
        if isinstance(settings, KnowledgeFile):
            return cls.from_file(settings)
        path = getattr(settings, "yaml_path", None) if settings is not None else None
        return cls.from_file(load_knowledge_file(path))

    def updated(self, overrides: Mapping[str, Any] | None) -> RetrievalConfig:
        """A copy with the recognised keys of ``overrides`` applied (see :func:`from_run_config`).

        Accepts this class's field names, and the evaluation run's well-known
        keys: ``retrieval`` (a method name, or a mapping of fields), ``alpha``,
        ``reranker``, ``budget`` and ``query_transform``. Other keys are ignored.
        """
        if not overrides:
            return self
        names = {f.name for f in fields(self)}
        values: dict[str, Any] = {}
        retrieval = overrides.get("retrieval")
        if isinstance(retrieval, Mapping):
            values.update({k: v for k, v in retrieval.items() if k in names})
        elif isinstance(retrieval, str):
            alpha = overrides.get("alpha", overrides.get("weighted_alpha"))
            if retrieval == "weighted" and alpha is None:
                alpha = self.alpha if self.fusion == "weighted" else None
            values.update(self.method_fields(retrieval, alpha))
        values.update({k: v for k, v in overrides.items() if k in names})
        aliases = {"reranker": "rerank", "budget": "budget_tokens", "query_transform": "transform"}
        for alias, name in aliases.items():
            if overrides.get(alias) is not None:
                values[name] = overrides[alias]
        if "rerank" in values and values["rerank"] in (None, ""):
            values["rerank"] = "none"
        if "transform" in values and values["transform"] in (None, ""):
            values["transform"] = "none"
        return replace(self, **values)
