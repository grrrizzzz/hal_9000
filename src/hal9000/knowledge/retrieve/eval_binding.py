"""Retrieval for the evaluation harness (KWP5), found by ``eval.wiring.retriever_for``.

:func:`binding` returns a :class:`~hal9000.knowledge.eval.wiring.RetrievalBinding`
whose ``retrieve(question)`` is ``select``'s exact output as
:class:`~hal9000.knowledge.eval.types.RetrievalOutput`: the selected items in
prompt order with their ``prompt_tokens``, the abstain score (dense cosine or
reranker score, never RRF) and the decision, and the local latency.

- **Metering** (rule K9): query embeddings go through an
  :class:`EmbeddingService` with ``agent="knowledge-eval"`` and the command's
  meter; the multi-query and HyDE rewrites through ``meter.wrap(...,
  "knowledge-eval")`` on the fast tier. A paid call without a meter is refused.
- **Configuration**: ``config/knowledge.yaml``'s retrieval block, overridden by
  the run's ``config`` (an :class:`~hal9000.knowledge.contracts.EvalRun`):
  its ``profile`` names the index profile, and its retrieval keys
  (``retrieval``, ``alpha``, ``reranker``, ``budget``, ``query_transform`` or
  any :class:`RetrievalConfig` field, see :meth:`RetrievalConfig.updated`)
  let the sweep vary T7-T10 and the budget. An explicit ``config`` replaces
  both.
- **Budget**: unless the run or ``config`` sets one, ``select`` runs with the
  largest scored budget (``knowledge_eval.yaml`` ``scoring.budgets``, 4,000 by
  default), because the scorer truncates one selection to each budget itself.
  Selection is prefix-consistent (:mod:`.select`), so that truncation is exactly
  what a smaller budget would have selected.
- **No index**: with no active profile (or the run's profile missing) it raises
  :class:`~hal9000.knowledge.eval.wiring.NotWired`, which the commands already
  report (``gold generate --allow-keyword-check`` falls back to keyword checks).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from hal9000.knowledge.config import EVAL_AGENT, ConfigMissing
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem
from hal9000.knowledge.eval.wiring import NotWired, RetrievalBinding
from hal9000.knowledge.index.embedding import EmbeddingService
from hal9000.knowledge.retrieve.config import RetrievalConfig
from hal9000.knowledge.retrieve.rerank import Reranker
from hal9000.knowledge.retrieve.retriever import NoActiveProfile, Retriever

__all__ = ["EvalRetrieve", "binding", "scored_budget"]

_BUDGET_KEYS = ("budget", "budget_tokens")


def scored_budget(settings: Any) -> int:
    """The largest budget the scorer reports (``knowledge_eval.yaml`` ``scoring.budgets``)."""
    from hal9000.knowledge.eval.config import load_eval_config

    config_dir = getattr(settings, "config_dir", None)
    path = Path(config_dir) / "knowledge_eval.yaml" if config_dir is not None else None
    cfg = load_eval_config(path) if path is not None and path.exists() else load_eval_config()
    return max(cfg.scoring.budgets)


class EvalRetrieve:
    """``question -> RetrievalOutput`` over one :class:`Retriever` and configuration."""

    def __init__(self, retriever: Retriever, config: RetrievalConfig) -> None:
        self.retriever = retriever
        self.config = config

    def __call__(self, question: str) -> RetrievalOutput:
        started = time.perf_counter()
        result = self.retriever.retrieve(question, config=self.config)
        latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalOutput(
            selected=tuple(
                SelectedItem(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    prompt_tokens=item.prompt_tokens,
                    header=item.header,
                )
                for item in result.selected
            ),
            abstain_score=result.abstain_score,
            abstained=result.abstained,
            latency_ms=latency_ms,
        )


def _run_config(run: Any) -> Mapping[str, Any]:
    config = getattr(run, "config", None) if run is not None else None
    return config if isinstance(config, Mapping) else {}


def binding(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    meter: Any = None,
    run: Any = None,
    config: RetrievalConfig | None = None,
    generate: Callable[..., str] | None = None,
    embed_fn: Callable[..., Any] | None = None,
    reranker: Reranker | None = None,
) -> RetrievalBinding:
    """Retrieval over the run's profile (or the active one), metered as ``knowledge-eval``.

    ``generate``, ``embed_fn`` and ``reranker`` are injectable for offline tests;
    by default they are the metered fast tier, the configured embedding
    provider and the configured cross-encoder.
    """
    overrides = _run_config(run)
    if config is None:
        base = RetrievalConfig.from_settings(settings)
        if not any(overrides.get(key) is not None for key in _BUDGET_KEYS):
            base = replace(base, budget_tokens=max(base.budget_tokens, scored_budget(settings)))
        config = base.updated(overrides)
    if config.transform != "none" and generate is None:
        if meter is None:
            raise ConfigMissing(
                f"the {config.transform!r} query transform calls the LLM: it needs a Meter "
                "(rule K9)"
            )
        from hal9000 import config as hal_config

        hal_config.require_llm_key()
        from hal9000.llm import generate_response

        generate = meter.wrap(generate_response, EVAL_AGENT)
    embeddings = EmbeddingService(conn, embed_fn=embed_fn, meter=meter, agent=EVAL_AGENT)
    retriever = Retriever(
        conn,
        embeddings,
        generate=generate,
        reranker=reranker,
        config=config,
        profile=overrides.get("profile") or None,
    )
    try:
        profile = retriever.resolve_profile()
    except NoActiveProfile as error:  # no index to retrieve from: the harness says so
        # the wording extends eval.wiring's own NotWired message, which KWP5's CLI tests match
        raise NotWired(
            f"retrieval is not available yet (KWP4 retrieval has no index to search: {error})"
        ) from error
    retriever.profile = profile
    return RetrievalBinding(
        retrieve=EvalRetrieve(retriever, config),
        profile_id=profile.id,
        profile_name=profile.name,
        chunk_size=profile.parent_size or profile.size,
    )
