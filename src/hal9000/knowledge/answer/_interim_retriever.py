"""INTERIM stand-in for KWP4's ``hal9000.knowledge.retrieve`` (REPLACE when KWP4 merges).

KWP4 implements the retriever on another branch against the orchestrator-fixed
interface in ``RETRIEVER_INTERFACE.md``. Until it merges, the answer pipeline
and ``kb serve --fake`` need something that satisfies that interface, so this
module adapts KWP3's primitives (``embed_query``, ``vector_search``,
``keyword_search``, ``fuse``) with the same names and signatures:
``RetrievalConfig``, ``Turn`` and ``Retriever(conn, embeddings, *, generate,
reranker, config, profile).retrieve(question, history, filters, config)``.

What it deliberately does not do (KWP4 does): reranking, parent expansion for
``parent_child`` profiles, multi-query / HyDE, near-duplicate detection beyond
identical text, and header token accounting from ``index_tokens``. Stages are
timed with the plan names; an embedding failure falls back to BM25 only
(``mode="bm25"``). :mod:`hal9000.knowledge.answer.retrieval` imports the real
module first and falls back to this one, so deleting this file is the whole
switch-over.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from hal9000.core.prompt import Prompt
from hal9000.knowledge.contracts import IndexProfile, RetrievalResult, RetrievedChunk, SearchFilters
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = ["RetrievalConfig", "Retriever", "Turn"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "rrf"
    alpha: float = 0.5
    top_k_each: int = 30
    rerank: Literal["none", "minilm", "bge-base"] = "none"
    budget_tokens: int = 2000
    send_headers: bool = True
    transform: Literal["none", "multi", "hyde"] = "none"

    @classmethod
    def from_settings(cls, settings: Any) -> RetrievalConfig:
        """From ``config/knowledge.yaml``'s ``retrieval:`` block (settings or a KnowledgeFile)."""
        from hal9000.knowledge.config import KnowledgeFile, load_knowledge_file

        kf = settings if isinstance(settings, KnowledgeFile) else None
        if kf is None:
            path = getattr(settings, "yaml_path", None)
            kf = load_knowledge_file(path)
        r = kf.retrieval
        mode = {"dense": "dense", "bm25": "bm25"}.get(r.method, "hybrid")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            fusion="weighted" if r.method == "weighted" else "rrf",
            alpha=r.weighted_alpha if r.weighted_alpha is not None else 0.5,
            top_k_each=max(r.top_k_vector, r.top_k_bm25),
            budget_tokens=r.context_budget_tokens,
        )


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str


CONDENSE_SYSTEM = (
    "Rewrite the last question as one standalone question about the house, using the "
    "conversation only to resolve what it refers to. Reply with the question only."
)


def condense_prompt(question: str, history: Sequence[Turn]) -> Prompt:
    lines = "\n".join(f"{turn.role}: {turn.text}" for turn in history)
    return Prompt(
        messages=[
            {"role": "system", "content": CONDENSE_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{lines}\n\nLast question: {question}"},
        ],
        metadata={"purpose": "kb-condense"},
    )


class Retriever:
    def __init__(
        self,
        conn: sqlite3.Connection,
        embeddings: Any,
        *,
        generate: Callable[..., str] | None = None,
        reranker: Any = None,
        config: RetrievalConfig | None = None,
        profile: IndexProfile | None = None,
    ) -> None:
        self.conn = conn
        self.embeddings = embeddings
        self.generate = generate
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self.profile = profile

    def _profile(self) -> IndexProfile:
        from hal9000.knowledge.index.profiles import active_profile

        profile = self.profile or active_profile(self.conn)
        if profile is None:
            raise LookupError("no active index profile")
        return profile

    def _threshold(self, profile: IndexProfile) -> float | None:
        row = self.conn.execute(
            "SELECT threshold FROM abstain_thresholds WHERE profile_id = ? AND reranker = ''",
            (profile.id,),
        ).fetchone()
        return float(row[0]) if row else None

    def retrieve(
        self,
        question: str,
        history: Sequence[Turn] = (),
        filters: SearchFilters | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        from hal9000.cost.metering import BudgetExceeded
        from hal9000.knowledge.index.search import (
            as_scored,
            embed_query,
            fuse,
            keyword_search,
            vector_search,
        )

        cfg = config or self.config
        profile = self._profile()
        standalone = question
        if history and self.generate is not None:
            with stage(Stage.CONDENSE):
                standalone = self.generate(condense_prompt(question, history)).strip() or question

        mode = cfg.mode
        dense = []
        if mode in ("hybrid", "dense"):
            try:
                vector = embed_query(self.embeddings, profile, standalone)
                dense = vector_search(self.conn, profile, vector, k=cfg.top_k_each, filters=filters)
            except BudgetExceeded:
                raise
            except Exception as error:  # the embedding call failed: BM25 only (plan §7.4)
                log.warning("query embedding failed (%s); BM25 only", type(error).__name__)
                cfg = replace(cfg, mode="bm25")
                mode = "bm25"
                dense = []
        keyword = []
        if mode in ("hybrid", "bm25"):
            keyword = keyword_search(
                self.conn, profile, standalone, k=cfg.top_k_each, filters=filters
            )
        method = {"dense": "dense", "bm25": "bm25"}.get(mode, cfg.fusion)
        fused = fuse(
            as_scored(dense),
            as_scored(keyword),
            method=method,
            alpha=cfg.alpha,  # type: ignore[arg-type]
        )

        with stage(Stage.SELECT):
            candidates = self._chunks(fused)
            selected: list[RetrievedChunk] = []
            seen: set[str] = set()
            used = 0
            for chunk in candidates:
                cost = chunk.tokens
                if chunk.text in seen:
                    continue
                if used + cost > cfg.budget_tokens:
                    if not selected:
                        continue  # a first chunk bigger than the budget can't be sent
                    break
                seen.add(chunk.text)
                selected.append(chunk)
                used += cost

        with stage(Stage.ABSTAIN):
            score = None
            if selected and mode != "bm25":
                score = selected[0].dense_score
            threshold = self._threshold(profile)
            abstained = score is not None and threshold is not None and score < threshold

        return RetrievalResult(
            query=question,
            standalone_query=standalone,
            profile=profile.name,
            mode=mode,  # type: ignore[arg-type]
            candidates=candidates,
            selected=selected,
            context_tokens=used,
            abstain_score=score,
            threshold=threshold,
            abstained=abstained,
        )

    def _chunks(self, fused: Sequence[Any]) -> list[RetrievedChunk]:
        from hal9000.knowledge.index.library import _document

        if not fused:
            return []
        ids = [hit.chunk_id for hit in fused]
        marks = ",".join("?" * len(ids))
        rows = {
            row["id"]: row
            for row in self.conn.execute(
                f"SELECT c.*, d.file_sha, d.page_start AS d_start, d.page_end AS d_end, d.title, "
                f"d.doc_type, d.vendor, d.doc_date, d.meta_source FROM chunks c "
                f"JOIN documents d ON d.id = c.document_id WHERE c.id IN ({marks})",
                ids,
            )
        }
        out: list[RetrievedChunk] = []
        for hit in fused:
            row = rows.get(hit.chunk_id)
            if row is None:
                continue
            document = _document(
                {
                    "id": row["document_id"],
                    "file_sha": row["file_sha"],
                    "page_start": row["d_start"],
                    "page_end": row["d_end"],
                    "title": row["title"],
                    "doc_type": row["doc_type"],
                    "vendor": row["vendor"],
                    "doc_date": row["doc_date"],
                    "meta_source": row["meta_source"],
                }
            )
            out.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    document_id=row["document_id"],
                    text=row["text"],
                    header=row["header"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    tokens=row["tokens"],
                    rank=len(out) + 1,
                    score=hit.score,
                    dense_score=hit.dense_score,
                    bm25_score=hit.bm25_score,
                    vector_rank=hit.vector_rank,
                    bm25_rank=hit.bm25_rank,
                    document=document,
                    parent_id=row["parent_id"],
                )
            )
        return out
