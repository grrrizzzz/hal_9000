"""The retriever: question → ranked, budgeted context and the abstain decision (plan §7.1).

Stages, each timed with the shared ``stage()`` under its plan §9.2 name:

``condense`` (only with history) → ``embed_query`` → ``vector`` → ``bm25`` →
``fuse`` → ``rerank`` (only when configured) → ``select`` → ``abstain``.

- **Condense** runs only when there is history (the last ``history_turns``
  turns). A failed or empty rewrite falls back to the question as asked, with a
  warning.
- **Query embedding** goes through KWP3's timed ``embed_query`` (the
  persistent ``embed_cache``), with an in-process cache in front of it
  (:class:`QueryVectorCache`, keyed by model, dimensions and text) so a repeated
  question costs no database round trip. A cache hit is still timed as
  ``embed_query``.
- **Degraded mode** (plan §7.4): if the embedding call raises (provider down,
  monthly cap reached), retrieval continues BM25-only and the result says
  ``mode="bm25"``, ``degraded="embedding_unavailable"``. Configuration errors
  (fake mode off, an unmetered paid model) are bugs and still raise.
- **Fusion**: ``hybrid`` fuses per ``fusion`` (RRF with ``rrf_k``, or weighted
  with ``alpha``); ``dense`` and ``bm25`` use one retriever.
- **Rerank** re-scores the top ``top_k_each`` fused candidates with the
  cross-encoder; the candidates list is then those, in reranked order.
- **Select**: see :mod:`.select` (budget, parent expansion, near-duplicates).
- **Abstain**: the abstain score is the top *selected* item's reranker score
  when reranking, else its **dense cosine similarity** to the query vector
  (looked up directly when that chunk came from BM25 alone), and never the fused
  score. The threshold is ``abstain_thresholds`` for (profile, reranker);
  ``abstained = score < threshold`` when both are known. With no query vector
  (mode ``bm25`` or degraded) there is no retrieval abstain, reranked or not.

**Query transforms** (evaluation only, off by default): ``multi`` adds three
rewrites; each query (the question and its rewrites) is retrieved as above and
the ranked lists are fused by RRF across queries. The component scores and
ranks recorded on each chunk are those of the question itself. ``hyde``
embeds a hypothetical answer as the dense query (BM25 still uses the question),
so the dense scores, and the abstain score, are cosines to that passage. The
transform's generate call is not a plan §7.1 stage and is not timed as one;
its cost is metered by the injected ``generate``.

The retriever never commits: the caller owns the connection. (The embedding
service commits its own cache writes.)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from hal9000 import config as hal_config
from hal9000.knowledge.config import ConfigMissing
from hal9000.knowledge.contracts import IndexProfile, RetrievalResult, RetrievedChunk, SearchFilters
from hal9000.knowledge.index.embedding import EmbeddingService, UnmeteredCall
from hal9000.knowledge.index.fusion import FusedHit, fuse
from hal9000.knowledge.index.profiles import active_profile, get_profile
from hal9000.knowledge.index.search import as_scored, embed_query, keyword_search, vector_search
from hal9000.knowledge.obs.timing import Stage, stage
from hal9000.knowledge.retrieve import transforms
from hal9000.knowledge.retrieve.config import RetrievalConfig, Turn, reranker_key
from hal9000.knowledge.retrieve.rerank import Reranker, reranker_for
from hal9000.knowledge.retrieve.select import prompt_tokens, select
from hal9000.knowledge.retrieve.thresholds import get_threshold
from hal9000.knowledge.storage import vec_table
from hal9000.knowledge.tokens import encoding_for_model

__all__ = [
    "NoActiveProfile",
    "QueryVectorCache",
    "Retriever",
    "search_documents",
]

log = logging.getLogger(__name__)

Generate = Callable[..., str]


class NoActiveProfile(ConfigMissing):
    """There is no index profile to search."""


class QueryVectorCache:
    """A small thread-safe LRU of query vectors, keyed by (model, dimensions, text)."""

    def __init__(self, size: int = 512) -> None:
        self.size = size
        self._items: OrderedDict[tuple[str, int, str], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, int, str]) -> np.ndarray | None:
        with self._lock:
            vector = self._items.get(key)
            if vector is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return vector

    def put(self, key: tuple[str, int, str], vector: np.ndarray) -> None:
        with self._lock:
            self._items[key] = vector
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)


@dataclass
class _Query:
    """One query of a question: the text for BM25 and the text embedded for the KNN."""

    bm25_text: str
    embed_text: str
    vector: np.ndarray | None = None
    fused: list[FusedHit] = field(default_factory=list)


_ROW_SQL = (
    "SELECT c.id, c.document_id, c.text, c.header, c.page_start, c.page_end, c.tokens, "
    "c.parent_id, d.file_sha, d.title, d.doc_type, d.vendor, d.doc_date, f.collection "
    "FROM chunks c JOIN documents d ON d.id = c.document_id "
    "JOIN files f ON f.sha256 = d.file_sha WHERE c.profile_id = ? AND c.id IN ({marks})"
)


def _date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


class Retriever:
    """Retrieval over one index profile (the fixed interface: ``RETRIEVER_INTERFACE.md``).

    Args:
        conn: ``knowledge.db`` (the caller owns it; the retriever never commits).
        embeddings: KWP3's :class:`EmbeddingService` (metered by its own Meter).
        generate: ``(prompt, tier) -> str`` for condense, multi-query and HyDE
            (the fast tier), metered by the caller. Without it, condense falls back
            to the question and the transforms are refused.
        reranker: the cross-encoder to use when the config asks for reranking
            (default: the configured model, :func:`reranker_for`).
        config: the default :class:`RetrievalConfig` (default: the class defaults).
        profile: an :class:`IndexProfile` or profile name; default the DB's
            active profile at call time.
        tier: the tier ``generate`` is called with.
        query_cache: the in-process query-vector cache (default: one per retriever).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embeddings: EmbeddingService,
        *,
        generate: Generate | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
        profile: IndexProfile | str | None = None,
        tier: str = "fast",
        query_cache: QueryVectorCache | None = None,
    ) -> None:
        self.conn = conn
        self.embeddings = embeddings
        self.generate = generate
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self.profile = profile
        self.tier = tier
        self.query_cache = query_cache if query_cache is not None else QueryVectorCache()

    # ------------------------------------------------------------ profile
    def resolve_profile(self) -> IndexProfile:
        """The profile to search now: the configured one, else the active one."""
        if isinstance(self.profile, IndexProfile) and self.profile.id is not None:
            return self.profile
        name = self.profile.name if isinstance(self.profile, IndexProfile) else self.profile
        found = get_profile(self.conn, name) if name else active_profile(self.conn)
        if found is None:
            raise NoActiveProfile(
                f"no index profile named {name!r}"
                if name
                else "no active index profile: ingest documents or activate a profile first"
            )
        return found

    # ------------------------------------------------------------ stages
    def _condense(self, question: str, history: Sequence[Turn], cfg: RetrievalConfig) -> str:
        turns = list(history)[-cfg.history_turns :] if cfg.history_turns else []
        if not turns:
            return question
        with stage(Stage.CONDENSE):
            if self.generate is None:
                log.warning("condense skipped: no generate function; using the question as asked")
                return question
            try:
                rewritten = transforms.condense(self.generate, question, turns, tier=self.tier)
            except Exception as error:
                log.warning("condense failed (%s); using the question as asked", error)
                return question
            if not rewritten:
                log.warning("condense returned no question; using the question as asked")
                return question
            return rewritten

    def _queries(self, standalone: str, cfg: RetrievalConfig) -> list[_Query]:
        queries = [_Query(standalone, standalone)]
        if cfg.transform == "none":
            return queries
        if self.generate is None:
            raise ValueError(f"query transform {cfg.transform!r} needs a generate function")
        try:
            if cfg.transform == "multi":
                rewrites = transforms.multi_query(self.generate, standalone, tier=self.tier)
                queries += [_Query(text, text) for text in rewrites]
            else:  # hyde
                passage = transforms.hyde(self.generate, standalone, tier=self.tier)
                if passage:
                    queries[0].embed_text = passage
        except Exception as error:
            log.warning("%s transform failed (%s); using the question alone", cfg.transform, error)
        return queries

    def _embed(self, profile: IndexProfile, text: str) -> np.ndarray:
        key = (profile.embed_model, profile.dimensions, text)
        cached = self.query_cache.get(key)
        if cached is not None:
            with stage(Stage.EMBED_QUERY):
                return cached
        vector = np.asarray(embed_query(self.embeddings, profile, text), dtype=np.float32)
        self.query_cache.put(key, vector)
        return vector

    def _embed_all(self, profile: IndexProfile, queries: list[_Query]) -> bool:
        """Embed every query; False (degraded) when the embedding call fails."""
        try:
            for query in queries:
                query.vector = self._embed(profile, query.embed_text)
        except (hal_config.ConfigError, UnmeteredCall):
            raise
        except Exception as error:
            from hal9000.knowledge.obs.metrics import record_error

            log.warning("query embedding failed (%s); retrieving with BM25 only", error)
            record_error(Stage.EMBED_QUERY, error)
            for query in queries:
                query.vector = None
            return False
        return True

    def _search(
        self,
        profile: IndexProfile,
        query: _Query,
        mode: str,
        cfg: RetrievalConfig,
        filters: SearchFilters | None,
    ) -> list[FusedHit]:
        dense = (
            vector_search(self.conn, profile, query.vector, k=cfg.top_k_each, filters=filters)
            if mode != "bm25" and query.vector is not None
            else []
        )
        lexical = (
            keyword_search(self.conn, profile, query.bm25_text, k=cfg.top_k_each, filters=filters)
            if mode != "dense"
            else []
        )
        method = cfg.fusion if mode == "hybrid" else mode
        return fuse(
            as_scored(dense),
            as_scored(lexical),
            method=method,
            k=cfg.rrf_k,
            alpha=cfg.alpha if method == "weighted" else None,
        )

    @staticmethod
    def _across(lists: list[list[FusedHit]], k: int) -> list[FusedHit]:
        """RRF across the queries' ranked lists; component scores from the first (the question)."""
        with stage(Stage.FUSE):
            first = {hit.chunk_id: hit for hit in lists[0]}
            totals: dict[int, float] = {}
            best: dict[int, int] = {}
            for hits in lists:
                for hit in hits:
                    totals[hit.chunk_id] = totals.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
                    best[hit.chunk_id] = min(best.get(hit.chunk_id, hit.rank), hit.rank)
            order = sorted(totals, key=lambda c: (-totals[c], best[c], c))
            out = []
            for rank, chunk_id in enumerate(order, start=1):
                own = first.get(chunk_id)
                out.append(
                    replace(own, score=totals[chunk_id], rank=rank)
                    if own
                    else FusedHit(chunk_id, totals[chunk_id], rank)
                )
            return out

    def _rows(self, profile: IndexProfile, ids: Sequence[int]) -> dict[int, tuple]:
        out: dict[int, tuple] = {}
        ids = list(dict.fromkeys(ids))
        for start in range(0, len(ids), 500):
            part = ids[start : start + 500]
            sql = _ROW_SQL.format(marks=",".join("?" * len(part)))
            for row in self.conn.execute(sql, (profile.id, *part)):
                out[int(row[0])] = tuple(row)
        return out

    def _chunk(
        self, row: tuple, *, rank: int, score: float, cfg: RetrievalConfig, encoding: str, **kw
    ) -> RetrievedChunk:
        (cid, doc_id, text, header, p0, p1, tokens, parent, sha, title, dtype, vendor, ddate,
         collection) = row  # fmt: skip
        return RetrievedChunk(
            chunk_id=cid,
            document_id=doc_id,
            text=text,
            header=header or "",
            page_start=p0,
            page_end=p1,
            tokens=tokens,
            rank=rank,
            score=score,
            parent_id=parent,
            file_sha=sha,
            title=title,
            doc_type=dtype or "other",
            vendor=vendor,
            doc_date=_date(ddate),
            collection=collection or "",
            prompt_tokens=prompt_tokens(
                header or "", text, tokens, send_headers=cfg.send_headers, encoding=encoding
            ),
            **kw,
        )

    def _rerank(
        self, question: str, candidates: list[RetrievedChunk], cfg: RetrievalConfig
    ) -> list[RetrievedChunk]:
        with stage(Stage.RERANK):
            reranker = self.reranker or reranker_for(cfg.rerank)
            top = candidates[: cfg.top_k_each]
            passages = [f"{c.header}\n{c.text}" if c.header else c.text for c in top]
            scores = reranker.score(question, passages) if top else []
            if len(scores) != len(top):
                raise ValueError(f"the reranker returned {len(scores)} scores for {len(top)}")
            order = sorted(range(len(top)), key=lambda i: (-scores[i], top[i].rank))
            return [
                top[i].model_copy(
                    update={
                        "rank": rank,
                        "score": float(scores[i]),
                        "rerank_score": float(scores[i]),
                    }
                )
                for rank, i in enumerate(order, start=1)
            ]

    def _cosine(self, profile: IndexProfile, chunk_id: int, vector: np.ndarray) -> float | None:
        row = self.conn.execute(
            f"SELECT embedding FROM {vec_table(profile.id)} WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        stored = np.frombuffer(row[0], dtype=np.float32)
        norm = float(np.linalg.norm(stored) * np.linalg.norm(vector))
        return float(np.dot(stored, vector) / norm) if norm else None

    # -------------------------------------------------------------- retrieve
    def retrieve(
        self,
        question: str,
        history: Sequence[Turn] = (),
        filters: SearchFilters | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        """Retrieve, select and check abstention for one question (see the module docstring)."""
        cfg = config or self.config
        profile = self.resolve_profile()
        standalone = self._condense(question, history, cfg) if history else question
        queries = self._queries(standalone, cfg)
        mode = cfg.mode
        degraded = None
        if mode != "bm25" and not self._embed_all(profile, queries):
            mode, degraded = "bm25", "embedding_unavailable"
        lists = [self._search(profile, q, mode, cfg, filters) for q in queries]
        fused = lists[0] if len(lists) == 1 else self._across(lists, cfg.rrf_k)

        encoding = encoding_for_model(profile.embed_model)
        rows = self._rows(profile, [hit.chunk_id for hit in fused])
        candidates: list[RetrievedChunk] = []
        for hit in fused:
            if hit.chunk_id not in rows:  # pragma: no cover - indexed chunks always have rows
                continue
            candidates.append(
                self._chunk(
                    rows[hit.chunk_id],
                    rank=len(candidates) + 1,
                    score=hit.score,
                    cfg=cfg,
                    encoding=encoding,
                    dense_score=hit.dense_score,
                    bm25_score=hit.bm25_score,
                    vector_rank=hit.vector_rank,
                    bm25_rank=hit.bm25_rank,
                    fused_score=hit.score,
                )
            )
        if cfg.rerank != "none":
            candidates = self._rerank(standalone, candidates, cfg)

        with stage(Stage.SELECT):
            parent_ids = {c.parent_id for c in candidates if c.parent_id is not None}
            parents = {
                pid: self._chunk(row, rank=1, score=0.0, cfg=cfg, encoding=encoding)
                for pid, row in self._rows(profile, sorted(parent_ids)).items()
            }
            chosen = select(candidates, cfg.budget_tokens, parents=parents)

        with stage(Stage.ABSTAIN):
            score = None
            primary = queries[0].vector
            # BM25 only (configured or degraded): no dense score, no retrieval abstain (§7.1)
            if chosen.items and primary is not None:
                top, source = chosen.items[0], chosen.sources[0]
                if cfg.rerank != "none":
                    score = top.rerank_score
                else:
                    score = source.dense_score
                    if score is None:  # it came from BM25 alone: its cosine, looked up
                        score = self._cosine(profile, source.chunk_id, primary)
            threshold = get_threshold(self.conn, profile.id, reranker_key(cfg.rerank))
            abstained = score is not None and threshold is not None and score < threshold
        return RetrievalResult(
            query=question,
            standalone_query=standalone,
            profile=profile.name,
            mode=mode,
            candidates=candidates,
            selected=chosen.items,
            context_tokens=chosen.tokens,
            abstain_score=score,
            threshold=threshold,
            abstained=abstained,
            degraded=degraded,
        )


def search_documents(
    conn: sqlite3.Connection,
    embeddings: EmbeddingService,
    query: str,
    *,
    k: int = 10,
    filters: SearchFilters | None = None,
    config: RetrievalConfig | None = None,
    reranker: Reranker | None = None,
    profile: IndexProfile | str | None = None,
) -> RetrievalResult:
    """Retrieval only, without generation: ``GET /api/search`` and HAL's knowledge tool.

    No history and no query transform (so nothing but the query embedding is
    paid). ``selected`` is cut to the top ``k`` items and ``context_tokens``
    recounted; ``candidates`` keeps every fused candidate with its scores.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    cfg = replace(config or RetrievalConfig(), transform="none")
    retriever = Retriever(conn, embeddings, reranker=reranker, config=cfg, profile=profile)
    started = time.perf_counter()
    result = retriever.retrieve(query, filters=filters)
    log.debug("search took %.1f ms", (time.perf_counter() - started) * 1000)
    selected = result.selected[:k]
    return result.model_copy(
        update={"selected": selected, "context_tokens": sum(c.prompt_tokens for c in selected)}
    )
