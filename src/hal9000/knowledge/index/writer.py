"""Chunk → embed → write: the steps an ingested file and a profile build share (plan §5.1 7–9).

- :func:`chunk_documents` adapts each document's cleaned text with
  ``ChunkInput.from_paged`` and chunks it with the profile (KWP2 times each
  document as ``stage("chunk")``). Documents with no text are skipped.
- :func:`embed_chunks` embeds only ``searchable`` drafts' index text (header +
  text): ``parent_child`` parents are stored for prompt expansion but never
  searched. Timed as ``stage("embed")``.
- :func:`write_chunks` inserts the drafts into ``chunks`` (a parent before its
  children, so ``parent_id`` exists), and the searchable ones into the
  profile's FTS5 and ``vec0`` tables. It doesn't commit; the caller wraps it in
  ``stage("index")`` and a transaction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from hal9000.knowledge.chunking import ChunkDraft, ChunkInput, chunk_document
from hal9000.knowledge.contracts import DocumentMeta, IndexProfile
from hal9000.knowledge.index.embedding import EmbeddingService
from hal9000.knowledge.index.keyword import FtsIndex
from hal9000.knowledge.index.vector import ChunkMeta, SqliteVecIndex, VectorRow
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = [
    "PreparedDocument",
    "chunk_documents",
    "embed_chunks",
    "searchable",
    "write_chunks",
]


@dataclass(frozen=True)
class PreparedDocument:
    """A document ready to chunk: its metadata, cleaned text (``PagedText``) and collection."""

    meta: DocumentMeta
    paged: object  # hal9000.knowledge.ingest.clean.PagedText (duck-typed)
    collection: str = ""

    @property
    def chunk_meta(self) -> ChunkMeta:
        return ChunkMeta(
            doc_type=self.meta.doc_type or "",
            collection=self.collection or "",
            doc_date=self.meta.doc_date.isoformat() if self.meta.doc_date else "",
        )


def chunk_documents(
    documents: Sequence[PreparedDocument], profile: IndexProfile
) -> list[list[ChunkDraft]]:
    """Drafts per document, aligned with ``documents`` (``[]`` for a document without text).

    ``ChunkDraft.document_id`` is the document's position in ``documents``;
    :func:`write_chunks` maps it to the stored ID.
    """
    out: list[list[ChunkDraft]] = []
    for position, doc in enumerate(documents):
        if not getattr(doc.paged, "text", ""):
            out.append([])
            continue
        inp = ChunkInput.from_paged(
            doc.paged,
            position,
            title=doc.meta.title,
            doc_type=doc.meta.doc_type,
            vendor=doc.meta.vendor,
            doc_date=doc.meta.doc_date,
        )
        out.append(chunk_document(inp, profile))
    return out


def searchable(chunked: Sequence[Sequence[ChunkDraft]]) -> list[ChunkDraft]:
    return [draft for drafts in chunked for draft in drafts if draft.searchable]


def embed_chunks(
    embeddings: EmbeddingService,
    profile: IndexProfile,
    chunked: Sequence[Sequence[ChunkDraft]],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Vectors for the searchable drafts, in :func:`searchable` order."""
    with stage(Stage.EMBED):
        texts = [draft.index_text for draft in searchable(chunked)]
        return embeddings.embed(texts, profile.embed_model, profile.dimensions, progress=progress)


def write_chunks(
    conn: sqlite3.Connection,
    profile: IndexProfile,
    documents: Sequence[PreparedDocument],
    chunked: Sequence[Sequence[ChunkDraft]],
    vectors: np.ndarray,
) -> int:
    """Insert every draft; index the searchable ones. Returns the number of chunks stored."""
    if profile.id is None:
        raise ValueError("the profile must be stored (have an id) before chunks are written")
    fts = FtsIndex(conn, profile.id)
    vec = SqliteVecIndex(conn, profile.id, profile.dimensions)
    fts_rows: list[tuple[int, str, str]] = []
    vec_rows: list[VectorRow] = []
    next_vector = 0
    stored = 0
    for doc, drafts in zip(documents, chunked):
        if doc.meta.id is None:
            raise ValueError("documents must be stored before their chunks")
        ids: dict[int, int] = {}
        for draft in drafts:
            parent_id = ids.get(draft.parent_key) if draft.parent_key is not None else None
            if draft.parent_key is not None and parent_id is None:
                raise ValueError("a child chunk came before its parent")
            chunk = draft.to_chunk(profile.id, parent_id)
            cur = conn.execute(
                "INSERT INTO chunks (profile_id, document_id, page_start, page_end, char_start, "
                "char_end, text, header, tokens, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    profile.id,
                    doc.meta.id,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.text,
                    chunk.header,
                    chunk.tokens,
                    chunk.parent_id,
                ),
            )
            chunk_id = int(cur.lastrowid)
            ids[draft.key] = chunk_id
            stored += 1
            if draft.searchable:
                fts_rows.append((chunk_id, draft.header, draft.text))
                vec_rows.append(VectorRow(chunk_id, vectors[next_vector], doc.chunk_meta))
                next_vector += 1
    if next_vector != len(vectors):
        raise ValueError(f"{len(vectors)} vectors for {next_vector} searchable chunks")
    fts.add(fts_rows)
    vec.add(vec_rows)
    return stored
