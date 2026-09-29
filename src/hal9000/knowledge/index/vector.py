"""Dense search: the ``VectorIndex`` protocol and its sqlite-vec implementation (plan §3.3, §12).

One ``vec0`` table per index profile, ``vec_<profile_id>``, created by
:func:`hal9000.knowledge.storage.create_profile_tables` with
``distance_metric=cosine`` and the metadata columns ``doc_type``,
``collection`` and ``doc_date`` (ISO text; ``''`` when unknown, because
sqlite-vec 0.1.9 rejects NULL metadata). Row IDs are chunk IDs.

Filters are ``vec0`` metadata constraints in the KNN query itself, so they
apply **inside** the search: a filtered search still returns *k* results when
at least *k* chunks match (a post-filter would return fewer). A date range
excludes chunks whose document date is unknown.

The protocol keeps Qdrant or LanceDB as drop-in alternatives (plan §3.3).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from hal9000.knowledge.contracts import SearchFilters
from hal9000.knowledge.storage import vec_table

__all__ = ["MAX_K", "ChunkMeta", "SqliteVecIndex", "VectorHit", "VectorIndex", "VectorRow"]

#: sqlite-vec's largest ``k``.
MAX_K = 4096


@dataclass(frozen=True)
class ChunkMeta:
    """A chunk's filterable metadata (its document's). Unknown values are ``''``."""

    doc_type: str = ""
    collection: str = ""
    doc_date: str = ""


@dataclass(frozen=True)
class VectorRow:
    chunk_id: int
    vector: Sequence[float] | np.ndarray
    meta: ChunkMeta = ChunkMeta()


@dataclass(frozen=True)
class VectorHit:
    """A KNN result: ``similarity`` is the cosine similarity (1 − cosine distance)."""

    chunk_id: int
    similarity: float
    rank: int

    @property
    def score(self) -> float:
        return self.similarity


@runtime_checkable
class VectorIndex(Protocol):
    """What retrieval needs from a vector store."""

    dimensions: int

    def add(self, rows: Iterable[VectorRow]) -> int: ...  # pragma: no cover

    def delete(self, chunk_ids: Iterable[int]) -> None: ...  # pragma: no cover

    def search(
        self, vector: Sequence[float] | np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[VectorHit]: ...  # pragma: no cover

    def count(self) -> int: ...  # pragma: no cover


def _blob(vector: Sequence[float] | np.ndarray, dimensions: int) -> bytes:
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (dimensions,):
        raise ValueError(f"expected a {dimensions}-dimension vector, got shape {array.shape}")
    return array.tobytes()


def vec_filters(filters: SearchFilters | None) -> tuple[str, list[object]]:
    """``AND ...`` constraints on the ``vec0`` metadata columns, and their parameters."""
    if filters is None:
        return "", []
    sql: list[str] = []
    params: list[object] = []
    if filters.doc_type:
        sql.append("doc_type = ?")
        params.append(filters.doc_type)
    if filters.collection is not None and filters.collection != "":
        sql.append("collection = ?")
        params.append(filters.collection)
    if filters.date_from is not None:
        sql.append("doc_date >= ?")
        params.append(filters.date_from.isoformat())
    if filters.date_to is not None:
        sql.append("doc_date <= ?")
        params.append(filters.date_to.isoformat())
        sql.append("doc_date != ''")
    return "".join(f" AND {clause}" for clause in sql), params


class SqliteVecIndex:
    """:class:`VectorIndex` over one profile's ``vec_<profile_id>`` table."""

    def __init__(self, conn: sqlite3.Connection, profile_id: int, dimensions: int) -> None:
        self.conn = conn
        self.profile_id = profile_id
        self.dimensions = dimensions
        self.table = vec_table(profile_id)

    def add(self, rows: Iterable[VectorRow]) -> int:
        """Insert rows (no commit: the caller owns the transaction)."""
        values = [
            (
                row.chunk_id,
                _blob(row.vector, self.dimensions),
                row.meta.doc_type or "",
                row.meta.collection or "",
                row.meta.doc_date or "",
            )
            for row in rows
        ]
        self.conn.executemany(
            f"INSERT INTO {self.table} (chunk_id, embedding, doc_type, collection, doc_date) "
            "VALUES (?, ?, ?, ?, ?)",
            values,
        )
        return len(values)

    def delete(self, chunk_ids: Iterable[int]) -> None:
        self.conn.executemany(
            f"DELETE FROM {self.table} WHERE chunk_id = ?", [(int(i),) for i in chunk_ids]
        )

    def update_meta(self, chunk_ids: Iterable[int], meta: ChunkMeta) -> None:
        """Rewrite the filter columns (after a document's metadata was edited)."""
        self.conn.executemany(
            f"UPDATE {self.table} SET doc_type = ?, collection = ?, doc_date = ? "
            "WHERE chunk_id = ?",
            [(meta.doc_type, meta.collection, meta.doc_date, int(i)) for i in chunk_ids],
        )

    def search(
        self, vector: Sequence[float] | np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[VectorHit]:
        """The ``k`` nearest chunks by cosine distance, filters applied inside the KNN."""
        if k < 1:
            return []
        where, params = vec_filters(filters)
        rows = self.conn.execute(
            f"SELECT chunk_id, distance FROM {self.table} "
            f"WHERE embedding MATCH ? AND k = ?{where} ORDER BY distance",
            (_blob(vector, self.dimensions), min(k, MAX_K), *params),
        ).fetchall()
        return [
            VectorHit(int(chunk_id), 1.0 - float(distance), rank)
            for rank, (chunk_id, distance) in enumerate(rows, start=1)
        ]

    def count(self) -> int:
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])
