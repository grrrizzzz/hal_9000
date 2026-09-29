"""Keyword search: one FTS5 table per profile, ranked by BM25 (plan §7.1, §12).

``fts_<profile_id>`` has two columns, ``header`` and ``text``: together they
are the chunk's index text (the contextual header, then the text), the same
text that is embedded. Row IDs are chunk IDs. One table per profile keeps
BM25's term statistics (document frequencies, average length) from being
pooled across chunk sizes, so a profile's scores don't move when another
profile is added.

Queries go through :func:`~hal9000.knowledge.index.fts_query.build_fts_query`,
never raw. :attr:`KeywordHit.score` is ``-bm25()``: FTS5's ``bm25()`` is
negative with lower meaning better, and every score in retrieval is
higher-is-better. Filters join the chunk to its document and file.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from hal9000.knowledge.contracts import SearchFilters
from hal9000.knowledge.index.fts_query import build_fts_query
from hal9000.knowledge.storage import fts_table

__all__ = ["FtsIndex", "KeywordHit", "sql_filters"]


@dataclass(frozen=True)
class KeywordHit:
    """A BM25 result; ``score`` is ``-bm25()`` (higher is better)."""

    chunk_id: int
    score: float
    rank: int


def sql_filters(filters: SearchFilters | None) -> tuple[str, list[object]]:
    """``AND ...`` constraints on ``d`` (documents) and ``f`` (files), and their parameters."""
    if filters is None:
        return "", []
    sql: list[str] = []
    params: list[object] = []
    if filters.doc_type:
        sql.append("d.doc_type = ?")
        params.append(filters.doc_type)
    if filters.collection is not None and filters.collection != "":
        sql.append("f.collection = ?")
        params.append(filters.collection)
    if filters.date_from is not None:
        sql.append("d.doc_date >= ?")
        params.append(filters.date_from.isoformat())
    if filters.date_to is not None:
        sql.append("d.doc_date <= ?")
        params.append(filters.date_to.isoformat())
    if filters.date_from is not None or filters.date_to is not None:
        sql.append("COALESCE(d.doc_date, '') != ''")
    return "".join(f" AND {clause}" for clause in sql), params


class FtsIndex:
    """BM25 over one profile's ``fts_<profile_id>`` table."""

    def __init__(self, conn: sqlite3.Connection, profile_id: int) -> None:
        self.conn = conn
        self.profile_id = profile_id
        self.table = fts_table(profile_id)

    def add(self, rows: Iterable[tuple[int, str, str]]) -> int:
        """Insert ``(chunk_id, header, text)`` rows (no commit)."""
        values = [(int(chunk_id), header or "", text) for chunk_id, header, text in rows]
        self.conn.executemany(
            f"INSERT INTO {self.table} (rowid, header, text) VALUES (?, ?, ?)", values
        )
        return len(values)

    def delete(self, chunk_ids: Iterable[int]) -> None:
        self.conn.executemany(
            f"DELETE FROM {self.table} WHERE rowid = ?", [(int(i),) for i in chunk_ids]
        )

    def search(self, query: str, k: int, filters: SearchFilters | None = None) -> list[KeywordHit]:
        """The best ``k`` chunks for a user's question (built into a safe ``MATCH``)."""
        expression = build_fts_query(query)
        if expression is None:
            return []
        return self.match(expression, k, filters)

    def match(
        self, expression: str, k: int, filters: SearchFilters | None = None
    ) -> list[KeywordHit]:
        """Run an already-built ``MATCH`` expression."""
        if k < 1:
            return []
        where, params = sql_filters(filters)
        if where:
            sql = (
                f"SELECT t.rowid, bm25({self.table}) AS s FROM {self.table} t "
                "JOIN chunks c ON c.id = t.rowid JOIN documents d ON d.id = c.document_id "
                f"JOIN files f ON f.sha256 = d.file_sha WHERE {self.table} MATCH ?{where} "
                "ORDER BY s, t.rowid LIMIT ?"
            )
        else:
            sql = (
                f"SELECT rowid, bm25({self.table}) AS s FROM {self.table} "
                f"WHERE {self.table} MATCH ? ORDER BY s, rowid LIMIT ?"
            )
        rows = self.conn.execute(sql, (expression, *params, k)).fetchall()
        return [
            KeywordHit(int(chunk_id), -float(score), rank)
            for rank, (chunk_id, score) in enumerate(rows, start=1)
        ]

    def count(self) -> int:
        return int(self.conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])
