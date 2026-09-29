"""Search filters as SQL: ``doc_type``, collection (with its sub-folders) and a date range.

**Collections** (decision D2) are folder paths normalised to ``A/B/``: parts
joined with ``/`` and a trailing ``/``; ``''`` is the root.
:func:`normalize_collection` is applied everywhere a collection enters: the
folder-path default, ``--collection``, the API and the filter value. A filter
on ``Appliances/`` matches ``Appliances/`` and every sub-folder
(``Appliances/Furnace/``), but not ``Appliances-old/``: with the trailing
``/`` replaced by ``0`` (the next code point), the range
``[Appliances/, Appliances0)`` holds exactly the paths under it. (A sentinel
such as ``U+FFFF`` would break on characters outside the BMP.) The root
filters nothing.

- **BM25** (:func:`sql_filters`) filters the joined ``files`` row with that range.
- **Vectors** (:func:`vec_filters`): sqlite-vec 0.1.9 mis-evaluates text range
  constraints on metadata values of 12 bytes or more, so the collections in
  range are looked up in ``files`` with plain SQL and the KNN filters
  ``collection IN (...)``, still inside the KNN. No collection in range means
  no result.

A date range excludes documents whose date is unknown (stored ``''`` in
``vec0``, NULL in ``documents``).
"""

from __future__ import annotations

import sqlite3

from hal9000.knowledge.contracts import SearchFilters

__all__ = [
    "NO_MATCH",
    "collection_bounds",
    "normalize_collection",
    "sql_filters",
    "vec_filters",
]

#: :func:`vec_filters` returns this when nothing can match (a collection with no files).
NO_MATCH = None


def normalize_collection(value: str | None) -> str:
    """``' Appliances\\\\Furnace '`` → ``'Appliances/Furnace/'``; empty → ``''`` (the root)."""
    parts = [
        part.strip()
        for part in str(value or "").replace("\\", "/").split("/")
        if part.strip() not in ("", ".", "..")
    ]
    for part in parts:
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in part):
            raise ValueError(f"a collection can't contain control characters: {value!r}")
    return "/".join(parts) + "/" if parts else ""


def collection_bounds(value: str | None) -> tuple[str, str] | None:
    """``[low, high)`` covering a collection and its sub-folders; ``None`` for the root."""
    low = normalize_collection(value)
    if not low:
        return None
    return low, low[:-1] + "0"


def sql_filters(filters: SearchFilters | None) -> tuple[str, list[object]]:
    """``AND ...`` constraints on ``d`` (documents) and ``f`` (files), and their parameters."""
    if filters is None:
        return "", []
    sql: list[str] = []
    params: list[object] = []
    if filters.doc_type:
        sql.append("d.doc_type = ?")
        params.append(filters.doc_type)
    bounds = collection_bounds(filters.collection)
    if bounds is not None:
        sql.append("f.collection >= ? AND f.collection < ?")
        params.extend(bounds)
    if filters.date_from is not None:
        sql.append("d.doc_date >= ?")
        params.append(filters.date_from.isoformat())
    if filters.date_to is not None:
        sql.append("d.doc_date <= ?")
        params.append(filters.date_to.isoformat())
    if filters.date_from is not None or filters.date_to is not None:
        sql.append("COALESCE(d.doc_date, '') != ''")
    return "".join(f" AND {clause}" for clause in sql), params


def vec_filters(
    conn: sqlite3.Connection, filters: SearchFilters | None
) -> tuple[str, list[object]] | None:
    """``AND ...`` constraints on the ``vec0`` metadata columns, or :data:`NO_MATCH`."""
    if filters is None:
        return "", []
    sql: list[str] = []
    params: list[object] = []
    if filters.doc_type:
        sql.append("doc_type = ?")
        params.append(filters.doc_type)
    bounds = collection_bounds(filters.collection)
    if bounds is not None:
        collections = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT collection FROM files WHERE collection >= ? AND collection < ? "
                "ORDER BY collection",
                bounds,
            )
        ]
        if not collections:
            return NO_MATCH
        sql.append(f"collection IN ({', '.join('?' * len(collections))})")
        params.extend(collections)
    if filters.date_from is not None:
        sql.append("doc_date >= ?")
        params.append(filters.date_from.isoformat())
    if filters.date_to is not None:
        sql.append("doc_date <= ?")
        params.append(filters.date_to.isoformat())
        sql.append("doc_date != ''")
    return "".join(f" AND {clause}" for clause in sql), params
