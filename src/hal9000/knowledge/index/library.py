"""Rows of the library: files, documents, pages and the persistent segmentation cache (plan §12).

Small, explicit SQL helpers shared by the ingestion pipeline and profile
builds. None of them commits: the caller owns the transaction.

:class:`DbSegmentCache` is KWP1's ``SegmentCache`` protocol on
``segment_cache`` (migration 2), keyed by ``<file sha>:<SEGMENT_VERSION>``. A
document the user edited in the Library (``meta_source = 'user'``) always
wins: while a file has one, :meth:`DbSegmentCache.get` returns the stored
documents instead of cached labels, and the pipeline keeps them as they are.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, date, datetime

from hal9000.knowledge.contracts import DocumentMeta, FileRecord, Page
from hal9000.knowledge.storage import fts_table, vec_table

__all__ = [
    "DbSegmentCache",
    "delete_file_index",
    "documents_for",
    "file_has_index",
    "get_file",
    "insert_documents",
    "insert_file",
    "latest_version",
    "list_files",
    "load_pages",
    "profile_ids",
    "save_pages",
    "user_documents",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# -------------------------------------------------------------------- files
def insert_file(conn: sqlite3.Connection, record: FileRecord) -> None:
    conn.execute(
        "INSERT INTO files (sha256, original_name, relative_path, collection, mime_type, bytes, "
        "version_of, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.sha256,
            record.original_name,
            record.relative_path,
            record.collection,
            record.mime_type,
            record.bytes,
            record.version_of,
            record.uploaded_at.isoformat(timespec="seconds"),
        ),
    )


def _file(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        sha256=row["sha256"],
        original_name=row["original_name"],
        relative_path=row["relative_path"],
        collection=row["collection"],
        mime_type=row["mime_type"],
        bytes=row["bytes"],
        version_of=row["version_of"],
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
    )


def get_file(conn: sqlite3.Connection, sha: str) -> FileRecord | None:
    row = conn.execute("SELECT * FROM files WHERE sha256 = ?", (sha,)).fetchone()
    return _file(row) if row else None


def list_files(conn: sqlite3.Connection) -> list[FileRecord]:
    rows = conn.execute("SELECT * FROM files ORDER BY uploaded_at, rowid").fetchall()
    return [_file(row) for row in rows]


def latest_version(conn: sqlite3.Connection, relative_path: str) -> str | None:
    """The newest file stored under ``relative_path`` (a new upload there is its next version)."""
    row = conn.execute(
        "SELECT sha256 FROM files WHERE relative_path = ? ORDER BY uploaded_at DESC, rowid DESC "
        "LIMIT 1",
        (relative_path,),
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------- documents
def _document(row: sqlite3.Row) -> DocumentMeta:
    return DocumentMeta(
        id=row["id"],
        file_sha=row["file_sha"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        title=row["title"],
        doc_type=row["doc_type"],
        vendor=row["vendor"],
        doc_date=date.fromisoformat(row["doc_date"]) if row["doc_date"] else None,
        meta_source=row["meta_source"],
    )


def documents_for(conn: sqlite3.Connection, sha: str) -> list[DocumentMeta]:
    rows = conn.execute(
        "SELECT * FROM documents WHERE file_sha = ? ORDER BY page_start, id", (sha,)
    ).fetchall()
    return [_document(row) for row in rows]


def user_documents(conn: sqlite3.Connection, sha: str) -> list[DocumentMeta]:
    return [doc for doc in documents_for(conn, sha) if doc.meta_source == "user"]


def file_has_index(conn: sqlite3.Connection, sha: str) -> bool:
    """Whether the file's ingestion completed (it has documents)."""
    return (
        conn.execute("SELECT 1 FROM documents WHERE file_sha = ? LIMIT 1", (sha,)).fetchone()
        is not None
    )


def insert_documents(
    conn: sqlite3.Connection, documents: Iterable[DocumentMeta]
) -> list[DocumentMeta]:
    """Insert documents; returns them with their new IDs."""
    out: list[DocumentMeta] = []
    for doc in documents:
        cur = conn.execute(
            "INSERT INTO documents (file_sha, page_start, page_end, title, doc_type, vendor, "
            "doc_date, meta_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc.file_sha,
                doc.page_start,
                doc.page_end,
                doc.title,
                doc.doc_type,
                doc.vendor,
                doc.doc_date.isoformat() if doc.doc_date else None,
                doc.meta_source,
            ),
        )
        out.append(doc.model_copy(update={"id": int(cur.lastrowid)}))
    return out


def profile_ids(conn: sqlite3.Connection) -> list[int]:
    return [int(row[0]) for row in conn.execute("SELECT id FROM profiles ORDER BY id")]


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def delete_file_index(conn: sqlite3.Connection, sha: str, *, keep_user_documents: bool) -> None:
    """Remove a file's chunks from every profile (tables included) and its documents.

    Chunks cascade from documents in ``chunks``, but not out of the FTS5 and
    ``vec0`` tables, so those rows are deleted explicitly first. With
    ``keep_user_documents`` the user-edited documents stay (their chunks go).
    """
    tables = _existing_tables(conn)
    for profile_id in profile_ids(conn):
        ids = [
            (int(row[0]),)
            for row in conn.execute(
                "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.file_sha = ? AND c.profile_id = ?",
                (sha, profile_id),
            )
        ]
        if not ids:
            continue
        if fts_table(profile_id) in tables:
            conn.executemany(f"DELETE FROM {fts_table(profile_id)} WHERE rowid = ?", ids)
        if vec_table(profile_id) in tables:
            conn.executemany(f"DELETE FROM {vec_table(profile_id)} WHERE chunk_id = ?", ids)
    conn.execute(
        "DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE file_sha = ?)",
        (sha,),
    )
    if keep_user_documents:
        conn.execute("DELETE FROM documents WHERE file_sha = ? AND meta_source != 'user'", (sha,))
    else:
        conn.execute("DELETE FROM documents WHERE file_sha = ?", (sha,))


# -------------------------------------------------------------------- pages
def save_pages(conn: sqlite3.Connection, pages: Iterable[Page]) -> int:
    """Store cleaned pages, replacing any earlier copy for the same file and parser."""
    rows = [
        (
            page.file_sha,
            page.page,
            page.parser,
            page.text,
            int(page.has_text_layer),
            json.dumps([list(pair) for pair in page.offset_map]),
        )
        for page in pages
    ]
    for sha, parser in {(row[0], row[2]) for row in rows}:
        conn.execute("DELETE FROM pages WHERE file_sha = ? AND parser = ?", (sha, parser))
    conn.executemany(
        "INSERT INTO pages (file_sha, page, parser, text, has_text_layer, offset_map) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def load_pages(conn: sqlite3.Connection, sha: str, parser: str) -> list[Page]:
    rows = conn.execute(
        "SELECT * FROM pages WHERE file_sha = ? AND parser = ? ORDER BY page", (sha, parser)
    ).fetchall()
    return [
        Page(
            file_sha=row["file_sha"],
            page=row["page"],
            parser=row["parser"],
            text=row["text"],
            has_text_layer=bool(row["has_text_layer"]),
            offset_map=[tuple(pair) for pair in json.loads(row["offset_map"])],
        )
        for row in rows
    ]


# ------------------------------------------------------------ segment cache
class DbSegmentCache:
    """KWP1's ``SegmentCache`` on ``knowledge.db`` (commits its own writes)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get(self, key: str) -> list[DocumentMeta] | None:
        sha = key.split(":", 1)[0]
        if user_documents(self.conn, sha):
            return [doc.model_copy(update={"id": None}) for doc in documents_for(self.conn, sha)]
        row = self.conn.execute(
            "SELECT documents FROM segment_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return [DocumentMeta.model_validate(item) for item in json.loads(row[0])]

    def put(self, key: str, documents: list[DocumentMeta]) -> None:
        body = json.dumps([doc.model_dump(mode="json", exclude={"id"}) for doc in documents])
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO segment_cache (key, file_sha, documents, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, key.split(":", 1)[0], body, _now()),
            )
