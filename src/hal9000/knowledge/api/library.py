"""The Library's queries and edits (plan §10, §11): documents, pages with chunk spans, deletes.

Collections are stored normalised as ``A/B/`` (decision D2) and shown without
the trailing slash (``A/B``), the way the UI groups them; a filter accepts
either form and includes sub-folders.

A metadata edit (``PATCH /api/documents/{id}``) marks the document
``meta_source = 'user'``, which wins over segmentation from then on, including
on re-ingestion. Changing a page range may leave pages that no document
covers; those get an automatic ``other`` document so nothing drops out of the
index. The file is then re-indexed in the background so its chunks, headers
and search filters carry the new metadata.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import date

from hal9000.knowledge.contracts import (
    ChunkSpan,
    DocumentDetail,
    DocumentMeta,
    DocumentPatch,
    DocumentSummary,
    FileRecord,
    PageView,
)
from hal9000.knowledge.index.filters import collection_bounds, normalize_collection
from hal9000.knowledge.index.library import (
    _document,
    _file,
    delete_file_chunks,
    documents_for,
    get_file,
    load_pages,
)
from hal9000.knowledge.index.profiles import active_profile, parser_key
from hal9000.knowledge.storage import fts_table, vec_table

__all__ = [
    "EditError",
    "delete_document",
    "display_collection",
    "document_detail",
    "file_parser",
    "list_documents",
    "page_view",
    "patch_document",
    "shown",
]


class EditError(ValueError):
    """An edit the Library can't apply (a page range outside the file, overlapping another)."""


def display_collection(collection: str) -> str:
    """``A/B/`` → ``A/B`` (the root stays ``''``)."""
    return collection[:-1] if collection.endswith("/") else collection


def shown(record: FileRecord) -> FileRecord:
    return record.model_copy(update={"collection": display_collection(record.collection)})


def file_parser(conn: sqlite3.Connection, record: FileRecord) -> str | None:
    """The parser whose pages the viewer shows: the active profile's, else any stored."""
    profile = active_profile(conn)
    if profile is not None:
        return parser_key(record, profile.parser)
    row = conn.execute(
        "SELECT parser FROM pages WHERE file_sha = ? ORDER BY parser LIMIT 1", (record.sha256,)
    ).fetchone()
    return row[0] if row else None


def _status(conn: sqlite3.Connection, sha: str) -> str:
    row = conn.execute(
        "SELECT status FROM jobs WHERE file_sha = ? ORDER BY id DESC LIMIT 1", (sha,)
    ).fetchone()
    if row is None or row[0] == "done":
        return "indexed"
    return {"queued": "queued", "running": "indexing", "failed": "failed"}[row[0]]


def _page_counts(
    conn: sqlite3.Connection, record: FileRecord, doc: DocumentMeta
) -> tuple[int, int]:
    parser = file_parser(conn, record)
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(1 - has_text_layer), 0) FROM pages WHERE file_sha = ? "
        "AND parser = ? AND page BETWEEN ? AND ?",
        (record.sha256, parser, doc.page_start, doc.page_end),
    ).fetchone()
    pages = int(row[0]) or (doc.page_end - doc.page_start + 1)
    return pages, int(row[1])


def _chunk_count(conn: sqlite3.Connection, document_id: int) -> int:
    profile = active_profile(conn)
    if profile is None:
        return 0
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE profile_id = ? AND document_id = ?",
            (profile.id, document_id),
        ).fetchone()[0]
    )


def _summary(conn: sqlite3.Connection, record: FileRecord, doc: DocumentMeta) -> DocumentSummary:
    pages, without = _page_counts(conn, record, doc)
    assert doc.id is not None
    return DocumentSummary(
        document=doc,
        original_name=record.original_name,
        relative_path=record.relative_path,
        collection=display_collection(record.collection),
        pages=pages,
        pages_without_text=without,
        chunks=_chunk_count(conn, doc.id),
        status=_status(conn, record.sha256),  # type: ignore[arg-type]
    )


def list_documents(
    conn: sqlite3.Connection, *, collection: str | None = None, doc_type: str | None = None
) -> list[DocumentSummary]:
    """Every document of the current file versions, by collection and path."""
    clauses = ["f.superseded_by IS NULL"]
    params: list[object] = []
    bounds = collection_bounds(collection) if collection else None
    if bounds is not None:
        clauses.append("f.collection >= ? AND f.collection < ?")
        params.extend(bounds)
    if doc_type:
        clauses.append("d.doc_type = ?")
        params.append(doc_type)
    rows = conn.execute(
        "SELECT d.*, f.sha256 AS f_sha FROM documents d JOIN files f ON f.sha256 = d.file_sha "
        f"WHERE {' AND '.join(clauses)} ORDER BY f.collection, f.relative_path, d.page_start",
        params,
    ).fetchall()
    files: dict[str, FileRecord] = {}
    out: list[DocumentSummary] = []
    for row in rows:
        sha = row["f_sha"]
        if sha not in files:
            record = get_file(conn, sha)
            assert record is not None
            files[sha] = record
        out.append(_summary(conn, files[sha], _document(row)))
    return out


def _get_document(conn: sqlite3.Connection, document_id: int) -> DocumentMeta | None:
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return _document(row) if row else None


def document_detail(conn: sqlite3.Connection, document_id: int) -> DocumentDetail | None:
    doc = _get_document(conn, document_id)
    if doc is None:
        return None
    record = get_file(conn, doc.file_sha)
    assert record is not None
    summary = _summary(conn, record, doc)
    siblings = [d for d in documents_for(conn, doc.file_sha) if d.id != doc.id]
    return DocumentDetail(**summary.model_dump(), file=shown(record), siblings=siblings)


def _page_total(conn: sqlite3.Connection, record: FileRecord) -> int:
    parser = file_parser(conn, record)
    row = conn.execute(
        "SELECT MAX(page) FROM pages WHERE file_sha = ? AND parser = ?", (record.sha256, parser)
    ).fetchone()
    if row and row[0]:
        return int(row[0])
    return max((d.page_end for d in documents_for(conn, record.sha256)), default=0)


def _gaps(ranges: Iterable[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    covered = sorted(ranges)
    gaps: list[tuple[int, int]] = []
    nxt = 1
    for start, end in covered:
        if start > nxt:
            gaps.append((nxt, start - 1))
        nxt = max(nxt, end + 1)
    if total >= nxt:
        gaps.append((nxt, total))
    return gaps


def patch_document(
    conn: sqlite3.Connection, document_id: int, patch: DocumentPatch
) -> DocumentMeta | None:
    """Apply a Library edit (the user's labels win from now on); ``None`` if no such document.

    Raises :class:`EditError` for a page range outside the file or overlapping
    a sibling. The caller re-indexes the file.
    """
    doc = _get_document(conn, document_id)
    if doc is None:
        return None
    changes = patch.model_dump(exclude_unset=True)
    if "title" in changes and not (changes["title"] or "").strip():
        raise EditError("a document needs a title")
    if "doc_type" in changes and changes["doc_type"] is None:
        changes.pop("doc_type")
    updated = doc.model_copy(update={**changes, "meta_source": "user"})
    try:
        updated = DocumentMeta.model_validate(updated.model_dump())
    except ValueError as error:
        raise EditError(str(error)) from None
    record = get_file(conn, doc.file_sha)
    assert record is not None
    total = _page_total(conn, record)
    if total and updated.page_end > total:
        raise EditError(f"the file has {total} page(s); page {updated.page_end} is outside it")
    siblings = [d for d in documents_for(conn, doc.file_sha) if d.id != doc.id]
    for other in siblings:
        if updated.page_start <= other.page_end and other.page_start <= updated.page_end:
            raise EditError(
                f"pages {updated.page_start}-{updated.page_end} overlap "
                f"{other.title!r} (pages {other.page_start}-{other.page_end})"
            )
    with conn:
        conn.execute(
            "UPDATE documents SET title = ?, doc_type = ?, vendor = ?, doc_date = ?, "
            "page_start = ?, page_end = ?, meta_source = 'user' WHERE id = ?",
            (
                updated.title.strip(),
                updated.doc_type,
                updated.vendor,
                updated.doc_date.isoformat() if isinstance(updated.doc_date, date) else None,
                updated.page_start,
                updated.page_end,
                document_id,
            ),
        )
        ranges = [(d.page_start, d.page_end) for d in siblings]
        ranges.append((updated.page_start, updated.page_end))
        for start, end in _gaps(ranges, total):
            pages = f"page {start}" if start == end else f"pages {start}-{end}"
            conn.execute(
                "INSERT INTO documents (file_sha, page_start, page_end, title, doc_type, "
                "meta_source) VALUES (?, ?, ?, ?, 'other', 'user')",
                (doc.file_sha, start, end, f"{record.original_name} ({pages})"),
            )
    return _get_document(conn, document_id)


def _delete_document_chunks(conn: sqlite3.Connection, document_id: int) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    rows = conn.execute(
        "SELECT id, profile_id FROM chunks WHERE document_id = ?", (document_id,)
    ).fetchall()
    for chunk_id, profile_id in rows:
        if fts_table(profile_id) in tables:
            conn.execute(f"DELETE FROM {fts_table(profile_id)} WHERE rowid = ?", (chunk_id,))
        if vec_table(profile_id) in tables:
            conn.execute(f"DELETE FROM {vec_table(profile_id)} WHERE chunk_id = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))


def delete_document(conn: sqlite3.Connection, document_id: int, files_dir) -> bool:
    """Remove a document from the library (its chunks leave every index).

    When it was the file's last document the file goes too: its rows, pages,
    jobs and the stored original.
    """
    doc = _get_document(conn, document_id)
    if doc is None:
        return False
    with conn:
        _delete_document_chunks(conn, document_id)
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        remaining = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE file_sha = ?", (doc.file_sha,)
        ).fetchone()[0]
        if remaining == 0:
            delete_file_chunks(conn, doc.file_sha)
            conn.execute("DELETE FROM jobs WHERE file_sha = ?", (doc.file_sha,))
            conn.execute("DELETE FROM segment_cache WHERE file_sha = ?", (doc.file_sha,))
            conn.execute("UPDATE files SET version_of = NULL WHERE version_of = ?", (doc.file_sha,))
            conn.execute(
                "UPDATE files SET superseded_by = NULL WHERE superseded_by = ?", (doc.file_sha,)
            )
            conn.execute("DELETE FROM files WHERE sha256 = ?", (doc.file_sha,))
    if remaining == 0 and files_dir is not None:
        stored = files_dir / doc.file_sha
        stored.unlink(missing_ok=True)
    return True


def page_view(conn: sqlite3.Connection, sha: str, number: int) -> PageView | None:
    """Cleaned text of page ``number`` and the active profile's chunk spans on it."""
    record = get_file(conn, sha)
    if record is None:
        return None
    parser = file_parser(conn, record)
    if parser is None:
        return None
    pages = load_pages(conn, sha, parser)
    page = next((p for p in pages if p.page == number), None)
    if page is None:
        return None
    return PageView(
        file_sha=sha,
        page=number,
        page_count=max(p.page for p in pages),
        text=page.text,
        has_text_layer=page.has_text_layer,
        spans=_spans(conn, record, pages, number),
    )


def _spans(conn: sqlite3.Connection, record: FileRecord, pages, number: int) -> list[ChunkSpan]:
    """Chunk spans (offsets into the document's joined text) clipped to page ``number``."""
    from hal9000.knowledge.ingest.clean import CleanResult

    profile = active_profile(conn)
    if profile is None:
        return []
    doc = next(
        (d for d in documents_for(conn, record.sha256) if d.page_start <= number <= d.page_end),
        None,
    )
    if doc is None or doc.id is None:
        return []
    paged = CleanResult(pages=pages).document(doc.page_start, doc.page_end)
    numbers = [p if isinstance(p, int) else p.page for p in paged.pages]
    if number not in numbers:
        return []
    index = numbers.index(number)
    start = int(paged.starts[index])
    text = next(p.text for p in pages if p.page == number)
    end = start + len(text)
    rows = conn.execute(
        "SELECT id, char_start, char_end FROM chunks WHERE profile_id = ? AND document_id = ? "
        "AND page_start <= ? AND page_end >= ? AND parent_id IS NULL ORDER BY char_start, id",
        (profile.id, doc.id, number, number),
    ).fetchall()
    spans: list[ChunkSpan] = []
    for chunk_id, char_start, char_end in rows:
        low, high = max(char_start, start), min(char_end, end)
        if high > low:
            spans.append(
                ChunkSpan(chunk_id=chunk_id, char_start=low - start, char_end=high - start)
            )
    return spans


def file_record(conn: sqlite3.Connection, sha: str) -> FileRecord | None:
    row = conn.execute("SELECT * FROM files WHERE sha256 = ?", (sha,)).fetchone()
    return _file(row) if row else None


def normalized(collection: str | None) -> str | None:
    return normalize_collection(collection) if collection else None
