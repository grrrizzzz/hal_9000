"""Index profiles in storage: lookup, build and atomic activation (plan §6.2).

A profile (``pdfium.structure-768.hdr.te3s-1536``) is a row in ``profiles``
plus its ``fts_<id>`` and ``vec_<id>`` tables. Its ``status`` is ``building``
until every stored document is chunked, embedded and indexed into it, then
``ready``; a build that fails leaves it ``failed`` with its partial rows
removed.

**Exactly one profile is active** (a unique index enforces "at most one").
:func:`activate_profile` flips it in one transaction and refuses a profile
that isn't ``ready``, so queries keep hitting the old active profile until the
new one is complete, and a failed build leaves the old one active.
:func:`build_profile` rebuilds from stored pages and cached embeddings;
rebuilding an existing ready profile builds a shadow copy and swaps it in
atomically when complete.

Which profile is active: the ``profiles.active`` row. When none is,
``active_profile`` in ``config/knowledge.yaml`` names the one to build and
activate (see :func:`configured_profile`).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePath

from hal9000.knowledge.chunking import parse_profile_name
from hal9000.knowledge.contracts import FileRecord, IndexProfile, Page
from hal9000.knowledge.index.embedding import EmbeddingService
from hal9000.knowledge.index.library import documents_for, list_files
from hal9000.knowledge.index.writer import (
    PreparedDocument,
    chunk_documents,
    embed_chunks,
    write_chunks,
)
from hal9000.knowledge.obs.timing import Stage, stage
from hal9000.knowledge.storage import create_profile_tables, drop_profile_tables

__all__ = [
    "BuildReport",
    "ProfileError",
    "ProfileNotReady",
    "activate_profile",
    "active_profile",
    "build_profile",
    "configured_profile",
    "ensure_profile",
    "get_profile",
    "list_profiles",
    "parser_key",
    "profile_status",
]

log = logging.getLogger(__name__)

#: Loads the cleaned pages of ``file`` as parsed by ``parser`` (parsing them if needed).
PagesFor = Callable[[sqlite3.Connection, FileRecord, str], list[Page]]
Progress = Callable[[float, str], None]


class ProfileError(RuntimeError):
    """A profile can't be built or activated."""


class ProfileNotReady(ProfileError):
    """Only a ``ready`` profile can be activated."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _profile(row: sqlite3.Row) -> IndexProfile:
    return IndexProfile(
        id=row["id"],
        name=row["name"],
        parser=row["parser"],
        chunker=row["chunker"],
        size=row["size"],
        overlap=row["overlap"],
        parent_size=row["parent_size"],
        header=bool(row["header"]),
        embed_model=row["embed_model"],
        dimensions=row["dimensions"],
        active=bool(row["active"]),
    )


def get_profile(conn: sqlite3.Connection, name: str) -> IndexProfile | None:
    row = conn.execute("SELECT * FROM profiles WHERE name = ?", (name,)).fetchone()
    return _profile(row) if row else None


def profile_status(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute("SELECT status FROM profiles WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def list_profiles(conn: sqlite3.Connection) -> list[IndexProfile]:
    rows = conn.execute("SELECT * FROM profiles WHERE name NOT LIKE '%#%' ORDER BY id").fetchall()
    return [_profile(row) for row in rows]


def active_profile(conn: sqlite3.Connection) -> IndexProfile | None:
    """The profile serving the chat: the one ``active`` row (always ``ready``)."""
    row = conn.execute("SELECT * FROM profiles WHERE active = 1").fetchone()
    return _profile(row) if row else None


def configured_profile(name: str | None) -> IndexProfile | None:
    """The profile ``config/knowledge.yaml`` names (unsaved), or ``None``."""
    return parse_profile_name(name) if name else None


def ensure_profile(
    conn: sqlite3.Connection, profile: IndexProfile, *, name: str | None = None
) -> IndexProfile:
    """The stored profile named ``name`` (default ``profile.name``), inserted if missing.

    A new row is ``building``; its FTS5 and ``vec0`` tables are created. Commits.
    """
    name = name or profile.name
    existing = get_profile(conn, name)
    if existing is None:
        with conn:
            cur = conn.execute(
                "INSERT INTO profiles (name, parser, chunker, size, overlap, parent_size, header, "
                "embed_model, dimensions, status, active, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'building', 0, ?)",
                (
                    name,
                    profile.parser,
                    profile.chunker,
                    profile.size,
                    profile.overlap,
                    profile.parent_size,
                    int(profile.header),
                    profile.embed_model,
                    profile.dimensions,
                    _now(),
                ),
            )
        existing = profile.model_copy(
            update={"id": int(cur.lastrowid), "name": name, "active": False}
        )
    elif (existing.chunker, existing.size, existing.overlap, existing.parent_size,
          existing.header, existing.embed_model, existing.dimensions, existing.parser) != (
        profile.chunker, profile.size, profile.overlap, profile.parent_size,
        profile.header, profile.embed_model, profile.dimensions, profile.parser,
    ):  # fmt: skip
        raise ProfileError(f"profile {name!r} is stored with different settings")
    create_profile_tables(conn, existing.id, existing.dimensions)
    return existing


def activate_profile(conn: sqlite3.Connection, name: str) -> IndexProfile:
    """Make ``name`` the one active profile, atomically. It must be ``ready``."""
    with conn:
        row = conn.execute("SELECT id, status FROM profiles WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise ProfileError(f"no index profile named {name!r}")
        if row["status"] != "ready":
            raise ProfileNotReady(
                f"profile {name!r} is {row['status']}; only a ready profile can be activated"
            )
        conn.execute("UPDATE profiles SET active = 0 WHERE active = 1 AND id != ?", (row["id"],))
        conn.execute("UPDATE profiles SET active = 1 WHERE id = ?", (row["id"],))
    profile = get_profile(conn, name)
    assert profile is not None
    return profile


def parser_key(file: FileRecord, pdf_parser: str) -> str:
    """The parser a profile uses for ``file``: its own parser for PDFs, the format's otherwise."""
    from hal9000.knowledge.ingest.parsers import SUFFIXES

    kind = SUFFIXES.get(PurePath(file.original_name).suffix.lower(), "pdf")
    return pdf_parser if kind == "pdf" else kind


def stored_pages(conn: sqlite3.Connection, file: FileRecord, parser: str) -> list[Page]:
    """The default :data:`PagesFor`: stored pages only."""
    from hal9000.knowledge.index.library import load_pages

    pages = load_pages(conn, file.sha256, parser)
    if not pages:
        raise ProfileError(f"no {parser} pages are stored for {file.original_name}")
    return pages


@dataclass
class BuildReport:
    profile: IndexProfile
    files: int = 0
    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    cache_hits: int = 0
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)


def _discard(conn: sqlite3.Connection, profile_id: int, *, delete_row: bool) -> None:
    with conn:
        conn.execute("DELETE FROM chunks WHERE profile_id = ?", (profile_id,))
        if delete_row:
            conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        else:
            conn.execute("UPDATE profiles SET status = 'failed' WHERE id = ?", (profile_id,))
    drop_profile_tables(conn, profile_id)


def build_profile(
    conn: sqlite3.Connection,
    profile: IndexProfile,
    embeddings: EmbeddingService,
    *,
    pages_for: PagesFor = stored_pages,
    activate: bool = False,
    rebuild: bool = False,
    progress: Progress | None = None,
) -> BuildReport:
    """Chunk, embed and index every stored document into ``profile``.

    - A profile that is already ``ready`` is left alone (``skipped``) unless
      ``rebuild``: then a shadow copy is built and swapped in, in one
      transaction, when complete (it stays active if it was).
    - A ``building`` or ``failed`` leftover is cleared and built again.
    - Embeddings come from ``embed_cache`` where the text was seen before, so
      only new chunk texts are paid for.
    - On any error the partial profile is removed (the row is kept as
      ``failed``) and the error re-raised; the active profile is untouched.
    - ``activate`` activates it once ready.
    """
    report = BuildReport(profile=profile)
    existing = get_profile(conn, profile.name)
    status = profile_status(conn, profile.name)
    shadow = False
    if existing is not None and status == "ready" and not rebuild:
        report.profile = activate_profile(conn, profile.name) if activate else existing
        report.skipped = True
        return report
    if existing is not None and status == "ready":
        shadow = True
        target = ensure_profile(conn, profile, name=f"{profile.name}#rebuild")
        _discard(conn, target.id, delete_row=True)
        target = ensure_profile(conn, profile, name=f"{profile.name}#rebuild")
    else:
        if existing is not None:  # building or failed: start clean
            _discard(conn, existing.id, delete_row=False)
            with conn:
                conn.execute("UPDATE profiles SET status = 'building' WHERE id = ?", (existing.id,))
        target = ensure_profile(conn, profile)

    try:
        files = [f for f in list_files(conn) if documents_for(conn, f.sha256)]
        for number, file in enumerate(files, start=1):
            if progress is not None:
                progress((number - 1) / max(len(files), 1), f"{profile.name}: {file.original_name}")
            pages = pages_for(conn, file, parser_key(file, profile.parser))
            from hal9000.knowledge.ingest.clean import CleanResult

            cleaned = CleanResult(pages=pages)
            prepared = [
                PreparedDocument(
                    doc, cleaned.document(doc.page_start, doc.page_end), file.collection
                )
                for doc in documents_for(conn, file.sha256)
            ]
            chunked = chunk_documents(prepared, target)
            vectors = embed_chunks(embeddings, target, chunked)
            report.embedded += embeddings.last.embedded
            report.cache_hits += embeddings.last.cache_hits
            with stage(Stage.INDEX), conn:
                report.chunks += write_chunks(conn, target, prepared, chunked, vectors)
            report.files += 1
            report.documents += len(prepared)
        if shadow:
            _swap(conn, profile.name, target)
        else:
            with conn:
                conn.execute("UPDATE profiles SET status = 'ready' WHERE id = ?", (target.id,))
    except BaseException:
        log.exception("building profile %s failed", profile.name)
        _discard(conn, target.id, delete_row=shadow)
        raise
    report.profile = (
        activate_profile(conn, profile.name) if activate else get_profile(conn, profile.name)
    )
    if progress is not None:
        progress(1.0, f"{profile.name}: ready")
    return report


def _swap(conn: sqlite3.Connection, name: str, shadow: IndexProfile) -> None:
    """Replace the ready profile ``name`` with its completed shadow build, atomically."""
    old = get_profile(conn, name)
    assert old is not None and old.id is not None
    from hal9000.knowledge.storage import fts_table, vec_table

    with conn:
        # calibrations and regression history belong to the profile, not to one build of it
        for table in ("abstain_thresholds", "regression_runs"):
            conn.execute(
                f"UPDATE {table} SET profile_id = ? WHERE profile_id = ?", (shadow.id, old.id)
            )
        conn.execute("DELETE FROM chunks WHERE profile_id = ?", (old.id,))
        conn.execute(f"DROP TABLE IF EXISTS {fts_table(old.id)}")
        conn.execute(f"DROP TABLE IF EXISTS {vec_table(old.id)}")
        conn.execute("DELETE FROM profiles WHERE id = ?", (old.id,))
        conn.execute(
            "UPDATE profiles SET name = ?, status = 'ready', active = ? WHERE id = ?",
            (name, int(old.active), shadow.id),
        )
