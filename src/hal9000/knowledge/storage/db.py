"""Opening ``data/knowledge.db``: WAL mode, foreign keys, sqlite-vec and migrations.

:func:`open_db` is the one way in. It is idempotent: opening an up-to-date
database changes nothing. :func:`create_profile_tables` builds a profile's
FTS5 and ``vec0`` tables (plan §12), which KWP3's index fills.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from hal9000.knowledge.storage.schema import MIGRATIONS, SCHEMA_VERSION

__all__ = [
    "FTS_TOKENIZER",
    "SchemaTooNew",
    "connect",
    "create_profile_tables",
    "drop_profile_tables",
    "fts_table",
    "load_sqlite_vec",
    "migrate",
    "open_db",
    "schema_version",
    "vec_table",
]

#: plan §7.1 / §12: queries must be tokenized with the same tokenizer.
FTS_TOKENIZER = "porter unicode61 remove_diacritics 2"
#: ``vec0`` metadata columns, filterable inside the KNN (plan §7.1).
VEC_METADATA = ("doc_type", "collection", "doc_date")


class SchemaTooNew(RuntimeError):
    """The database was migrated by newer code than this; refuse rather than guess."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into ``conn`` (knowledge group dependency)."""
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def connect(path: Path | str, *, vec: bool = True) -> sqlite3.Connection:
    """A connection with WAL, foreign keys and (by default) sqlite-vec loaded."""
    target = str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    if target != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    if vec:
        load_sqlite_vec(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if not exists:
        return 0
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
    return int(row[0])


_VERSION_TABLE = (
    "CREATE TABLE IF NOT EXISTS schema_version ("
    "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
)


def statements(script: str) -> list[str]:
    """Split a migration script into complete SQL statements."""
    out: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                out.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        raise ValueError(f"incomplete SQL at the end of a migration: {buffer.strip()[:60]!r}")
    return out


def _too_new(current: int) -> SchemaTooNew:
    return SchemaTooNew(
        f"knowledge.db is at schema version {current}, but this code knows only up to "
        f"{SCHEMA_VERSION}. Update hal9000 before opening it."
    )


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every migration newer than the database, each atomically. Returns the version.

    Safe when several processes open a new database at once (the web service
    and a CLI command): each migration runs under ``BEGIN IMMEDIATE`` (the
    write lock) and re-reads the version *inside* that transaction, so a
    migration another process applied while this one waited is skipped, not
    applied twice. ``executescript`` is avoided because it commits first.
    """
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise _too_new(current)
    if current == SCHEMA_VERSION:
        return current
    saved = conn.isolation_level
    conn.isolation_level = None  # transactions are managed explicitly below
    try:
        for version, script in MIGRATIONS:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(_VERSION_TABLE)
                current = schema_version(conn)
                if current > SCHEMA_VERSION:
                    raise _too_new(current)
                if version > current:
                    for statement in statements(script):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                        (version, _now()),
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
    finally:
        conn.isolation_level = saved
    return schema_version(conn)


def open_db(path: Path | str, *, vec: bool = True) -> sqlite3.Connection:
    """Connect and migrate. Safe to call on an empty file or an up-to-date one."""
    conn = connect(path, vec=vec)
    migrate(conn)
    return conn


def _profile_id(profile_id: int) -> int:
    # Table names can't be bound parameters, so the ID is checked to be a plain int.
    if isinstance(profile_id, bool) or not isinstance(profile_id, int) or profile_id < 1:
        raise ValueError(f"profile_id must be a positive integer, not {profile_id!r}")
    return profile_id


def fts_table(profile_id: int) -> str:
    """``fts_<id>``: named by integer ID because profile names contain ``.`` and ``-``."""
    return f"fts_{_profile_id(profile_id)}"


def vec_table(profile_id: int) -> str:
    return f"vec_{_profile_id(profile_id)}"


def create_profile_tables(conn: sqlite3.Connection, profile_id: int, dimensions: int) -> None:
    """Create a profile's FTS5 (BM25) and sqlite-vec (cosine KNN) tables if missing.

    Row IDs are chunk IDs in both. One table per profile keeps BM25's term
    statistics from being pooled across chunk sizes (plan §12). The ``vec0``
    metadata columns (``doc_type``, ``collection``, ``doc_date`` as ISO text)
    reject NULL in sqlite-vec 0.1.9, so an unknown value is stored as ``''``.
    """
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError(f"dimensions must be a positive integer, not {dimensions!r}")
    metadata = ", ".join(f"{name} text" for name in VEC_METADATA)
    with conn:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table(profile_id)} "
            f"USING fts5(header, text, tokenize = '{FTS_TOKENIZER}')"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_table(profile_id)} USING vec0("
            f"chunk_id integer primary key, embedding float[{dimensions}] "
            f"distance_metric=cosine, {metadata})"
        )


def drop_profile_tables(conn: sqlite3.Connection, profile_id: int) -> None:
    with conn:
        conn.execute(f"DROP TABLE IF EXISTS {fts_table(profile_id)}")
        conn.execute(f"DROP TABLE IF EXISTS {vec_table(profile_id)}")
