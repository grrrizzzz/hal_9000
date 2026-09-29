"""Knowledge-base storage: ``data/knowledge.db`` (SQLite + FTS5 + sqlite-vec, plan §12)."""

from hal9000.knowledge.storage.db import (
    FTS_TOKENIZER,
    SchemaTooNew,
    connect,
    create_profile_tables,
    drop_profile_tables,
    fts_table,
    load_sqlite_vec,
    migrate,
    open_db,
    schema_version,
    vec_table,
)
from hal9000.knowledge.storage.schema import MIGRATIONS, SCHEMA_VERSION, TABLES

__all__ = [
    "FTS_TOKENIZER",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "TABLES",
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
