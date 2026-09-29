"""``knowledge.db``: migrations, WAL, per-profile tables and the sqlite-vec spike (plan §12, §21)."""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hal9000.knowledge import storage
from hal9000.knowledge.storage import db as kdb

REPO = Path(__file__).resolve().parents[2]

#: Plan §12, written out independently of storage.TABLES.
PLAN_TABLES = {
    "files",
    "documents",
    "pages",
    "profiles",
    "abstain_thresholds",
    "chunks",
    "embed_cache",
    "jobs",
    "conversations",
    "messages",
    "feedback",
    "traces",
    "gold_items",
    "judge_labels",
    "eval_runs",
    "regression_runs",
}


def _tables(conn) -> set[str]:
    return {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }


def _schema(conn) -> list[tuple]:
    return conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()


def test_migrations_create_every_plan_table_on_an_empty_file(tmp_path):
    path = tmp_path / "knowledge.db"
    assert not path.exists()
    conn = storage.open_db(path)
    assert PLAN_TABLES <= _tables(conn)
    assert set(storage.TABLES) == PLAN_TABLES
    assert storage.schema_version(conn) == storage.SCHEMA_VERSION == len(storage.MIGRATIONS)


def test_wal_mode_is_on(tmp_path):
    path = tmp_path / "knowledge.db"
    conn = storage.open_db(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.execute("INSERT INTO jobs (kind, created_at, updated_at) VALUES ('ingest', 'now', 'now')")
    conn.commit()
    assert (tmp_path / "knowledge.db-wal").exists()
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "knowledge.db"
    first = storage.open_db(path)
    before = _schema(first)
    first.close()
    second = storage.open_db(path)
    assert storage.migrate(second) == storage.SCHEMA_VERSION
    assert _schema(second) == before
    assert second.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(
        storage.MIGRATIONS
    )


def test_a_newer_database_is_refused(tmp_path):
    conn = storage.open_db(tmp_path / "k.db")
    conn.execute("INSERT INTO schema_version VALUES (99, 'later')")
    conn.commit()
    with pytest.raises(storage.SchemaTooNew, match="version 99"):
        storage.migrate(conn)


def test_a_failing_migration_leaves_nothing_behind(tmp_path, monkeypatch):
    conn = storage.open_db(tmp_path / "k.db")
    broken = "CREATE TABLE half_done (x INTEGER);\nTHIS IS NOT SQL;"
    nxt = storage.SCHEMA_VERSION + 1
    monkeypatch.setattr(kdb, "MIGRATIONS", (*storage.MIGRATIONS, (nxt, broken)))
    monkeypatch.setattr(kdb, "SCHEMA_VERSION", nxt)
    with pytest.raises(sqlite3.Error):
        storage.migrate(conn)
    assert storage.schema_version(conn) == nxt - 1
    assert "half_done" not in _tables(conn)


def test_a_new_migration_applies_forward(tmp_path, monkeypatch):
    conn = storage.open_db(tmp_path / "k.db")
    nxt = storage.SCHEMA_VERSION + 1
    monkeypatch.setattr(
        kdb, "MIGRATIONS", (*storage.MIGRATIONS, (nxt, "ALTER TABLE jobs ADD COLUMN note TEXT;"))
    )
    monkeypatch.setattr(kdb, "SCHEMA_VERSION", nxt)
    assert storage.migrate(conn) == nxt
    assert "note" in {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}


def test_a_migration_applied_by_another_process_while_waiting_is_skipped(tmp_path):
    """Review (d): two connections open a new DB; the second waits, re-reads, skips."""
    import threading
    import time

    path = tmp_path / "k.db"
    other = storage.connect(path)  # the process that wins the race
    mine = storage.connect(path)
    other.isolation_level = None
    other.execute("BEGIN IMMEDIATE")
    other.execute(kdb._VERSION_TABLE)
    for version, script in storage.MIGRATIONS:
        for statement in kdb.statements(script):
            other.execute(statement)
        other.execute("INSERT INTO schema_version VALUES (?, 'other process')", (version,))

    outcome: dict = {}

    def run() -> None:
        try:
            outcome["version"] = storage.migrate(mine)
        except BaseException as error:  # reported below
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.3)
    assert thread.is_alive(), "migrate() should be waiting for the write lock"
    other.execute("COMMIT")
    thread.join(timeout=30)

    assert "error" not in outcome, outcome.get("error")
    assert outcome["version"] == storage.SCHEMA_VERSION
    rows = mine.execute("SELECT version, applied_at FROM schema_version").fetchall()
    assert [tuple(r) for r in rows] == [(v, "other process") for v, _ in storage.MIGRATIONS]
    assert PLAN_TABLES <= _tables(mine)


def test_the_version_is_reread_inside_the_write_lock(tmp_path, monkeypatch):
    """Another process migrates between our first version read and our write: skip, don't redo."""
    path = tmp_path / "k.db"
    other, mine = storage.connect(path), storage.connect(path)
    real = kdb.schema_version
    raced: list[int] = []

    def racing(conn):
        value = real(conn)
        if conn is mine and not raced:
            raced.append(value)
            storage.migrate(other)  # lands right after our (now stale) read
        return value

    monkeypatch.setattr(kdb, "schema_version", racing)
    assert storage.migrate(mine) == storage.SCHEMA_VERSION
    assert raced == [0]
    count = mine.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert count == len(storage.MIGRATIONS)


def test_many_connections_migrating_a_new_file_at_once(tmp_path):
    import threading

    for attempt in range(3):
        path = tmp_path / f"race{attempt}.db"
        conns = [storage.connect(path) for _ in range(4)]
        barrier = threading.Barrier(len(conns))
        errors: list[BaseException] = []

        def run(conn, barrier=barrier, errors=errors) -> None:
            barrier.wait()
            try:
                storage.migrate(conn)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=run, args=(c,)) for c in conns]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert errors == []
        count = conns[0].execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == len(storage.MIGRATIONS)
        assert PLAN_TABLES <= _tables(conns[0])


def test_statements_splits_complete_sql():
    assert kdb.statements("CREATE TABLE a (x TEXT DEFAULT ';');\n\nCREATE INDEX i ON a(x);\n") == [
        "CREATE TABLE a (x TEXT DEFAULT ';');",
        "CREATE INDEX i ON a(x);",
    ]
    with pytest.raises(ValueError, match="incomplete SQL"):
        kdb.statements("CREATE TABLE b (x")


def _profile(conn, name="p", active=0):
    cur = conn.execute(
        "INSERT INTO profiles (name, parser, chunker, size, header, embed_model, dimensions, "
        "active, created_at) VALUES (?, 'pdfium', 'fixed', 512, 1, 'm', 4, ?, 'now')",
        (name, active),
    )
    conn.commit()
    return cur.lastrowid


def test_only_one_profile_can_be_active(kb_db):
    _profile(kb_db, "a", active=1)
    _profile(kb_db, "b", active=0)
    with pytest.raises(sqlite3.IntegrityError):
        _profile(kb_db, "c", active=1)


def test_foreign_keys_cascade(kb_db):
    sha = "a" * 64
    kb_db.execute(
        "INSERT INTO files VALUES (?, 'f.pdf', 'f.pdf', '', 'application/pdf', 1, NULL, 'now')",
        (sha,),
    )
    kb_db.execute(
        "INSERT INTO documents (file_sha, page_start, page_end, title) VALUES (?, 1, 2, 't')",
        (sha,),
    )
    kb_db.execute("DELETE FROM files WHERE sha256 = ?", (sha,))
    assert kb_db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    with pytest.raises(sqlite3.IntegrityError):
        kb_db.execute(
            "INSERT INTO documents (file_sha, page_start, page_end, title) "
            "VALUES ('nope', 1, 1, 't')"
        )


def test_profile_tables_fts_and_vec(kb_db):
    pid = _profile(kb_db)
    storage.create_profile_tables(kb_db, pid, 4)
    storage.create_profile_tables(kb_db, pid, 4)  # idempotent
    fts, vec = storage.fts_table(pid), storage.vec_table(pid)
    assert {fts, vec} <= _tables(kb_db)

    kb_db.execute(
        f"INSERT INTO {fts} (rowid, header, text) VALUES (7, 'Plumbing', 'The valve is leaking')"
    )
    hits = kb_db.execute(f"SELECT rowid FROM {fts} WHERE {fts} MATCH '\"leaks\"'").fetchall()
    assert [r[0] for r in hits] == [7]  # porter stemming: leaks ~ leaking

    rows = [
        (1, "[1,0,0,0]", "invoice", "Plumbing", "2023-06-27"),
        (2, "[0.9,0.1,0,0]", "estimate", "Plumbing", "2023-06-27"),
        (3, "[0,1,0,0]", "invoice", "HVAC", "2023-07-07"),
    ]
    kb_db.executemany(
        f"INSERT INTO {vec} (chunk_id, embedding, doc_type, collection, doc_date) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    near = kb_db.execute(
        f"SELECT chunk_id FROM {vec} WHERE embedding MATCH ? AND k = 3 ORDER BY distance",
        ("[1,0,0,0]",),
    ).fetchall()
    assert [r[0] for r in near] == [1, 2, 3]
    filtered = kb_db.execute(
        f"SELECT chunk_id FROM {vec} WHERE embedding MATCH ? AND k = 3 AND doc_type = ? "
        "ORDER BY distance",
        ("[1,0,0,0]", "invoice"),
    ).fetchall()
    assert [r[0] for r in filtered] == [1, 3]

    storage.drop_profile_tables(kb_db, pid)
    assert not ({fts, vec} & _tables(kb_db))


def test_vec_metadata_columns_reject_null(kb_db):
    """sqlite-vec 0.1.9 metadata columns take no NULL: an unknown doc_date is stored as ''."""
    pid = _profile(kb_db)
    storage.create_profile_tables(kb_db, pid, 2)
    with pytest.raises(sqlite3.OperationalError, match="Expected text"):
        kb_db.execute(
            f"INSERT INTO {storage.vec_table(pid)} VALUES (1, '[1,0]', 'other', '', NULL)"
        )


@pytest.mark.parametrize("bad", [0, -1, "1; DROP TABLE files", True, 1.5])
def test_profile_table_names_reject_anything_but_positive_ints(bad):
    with pytest.raises(ValueError):
        storage.fts_table(bad)


def test_profile_tables_reject_bad_dimensions(kb_db):
    with pytest.raises(ValueError):
        storage.create_profile_tables(kb_db, 1, 0)


def test_connect_without_vec(tmp_path):
    conn = storage.connect(tmp_path / "plain.db", vec=False)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("SELECT vec_version()")


# ------------------------------------------------------------- the spike
def test_sqlite_vec_spike_1000_vectors_match_numpy_brute_force(kb_db):
    """Plan §21 / handoff KWP0: 1,000 random 1,536-d vectors, KNN order == brute force."""
    pid = _profile(kb_db)
    storage.create_profile_tables(kb_db, pid, 1536)
    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((1000, 1536)).astype(np.float32)
    kb_db.executemany(
        f"INSERT INTO {storage.vec_table(pid)} (chunk_id, embedding, doc_type, collection, "
        "doc_date) VALUES (?, ?, 'other', '', '')",
        [(i + 1, vectors[i].tobytes()) for i in range(1000)],
    )
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    for _ in range(5):
        query = rng.standard_normal(1536).astype(np.float32)
        distance = 1 - unit @ (query / np.linalg.norm(query))
        expected = [int(i) + 1 for i in np.argsort(distance)[:20]]
        rows = kb_db.execute(
            f"SELECT chunk_id, distance FROM {storage.vec_table(pid)} "
            "WHERE embedding MATCH ? AND k = 20 ORDER BY distance",
            (query.tobytes(),),
        ).fetchall()
        assert [r[0] for r in rows] == expected
        assert np.allclose([r[1] for r in rows], np.sort(distance)[:20], atol=1e-5)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "kb_sqlite_vec_check", REPO / "scripts" / "kb_sqlite_vec_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gate_script_runs_the_same_check():
    result = _load_script().run_check()
    assert result["ok"] and result["vectors"] == 1000 and result["dimensions"] == 1536
    assert result["sqlite_vec"] == "v0.1.9"


def test_the_gate_script_catches_a_wrong_order(monkeypatch):
    """Guard against a check that can't fail: corrupt numpy's ranking and it must object."""
    module = _load_script()
    real = np.argsort
    monkeypatch.setattr(np, "argsort", lambda a, **kw: real(a, **kw)[::-1])
    with pytest.raises(AssertionError, match="KNN order differs"):
        module.run_check(n=50, dim=8, k=5, queries=1)


def test_the_gate_script_as_a_process():
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "kb_sqlite_vec_check.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "PASS" in done.stdout


def test_the_gate_script_reports_failure(monkeypatch, capsys):
    module = _load_script()
    monkeypatch.setattr(module, "run_check", lambda: (_ for _ in ()).throw(AssertionError("x")))
    assert module.main() == 1
    monkeypatch.setattr(
        module, "run_check", lambda: (_ for _ in ()).throw(AttributeError("no ext"))
    )
    assert module.main() == 1
    assert "cannot load extensions" in capsys.readouterr().err
