"""``hal9000 kb stats``, ``profile list|activate``, ``backup`` and ``restore`` (plan §15, §17)."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sqlite3
import tarfile
import threading
import time
from pathlib import Path

import pytest

from hal9000.__main__ import main
from hal9000.knowledge.cli import ops
from hal9000.knowledge.index.embedding import EmbeddingService
from hal9000.knowledge.index.fake import FakeEmbedder
from hal9000.knowledge.ingest.jobs import WriterLock
from hal9000.knowledge.retrieve import search_documents, set_threshold
from hal9000.knowledge.storage import open_db
from tests.knowledge.ops.conftest import ACTIVE, EVAL_LINE, PARENT_CHILD

QUESTIONS = (
    "How old is the water heater?",
    "Where is the main water shutoff valve?",
    "What is the total of the plumbing estimate?",
    "garage door opener",
)


def kb(*argv: str) -> int:
    return main(["kb", *argv])


def one_backup(root: Path) -> Path:
    (folder,) = [p for p in root.iterdir() if p.name.startswith("knowledge-")]
    return folder


# ------------------------------------------------------------------ stats
def test_stats_reports_the_library_profiles_spend_and_activity(data_dir, capsys):
    assert kb("stats") == 0
    out = capsys.readouterr().out
    assert f"Active profile: {ACTIVE}" in out
    assert f"* {ACTIVE}  ready:" in out and f"  {PARENT_CHILD}  ready:" in out
    assert "Library: 7 files (0 superseded" in out
    assert "Spend this month: $0.00 of $10.00 (knowledge), $0.00 of $10.00 (evaluation)" in out
    assert "(no ledger yet)" in out
    assert "Last answered question: never" in out
    assert "Last ingestion: never" not in out
    assert str(data_dir / "knowledge.db") in out


def test_stats_counts_match_the_database(data_dir):
    conn = open_db(data_dir / "knowledge.db")
    try:
        body = ops.stats(conn)
        chunks = dict(
            conn.execute(
                "SELECT p.name, COUNT(c.id) FROM profiles p JOIN chunks c ON c.profile_id = p.id "
                "GROUP BY p.name"
            ).fetchall()
        )
    finally:
        conn.close()
    assert body["documents"] >= 7 and body["pages"] >= 12
    by_name = {p["name"]: p for p in body["profiles"]}
    assert {name: p["chunks"] for name, p in by_name.items()} == chunks
    assert by_name[ACTIVE]["documents"] == body["documents"]
    assert by_name[ACTIVE]["pages"] >= body["pages"] - 1  # the photo page has no text


def test_stats_reads_spend_and_the_last_question(data_dir, capsys):
    from hal9000.storage.db import Database

    ledger = Database(data_dir / "hal9000.db")
    ledger.add_ledger("r1", "knowledge", {"model": "m"}, 1.5)
    ledger.add_ledger("r2", "knowledge-eval", {"model": "m"}, 0.25)
    ledger.add_ledger("r3", "solar", {"model": "m"}, 9.0)
    ledger.close()
    conn = open_db(data_dir / "knowledge.db")
    with conn:
        conn.execute(
            "INSERT INTO traces (id, created_at, kind, outcome, body) "
            "VALUES ('t', '2026-09-01T10:00:00+00:00', 'question', 'answered', '{}')"
        )
    conn.close()
    assert kb("stats") == 0
    out = capsys.readouterr().out
    assert "$1.50 of $10.00 (knowledge), $0.25 of $10.00 (evaluation)" in out
    assert "no ledger" not in out
    assert "Last answered question: 2026-09-01T10:00:00+00:00" in out


def test_stats_without_a_library_says_so_and_creates_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    assert kb("stats") == 0
    assert "No knowledge base yet" in capsys.readouterr().out
    assert kb("profile", "list") == 0
    assert not (tmp_path / "knowledge.db").exists()


# ------------------------------------------------------------------ profiles
def test_profile_list_shows_built_active_and_thresholds(data_dir, capsys):
    assert kb("profile", "list") == 0
    out = capsys.readouterr().out
    assert f"* {ACTIVE}\n    active;" in out and f"  {PARENT_CHILD}\n    built;" in out
    assert out.count("abstain threshold: not calibrated") == 2
    conn = open_db(data_dir / "knowledge.db")
    active_id = conn.execute("SELECT id FROM profiles WHERE name = ?", (ACTIVE,)).fetchone()[0]
    set_threshold(conn, active_id, None, 0.42)
    set_threshold(conn, active_id, "minilm", 0.61)
    conn.close()
    assert kb("profile", "list") == 0
    assert "abstain threshold: minilm 0.610, none 0.420" in capsys.readouterr().out


def test_profile_activate_flips_the_database_flag(data_dir, capsys):
    assert kb("profile", "activate", PARENT_CHILD) == 0
    assert f"{PARENT_CHILD} now serves the chat." in capsys.readouterr().out
    conn = open_db(data_dir / "knowledge.db")
    try:
        active = [r[0] for r in conn.execute("SELECT name FROM profiles WHERE active = 1")]
    finally:
        conn.close()
    assert active == [PARENT_CHILD]


def test_profile_activate_refuses_unbuilt_and_unknown_profiles(data_dir, capsys):
    conn = open_db(data_dir / "knowledge.db")
    with conn:
        conn.execute(
            "INSERT INTO profiles (name, parser, chunker, size, overlap, header, embed_model, "
            "dimensions, status, active, created_at) VALUES ('pdfium.fixed-128.hdr.fake-64', "
            "'pdfium', 'fixed', 128, 0, 1, 'fake/embedding-64', 64, 'building', 0, 'now')"
        )
    conn.close()
    assert kb("profile", "list") == 0
    assert "pdfium.fixed-128.hdr.fake-64\n    building;" in capsys.readouterr().out
    assert kb("profile", "activate", "pdfium.fixed-128.hdr.fake-64") == 1
    assert "only a ready profile can be activated" in capsys.readouterr().err
    assert kb("profile", "activate", "no.such-profile") == 1
    assert "no index profile named" in capsys.readouterr().err
    conn = open_db(data_dir / "knowledge.db")
    try:
        active = [r[0] for r in conn.execute("SELECT name FROM profiles WHERE active = 1")]
    finally:
        conn.close()
    assert active == [ACTIVE]


def test_profile_activate_without_a_library_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    assert kb("profile", "activate", ACTIVE) == 1
    assert "No knowledge base yet" in capsys.readouterr().err


def test_profile_list_notes_a_yaml_that_names_another_profile(data_dir, tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "knowledge.yaml").write_text(f"active_profile: {PARENT_CHILD}\n")
    monkeypatch.setenv("HAL_CONFIG_DIR", str(config))
    out = io.StringIO()
    args = argparse.Namespace(kb_command="profile list")
    assert ops.run(args, out=out) == 0
    assert "only sets the profile for the first ingest" in out.getvalue()


# ------------------------------------------------------------------ backup
def test_backup_writes_a_timestamped_folder_with_db_archive_and_manifest(data_dir, capsys):
    assert kb("backup") == 0
    assert "Backed up 7 files" in capsys.readouterr().out
    folder = one_backup(data_dir / "backups")
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [ops.BACKUP_DB, ops.BACKUP_TAR, ops.MANIFEST]
    )
    manifest = json.loads((folder / ops.MANIFEST).read_text())
    assert manifest["counts"]["files"] == 7 and manifest["missing_files"] == []
    assert manifest["schema_version"] >= 5
    with tarfile.open(folder / ops.BACKUP_TAR) as tar:
        names = set(tar.getnames())
    stored = {f"files/{p.name}" for p in (data_dir / "knowledge" / "files").iterdir()}
    assert stored <= names and {"eval/gold.jsonl", "eval/runs/run-1.json"} <= names
    assert not [n for n in names if not n.startswith(("files/", "eval/"))]
    copy = sqlite3.connect(folder / ops.BACKUP_DB)
    try:
        assert copy.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        copy.close()
    assert not list((data_dir / "backups").glob("*.partial"))


def test_backup_to_a_chosen_folder_twice_makes_two_backups(data_dir, tmp_path, capsys):
    target = tmp_path / "elsewhere"
    assert kb("backup", "--to", str(target)) == 0
    assert kb("backup", "--to", str(target)) == 0
    assert len([p for p in target.iterdir() if p.name.startswith("knowledge-")]) == 2


def test_backup_skips_uploads_in_progress_and_reports_missing_originals(data_dir, capsys):
    files = data_dir / "knowledge" / "files"
    (files / ".upload-abc123").write_bytes(b"partial")
    gone = sorted(p for p in files.iterdir() if not p.name.startswith("."))[0]
    gone.unlink()
    assert kb("backup") == 0
    assert "1 stored originals were missing" in capsys.readouterr().err
    folder = one_backup(data_dir / "backups")
    manifest = json.loads((folder / ops.MANIFEST).read_text())
    assert manifest["missing_files"] == [gone.name]
    with tarfile.open(folder / ops.BACKUP_TAR) as tar:
        assert not [n for n in tar.getnames() if ".upload-" in n]


def test_backup_without_a_library_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    assert kb("backup") == 1
    assert "no knowledge base to back up" in capsys.readouterr().err
    assert not (tmp_path / "backups").exists() or not any((tmp_path / "backups").iterdir())


def test_a_failed_backup_leaves_no_partial_folder(data_dir, tmp_path, monkeypatch):
    def broken(*args, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(ops, "_library_members", broken)
    with pytest.raises(OSError, match="disk full"):
        ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    assert list((tmp_path / "b").iterdir()) == []


def test_backup_is_consistent_while_a_writer_is_active(data_dir, tmp_path):
    """A writer commits pairs of rows in one transaction; every backup holds whole pairs."""
    stop = threading.Event()
    written = [0]

    def writer() -> None:
        conn = open_db(data_dir / "knowledge.db")
        try:
            while not stop.is_set():
                n = written[0]
                with conn:
                    for half in ("a", "b"):
                        conn.execute(
                            "INSERT INTO jobs (kind, status, detail, created_at, updated_at) "
                            "VALUES ('regression', 'queued', ?, 'now', 'now')",
                            (f"pair-{n}-{half}",),
                        )
                    time.sleep(0.001)  # hold the write transaction open a little
                written[0] = n + 1
        finally:
            conn.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        while written[0] < 5:
            time.sleep(0.005)
        reports = []
        for _ in range(4):
            reports.append(ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path))
    finally:
        stop.set()
        thread.join()
    counts = []
    for report in reports:
        copy = sqlite3.connect(report.folder / ops.BACKUP_DB)
        try:
            assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            a, b = (
                copy.execute(
                    "SELECT COUNT(*) FROM jobs WHERE detail LIKE ?", (f"pair-%-{half}",)
                ).fetchone()[0]
                for half in ("a", "b")
            )
        finally:
            copy.close()
        assert a == b, "a backup caught half a transaction"
        counts.append(a)
    assert counts == sorted(counts) and counts[-1] > counts[0] > 0, counts
    assert written[0] > counts[0], "the writer kept writing during the backups"


# ------------------------------------------------------------------ restore
def _rows(db: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return [tuple(row) for row in conn.execute(sql)]
    finally:
        conn.close()


def _searches(db: Path) -> list[list[tuple]]:
    conn = open_db(db)
    try:
        results = []
        for question in QUESTIONS:
            for filters in (None, {"doc_type": "estimate"}):
                from hal9000.knowledge.contracts import SearchFilters

                result = search_documents(
                    conn,
                    EmbeddingService(conn, embed_fn=FakeEmbedder()),
                    question,
                    k=8,
                    filters=SearchFilters(**filters) if filters else None,
                )
                results.append(
                    [(c.chunk_id, c.text, round(c.score, 9), c.title) for c in result.selected]
                )
        return results
    finally:
        conn.close()


def test_restore_round_trips_the_fixture_library(data_dir, tmp_path, capsys):
    assert kb("backup", "--to", str(tmp_path / "b")) == 0
    folder = one_backup(tmp_path / "b")
    target = tmp_path / "restored"
    assert kb("restore", str(folder), "--into", str(target)) == 0
    assert "Restored 7 files" in capsys.readouterr().out

    src, dst = data_dir / "knowledge.db", target / "knowledge.db"
    for sql in (
        "SELECT * FROM files ORDER BY sha256",
        "SELECT * FROM documents ORDER BY id",
        "SELECT * FROM pages ORDER BY file_sha, parser, page",
        "SELECT * FROM profiles ORDER BY id",
        "SELECT * FROM chunks ORDER BY id",
        "SELECT model, dimensions, text_sha, vector FROM embed_cache ORDER BY text_sha",
    ):
        assert _rows(src, sql) == _rows(dst, sql), sql
    assert _rows(dst, "SELECT COUNT(*) FROM chunks")[0][0] > 0
    searches = _searches(dst)
    assert any(searches) and searches == _searches(src)
    for name in ("files", "eval"):
        before = sorted(
            p.relative_to(data_dir / "knowledge")
            for p in (data_dir / "knowledge" / name).rglob("*")
        )
        after = sorted(
            p.relative_to(target / "knowledge") for p in (target / "knowledge" / name).rglob("*")
        )
        assert before == after
    for path in (data_dir / "knowledge" / "files").iterdir():
        assert (target / "knowledge" / "files" / path.name).read_bytes() == path.read_bytes()
    assert (target / "knowledge" / "eval" / "gold.jsonl").read_text() == EVAL_LINE


def test_restore_uses_kb_data_dir_by_default_and_ignores_other_data(
    data_dir, tmp_path, monkeypatch
):
    report = ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "hal9000.db").write_bytes(b"the solar agent's")  # shared data dir: left alone
    (fresh / "knowledge" / "files").mkdir(parents=True)  # empty folders are fine
    monkeypatch.setenv("KB_DATA_DIR", str(fresh))
    assert kb("restore", str(report.folder)) == 0
    assert (fresh / "hal9000.db").read_bytes() == b"the solar agent's"
    assert _rows(fresh / "knowledge.db", "SELECT COUNT(*) FROM files")[0][0] == 7


def test_restore_refuses_a_non_empty_data_dir_without_force(data_dir, tmp_path, capsys):
    report = ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    conn = open_db(data_dir / "knowledge.db")
    with conn:
        conn.execute("DELETE FROM chunks")
    conn.close()
    assert kb("restore", str(report.folder), "--into", str(data_dir)) == 1
    assert "already holds a knowledge library" in capsys.readouterr().err
    assert _rows(data_dir / "knowledge.db", "SELECT COUNT(*) FROM chunks")[0][0] == 0

    eval_only = tmp_path / "eval-only"
    (eval_only / "knowledge" / "eval").mkdir(parents=True)
    (eval_only / "knowledge" / "eval" / "gold.jsonl").write_text(EVAL_LINE)
    assert not ops.library_is_empty(eval_only)
    assert kb("restore", str(report.folder), "--into", str(eval_only)) == 1


def test_restore_with_force_moves_the_old_library_aside(data_dir, tmp_path, capsys):
    report = ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    conn = open_db(data_dir / "knowledge.db")
    with conn:
        conn.execute("DELETE FROM chunks")
    conn.close()
    (data_dir / "knowledge" / "eval" / "later.jsonl").write_text("{}\n")
    assert kb("restore", str(report.folder), "--into", str(data_dir), "--force") == 0
    out = capsys.readouterr().out
    (aside,) = list(data_dir.glob("knowledge-replaced-*"))
    assert str(aside) in out
    assert _rows(aside / "knowledge.db", "SELECT COUNT(*) FROM chunks")[0][0] == 0
    assert (aside / "eval" / "later.jsonl").exists()
    assert _rows(data_dir / "knowledge.db", "SELECT COUNT(*) FROM chunks")[0][0] > 0
    assert not (data_dir / "knowledge" / "eval" / "later.jsonl").exists()


def test_restore_refuses_while_the_service_holds_the_writer_lock(data_dir, tmp_path, capsys):
    report = ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    target = tmp_path / "target"
    with WriterLock(target / "knowledge" / "worker.lock") as lock:
        assert lock.acquire()
        assert kb("restore", str(report.folder), "--into", str(target), "--force") == 1
    assert "stop it before restoring" in capsys.readouterr().err
    assert not (target / "knowledge.db").exists()


def test_restore_refuses_a_damaged_or_foreign_backup(data_dir, tmp_path, capsys):
    report = ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    damaged = tmp_path / "damaged"
    shutil.copytree(report.folder, damaged)
    with open(damaged / ops.BACKUP_DB, "r+b") as handle:
        handle.seek(4096)
        handle.write(b"\x00" * 64)
    assert kb("restore", str(damaged), "--into", str(tmp_path / "t1")) == 1
    assert "doesn't match its checksum" in capsys.readouterr().err

    incomplete = tmp_path / "incomplete"
    shutil.copytree(report.folder, incomplete)
    (incomplete / ops.BACKUP_TAR).unlink()
    assert kb("restore", str(incomplete), "--into", str(tmp_path / "t2")) == 1
    assert "incomplete" in capsys.readouterr().err

    assert kb("restore", str(tmp_path), "--into", str(tmp_path / "t3")) == 1
    assert "is not a `kb backup` folder" in capsys.readouterr().err
    for name in ("t1", "t2", "t3"):
        assert not (tmp_path / name / "knowledge.db").exists()


def test_restore_refuses_an_archive_that_escapes_its_folders(data_dir, tmp_path, capsys):
    report = ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    evil = tmp_path / "evil"
    shutil.copytree(report.folder, evil)
    payload = tmp_path / "payload"
    payload.write_text("gotcha")
    with tarfile.open(evil / ops.BACKUP_TAR, "w:gz") as tar:
        tar.add(payload, arcname="files/../../escaped")
    manifest = json.loads((evil / ops.MANIFEST).read_text())
    manifest["sha256"][ops.BACKUP_TAR] = ops._sha256(evil / ops.BACKUP_TAR)
    (evil / ops.MANIFEST).write_text(json.dumps(manifest))
    target = tmp_path / "target"
    assert kb("restore", str(evil), "--into", str(target)) == 1
    assert "unexpected entry" in capsys.readouterr().err
    assert not (tmp_path / "escaped").exists() and not (target / "escaped").exists()
    assert not (target / "knowledge.db").exists()


def test_stats_and_profiles_on_an_empty_library(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path))
    open_db(tmp_path / "knowledge.db").close()
    assert kb("stats") == 0
    out = capsys.readouterr().out
    assert "Library: 0 files" in out and "Active profile: (none)" in out
    assert "(none built yet)" in out and "Last ingestion: never" in out
    assert kb("profile", "list") == 0
    assert "No index profiles yet" in capsys.readouterr().out


def test_restore_refuses_a_manifest_missing_a_part(data_dir, tmp_path, capsys):
    report = ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b")
    manifest = json.loads((report.folder / ops.MANIFEST).read_text())
    del manifest["sha256"][ops.BACKUP_TAR]
    (report.folder / ops.MANIFEST).write_text(json.dumps(manifest))
    assert kb("restore", str(report.folder), "--into", str(tmp_path / "t")) == 1
    assert "must list exactly" in capsys.readouterr().err


# ------------------------------------------------------------------ refused before touching
def snapshot_tree(root: Path) -> dict[str, object]:
    """Every path under ``root`` (and its parent's entries) with its bytes: to prove "untouched"."""
    tree: dict[str, object] = {"<parent>": sorted(p.name for p in root.parent.iterdir())}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            tree[key] = ("link", str(path.readlink()))
        elif path.is_dir():
            tree[key] = "dir"
        else:
            tree[key] = (path.read_bytes(), path.stat().st_mtime_ns)
    return tree


def reseal(folder: Path) -> None:
    """Recompute the manifest's checksums after a test tampers with a part."""
    manifest = json.loads((folder / ops.MANIFEST).read_text())
    for name in manifest["sha256"]:
        manifest["sha256"][name] = ops._sha256(folder / name)
    (folder / ops.MANIFEST).write_text(json.dumps(manifest))


@pytest.fixture
def backup_copy(data_dir: Path, tmp_path: Path) -> Path:
    """A fresh backup of the fixture library, to tamper with."""
    return ops.backup(data_dir / "knowledge.db", data_dir / "knowledge", tmp_path / "b").folder


def _refused_untouched(backup: Path, data_dir: Path, capsys, message: str) -> None:
    """A forced restore over the populated library is refused and changes nothing."""
    before = snapshot_tree(data_dir)
    assert kb("restore", str(backup), "--into", str(data_dir), "--force") == 1
    assert message in capsys.readouterr().err
    assert snapshot_tree(data_dir) == before
    assert not list(data_dir.parent.glob(".kb-restore-*"))


def _tar_with(folder: Path, bad: tarfile.TarInfo, payload: bytes | None = None) -> None:
    ok = tarfile.TarInfo("files/" + "a" * 64)
    ok.size = 3
    with tarfile.open(folder / ops.BACKUP_TAR, "w:gz") as tar:
        tar.addfile(ok, io.BytesIO(b"abc"))
        tar.addfile(bad, io.BytesIO(payload) if payload is not None else None)
    reseal(folder)


def _entry(name: str, kind: bytes, linkname: str = "", size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.size = kind, linkname, size
    if kind in (tarfile.CHRTYPE, tarfile.BLKTYPE):
        info.devmajor, info.devminor = 1, 3
    return info


@pytest.mark.parametrize(
    ("entry", "payload"),
    [
        (_entry("files/inside-link", tarfile.SYMTYPE, "a" * 64), None),
        (_entry("files/outside-link", tarfile.SYMTYPE, "/etc/passwd"), None),
        (_entry("eval/up-link", tarfile.SYMTYPE, "../../../escaped"), None),
        (_entry("files/hard", tarfile.LNKTYPE, "files/" + "a" * 64), None),
        (_entry("files/hard-out", tarfile.LNKTYPE, "/etc/passwd"), None),
        (_entry("/files/absolute", tarfile.REGTYPE, size=4), b"evil"),
        (_entry("/etc/cron.d/absolute", tarfile.REGTYPE, size=4), b"evil"),
        (_entry("files/device", tarfile.CHRTYPE), None),
        (_entry("files/block", tarfile.BLKTYPE), None),
        (_entry("files/fifo", tarfile.FIFOTYPE), None),
        (_entry("other/file", tarfile.REGTYPE, size=4), b"evil"),
        (_entry("knowledge.db", tarfile.REGTYPE, size=4), b"evil"),
        (_entry("files", tarfile.DIRTYPE), None),
        (_entry("files/../../escaped", tarfile.REGTYPE, size=4), b"evil"),
    ],
    ids=lambda v: v.name if isinstance(v, tarfile.TarInfo) else "",
)
def test_restore_refuses_unsafe_archive_entries_before_touching_anything(
    backup_copy, data_dir, capsys, entry, payload
):
    _tar_with(backup_copy, entry, payload)
    _refused_untouched(backup_copy, data_dir, capsys, "unexpected entry")
    assert not (data_dir.parent / "escaped").exists()


def test_the_unsafe_entry_check_accepts_the_real_archive(backup_copy):
    with tarfile.open(backup_copy / ops.BACKUP_TAR) as tar:
        members = ops._safe_members(tar)
    assert members and all(m.isreg() for m in members)


def test_restore_checks_the_counts_before_touching_anything(backup_copy, data_dir, capsys):
    manifest = json.loads((backup_copy / ops.MANIFEST).read_text())
    manifest["counts"]["chunks"] += 1
    (backup_copy / ops.MANIFEST).write_text(json.dumps(manifest))
    _refused_untouched(backup_copy, data_dir, capsys, "but its manifest says")


def test_restore_checks_integrity_before_touching_anything(backup_copy, data_dir, capsys):
    db = backup_copy / ops.BACKUP_DB
    data = db.read_bytes()
    db.write_bytes(data[: len(data) // 2])
    reseal(backup_copy)
    _refused_untouched(backup_copy, data_dir, capsys, "failed its integrity check")


def test_restore_refuses_a_backup_from_newer_code(backup_copy, data_dir, capsys):
    conn = sqlite3.connect(backup_copy / ops.BACKUP_DB)
    with conn:
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (999, 'x')")
    conn.close()
    reseal(backup_copy)
    _refused_untouched(backup_copy, data_dir, capsys, "made by newer code")


def test_a_refused_non_empty_target_is_untouched(backup_copy, data_dir, capsys):
    before = snapshot_tree(data_dir)
    assert kb("restore", str(backup_copy), "--into", str(data_dir)) == 1
    assert "already holds a knowledge library" in capsys.readouterr().err
    assert snapshot_tree(data_dir) == before


@pytest.mark.parametrize("parts", [{ops.BACKUP_DB}, {ops.BACKUP_DB, ops.BACKUP_TAR, "run.sh"}])
def test_the_manifest_must_name_exactly_the_parts_before_anything_is_hashed(
    backup_copy, data_dir, capsys, monkeypatch, parts
):
    manifest = json.loads((backup_copy / ops.MANIFEST).read_text())
    manifest["sha256"] = {name: "0" * 64 for name in parts}
    (backup_copy / ops.MANIFEST).write_text(json.dumps(manifest))

    def no_hashing(path):
        raise AssertionError(f"hashed {path} before checking the manifest")

    monkeypatch.setattr(ops, "_sha256", no_hashing)
    _refused_untouched(backup_copy, data_dir, capsys, "must list exactly")


def test_an_unreadable_manifest_is_refused(backup_copy, data_dir, capsys):
    (backup_copy / ops.MANIFEST).write_text("{not json")
    _refused_untouched(backup_copy, data_dir, capsys, "unreadable")


def test_a_failure_after_moving_the_library_aside_says_where_it_went(
    backup_copy, data_dir, capsys, monkeypatch
):
    def disk_full(*args, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(ops.shutil, "move", disk_full)
    assert kb("restore", str(backup_copy), "--into", str(data_dir), "--force") == 1
    err = capsys.readouterr().err
    (aside,) = list(data_dir.glob("knowledge-replaced-*"))
    assert "disk full" in err and f"the previous library was moved to {aside}" in err
    assert (aside / "knowledge.db").is_file() and (aside / "files").is_dir()
