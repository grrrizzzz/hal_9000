"""``hal9000 kb profile list|activate``, ``backup``, ``restore`` and ``stats`` (plan §15, §17).

**Profiles.** The database's ``profiles.active`` flag decides which profile
serves the chat (decision D3); ``activate`` flips it atomically and refuses a
profile that isn't ``ready`` (fully built). ``list`` shows every profile's
status, chunk count, whether it is active, and its calibrated abstain
thresholds (per reranker).

**Backup** (``kb backup [--to DIR]``, default ``data/backups``) writes a
timestamped folder ``knowledge-<UTC time>/`` holding:

- ``knowledge.db``: an online copy made with SQLite's backup API in one step,
  so it is a consistent snapshot even while the service is writing (WAL
  readers don't block writers, and the copy never sees half a transaction);
  it is switched to a single-file journal and checked with
  ``PRAGMA integrity_check``;
- ``library.tar.gz``: ``data/knowledge/files/`` (the originals, by hash) and
  ``data/knowledge/eval/`` (the question set and runs), taken *after* the
  database, so every file the snapshot references is already on disk (a file
  deleted in between is reported as missing);
- ``manifest.json``: schema version, counts and SHA-256 checksums (no
  document text, file names or questions).

**Restore** (``kb restore FROM [--into DIR] [--force]``) checks the checksums,
then restores into the data directory (default ``KB_DATA_DIR``). It refuses a
data directory that already holds a library (``knowledge.db`` or anything in
``knowledge/files`` or ``knowledge/eval``) unless ``--force``, which moves the
old library aside to ``knowledge-replaced-<time>/`` rather than deleting it. It
never touches anything else in the directory (``hal9000.db`` belongs to the
solar agent), and refuses while the service's writer lock is held. The restored
database is migrated and integrity-checked before the command reports success.

**Stats**: documents, pages and chunks per profile, the active profile, the
database size, the month's knowledge spend against its cap, the last ingestion
and the last answered question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "BACKUP_DB",
    "BACKUP_TAR",
    "MANIFEST",
    "BackupError",
    "backup",
    "library_is_empty",
    "restore",
    "run",
    "stats",
]

EXIT_OK = 0
EXIT_ERROR = 1

BACKUP_DB = "knowledge.db"
BACKUP_TAR = "library.tar.gz"
MANIFEST = "manifest.json"
#: the folders under ``data/knowledge/`` a backup carries
LIBRARY_DIRS = ("files", "eval")


class BackupError(RuntimeError):
    """A backup or restore can't go ahead; the message says why."""


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integrity(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    if result != "ok":
        raise BackupError(f"{path} failed its integrity check: {result}")


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("files", "documents", "pages", "profiles", "chunks")
    }


# ------------------------------------------------------------------ backup
@dataclass
class BackupReport:
    folder: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def missing_files(self) -> list[str]:
        return list(self.manifest.get("missing_files", []))


def _library_members(knowledge_dir: Path) -> list[tuple[Path, str]]:
    """``(path, name in the tarball)`` for every regular file under files/ and eval/."""
    members = []
    for name in LIBRARY_DIRS:
        root = knowledge_dir / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            if name == "files" and path.name.startswith(".upload-"):
                continue  # an upload still being written; its database row doesn't exist yet
            members.append((path, path.relative_to(knowledge_dir).as_posix()))
    return members


def backup(db_path: Path, knowledge_dir: Path, to: Path) -> BackupReport:
    """Back up ``db_path`` and ``knowledge_dir/{files,eval}`` into a new folder under ``to``."""
    from hal9000.knowledge.storage import schema_version

    if not Path(db_path).is_file():
        raise BackupError(f"there is no knowledge base to back up at {db_path}")
    to = Path(to)
    folder = to / f"knowledge-{_stamp()}"
    suffix = 1
    while folder.exists():
        suffix += 1
        folder = to / f"knowledge-{_stamp()}-{suffix}"
    partial = folder.with_name(folder.name + ".partial")
    partial.mkdir(parents=True)
    try:
        # 1. the database, online, in one step: a consistent snapshot
        target = partial / BACKUP_DB
        source = sqlite3.connect(db_path, timeout=30)  # only read: a WAL reader blocks nobody
        dest = sqlite3.connect(target)
        try:
            source.execute("PRAGMA busy_timeout = 30000")
            source.backup(dest)
            dest.execute("PRAGMA journal_mode = DELETE")
            referenced = {row[0] for row in dest.execute("SELECT sha256 FROM files")}
            counts = _counts(dest)
            version = schema_version(dest)
        finally:
            dest.close()
            source.close()
        _integrity(target)

        # 2. the originals and the evaluation data, after the database
        members = _library_members(Path(knowledge_dir))
        with tarfile.open(partial / BACKUP_TAR, "w:gz") as tar:
            for path, name in members:
                tar.add(path, arcname=name, recursive=False)
        stored = {name.split("/", 1)[1] for _, name in members if name.startswith("files/")}
        manifest = {
            "format": 1,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "schema_version": version,
            "counts": counts,
            "library_files": len(members),
            "missing_files": sorted(referenced - stored),
            "sha256": {BACKUP_DB: _sha256(target), BACKUP_TAR: _sha256(partial / BACKUP_TAR)},
        }
        (partial / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        partial.rename(folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return BackupReport(folder=folder, manifest=manifest)


# ------------------------------------------------------------------ restore
def _library_paths(data_dir: Path) -> list[Path]:
    """What a restore replaces in ``data_dir`` (nothing else is touched)."""
    db = data_dir / "knowledge.db"
    knowledge = data_dir / "knowledge"
    return [
        db,
        db.with_name("knowledge.db-wal"),
        db.with_name("knowledge.db-shm"),
        *(knowledge / name for name in LIBRARY_DIRS),
    ]


def library_is_empty(data_dir: Path) -> bool:
    """No ``knowledge.db`` and nothing in ``knowledge/files`` or ``knowledge/eval``."""
    for path in _library_paths(Path(data_dir)):
        if path.is_dir():
            if any(path.iterdir()):
                return False
        elif path.exists():
            return False
    return True


def _check(source: Path) -> dict[str, Any]:
    manifest_path = source / MANIFEST
    if not manifest_path.is_file():
        raise BackupError(f"{source} is not a `kb backup` folder (no {MANIFEST})")
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.get("sha256", {}).items():
        path = source / name
        if not path.is_file():
            raise BackupError(f"the backup is incomplete: {name} is missing")
        if _sha256(path) != expected:
            raise BackupError(f"the backup is damaged: {name} doesn't match its checksum")
    if set(manifest.get("sha256", {})) != {BACKUP_DB, BACKUP_TAR}:
        raise BackupError("the backup's manifest doesn't list both of its parts")
    return manifest


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for member in tar.getmembers():
        parts = Path(member.name).parts
        if (
            not member.isfile()
            or member.name.startswith("/")
            or ".." in parts
            or not parts
            or parts[0] not in LIBRARY_DIRS
        ):
            raise BackupError(f"the backup's archive holds an unexpected entry: {member.name!r}")
        members.append(member)
    return members


@dataclass
class RestoreReport:
    data_dir: Path
    counts: dict[str, int]
    replaced: Path | None = None


def restore(source: Path, data_dir: Path, *, force: bool = False) -> RestoreReport:
    """Restore the backup folder ``source`` into ``data_dir`` (see the module docstring)."""
    from hal9000.knowledge.ingest.jobs import WriterLock
    from hal9000.knowledge.storage import SCHEMA_VERSION, open_db

    source, data_dir = Path(source), Path(data_dir)
    manifest = _check(source)
    if int(manifest.get("schema_version", 0)) > SCHEMA_VERSION:
        raise BackupError(
            f"the backup was made by newer code (schema {manifest['schema_version']}, this code "
            f"knows {SCHEMA_VERSION}); update hal9000 before restoring it"
        )
    knowledge = data_dir / "knowledge"
    lock = WriterLock(knowledge / "worker.lock")
    if not lock.acquire():
        raise BackupError(
            "the knowledge service (or an ingest) is running on this data directory; stop it "
            "before restoring"
        )
    try:
        replaced = None
        if not library_is_empty(data_dir):
            if not force:
                raise BackupError(
                    f"{data_dir} already holds a knowledge library; restore into an empty data "
                    "directory, or pass --force to move the current one aside"
                )
            replaced = data_dir / f"knowledge-replaced-{_stamp()}"
            replaced.mkdir(parents=True)
            for path in _library_paths(data_dir):
                if path.exists():
                    path.rename(replaced / path.name)

        # the archive first: a failure there leaves no database pointing at missing files
        knowledge.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source / BACKUP_TAR, "r:gz") as tar:
            tar.extractall(knowledge, members=_safe_members(tar), filter="data")
        db = data_dir / "knowledge.db"
        tmp = db.with_name("knowledge.db.restoring")
        shutil.copyfile(source / BACKUP_DB, tmp)
        _integrity(tmp)
        tmp.replace(db)
        conn = open_db(db)  # migrates a backup made by older code; refuses a newer one
        try:
            counts = _counts(conn)
        finally:
            conn.close()
        if counts != manifest.get("counts", counts):  # pragma: no cover - checksums guard this
            raise BackupError(f"restored counts {counts} differ from the backup's")
    finally:
        lock.release()
    return RestoreReport(data_dir=data_dir, counts=counts, replaced=replaced)


# ------------------------------------------------------------------ stats
_NO_DB = "No knowledge base yet at {path}: add documents with `hal9000 kb ingest`."


def _size(db_path: Path) -> int:
    return sum(
        path.stat().st_size
        for path in (db_path, db_path.with_name(db_path.name + "-wal"))
        if path.exists()
    )


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def month_spend(agents: list[str]) -> dict[str, float] | None:
    """Month-to-date spend per agent from the shared ledger (``hal9000.db``), or ``None``."""
    from hal9000 import config as hal_config
    from hal9000.storage.db import Database

    hal = hal_config.settings()
    ledger = Path(hal.data_dir) / "hal9000.db"
    if not ledger.is_file():
        return None
    now = datetime.now(hal.tz)
    db = Database(ledger)
    try:
        return {agent: db.month_cost_for(now.year, now.month, agents=[agent]) for agent in agents}
    finally:
        db.close()


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Library and index statistics (no document text)."""
    from hal9000.knowledge.health import last_ingestion_at, last_question_at
    from hal9000.knowledge.index.profiles import active_profile

    files = conn.execute(
        "SELECT COUNT(*), COUNT(superseded_by), COALESCE(SUM(bytes), 0) FROM files"
    ).fetchone()
    profiles = []
    for row in conn.execute(
        "SELECT p.id, p.name, p.status, p.active, "
        "(SELECT COUNT(*) FROM chunks c WHERE c.profile_id = p.id) AS chunks, "
        "(SELECT COUNT(DISTINCT c.document_id) FROM chunks c WHERE c.profile_id = p.id) AS docs "
        "FROM profiles p WHERE p.name NOT LIKE '%#%' ORDER BY p.id"
    ):
        pages = conn.execute(
            "SELECT COALESCE(SUM(d.page_end - d.page_start + 1), 0) FROM documents d "
            "WHERE d.id IN (SELECT DISTINCT document_id FROM chunks WHERE profile_id = ?)",
            (row["id"],),
        ).fetchone()[0]
        profiles.append(
            {
                "name": row["name"],
                "status": row["status"],
                "active": bool(row["active"]),
                "documents": int(row["docs"]),
                "pages": int(pages),
                "chunks": int(row["chunks"]),
            }
        )
    active = active_profile(conn)
    return {
        "files": int(files[0]),
        "superseded_files": int(files[1]),
        "file_bytes": int(files[2]),
        "documents": int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
        "pages": int(
            conn.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT file_sha, page FROM pages)"
            ).fetchone()[0]
        ),
        "profiles": profiles,
        "active_profile": active.name if active else None,
        "last_ingestion_at": last_ingestion_at(conn),
        "last_question_at": last_question_at(conn),
    }


def _print_stats(settings: Any, out: TextIO) -> int:
    from hal9000.knowledge.config import AGENT, EVAL_AGENT
    from hal9000.knowledge.storage import open_db

    db_path = Path(settings.db_path)
    if not db_path.is_file():
        print(_NO_DB.format(path=db_path), file=out)
        return EXIT_OK
    conn = open_db(db_path)
    try:
        body = stats(conn)
    finally:
        conn.close()
    print(f"Database: {db_path} ({_human(_size(db_path))})", file=out)
    print(
        f"Library: {body['files']} files ({body['superseded_files']} superseded, "
        f"{_human(body['file_bytes'])}), {body['documents']} documents, {body['pages']} pages",
        file=out,
    )
    print(f"Active profile: {body['active_profile'] or '(none)'}", file=out)
    print("Profiles:", file=out)
    if not body["profiles"]:
        print("  (none built yet)", file=out)
    for p in body["profiles"]:
        mark = "*" if p["active"] else " "
        print(
            f"  {mark} {p['name']}  {p['status']}: {p['documents']} documents, "
            f"{p['pages']} pages, {p['chunks']} chunks",
            file=out,
        )
    spend = month_spend([AGENT, EVAL_AGENT])
    spent = spend or {AGENT: 0.0, EVAL_AGENT: 0.0}
    note = "" if spend is not None else " (no ledger yet)"
    print(
        f"Spend this month: ${spent[AGENT]:.2f} of ${settings.monthly_cap_usd:.2f} (knowledge), "
        f"${spent[EVAL_AGENT]:.2f} of ${settings.eval_cap_usd:.2f} (evaluation){note}",
        file=out,
    )
    print(f"Last ingestion: {body['last_ingestion_at'] or 'never'}", file=out)
    print(f"Last answered question: {body['last_question_at'] or 'never'}", file=out)
    return EXIT_OK


# ------------------------------------------------------------------ profiles
def _thresholds(conn: sqlite3.Connection) -> dict[int, dict[str, float]]:
    found: dict[int, dict[str, float]] = {}
    for row in conn.execute("SELECT profile_id, reranker, threshold FROM abstain_thresholds"):
        found.setdefault(row[0], {})[row[1] or "none"] = float(row[2])
    return found


def _profile_list(settings: Any, out: TextIO) -> int:
    from hal9000.knowledge.config import load_knowledge_file
    from hal9000.knowledge.storage import open_db

    if not Path(settings.db_path).is_file():
        print(_NO_DB.format(path=settings.db_path), file=out)
        return EXIT_OK
    conn = open_db(settings.db_path)
    try:
        thresholds = _thresholds(conn)
        rows = conn.execute(
            "SELECT p.id, p.name, p.status, p.active, "
            "(SELECT COUNT(*) FROM chunks c WHERE c.profile_id = p.id) AS chunks "
            "FROM profiles p WHERE p.name NOT LIKE '%#%' ORDER BY p.id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        print("No index profiles yet: the first `hal9000 kb ingest` builds one.", file=out)
    for row in rows:
        state = (
            "active" if row["active"] else ("built" if row["status"] == "ready" else row["status"])
        )
        abstain = thresholds.get(row["id"], {})
        shown = ", ".join(f"{k} {v:.3f}" for k, v in sorted(abstain.items())) or "not calibrated"
        print(
            f"{'*' if row['active'] else ' '} {row['name']}\n"
            f"    {state}; {row['chunks']} chunks; abstain threshold: {shown}",
            file=out,
        )
    configured = load_knowledge_file(settings.yaml_path).active_profile
    active = next((row["name"] for row in rows if row["active"]), None)
    if configured and active and configured != active:
        print(
            f"\nconfig/knowledge.yaml names {configured}; it only sets the profile for the "
            "first ingest. The database's active flag decides (activate to change it).",
            file=out,
        )
    return EXIT_OK


def _profile_activate(settings: Any, name: str, out: TextIO) -> int:
    from hal9000.knowledge.index.profiles import ProfileError, activate_profile
    from hal9000.knowledge.storage import open_db

    if not Path(settings.db_path).is_file():
        raise BackupError(_NO_DB.format(path=settings.db_path))
    conn = open_db(settings.db_path)
    try:
        profile = activate_profile(conn, name)
    except ProfileError as error:
        print(f"hal9000 kb profile activate: {error}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        conn.close()
    print(f"{profile.name} now serves the chat.", file=out)
    return EXIT_OK


# ------------------------------------------------------------------ dispatch
def run(args: argparse.Namespace, *, out: TextIO | None = None) -> int:
    from hal9000 import config as hal_config
    from hal9000.knowledge.config import settings as kb_settings

    out = out or sys.stdout
    command = args.kb_command
    try:
        settings = kb_settings()
        if command == "stats":
            return _print_stats(settings, out)
        if command == "profile list":
            return _profile_list(settings, out)
        if command == "profile activate":
            return _profile_activate(settings, args.name, out)
        if command == "backup":
            to = Path(args.to) if args.to else settings.data_dir / "backups"
            report = backup(settings.db_path, settings.data_dir / "knowledge", to)
            counts = report.manifest["counts"]
            print(
                f"Backed up {counts['files']} files, {counts['documents']} documents and "
                f"{counts['chunks']} chunks to {report.folder}",
                file=out,
            )
            if report.missing_files:
                print(
                    f"warning: {len(report.missing_files)} stored originals were missing when "
                    "the archive was written (deleted during the backup?)",
                    file=sys.stderr,
                )
            return EXIT_OK
        if command == "restore":
            data_dir = Path(args.into) if args.into else settings.data_dir
            restored = restore(Path(args.source), data_dir, force=args.force)
            counts = restored.counts
            print(
                f"Restored {counts['files']} files, {counts['documents']} documents and "
                f"{counts['chunks']} chunks into {restored.data_dir}",
                file=out,
            )
            if restored.replaced is not None:
                print(f"The previous library was moved to {restored.replaced}", file=out)
            return EXIT_OK
    except (BackupError, hal_config.ConfigError) as error:
        print(f"hal9000 kb {command}: {error}", file=sys.stderr)
        return EXIT_ERROR
    raise ValueError(f"unknown ops command {command!r}")  # pragma: no cover
