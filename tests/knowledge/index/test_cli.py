"""``hal9000 kb ingest`` and ``hal9000 kb check`` (plan §17)."""

from __future__ import annotations

import shutil
import sqlite3
import time

import pytest

from hal9000.__main__ import main
from hal9000.knowledge.index.fake import FakeEmbedder
from hal9000.knowledge.storage import open_db
from tests.knowledge.fixtures import corpus
from tests.knowledge.index.conftest import PROFILE


@pytest.fixture
def kb_env(tmp_path, monkeypatch):
    """A data dir and a config dir whose knowledge.yaml names ``profile``."""
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("HAL_CONFIG_DIR", str(config))
    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path / "hal-data"))
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "data"))

    def configure(profile: str | None = PROFILE) -> None:
        value = profile if profile else "null"
        (config / "knowledge.yaml").write_text(f"active_profile: {value}\n")

    configure()
    return configure


@pytest.fixture
def library_folder(tmp_path):
    root = tmp_path / "House"
    (root / "Appliances" / "Heat pump").mkdir(parents=True)
    (root / "Reports").mkdir()
    shutil.copy(corpus.DOCX.path, root / "Appliances" / "Heat pump" / corpus.DOCX.name)
    shutil.copy(corpus.INVOICE.path, root / "Reports" / corpus.INVOICE.name)
    (root / "Reports" / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    (root / ".DS_Store").write_bytes(b"junk")
    return root


def test_ingest_walks_folders_and_reports_each_file(kb_env, library_folder, tmp_path, capsys):
    loose = tmp_path / "pool_log.txt"
    shutil.copy(corpus.TEXT.path, loose)
    code = main(["kb", "ingest", str(library_folder), str(loose), "--no-llm"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert (
        "added      House/Appliances/Heat pump/heat_pump_guide.docx  [House/Appliances/Heat pump/]"
        in out
    )
    assert "added      House/Reports/invoice.pdf  [House/Reports/]" in out
    assert "added      pool_log.txt" in out
    assert (
        "skipped    House/Reports/photo.jpg  [House/Reports/]: .jpg is not a supported type" in out
    )
    assert ".DS_Store" not in out
    assert "embedding $0.0000" in out
    assert "3 added, 0 duplicate, 1 skipped, 0 failed. Embedding cost $0.0000." in out

    assert main(["kb", "ingest", str(loose), "--collection", "Pool"]) == 0
    out = capsys.readouterr().out
    # a duplicate keeps the collection it was first stored with
    assert "duplicate  pool_log.txt: already in library as pool_log.txt" in out
    conn = open_db(tmp_path / "data" / "knowledge.db")
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0
    conn.close()


def test_ingest_reports_missing_paths_and_failures(kb_env, tmp_path, capsys):
    assert main(["kb", "ingest", str(tmp_path / "nope.pdf")]) == 1
    assert "no such file or folder" in capsys.readouterr().err
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nnot a pdf\n")
    assert main(["kb", "ingest", str(broken), "--no-llm"]) == 1
    out = capsys.readouterr().out
    assert "failed     broken.pdf: ParseError" in out and "1 failed" in out


def test_ingest_without_a_profile_stores_files_for_a_later_build(kb_env, tmp_path, capsys):
    kb_env(None)
    assert main(["kb", "ingest", str(corpus.ESTIMATE.path), "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert "No active index profile" in out and "not chunked yet" in out


def test_a_paid_profile_needs_a_key(kb_env, capsys):
    kb_env("pdfium.structure-768.hdr.te3s-1536")
    assert main(["kb", "ingest", str(corpus.INVOICE.path)]) == 1
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


def test_paid_ingestion_prints_an_estimate_and_needs_yes(kb_env, tmp_path, monkeypatch, capsys):
    kb_env("pdfium.structure-768.hdr.te3s-1536")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake = FakeEmbedder(dimensions=1536)
    segment_calls = []

    def fake_generate(prompt, tier="fast", *, on_usage=None):
        segment_calls.append(tier)
        if on_usage is not None:
            on_usage({"model": "openai/gpt-5-mini", "prompt": 900, "cached": 0, "completion": 100})
        return "not json"  # the segmenter falls back to heuristics

    monkeypatch.setattr("hal9000.llm.embed", fake)
    monkeypatch.setattr("hal9000.llm.generate_response", fake_generate)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)

    assert main(["kb", "ingest", str(corpus.INVOICE.path)]) == 1
    captured = capsys.readouterr()
    assert "Estimated cost: up to $" in captured.out and "monthly cap $10.00" in captured.out
    assert "pass --yes" in captured.err
    assert fake.calls == 0 and segment_calls == []

    assert main(["kb", "ingest", str(corpus.INVOICE.path), "--yes"]) == 0
    out = capsys.readouterr().out
    assert "added      invoice.pdf" in out and "Total metered spend $" in out
    assert fake.calls == 1 and len(segment_calls) == 2  # one retry after the invalid reply
    ledger = sqlite3.connect(tmp_path / "hal-data" / "hal9000.db")
    agents = ledger.execute("SELECT agent, model FROM ledger ORDER BY id").fetchall()
    ledger.close()
    assert ("knowledge", "openai/text-embedding-3-small") in agents
    assert agents.count(("knowledge", "openai/gpt-5-mini")) == 2


def test_check_passes_offline_and_skips_openai_without_a_key(kb_env, capsys):
    assert main(["kb", "check"]) == 0
    out = capsys.readouterr().out
    for name in ("knowledge.db migrated", "sqlite-vec loads", "FTS5 tokenizer", "tiktoken offline"):
        assert f"ok    {name}" in out
    # before any ingest: nothing is active; the yaml only names the first ingest's default
    assert (
        f"ok    index profile: no active profile yet; default for the first ingest: {PROFILE}"
        in out
    )
    assert "skip  OpenAI embedding and chat: OPENAI_API_KEY is not set" in out
    assert "All checks passed." in out


def test_check_calls_openai_only_with_a_key_and_reports_failures(
    kb_env, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake = FakeEmbedder(dimensions=1536)
    monkeypatch.setattr("hal9000.llm.embed", fake)

    def reply(prompt, tier="fast", *, on_usage=None):
        return "OK"

    monkeypatch.setattr("hal9000.llm.generate_response", reply)
    assert main(["kb", "check"]) == 0
    out = capsys.readouterr().out
    assert (
        "ok    OpenAI embedding: openai/text-embedding-3-small returned a 1536-dimension vector"
        in out
    )
    assert "ok    OpenAI chat: openai/gpt-5-mini replied 'OK'" in out

    def down(prompt, tier="fast", *, on_usage=None):
        raise ConnectionError("unreachable")

    monkeypatch.setattr("hal9000.llm.generate_response", down)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "knowledge.db").unlink()
    (tmp_path / "data" / "knowledge.db").write_text("this is not a database")
    assert main(["kb", "check"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  knowledge.db migrated: DatabaseError" in out
    assert "FAIL  OpenAI chat: ConnectionError: unreachable" in out
    assert "ok    sqlite-vec loads" in out and "ok    tiktoken offline" in out
    assert "4 check(s) failed." in out  # the database, the profile, the embedding model, chat


def test_check_reports_the_database_active_profile_not_the_yaml(kb_env, tmp_path, capsys):
    assert main(["kb", "ingest", str(corpus.TEXT.path), "--no-llm"]) == 0
    other = "pdfium.fixed-64.hdr.fake-64"
    conn = open_db(tmp_path / "data" / "knowledge.db")
    from hal9000.knowledge.chunking import parse_profile_name
    from hal9000.knowledge.index.embedding import EmbeddingService
    from hal9000.knowledge.index.profiles import build_profile
    from hal9000.knowledge.ingest.pipeline import IngestPipeline

    pipeline = IngestPipeline(tmp_path / "data" / "knowledge" / "files", bus=None)
    build_profile(
        conn,
        parse_profile_name(other),
        EmbeddingService(conn),
        pages_for=pipeline.pages_for,
        activate=True,
    )
    pipeline.close()
    conn.close()
    capsys.readouterr()
    assert main(["kb", "check"]) == 0
    out = capsys.readouterr().out
    assert (
        f"ok    index profile: active profile {other}; default for the first ingest: {PROFILE}"
        in out
    )


def test_a_fake_profile_is_refused_outside_fake_mode(kb_env, monkeypatch, capsys):
    monkeypatch.delenv("KB_FAKE")
    assert main(["kb", "ingest", str(corpus.TEXT.path), "--no-llm"]) == 1
    assert "KB_FAKE=1" in capsys.readouterr().err


def test_ingest_only_queues_when_the_service_holds_the_lock(kb_env, tmp_path, capsys):
    from hal9000.knowledge.config import KnowledgeFile
    from hal9000.knowledge.ingest.jobs import WriterLock
    from hal9000.knowledge.ingest.pipeline import IngestPipeline

    data = tmp_path / "data"
    service_lock = WriterLock(data / "knowledge" / "worker.lock")
    assert service_lock.acquire()  # the running service
    try:
        assert (
            main(["kb", "ingest", str(corpus.TEXT.path), str(corpus.INVOICE.path), "--no-llm"]) == 0
        )
        out = capsys.readouterr().out
        assert "queued     pool_log.txt" in out and "queued     invoice.pdf" in out
        assert (
            "The knowledge service is running and holds the library lock: 2 file(s) queued" in out
        )
        conn = open_db(data / "knowledge.db")
        assert [r[0] for r in conn.execute("SELECT status FROM jobs")] == ["queued", "queued"]
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0

        # the service's worker (it holds the lock) picks the queued jobs up
        pipeline = IngestPipeline(
            data / "knowledge" / "files", bus=None, use_llm=False,
            knowledge_file=KnowledgeFile(active_profile=PROFILE),
        )  # fmt: skip
        worker = pipeline.worker(data / "knowledge.db")
        worker.lock = service_lock
        worker.poll_s = 0.1
        worker.start()
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if {r[0] for r in conn.execute("SELECT status FROM jobs")} == {"done"}:
                    break
                time.sleep(0.05)
        finally:
            worker.stop()
            pipeline.close()
        assert {r[0] for r in conn.execute("SELECT status FROM jobs")} == {"done"}
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0
        conn.close()
    finally:
        service_lock.release()


def test_ingest_runs_inline_and_holds_the_lock_while_it_does(kb_env, tmp_path, monkeypatch, capsys):
    from hal9000.knowledge.ingest import jobs
    from hal9000.knowledge.ingest.jobs import WriterLock

    seen: list[bool] = []
    real_run = jobs.JobWorker.run_job

    def run_job(self, conn, job_id):
        other = WriterLock(tmp_path / "data" / "knowledge" / "worker.lock")
        seen.append(other.acquire())  # a service starting now can't take it
        other.release()
        return real_run(self, conn, job_id)

    monkeypatch.setattr(jobs.JobWorker, "run_job", run_job)
    assert main(["kb", "ingest", str(corpus.TEXT.path), "--no-llm"]) == 0
    assert seen == [False]
    assert "added      pool_log.txt" in capsys.readouterr().out
    after = WriterLock(tmp_path / "data" / "knowledge" / "worker.lock")
    assert after.acquire()  # released on exit
    after.release()


KILLED_INGEST = """
import sys, time
from hal9000.__main__ import main
import hal9000.knowledge.index.fake as fake

real = fake.FakeEmbedder.__call__
def hang(self, texts, *a, **kw):
    print("embedding", flush=True)
    time.sleep(600)
fake.FakeEmbedder.__call__ = hang

if __name__ == "__main__":
    sys.exit(main(["kb", "ingest", sys.argv[1], "--no-llm"]))
"""


def test_a_killed_kb_ingest_is_recovered_by_the_next_one(kb_env, tmp_path, capsys):
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    script = tmp_path / "killed.py"
    script.write_text(KILLED_INGEST)
    child = subprocess.Popen(
        [sys.executable, str(script), str(corpus.TEXT.path)],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(repo), "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        line = child.stdout.readline()
        assert line.strip() == "embedding", child.stderr.read()
        child.send_signal(signal.SIGKILL)
        child.wait(30)
    finally:
        if child.poll() is None:  # pragma: no cover
            child.kill()
        child.stdout.close()
        child.stderr.close()

    conn = open_db(tmp_path / "data" / "knowledge.db")
    assert [r[0] for r in conn.execute("SELECT status FROM jobs")] == ["running"]
    conn.close()
    # the dead process's lock went with it; the next kb ingest recovers and retries the file
    assert main(["kb", "ingest", str(corpus.TEXT.path), "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert "1 job(s) interrupted by a stopped process were marked failed" in out
    assert "added      pool_log.txt" in out
    conn = open_db(tmp_path / "data" / "knowledge.db")
    assert [r[0] for r in conn.execute("SELECT status FROM jobs")] == ["done"]
    conn.close()
