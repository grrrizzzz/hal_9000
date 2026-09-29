"""Fixtures for the retriever tests: the fixture corpus, ingested once, copied per test.

The corpus is ingested with the KWP3 pipeline in fake mode into two profiles:
``structure-256`` (active) and ``parent_child-128-p512``. Files sit in nested
folders so collection filters (with sub-folders) can be tested, and
``bundle.pdf`` repeats the inspection, estimate and invoice text, which is what
near-duplicate removal is for.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from hal9000.knowledge.config import KnowledgeFile
from hal9000.knowledge.index.embedding import EmbeddingService
from hal9000.knowledge.index.fake import FAKE_ENV, FakeEmbedder, enable_fake_mode
from hal9000.knowledge.ingest.pipeline import IngestPipeline
from hal9000.knowledge.storage import open_db
from tests.knowledge.fixtures import corpus

ACTIVE = "pdfium.structure-256.hdr.fake-64"
PARENT_CHILD = "pdfium.parent_child-128-p512.hdr.fake-64"

#: relative upload path -> fixture file; the folder is the collection
LAYOUT: dict[str, corpus.CorpusFile] = {
    "House/inspection.pdf": corpus.INSPECTION,
    "House/Bills/estimate.pdf": corpus.ESTIMATE,
    "House/Bills/invoice.pdf": corpus.INVOICE,
    "Appliances/heat_pump_guide.docx": corpus.DOCX,
    "Appliances/Garage/garage_door.md": corpus.MARKDOWN,
    "Yard/irrigation_notes.html": corpus.HTML,
    "Yard/pool_log.txt": corpus.TEXT,
    "Scans/bundle.pdf": corpus.BUNDLE,
}


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FAKE_ENV, "1")


@pytest.fixture(scope="session")
def template_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The ingested corpus, built once per session (about a second)."""
    root = tmp_path_factory.mktemp("retrieve-corpus")
    enable_fake_mode(True)
    pipeline = IngestPipeline(
        root / "files",
        knowledge_file=KnowledgeFile(active_profile=ACTIVE),
        embed_fn=FakeEmbedder(),
        use_llm=False,
        bus=None,
        isolate_parse=False,
        extra_profiles=[PARENT_CHILD],
    )
    conn = open_db(root / "knowledge.db")
    try:
        ids = []
        for relative, item in LAYOUT.items():
            result = pipeline.submit(conn, item.path, relative)
            assert result.status == "added", result
            ids.append(result.job_id)
        worker = pipeline.worker(":unused:")
        jobs = [worker.run_job(conn, job_id) for job_id in ids]
        assert all(job.status == "done" for job in jobs), jobs
    finally:
        conn.close()
        pipeline.close()
        enable_fake_mode(False)
    return root / "knowledge.db"


@pytest.fixture
def conn(template_db: Path, tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A private copy of the ingested corpus."""
    target = tmp_path / "knowledge.db"
    shutil.copyfile(template_db, target)
    connection = open_db(target)
    yield connection
    connection.close()


@pytest.fixture
def fake() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def embeddings(conn: sqlite3.Connection, fake: FakeEmbedder) -> EmbeddingService:
    return EmbeddingService(conn, embed_fn=fake)


def profile_id(conn: sqlite3.Connection, name: str) -> int:
    return int(conn.execute("SELECT id FROM profiles WHERE name = ?", (name,)).fetchone()[0])
