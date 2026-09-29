"""Fixtures for the index and ingestion tests: a pipeline over a temp data dir, the fake embedder."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hal9000.knowledge.config import KnowledgeFile
from hal9000.knowledge.index.fake import FAKE_ENV, FakeEmbedder
from hal9000.knowledge.ingest.pipeline import IngestPipeline
from hal9000.knowledge.storage import open_db
from tests.knowledge.fixtures import corpus

#: The profile most tests index into: structure chunks of 256 tokens, fake 64-d vectors.
PROFILE = "pdfium.structure-256.hdr.fake-64"


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests index with the fake embedder, which needs explicit fake mode."""
    monkeypatch.setenv(FAKE_ENV, "1")


@pytest.fixture
def fake() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_pipeline(tmp_path: Path, fake: FakeEmbedder) -> Iterator[Callable[..., IngestPipeline]]:
    made: list[IngestPipeline] = []

    def make(active: str | None = PROFILE, **kw) -> IngestPipeline:
        kw.setdefault("embed_fn", fake)
        kw.setdefault("use_llm", False)
        kw.setdefault("bus", None)
        pipeline = IngestPipeline(
            tmp_path / "files", knowledge_file=KnowledgeFile(active_profile=active), **kw
        )
        made.append(pipeline)
        return pipeline

    yield make
    for pipeline in made:  # stop their parse processes
        pipeline.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "knowledge.db"


@pytest.fixture
def conn(db_path: Path):
    connection = open_db(db_path)
    yield connection
    connection.close()


def ingest_all(conn, pipeline: IngestPipeline, files=corpus.ALL_FILES, folder: str = "Docs"):
    """Submit and run every file; returns the finished jobs."""
    ids = []
    for item in files:
        result = pipeline.submit(conn, item.path, f"{folder}/{item.name}")
        assert result.status == "added", result
        ids.append(result.job_id)
    worker = pipeline.worker(":unused:")
    return [worker.run_job(conn, job_id) for job_id in ids]


def table_counts(conn) -> dict[str, int]:
    """Row counts of every table, the per-profile FTS5 and vec0 tables and their shadows included."""
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in names}
