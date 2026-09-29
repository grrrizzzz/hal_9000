"""Fixtures for KWP10: a data directory holding the ingested fixture library.

The synthetic corpus is ingested once per session with the KWP3 pipeline in fake
mode into ``data/knowledge.db`` and ``data/knowledge/files/`` (the real layout),
with two profiles: ``structure-256`` (active) and ``parent_child-128-p512``. A
synthetic evaluation file sits in ``data/knowledge/eval/``. Each test gets its
own copy.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from hal9000.knowledge.config import KnowledgeFile
from hal9000.knowledge.index.fake import FAKE_ENV, FakeEmbedder, enable_fake_mode
from hal9000.knowledge.ingest.pipeline import IngestPipeline
from hal9000.knowledge.storage import open_db
from tests.knowledge.fixtures import corpus

ACTIVE = "pdfium.structure-256.hdr.fake-64"
PARENT_CHILD = "pdfium.parent_child-128-p512.hdr.fake-64"

LAYOUT: dict[str, corpus.CorpusFile] = {
    "House/inspection.pdf": corpus.INSPECTION,
    "House/Bills/estimate.pdf": corpus.ESTIMATE,
    "House/Bills/invoice.pdf": corpus.INVOICE,
    "Appliances/heat_pump_guide.docx": corpus.DOCX,
    "Appliances/Garage/garage_door.md": corpus.MARKDOWN,
    "Yard/irrigation_notes.html": corpus.HTML,
    "Yard/pool_log.txt": corpus.TEXT,
}

#: a synthetic evaluation file, so backups carry data/knowledge/eval/ too
EVAL_LINE = '{"question": "How old is the water heater?", "category": "factoid"}\n'


def build_library(data_dir: Path) -> Path:
    """Ingest the fixture corpus into ``data_dir`` (fake mode); returns ``knowledge.db``."""
    knowledge = data_dir / "knowledge"
    enable_fake_mode(True)
    pipeline = IngestPipeline(
        knowledge / "files",
        knowledge_file=KnowledgeFile(active_profile=ACTIVE),
        embed_fn=FakeEmbedder(),
        use_llm=False,
        bus=None,
        isolate_parse=False,
        extra_profiles=[PARENT_CHILD],
    )
    db = data_dir / "knowledge.db"
    conn = open_db(db)
    try:
        ids = []
        for relative, item in LAYOUT.items():
            result = pipeline.submit(conn, item.path, relative)
            assert result.status == "added", result
            ids.append(result.job_id)
        worker = pipeline.worker(":unused:")
        jobs = [worker.run_job(conn, job_id) for job_id in ids]
        assert all(job.status == "done" for job in jobs), jobs
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
        pipeline.close()
        enable_fake_mode(False)
    (knowledge / "eval" / "runs").mkdir(parents=True)
    (knowledge / "eval" / "gold.jsonl").write_text(EVAL_LINE)
    (knowledge / "eval" / "runs" / "run-1.json").write_text('{"recall": 0.9}\n')
    return db


@pytest.fixture(scope="session")
def template_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ops-library") / "data"
    build_library(root)
    return root


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FAKE_ENV, "1")


@pytest.fixture(autouse=True)
def fresh_query_cache() -> Iterator[None]:
    from hal9000.knowledge.retrieve.retriever import DEFAULT_QUERY_CACHE

    DEFAULT_QUERY_CACHE.clear()
    yield
    DEFAULT_QUERY_CACHE.clear()


@pytest.fixture
def data_dir(template_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the library; ``KB_DATA_DIR`` and ``HAL_DATA_DIR`` point at it."""
    target = tmp_path / "data"
    shutil.copytree(template_data, target)
    monkeypatch.setenv("KB_DATA_DIR", str(target))
    monkeypatch.setenv("HAL_DATA_DIR", str(target))
    return target
