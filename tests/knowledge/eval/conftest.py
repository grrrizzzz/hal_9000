"""Fixtures for the evaluation-harness tests: the synthetic fixture corpus as documents.

Everything is synthetic (rule K2): the KWP1 fixture files, parsed and cleaned
with the real pipeline, split into the sub-documents their ground truth
declares. ``bundle.pdf`` is left out because it repeats three other files.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hal9000.knowledge.eval import wiring
from hal9000.knowledge.eval.gold import SourceDocument
from hal9000.knowledge.storage import open_db
from tests.knowledge.fixtures import corpus
from tests.knowledge.ingest.conftest import cleaned

FILES = (
    corpus.INSPECTION,
    corpus.ESTIMATE,
    corpus.INVOICE,
    corpus.UNNUMBERED,
    corpus.DOCX,
    corpus.HTML,
    corpus.MARKDOWN,
    corpus.TEXT,
)


def build_documents() -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for file in FILES:
        sha = hashlib.sha256(file.path.read_bytes()).hexdigest()
        pages = cleaned(file).pages
        for expected in file.documents:
            documents.append(
                SourceDocument(
                    document_id=len(documents) + 1,
                    file_sha=sha,
                    title=expected.title,
                    doc_type=expected.doc_type,
                    vendor=expected.vendor,
                    doc_date=expected.doc_date,
                    pages={
                        p.page: p.text
                        for p in pages
                        if expected.page_start <= p.page <= expected.page_end and p.text.strip()
                    },
                )
            )
    return documents


_CACHE: list[SourceDocument] = []


@pytest.fixture
def fixture_docs() -> list[SourceDocument]:
    """The fixture corpus as sub-documents with cleaned pages (built once per session)."""
    if not _CACHE:
        _CACHE.extend(build_documents())
    return list(_CACHE)


def store_documents(conn, documents: list[SourceDocument]) -> None:
    """Write ``documents`` (files, documents and pages rows) into a knowledge.db."""
    with conn:
        for sha in dict.fromkeys(d.file_sha for d in documents):
            conn.execute(
                "INSERT OR IGNORE INTO files (sha256, original_name, relative_path, mime_type, "
                "bytes, uploaded_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sha, f"{sha[:8]}.pdf", f"{sha[:8]}.pdf", "application/pdf", 1, "2026-01-01"),
            )
        for d in documents:
            conn.execute(
                "INSERT INTO documents (id, file_sha, page_start, page_end, title, doc_type, "
                "vendor, doc_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    d.document_id,
                    d.file_sha,
                    min(d.pages),
                    max(d.pages),
                    d.title,
                    d.doc_type,
                    d.vendor,
                    d.doc_date.isoformat() if d.doc_date else None,
                ),
            )
            for page, text in d.pages.items():
                conn.execute(
                    "INSERT OR IGNORE INTO pages (file_sha, page, parser, text, has_text_layer) "
                    "VALUES (?, ?, 'pdfium', ?, 1)",
                    (d.file_sha, page, text),
                )


#: the small quota mix of the gold-generation CLI tests (15 items, 7 generation tasks)
GOLD_QUOTAS = {
    "factoid": 4,
    "money": 2,
    "date": 2,
    "list": 2,
    "cross_document": 2,
    "unanswerable": 3,
}


@pytest.fixture
def gold_env(tmp_path, monkeypatch, openai_env, fixture_docs):
    data, config = tmp_path / "data", tmp_path / "config"
    config.mkdir()
    (config / "knowledge_eval.yaml").write_text(
        "gold:\n  quotas: " + json.dumps(GOLD_QUOTAS) + "\n"
    )
    monkeypatch.setenv("HAL_DATA_DIR", str(data))
    monkeypatch.setenv("HAL_CONFIG_DIR", str(config))
    conn = open_db(data / "knowledge.db")
    store_documents(conn, fixture_docs)
    yield conn
    conn.close()
    wiring.register_retriever(None)
