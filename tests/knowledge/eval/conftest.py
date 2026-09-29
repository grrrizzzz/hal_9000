"""Fixtures for the evaluation-harness tests: the synthetic fixture corpus as documents.

Everything is synthetic (rule K2): the KWP1 fixture files, parsed and cleaned
with the real pipeline, split into the sub-documents their ground truth
declares. ``bundle.pdf`` is left out because it repeats three other files.
"""

from __future__ import annotations

import hashlib

import pytest

from hal9000.knowledge.eval.gold import SourceDocument
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
