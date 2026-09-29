"""Helpers for the ingestion tests (handoff KWP1).

The opt-in ``house``, ``models`` and ``live`` markers are registered in
``pyproject.toml`` and gated in ``tests/conftest.py``.
"""

from __future__ import annotations

from hal9000.knowledge.ingest.clean import CleanResult, clean_pages
from hal9000.knowledge.ingest.parsers import ParsedFile, parse_file
from tests.knowledge.fixtures import corpus

SHA = "0" * 64


def parsed(file: corpus.CorpusFile) -> ParsedFile:
    return parse_file(file.path, isolate=False)


def cleaned(file: corpus.CorpusFile) -> CleanResult:
    result = parsed(file)
    return clean_pages(
        result.pages, file_sha=SHA, parser=result.parser, pseudo_pages=result.pseudo_pages
    )
