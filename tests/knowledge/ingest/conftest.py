"""Fixtures and markers for the ingestion tests (handoff KWP1).

``models`` (needs downloaded models, e.g. Docling's layout models) and ``house``
(needs the real inspection bundle at ``KB_HOUSE_PDF``) are opt-in: they skip
cleanly when what they need is absent (rule K3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hal9000.knowledge.ingest.clean import CleanResult, clean_pages
from hal9000.knowledge.ingest.parsers import ParsedFile, parse_file
from tests.knowledge.fixtures import corpus

OPT_IN = ("house", "live", "models")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "models: needs downloaded models (Docling, ONNX); opt-in")
    config.addinivalue_line("markers", "house: needs the real bundle at KB_HOUSE_PDF; opt-in")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Opt-in markers run only when selected with ``-m``: a plain ``pytest`` never reads
    the house bundle, downloads models or pays for a call, even on Mike's Mac."""
    selected = config.getoption("markexpr") or ""
    here = Path(__file__).resolve().parent
    for item in items:
        if here not in Path(str(item.fspath)).resolve().parents:
            continue
        for marker in OPT_IN:
            if item.get_closest_marker(marker) and marker not in selected:
                item.add_marker(pytest.mark.skip(reason=f"opt-in: select it with -m {marker}"))


SHA = "0" * 64


def parsed(file: corpus.CorpusFile) -> ParsedFile:
    return parse_file(file.path, isolate=False)


def cleaned(file: corpus.CorpusFile) -> CleanResult:
    result = parsed(file)
    return clean_pages(
        result.pages, file_sha=SHA, parser=result.parser, pseudo_pages=result.pseudo_pages
    )
