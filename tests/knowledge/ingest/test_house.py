"""Opt-in ``house`` checks against the real inspection bundle (plan §18, Appendix A).

Run by Mike on the Mac, never in CI: the bundle lives at ``KB_HOUSE_PDF``
(default ``data/knowledge/inbox/Home_Inspection.pdf``) and is never committed
(rule K2). These tests skip cleanly when it is absent. Only page ranges and
types are encoded here -- no names, amounts or reference numbers.

    uv run pytest -m house tests/knowledge/ingest/test_house.py --no-cov -s

The LLM-path test also needs ``-m "house and live"`` and an OpenAI key in
``.env``; it makes one metered fast-tier call (a few cents at most).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from hal9000.knowledge.config import settings as kb_settings
from hal9000.knowledge.ingest.clean import clean_pages
from hal9000.knowledge.ingest.parsers import parse_file
from hal9000.knowledge.ingest.segment import segment_file

#: Appendix A: (first page, last page, doc_type) of the eleven sub-documents.
APPENDIX_A: tuple[tuple[int, int, str], ...] = (
    (1, 2, "disclosure_form"),
    (3, 60, "inspection_report"),
    (61, 63, "invoice"),
    (64, 66, "estimate"),
    (67, 69, "estimate"),
    (70, 73, "estimate"),
    (74, 80, "pest_report"),
    (81, 81, "completion_notice"),
    (82, 87, "pest_report"),
    (88, 88, "invoice"),
    (89, 89, "invoice"),
)
HOUSE_PAGES = 89

# Read now: the suite's autouse fixtures strip KB_* and provider variables per test.
HOUSE_PDF = Path(os.environ.get("KB_HOUSE_PDF", "").strip() or kb_settings().house_pdf)
_ENV_AT_IMPORT = dict(os.environ)

pytestmark = [
    pytest.mark.house,
    pytest.mark.skipif(not HOUSE_PDF.is_file(), reason=f"no house bundle at {HOUSE_PDF}"),
]


@pytest.fixture(scope="module")
def house_pages():
    return parse_file(HOUSE_PDF, timeout_s=600).pages


def test_the_bundle_has_89_pages_all_with_a_text_layer(house_pages):
    assert len(house_pages) == HOUSE_PAGES
    assert all(page.has_text_layer for page in house_pages)


def test_heuristics_find_at_least_nine_of_the_eleven_boundaries(house_pages):
    documents = segment_file(house_pages, file_sha="0" * 64, use_llm=False).documents
    found = {d.page_start for d in documents}
    expected = {start for start, _, _ in APPENDIX_A}
    print(
        f"heuristic starts: {sorted(found)}; missed {sorted(expected - found)}; extra {sorted(found - expected)}"
    )
    assert len(found & expected) >= 9


def test_no_running_header_survives_cleaning(house_pages):
    # the pipeline order: segment the raw pages, then clean with the documents
    documents = segment_file(house_pages, file_sha="0" * 64, use_llm=False).documents
    result = clean_pages(house_pages, file_sha="0" * 64, parser="pdfium", documents=documents)
    for page in result.pages:
        assert "Envelope ID" not in page.text, page.page
        assert not re.search(r"Page \d+ of \d+", page.text), page.page


@pytest.mark.live
def test_llm_path_gives_the_exact_page_ranges_and_types(house_pages, monkeypatch, tmp_path):
    from dotenv import dotenv_values

    from hal9000 import config as hal_config
    from hal9000.cost.metering import Meter
    from hal9000.knowledge.config import AGENT, monthly_caps
    from hal9000.llm import generate_response
    from hal9000.storage.db import Database

    env = {**{k: v for k, v in dotenv_values().items() if v is not None}, **_ENV_AT_IMPORT}
    if not env.get("OPENAI_API_KEY"):
        pytest.skip("no OPENAI_API_KEY")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    hal = hal_config.settings()
    database = Database(hal.data_dir / "hal9000.db")
    meter = Meter(
        database, "kb-house-segment", soft_cap=0.5, hard_cap=1.0, monthly_caps=monthly_caps()
    )
    result = segment_file(
        house_pages,
        file_sha="0" * 64,
        generate=meter.wrap(generate_response, agent=AGENT),
        name=HOUSE_PDF.name,
    )
    database.close()
    print(
        f"source={result.source} calls={result.llm_calls} cost=${meter.spent:.4f} warnings={result.warnings}"
    )
    assert result.source == "llm"
    assert [(d.page_start, d.page_end, d.doc_type) for d in result.documents] == list(APPENDIX_A)
