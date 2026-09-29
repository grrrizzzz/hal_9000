"""The fixture corpus is reproducible and its ground truth matches the files.

Later work packages build on :mod:`tests.knowledge.fixtures.corpus`, so its
claims are checked here: page counts, text layers, finding pages, quotes, and
cleaning on the real fixture (running lines gone, the twice-repeated line kept,
letter spacing repaired).
"""

from __future__ import annotations

import re
import zipfile
import zlib

import pytest

from tests.knowledge.fixtures import build_fixture_corpus, corpus
from tests.knowledge.ingest.conftest import cleaned, parsed


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _content(path) -> object:
    """File content with compression undone, so a different zlib build can't fail the test."""
    data = path.read_bytes()
    if path.suffix == ".pdf":
        streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)
        inflated = []
        for stream in streams:
            try:
                inflated.append(zlib.decompress(stream))
            except zlib.error:
                inflated.append(stream)  # page streams are written uncompressed
        return re.sub(
            rb"stream\r?\n.*?\r?\nendstream|/Length \d+", b"", data, flags=re.DOTALL
        ), inflated
    if path.suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            return [
                (info.filename, info.date_time, archive.read(info)) for info in archive.infolist()
            ]
    return data


def test_builder_reproduces_the_committed_files(tmp_path):
    built = build_fixture_corpus.build(tmp_path)
    assert [p.name for p in built] == [f.name for f in corpus.ALL_FILES]
    for path in built:
        committed = corpus.FILES_DIR / path.name
        if committed.read_bytes() == path.read_bytes():
            continue  # byte-for-byte: the usual case with the locked versions
        assert _content(committed) == _content(path), (
            f"{path.name} differs from the committed fixture; rerun "
            "`uv run python -m tests.knowledge.fixtures.build_fixture_corpus`"
        )


def test_content_comparison_notices_a_real_change(tmp_path):
    built = build_fixture_corpus.build(tmp_path)
    pdf = built[0]
    pdf.write_bytes(pdf.read_bytes().replace(b"Valve Leak", b"Valve Lake"))
    assert _content(pdf) != _content(corpus.FILES_DIR / pdf.name)
    docx = tmp_path / "heat_pump_guide.docx"
    assert _content(docx) == _content(corpus.FILES_DIR / docx.name)


def test_corpus_lists_every_committed_file():
    committed = {p.name for p in corpus.FILES_DIR.iterdir() if p.is_file()}
    assert committed == {f.name for f in corpus.ALL_FILES}


@pytest.mark.parametrize("file", corpus.ALL_FILES, ids=lambda f: f.name)
def test_page_counts_and_documents_cover_every_page(file):
    assert parsed(file).page_count == file.page_count
    pages = [p for d in file.documents for p in range(d.page_start, d.page_end + 1)]
    assert pages == list(range(1, file.page_count + 1))
    assert file.path.exists() and file.mime_type


def test_bundle_is_the_three_pdfs_in_order():
    bundle = [p.text for p in parsed(corpus.BUNDLE).pages]
    parts = [
        p.text
        for f in (corpus.INSPECTION, corpus.ESTIMATE, corpus.INVOICE)
        for p in parsed(f).pages
    ]
    assert bundle == parts


def test_findings_are_on_their_pages_and_in_the_summary():
    pages = cleaned(corpus.INSPECTION).pages
    summary = pages[1].text
    assert len(corpus.FINDING_IDS) == len(set(corpus.FINDING_IDS)) == 13
    assert {f.id.split(".")[0] for f in corpus.FINDINGS} == {"1", "2", "3"}
    for finding in corpus.FINDINGS:
        assert finding.heading in pages[finding.page - 1].text
        assert finding.heading in summary
        assert f"Severity: {finding.severity}" in pages[finding.page - 1].text
    assert "3.1.2 Plumbing – Water Heater: Valve Leak" in pages[6].text


@pytest.mark.parametrize("fact", corpus.QA_FACTS, ids=lambda f: f.question)
def test_every_quote_is_on_its_cleaned_page(fact):
    file = next(f for f in corpus.ALL_FILES if f.name == fact.file)
    page = cleaned(file).pages[fact.page - 1]
    assert _squash(fact.quote) in _squash(page.text)


# -------------------------------------------- cleaning acceptance on the fixture
@pytest.mark.parametrize("file", [corpus.INSPECTION, corpus.BUNDLE], ids=lambda f: f.name)
def test_no_running_header_or_footer_survives_cleaning(file):
    raw = parsed(file)
    assert any(corpus.RUNNING_LINES[0] in p.text for p in raw.pages)  # it was there
    for page in cleaned(file).pages:
        for line in corpus.RUNNING_LINES:
            assert line not in page.text, (page.page, line)
        assert not re.search(r"Page \d+ of \d+", page.text)


def test_the_twice_repeated_finding_line_is_kept():
    pages = cleaned(corpus.INSPECTION).pages
    holders = [p.page for p in pages if corpus.REPEATED_LINE in p.text]
    assert holders == [3, 4]
    # it sits in the footer zone of both pages, so only its frequency saves it
    for number in holders:
        assert pages[number - 1].text.rstrip().endswith(corpus.REPEATED_LINE)


def test_letter_spaced_headings_are_repaired_in_the_estimate():
    text = "\n".join(p.text for p in cleaned(corpus.ESTIMATE).pages)
    for spaced, repaired in corpus.LETTER_SPACED.items():
        assert spaced not in text
        assert repaired in text.splitlines()


def test_hyphenated_line_ends_are_joined():
    standards = cleaned(corpus.INSPECTION).pages[9].text
    assert "not required to determine the remaining life" in _squash(standards)
    assert "licensed professional before" in _squash(standards)
    assert "￾" not in standards


def test_no_text_page_stays_flagged_after_cleaning():
    pages = cleaned(corpus.BUNDLE).pages
    assert [p.page for p in pages if not p.has_text_layer] == list(corpus.BUNDLE.no_text_pages)
    assert pages[8].text == ""


def test_mixed_bundle_cleans_to_content_once_segmented():
    from hal9000.knowledge.ingest.clean import clean_pages
    from hal9000.knowledge.ingest.segment import segment_file

    pages = parsed(corpus.MIXED).pages
    documents = segment_file(pages, file_sha="0" * 64, use_llm=False).documents
    result = clean_pages(pages, file_sha="0" * 64, parser="pdfium", documents=documents)
    text = "\n".join(page.text for page in result.pages)
    assert "Docusign" not in text
    assert not re.search(r"(?im)page \d+ of \d+\s*$", text)
    assert not re.search(r"(?m)^(Invoice|Estimate) #\d+", text)
    # the four-page pest report's form rows are its own furniture, wherever they sit
    for page in result.pages[13:17]:
        assert "NUMBER OF PAGES" not in page.text and " of 4" not in page.text
    # content stays: labels, line items, the letter's letterhead (one page, nothing repeats)
    for page in result.pages[2:6]:
        assert page.text.count("Moderate") == 2 and page.text.count("Recommendation") == 2
    assert "Invoice line 2.3: pipe work item 3" in text
    assert "Lic # 000111" in result.pages[20].text
