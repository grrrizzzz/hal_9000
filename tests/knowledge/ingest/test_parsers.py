"""Parsers (plan §2, §5.1 step 4; handoff KWP1): pages, text layers, pseudo-pages, limits."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hal9000.knowledge.contracts import Trace
from hal9000.knowledge.ingest.parsers import (
    DoclingParser,
    DocxParser,
    HtmlParser,
    ParsedFile,
    ParsedPage,
    ParseError,
    Parser,
    ParseTimeout,
    PdfiumParser,
    Section,
    TextParser,
    get_parser,
    pages_from_sections,
    parse_file,
    parser_for,
    split_text,
    top_level,
)
from hal9000.knowledge.ingest.parsers.docling import docling_available
from hal9000.knowledge.obs.timing import start_trace
from tests.knowledge.fixtures import corpus
from tests.knowledge.ingest.conftest import parsed


# ----------------------------------------------------------------------- PDF
@pytest.mark.parametrize("file", corpus.PDF_FILES, ids=lambda f: f.name)
def test_pdf_page_count_and_text_layer(file):
    result = parsed(file)
    assert result.parser == "pdfium"
    assert result.page_count == file.page_count
    assert [p.page for p in result.pages] == list(range(1, file.page_count + 1))
    flagged = tuple(p.page for p in result.pages if not p.has_text_layer)
    assert flagged == file.no_text_pages
    for page in result.pages:
        assert bool(page.text.strip()) == page.has_text_layer
    assert result.pseudo_pages is False


def test_the_photo_page_has_no_text_layer():
    page = parsed(corpus.INSPECTION).pages[8]
    assert page.page == 9 and page.has_text_layer is False and page.text == ""


def test_pdfium_extracts_findings_on_their_pages():
    pages = parsed(corpus.INSPECTION).pages
    for finding in corpus.FINDINGS:
        assert finding.heading in pages[finding.page - 1].text


def test_letter_spaced_headings_come_out_letter_spaced():
    text = parsed(corpus.ESTIMATE).pages[0].text
    assert "D ES C R I PT I O N O F W O R K" in text


def test_corrupt_pdf_fails_with_a_readable_error(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.7\n1 0 obj << /Type /Catalog >> endobj\ngarbage")
    with pytest.raises(
        ParseError, match=r"Could not parse broken\.pdf: the file is not a readable PDF"
    ):
        parse_file(bad, isolate=False)


def test_corrupt_pdf_fails_readably_in_the_isolated_worker(tmp_path):
    bad = tmp_path / "truncated.pdf"
    bad.write_bytes(corpus.INVOICE.path.read_bytes()[:300])
    with pytest.raises(ParseError, match=r"truncated\.pdf: the file is not a readable PDF"):
        parse_file(bad, timeout_s=60)


def test_password_protected_pdf_is_reported(tmp_path, monkeypatch):
    import pypdfium2 as pdfium

    from hal9000.knowledge.ingest.parsers.pdfium import open_pdf

    def refuse(_path):
        raise pdfium.PdfiumError("Failed to load document (PDFium: Incorrect password error).")

    monkeypatch.setattr(pdfium, "PdfDocument", refuse)
    with pytest.raises(ParseError, match="locked.pdf: the PDF is password-protected"):
        open_pdf(tmp_path / "locked.pdf")


# ---------------------------------------------------------- the time limit
class SlowParser:
    """Sleeps longer than the limit; module level so the spawned child can import it."""

    name = "slow"

    def parse(self, path: Path) -> ParsedFile:
        time.sleep(30)
        return ParsedFile(
            parser=self.name, pages=[ParsedPage(page=1, text="late")]
        )  # pragma: no cover


class CrashingParser:
    name = "crash"

    def parse(self, path: Path) -> ParsedFile:  # pragma: no cover - runs in the child
        import os

        os._exit(3)


class BuggyParser:
    name = "buggy"

    def parse(self, path: Path) -> ParsedFile:  # pragma: no cover - runs in the child
        raise KeyError("unexpected")


def test_parsing_is_bounded_by_a_timeout():
    started = time.monotonic()
    with pytest.raises(ParseTimeout, match=r"longer than 1 s"):
        parse_file(corpus.INVOICE.path, parser=SlowParser(), timeout_s=1)
    assert time.monotonic() - started < 15


def test_a_crashing_parser_fails_readably():
    with pytest.raises(ParseError, match=r"the crash parser crashed \(exit code 3\)"):
        parse_file(corpus.INVOICE.path, parser=CrashingParser(), timeout_s=60)


def test_an_unexpected_parser_error_is_wrapped():
    with pytest.raises(ParseError, match=r"invoice\.pdf: KeyError"):
        parse_file(corpus.INVOICE.path, parser=BuggyParser(), timeout_s=60)


def test_isolated_parse_returns_the_same_pages_as_in_process():
    isolated = parse_file(corpus.ESTIMATE.path, timeout_s=60)
    assert isolated == parsed(corpus.ESTIMATE)


def test_timeout_defaults_come_from_knowledge_yaml(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "knowledge.yaml").write_text(
        "ingest:\n  parse_timeout_seconds: 1\n  pseudo_page_chars: 500\n"
    )
    monkeypatch.setenv("HAL_CONFIG_DIR", str(config_dir))
    with pytest.raises(ParseTimeout, match="longer than 1 s"):
        parse_file(corpus.INVOICE.path, parser=SlowParser())
    assert parse_file(corpus.TEXT.path, isolate=False).page_count > 3


def test_parse_is_timed_as_the_parse_stage():
    with start_trace(Trace()) as trace:
        parse_file(corpus.INVOICE.path, isolate=False)
    assert [timing.stage for timing in trace.stages] == ["parse"]


# ------------------------------------------------------- selecting parsers
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.pdf", PdfiumParser),
        ("A.PDF", PdfiumParser),
        ("b.docx", DocxParser),
        ("c.html", HtmlParser),
        ("c.htm", HtmlParser),
        ("d.md", TextParser),
        ("e.txt", TextParser),
    ],
)
def test_parser_for_picks_by_suffix(name, expected):
    assert isinstance(parser_for(name), expected)


def test_parser_for_can_pick_docling_for_pdfs():
    assert isinstance(parser_for("x.pdf", pdf_parser="docling"), DoclingParser)


@pytest.mark.parametrize("name", ["photo.jpg", "sheet.xlsx", "noext"])
def test_unsupported_types_fail_readably(name):
    with pytest.raises(ParseError, match="not supported"):
        parser_for(name)


def test_unknown_parser_name():
    with pytest.raises(ParseError, match="Unknown parser 'ocr'"):
        get_parser("ocr")


def test_parsers_satisfy_the_protocol():
    for name in ("pdfium", "docling", "docx", "html", "text"):
        assert isinstance(get_parser(name), Parser)


def test_missing_file(tmp_path):
    with pytest.raises(ParseError, match="does not exist"):
        parse_file(tmp_path / "gone.pdf", isolate=False)


def test_empty_text_file_fails_readably(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n\n")
    with pytest.raises(ParseError, match="no text was found"):
        parse_file(empty, isolate=False)


def test_binary_text_file_fails_readably(tmp_path):
    binary = tmp_path / "blob.txt"
    binary.write_bytes(b"\x00\x01\x02binary")
    with pytest.raises(ParseError, match="binary"):
        parse_file(binary, isolate=False)


def test_corrupt_docx_fails_readably(tmp_path):
    bad = tmp_path / "broken.docx"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(ParseError, match="not a readable Word document"):
        parse_file(bad, isolate=False)


def test_latin1_text_is_decoded(tmp_path):
    path = tmp_path / "old.txt"
    path.write_bytes("Caf\xe9 wiring notes".encode("latin-1"))
    assert parse_file(path, isolate=False).pages[0].text == "Café wiring notes"


# ------------------------------------------------------------ pseudo-pages
def test_docx_with_three_top_level_headings_yields_three_pages():
    result = parsed(corpus.DOCX)
    assert result.pseudo_pages is True
    assert result.page_count == 3
    for page, heading in zip(result.pages, corpus.DOCX.page_headings):
        assert heading in page.text.splitlines()[:4]
    # the title and preamble join the first page; the level-2 heading stays inside it
    assert result.pages[0].text.startswith("Harborline HP-3000")
    assert "Checking the filter" in result.pages[0].text
    # tables are read in order as "cell | cell" rows
    assert "Heat | 68 F" in result.pages[1].text


def test_heading_less_7000_character_text_yields_three_pages():
    text = corpus.TEXT.path.read_text()
    assert 6500 <= len(text) <= 7500 and "#" not in text
    result = parsed(corpus.TEXT)
    assert result.page_count == 3
    assert all(len(page.text) <= 3100 for page in result.pages)
    assert "".join(page.text for page in result.pages) == text  # nothing lost


def test_html_is_split_at_its_repeated_top_level_heading_and_drops_scripts():
    result = parsed(corpus.HTML)
    assert result.page_count == 3
    for page, heading in zip(result.pages, corpus.HTML.page_headings):
        assert heading in page.text.splitlines()[:4]
    everything = "\n".join(page.text for page in result.pages)
    for leaked in ("trackingPixel", "console.log", "font-family", "Enable JavaScript"):
        assert leaked not in everything
    assert "1 | Front lawn | Rotors" in result.pages[0].text
    assert (
        "Shut the backflow valve in November, then blow out the lines at 50 psi.\nRestart in March."
        in result.pages[2].text
    )


def test_markdown_is_split_at_its_top_level_heading_not_inside_code():
    result = parsed(corpus.MARKDOWN)
    assert result.page_count == 3
    assert result.pages[0].text.startswith("# Garage Door Opener")
    assert "# this line is inside a code block" in result.pages[1].text
    for page, heading in zip(result.pages, corpus.MARKDOWN.page_headings):
        assert heading in page.text


def test_pseudo_page_size_is_configurable(tmp_path):
    result = parse_file(corpus.TEXT.path, isolate=False, max_chars=1000)
    assert result.page_count >= -(-len(corpus.TEXT.path.read_text()) // 1000)  # ceil: 8
    assert all(len(page.text) <= 1000 for page in result.pages)


def test_long_top_level_sections_are_split_further():
    long = "word " * 1500  # 7,500 characters under one heading
    sections = [Section(1, "# A\n" + long), Section(1, "# B\nshort\n")]
    pages = pages_from_sections(sections, max_chars=3000)
    assert len(pages) == 4
    assert pages[-1].text == "# B\nshort\n"


def test_top_level_prefers_a_repeated_level():
    assert top_level([0, 1, 2, 2, 2]) == 2
    assert top_level([1, 1, 2]) == 1
    assert top_level([3]) == 3
    assert top_level([0, 0]) is None


@pytest.mark.parametrize("size", [1, 50, 999, 3000, 7000])
def test_split_text_is_lossless_and_bounded(size):
    text = ("Sentence number one. Another one follows here.\n" * 200)[:size]
    pieces = split_text(text, 1000)
    assert "".join(pieces) == text
    assert all(0 < len(piece) <= 1000 for piece in pieces)


@pytest.mark.parametrize(
    ("size", "limit", "pieces"),
    [(7000, 3000, 3), (6100, 3000, 3), (3001, 3000, 2), (10000, 1000, 12)],
)
def test_split_text_balances_its_pieces(size, limit, pieces):
    text = ("A sentence of house notes. " * 1000)[:size]
    parts = split_text(text, limit)
    assert len(parts) == pieces
    lengths = [len(part) for part in parts]
    # balanced: every piece is near an even share, not limit-limit-remainder
    share = size / pieces
    assert all(abs(length - share) <= limit // 5 + 30 for length in lengths), lengths


def test_split_text_prefers_paragraph_breaks():
    text = "a" * 900 + "\n\n" + "b" * 900 + " c" * 200
    pieces = split_text(text, 1000)
    assert pieces[0].endswith("\n\n")


def test_split_text_rejects_nonsense():
    with pytest.raises(ValueError):
        split_text("x", 0)
    assert split_text("", 10) == []


def test_a_text_without_any_break_is_still_split():
    pieces = split_text("x" * 2500, 1000)
    assert "".join(pieces) == "x" * 2500 and all(len(p) <= 1000 for p in pieces)


# ------------------------------------------------------------------ docling
def test_docling_parser_explains_when_it_is_not_installed(monkeypatch):
    from hal9000.knowledge.ingest.parsers import docling as module

    monkeypatch.setattr(module, "docling_available", lambda: False)
    with pytest.raises(ParseError, match="uv sync --group knowledge-docling"):
        DoclingParser().parse(corpus.INVOICE.path)


def test_importing_the_docling_parser_does_not_import_docling():
    import sys

    assert DoclingParser.name == "docling"
    assert "docling" not in sys.modules or docling_available()


@pytest.mark.models
@pytest.mark.skipif(not docling_available(), reason="the knowledge-docling group is not installed")
def test_docling_parses_the_fixture_inspection():  # pragma: no cover - models only
    result = DoclingParser().parse(corpus.INSPECTION.path)
    assert result.page_count == corpus.INSPECTION.page_count
    assert [p.page for p in result.pages if not p.has_text_layer] == list(
        corpus.INSPECTION.no_text_pages
    )
    assert "3.1.2" in result.pages[6].text
