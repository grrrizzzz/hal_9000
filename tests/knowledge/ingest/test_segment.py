"""Segmentation (plan §5.1 step 6; handoff KWP1): heuristics, the LLM path, fallback, cache."""

from __future__ import annotations

import json
from datetime import date

import pytest

from hal9000.cost.metering import BudgetExceeded
from hal9000.knowledge.contracts import Trace
from hal9000.knowledge.ingest.parsers import ParsedPage
from hal9000.knowledge.ingest.segment import (
    SEGMENT_TIER,
    TOOL_NAME,
    InMemorySegmentCache,
    build_prompt,
    find_date,
    heuristic_documents,
    page_signals,
    parse_reply,
    propose_boundaries,
    segment_file,
)
from hal9000.knowledge.obs.timing import start_trace
from tests.fakes import FakeLLM, tool_call
from tests.knowledge.fixtures import corpus
from tests.knowledge.ingest.conftest import parsed

SHA = "b" * 64


def _ranges(documents) -> list[tuple[int, int]]:
    return [(d.page_start, d.page_end) for d in documents]


def _expected_ranges(file: corpus.CorpusFile) -> list[tuple[int, int]]:
    return [(d.page_start, d.page_end) for d in file.documents]


def _pages(*texts: str) -> list[ParsedPage]:
    return [
        ParsedPage(page=n, text=text, has_text_layer=bool(text.strip()))
        for n, text in enumerate(texts, start=1)
    ]


# ------------------------------------------------------------ heuristics
def test_heuristics_find_exactly_the_bundle_documents():
    pages = parsed(corpus.BUNDLE).pages
    documents = heuristic_documents(pages, file_sha=SHA, name="bundle.pdf")
    assert _ranges(documents) == _expected_ranges(corpus.BUNDLE)
    reasons = [b.reason for b in propose_boundaries(pages)]
    assert reasons == ["start of file", "page 1 of 2", "page 1 of 1"]


def test_heuristic_labels_on_the_bundle():
    documents = heuristic_documents(parsed(corpus.BUNDLE).pages, file_sha=SHA)
    for got, want in zip(documents, corpus.BUNDLE.documents):
        assert (got.doc_type, got.vendor, got.doc_date) == (
            want.doc_type,
            want.vendor,
            want.doc_date,
        )
        assert got.title == want.title
        assert got.meta_source == "auto" and got.file_sha == SHA


@pytest.mark.parametrize(
    "file",
    [f for f in corpus.ALL_FILES if f is not corpus.MIXED and f not in corpus.REVIEW_FILES],
    ids=lambda f: f.name,
)
def test_each_fixture_file_segments_to_its_documents(file):
    result = parsed(file)
    segmented = segment_file(
        result.pages, file_sha=SHA, name=file.name, pseudo_pages=result.pseudo_pages
    )
    assert segmented.source == "heuristic"
    assert _ranges(segmented.documents) == _expected_ranges(file)
    for got, want in zip(segmented.documents, file.documents):
        assert got.doc_type == want.doc_type
        assert got.title == want.title


def test_heuristics_split_the_unnumbered_bundle():
    pages = parsed(corpus.UNNUMBERED).pages
    boundaries = propose_boundaries(pages)
    assert [b.page for b in boundaries] == [1, 2, 3, 5]
    assert boundaries[1].reason == "form header INVOICE repeats"  # same vendor, no number
    documents = heuristic_documents(pages, file_sha=SHA)
    got = [(d.page_start, d.page_end, d.doc_type, d.vendor, d.doc_date, d.title) for d in documents]
    want = [
        (d.page_start, d.page_end, d.doc_type, d.vendor, d.doc_date, d.title)
        for d in corpus.UNNUMBERED.documents
    ]
    assert got == want


@pytest.mark.parametrize(
    "heading", ["Electrical Connections", "Plumbing Services Overview", "Roofing Services"]
)
def test_a_trade_heading_below_the_letterhead_is_not_a_new_vendor(heading):
    pages = _pages(
        "Alpha Roofing Inc.\nESTIMATE\nEstimate No. E-7\nScope",
        f"Scope continued\nTear-off\n{heading}\nMore scope",
    )
    assert page_signals(pages[1]).vendor is None
    assert [b.page for b in propose_boundaries(pages)] == [1]


def test_a_legal_suffix_names_a_vendor_below_the_top_lines():
    page = ParsedPage(page=1, text="RECEIPT\nDate: May 1, 2025\nIssued by\nOsprey Glass LLC")
    assert page_signals(page).vendor == "Osprey Glass LLC"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Invoice # BE-10442", (("invoice", "BE-10442"),)),
        ("Estimate No. TPR-1", (("estimate", "TPR-1"),)),
        ("INVOICE\nno number here", ()),
        ("Invoice number: pending", ()),
        ("Report number: CHI-2025-0314", (("report", "CHI-2025-0314"),)),
    ],
)
def test_reference_numbers(text, expected):
    assert page_signals(ParsedPage(page=1, text=text)).references == expected


def test_an_amount_line_is_not_a_form_header():
    signals = page_signals(
        ParsedPage(page=1, text="Scope\nProposal total: $1,480.00\nInvoice total: 5")
    )
    assert signals.form is None


def test_a_repeated_form_header_with_the_same_reference_continues():
    pages = _pages(
        "Osprey Glass LLC\nINVOICE\nInvoice # OG-1\nitems",
        "Osprey Glass LLC\nINVOICE\nInvoice # OG-1\nmore items",
        "Osprey Glass LLC\nINVOICE\nno number here",
    )
    reasons = [(b.page, b.reason) for b in propose_boundaries(pages)]
    assert reasons == [(1, "start of file"), (3, "form header INVOICE repeats")]


def test_pseudo_page_sections_are_not_documents():
    # the DOCX's "Warranty Registration" section would otherwise look like a warranty form
    pages = parsed(corpus.DOCX).pages
    assert len(propose_boundaries(pages)) == 2
    assert len(heuristic_documents(pages, file_sha=SHA, pseudo_pages=True)) == 1


def test_a_page_without_text_inside_a_numbered_document_is_absorbed():
    pages = _pages(
        "Acme\nPage 1 of 3", "", "Acme\nPage 3 of 3", "INVOICE\nBeta Plumbing Inc.\nno footer"
    )
    assert _ranges(heuristic_documents(pages, file_sha=SHA)) == [(1, 3), (4, 4)]


def test_numbering_that_changes_total_starts_a_document():
    pages = _pages("Page 1 of 3", "Page 2 of 3", "Page 2 of 4", "Page 3 of 4")
    boundaries = propose_boundaries(pages)
    assert [b.page for b in boundaries] == [1, 3]
    assert boundaries[1].reason == "page numbering changes to 2 of 4"


def test_unnumbered_documents_split_on_vendor_and_form_headers():
    pages = _pages(
        "Alpha Roofing Inc.\nESTIMATE\nReplace shingles",
        "Alpha Roofing Inc.\nterms continue here",
        "Alpha Roofing Inc.\nINVOICE\nShingles replaced",
        "Gamma Pest Control\nWOOD DESTROYING PESTS AND ORGANISMS INSPECTION REPORT\nNo activity",
        "plain continuation text",
    )
    boundaries = propose_boundaries(pages)
    assert [b.page for b in boundaries] == [1, 3, 4]
    assert boundaries[1].reason.startswith("form header INVOICE")
    assert "form header" in boundaries[2].reason or "vendor" in boundaries[2].reason
    types = [d.doc_type for d in heuristic_documents(pages, file_sha=SHA)]
    assert types == ["estimate", "invoice", "pest_report"]


def test_same_vendor_same_type_split_on_a_new_reference_number():
    pages = _pages(
        "Kestrel Plumbing Inc.\nESTIMATE\nEstimate No. E-1001\nRe-pipe",
        "Kestrel Plumbing Inc.\nESTIMATE\nEstimate No. E-1001\ncontinued",
        "Kestrel Plumbing Inc.\nESTIMATE\nEstimate No. E-1002\nRepairs",
    )
    boundaries = propose_boundaries(pages)
    assert [b.page for b in boundaries] == [1, 3]
    assert boundaries[1].reason == "new estimate number E-1002"
    assert [d.title for d in heuristic_documents(pages, file_sha=SHA)] == [
        "Estimate E-1001",
        "Estimate E-1002",
    ]


def test_vendor_change_without_a_form_header():
    pages = _pages("Alpha Roofing Inc.\nnotes", "Larkspur Mechanical, Inc.\nnotes")
    assert propose_boundaries(pages)[1].reason == "vendor changes to Larkspur Mechanical, Inc."


def test_a_new_letterhead_after_a_finished_numbered_document():
    pages = _pages("Report\nPage 1 of 1", "Delta Electric Co.\nnotes without numbering")
    assert (
        propose_boundaries(pages)[1].reason
        == "letterhead Delta Electric Co. where the document had none"
    )
    pages = _pages("Report\nPage 1 of 1", "notes without numbering")
    assert propose_boundaries(pages)[1].reason == "after page 1 of 1"


def test_nothing_detected_means_one_document():
    pages = _pages("some notes", "more notes", "", "last notes")
    documents = heuristic_documents(pages, file_sha=SHA, name="notes.pdf")
    assert _ranges(documents) == [(1, 4)]
    assert documents[0].doc_type == "other"
    assert documents[0].title == "some notes"


def test_a_file_with_no_text_at_all_is_one_untitled_document():
    documents = heuristic_documents(_pages("", ""), file_sha=SHA, name="scan.pdf")
    assert _ranges(documents) == [(1, 2)]
    assert documents[0].title == "scan"
    assert heuristic_documents([], file_sha=SHA) == []


def test_page_signals_read_markers_references_and_letter_spacing():
    signals = page_signals(
        ParsedPage(
            page=1,
            text="Tidewater Plumbing & Rooter, Inc.\nE S T I M A T E\nEstimate No. TPR-1\nPage 1 of 2",
        )
    )
    assert signals.marker == (1, 2)
    assert signals.form == "estimate" and signals.form_line == "ESTIMATE"
    assert signals.vendor == "Tidewater Plumbing & Rooter, Inc."
    assert signals.references == (("estimate", "TPR-1"),)


@pytest.mark.parametrize(
    "line",
    [
        "Docusign Envelope ID: 1234",
        "3.1.2 Plumbing – Water Heater: Valve Leak",
        "We recommend a plumbing contractor.",
        "License C36-000000 Plumbing",
    ],
)
def test_lines_that_are_not_vendors(line):
    assert page_signals(ParsedPage(page=1, text=line)).vendor is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Inspection date: March 14, 2025", date(2025, 3, 14)),
        ("Dated Sept. 3, 2024 and 2025-01-02", date(2024, 9, 3)),
        ("on 2023-06-22", date(2023, 6, 22)),
        ("signed 7/8/2023", date(2023, 7, 8)),
        ("signed 07/08/23", date(2023, 7, 8)),
        ("no date here, only 13/45/2023 and Smarch 3, 2020", None),
        ("Febuary 30, 2024 then May 1, 2024", date(2024, 5, 1)),
    ],
)
def test_find_date(text, expected):
    assert find_date(text) == expected


# -------------------------------------------------------------- LLM path
def _reply(documents: list[dict]) -> str:
    return tool_call(TOOL_NAME, documents=documents)


GOOD = [
    {"page_start": 1, "page_end": 12, "doc_type": "inspection_report", "vendor": "Coastline Home Inspections",
     "doc_date": "2025-03-14", "title": "Home inspection report"},
    {"page_start": 13, "page_end": 14, "doc_type": "estimate", "vendor": "Tidewater Plumbing & Rooter, Inc.",
     "doc_date": "2025-03-20", "title": "Plumbing estimate TPR-58213"},
    {"page_start": 15, "page_end": 15, "doc_type": "invoice", "vendor": " Bayside Electric Co. ",
     "doc_date": None, "title": "Electrical invoice BE-10442"},
]  # fmt: skip


def test_llm_labels_are_applied():
    llm = FakeLLM([_reply(GOOD)])
    result = segment_file(
        parsed(corpus.BUNDLE).pages, file_sha=SHA, generate=llm, name="bundle.pdf"
    )
    assert result.source == "llm" and result.llm_calls == 1 and result.warnings == []
    assert _ranges(result.documents) == _expected_ranges(corpus.BUNDLE)
    labels = [(d.doc_type, d.vendor, d.doc_date, d.title) for d in result.documents]
    assert labels == [
        (
            "inspection_report",
            "Coastline Home Inspections",
            date(2025, 3, 14),
            "Home inspection report",
        ),
        (
            "estimate",
            "Tidewater Plumbing & Rooter, Inc.",
            date(2025, 3, 20),
            "Plumbing estimate TPR-58213",
        ),
        ("invoice", "Bayside Electric Co.", None, "Electrical invoice BE-10442"),
    ]
    assert llm.tiers == [SEGMENT_TIER] == ["fast"]


def test_the_llm_may_override_the_heuristic_boundaries():
    merged = [dict(GOOD[0], page_end=15)]
    result = segment_file(
        parsed(corpus.BUNDLE).pages, file_sha=SHA, generate=FakeLLM([_reply(merged)])
    )
    assert _ranges(result.documents) == [(1, 15)]


def test_plain_json_and_fenced_json_replies_are_accepted():
    payload = json.dumps({"documents": GOOD})
    assert len(parse_reply(payload, 15)) == 3
    assert len(parse_reply(f"```json\n{payload}\n```", 15)) == 3


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        ([dict(GOOD[0], page_end=11), *GOOD[1:]], "page 12 was expected"),
        (GOOD[:2], "end at page 14, not 15"),
        ([dict(GOOD[0], doc_type="receipt"), *GOOD[1:]], "doc_type"),
        ([dict(GOOD[0], doc_date="14/03/2025"), *GOOD[1:]], "doc_date"),
        ([dict(GOOD[0], title=""), *GOOD[1:]], "title"),
        ([dict(GOOD[0], extra=1), *GOOD[1:]], "extra"),
        ([dict(GOOD[0], page_start=0), *GOOD[1:]], "page_start"),
        ([], "documents"),
    ],
)
def test_malformed_replies_are_rejected(documents, message):
    with pytest.raises(ValueError, match=message):
        parse_reply(_reply(documents), 15)


def test_non_json_reply_is_rejected():
    with pytest.raises(ValueError, match="not JSON"):
        parse_reply("Sure! Here are the documents: ...", 15)


def test_a_malformed_reply_is_retried_once_with_the_error():
    llm = FakeLLM(["not json at all", _reply(GOOD)])
    result = segment_file(parsed(corpus.BUNDLE).pages, file_sha=SHA, generate=llm)
    assert result.source == "llm" and result.llm_calls == 2
    assert len(result.warnings) == 1 and "not JSON" in result.warnings[0]
    retry = llm.calls[1].messages
    assert retry[-2] == {"role": "assistant", "content": "not json at all"}
    assert "invalid" in retry[-1]["content"] and TOOL_NAME in retry[-1]["content"]


def test_two_malformed_replies_fall_back_to_heuristics():
    bad = _reply([dict(GOOD[0], page_end=14)])
    llm = FakeLLM([bad, "{}"])
    result = segment_file(parsed(corpus.BUNDLE).pages, file_sha=SHA, generate=llm)
    assert result.source == "heuristic" and result.llm_calls == 2
    assert _ranges(result.documents) == _expected_ranges(corpus.BUNDLE)
    assert result.warnings[-1] == "segmentation LLM replies were invalid twice; using heuristics"
    assert llm.responses == []  # exactly two calls, no third


@pytest.mark.parametrize(
    "error",
    [BudgetExceeded("monthly cap reached"), ConnectionError("offline"), RuntimeError("boom")],
)
def test_an_llm_error_falls_back_without_raising(error):
    def failing(prompt, tier):
        raise error

    result = segment_file(parsed(corpus.BUNDLE).pages, file_sha=SHA, generate=failing)
    assert result.source == "heuristic" and result.llm_calls == 1
    assert _ranges(result.documents) == _expected_ranges(corpus.BUNDLE)
    assert type(error).__name__ in result.warnings[0]


def test_no_llm_uses_heuristics_only():
    llm = FakeLLM([])
    result = segment_file(parsed(corpus.BUNDLE).pages, file_sha=SHA, generate=llm, use_llm=False)
    assert result.source == "heuristic" and llm.calls == []
    assert segment_file([], file_sha=SHA, generate=llm).documents == []


def test_labels_are_cached_by_file_sha():
    cache = InMemorySegmentCache()
    pages = parsed(corpus.BUNDLE).pages
    first = segment_file(pages, file_sha=SHA, generate=FakeLLM([_reply(GOOD)]), cache=cache)
    llm = FakeLLM([])
    second = segment_file(pages, file_sha=SHA, generate=llm, cache=cache)
    assert second.source == "cache" and llm.calls == [] and second.llm_calls == 0
    assert second.documents == first.documents
    # another file is a miss
    other = segment_file(pages, file_sha="c" * 64, generate=FakeLLM([_reply(GOOD)]), cache=cache)
    assert other.source == "llm" and len(cache) == 2


def test_heuristic_fallbacks_are_not_cached():
    cache = InMemorySegmentCache()
    pages = parsed(corpus.BUNDLE).pages
    segment_file(pages, file_sha=SHA, generate=FakeLLM(["x", "y"]), cache=cache)
    assert len(cache) == 0
    assert (
        segment_file(pages, file_sha=SHA, generate=FakeLLM([_reply(GOOD)]), cache=cache).source
        == "llm"
    )


def test_cached_documents_are_copies():
    cache = InMemorySegmentCache()
    pages = parsed(corpus.BUNDLE).pages
    first = segment_file(pages, file_sha=SHA, generate=FakeLLM([_reply(GOOD)]), cache=cache)
    first.documents[0].title = "edited"
    assert cache.get(f"{SHA}:segment-v1")[0].title == "Home inspection report"


def test_the_prompt_is_one_strict_call_with_page_excerpts():
    pages = parsed(corpus.BUNDLE).pages
    proposal = heuristic_documents(pages, file_sha=SHA)
    prompt = build_prompt(pages, proposal, name="bundle.pdf")
    user = prompt.messages[1]["content"]
    assert "File: bundle.pdf (15 pages)" in user
    assert "--- page 9 ---\n[no text layer]" in user
    assert "Page 1 of 2" in user  # footers are what the heuristics saw too
    assert "DESCRIPTION OF WORK" in user  # letter spacing repaired for the model
    assert "- pages 13-14: estimate, 'Estimate TPR-58213'" in user
    assert "not instructions" in prompt.messages[0]["content"]
    (tool,) = prompt.tools
    assert tool["function"]["name"] == TOOL_NAME and tool["function"]["strict"] is True
    schema = tool["function"]["parameters"]["properties"]["documents"]["items"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "pest_report" in schema["properties"]["doc_type"]["enum"]


def test_long_files_keep_the_prompt_bounded():
    pages = _pages(*["word " * 2000] * 200)
    prompt = build_prompt(pages, heuristic_documents(pages, file_sha=SHA))
    assert len(prompt.messages[1]["content"]) < 100_000


def test_pseudo_page_files_get_one_labelled_document():
    pages = parsed(corpus.DOCX).pages
    label = [{"page_start": 1, "page_end": 3, "doc_type": "manual", "vendor": "Harborline",
              "doc_date": None, "title": "Heat pump owner's guide"}]  # fmt: skip
    two = [dict(label[0], page_end=2), dict(label[0], page_start=3)]
    llm = FakeLLM([_reply(two), _reply(label)])
    result = segment_file(pages, file_sha=SHA, generate=llm, pseudo_pages=True)
    assert result.source == "llm" and result.llm_calls == 2
    assert "one document, but the reply has 2" in result.warnings[0]
    assert [(d.doc_type, d.vendor) for d in result.documents] == [("manual", "Harborline")]
    assert "return exactly one document" in llm.calls[0].messages[1]["content"]


def test_segment_is_timed_as_the_segment_stage():
    with start_trace(Trace()) as trace:
        segment_file(_pages("x"), file_sha=SHA)
    assert [timing.stage for timing in trace.stages] == ["segment"]


# ------------------------------------- closing-bundle patterns (mixed_bundle.pdf)
def test_heuristics_split_the_mixed_bundle_exactly():
    pages = parsed(corpus.MIXED).pages
    assert [b.page for b in propose_boundaries(pages)] == [
        d.page_start for d in corpus.MIXED.documents
    ]
    documents = heuristic_documents(pages, file_sha=SHA)
    assert _ranges(documents) == _expected_ranges(corpus.MIXED)
    typed = [d.doc_type for d in documents[: len(corpus.MIXED_TYPED)]]
    assert typed == [d.doc_type for d in corpus.MIXED_TYPED]


@pytest.mark.parametrize(
    ("text", "marker"),
    [
        ("Header\nBody\n12 GULL LN PORT ALDER 90000 04/02/2025 2 of 7", (2, 7)),
        ("LABELS NUMBER OF PAGES\n12 GULL LN 04/02/2025 1 of 3\nbody", (1, 3)),
        ("Estimate #77 Page 2 of 4\nbody", (2, 4)),
        ("(FORM RR PAGE 1 OF 2)\nbody", (1, 2)),
    ],
)
def test_page_markers_with_and_without_the_word_page(text, marker):
    assert page_signals(ParsedPage(page=1, text=text)).marker == marker


def test_a_bare_k_of_n_in_the_middle_of_a_page_is_not_a_marker():
    body = "\n".join(f"line {n}" for n in range(12))
    text = f"top\n{body[:40]}\nWe replaced 2 of 7 valves\n{body}\nbottom"
    assert page_signals(ParsedPage(page=1, text=text)).marker is None
    assert page_signals(ParsedPage(page=1, text="Covered 12/2 of 3 areas")).marker is None


def test_a_field_label_row_is_not_a_vendor():
    signals = page_signals(
        ParsedPage(page=1, text="BLDG NO. STREET CITY ZIP INSPECTION DATE NUMBER OF PAGES\nrow")
    )
    assert signals.vendor is None


def test_a_vendor_line_loses_its_page_number():
    signals = page_signals(ParsedPage(page=1, text="Harbor Point Home Inspections Page 3 of 9\nx"))
    assert signals.vendor == "Harbor Point Home Inspections"
    assert signals.marker == (3, 9)


def test_a_long_form_title_is_still_a_form_header():
    signals = page_signals(
        ParsedPage(page=1, text="Standardized Notice of Work Completed; Items Pending\nx")
    )
    assert signals.form == "completion_notice"


def test_letterhead_top_lines_are_recognised():
    letter = "Marisol Vega Finish Carpentry\n88 Kelp Ave.\nPort Alder, CA. 90000\nLic # 000111\n(555) 010-0444\nJune 3, 2025"
    signals = page_signals(ParsedPage(page=1, text=letter))
    assert signals.letterhead is True
    assert signals.contacts == frozenset({"5550100444"})
    assert signals.top_date == date(2025, 6, 3)
    plain = page_signals(ParsedPage(page=1, text="Findings continue here.\nMore findings."))
    assert plain.letterhead is False and plain.contacts == frozenset()


def test_a_letterhead_after_a_finished_numbered_document_starts_a_new_one():
    pages = _pages(
        "Report\nbody\n12 GULL LN 04/02/2025 1 of 3",
        "12 GULL LN 04/02/2025 2 of 3\nbody",
        "a last page whose footer was not captured",
        "A Person Carpentry\n88 Kelp Ave.\nPort Alder, CA 90000\nLic # 000111\nnote",
    )
    reasons = [(b.page, b.reason) for b in propose_boundaries(pages)]
    assert reasons[1:] == [(4, "letterhead after a finished numbered document")]


def test_a_changed_contact_block_splits_only_with_corroboration():
    # alone: a letter's last page names a managing company with another phone number
    letter = _pages(
        "Coastal Bluffs HOA\nPO Box 100, Seaview, CA 92000\n(555) 010-0101\nMarch 3, 2024\nDear Owner,",
        "Owner, 12 Example Lane\nMarch 3, 2024\nPage 2\nbody\nManaged by Sunset Management (555) 010-0199",
    )
    assert [b.page for b in propose_boundaries(letter)] == [1]
    # with a reference number on the page: a new document
    pages = _pages(
        "note one\nCall (555) 010-0444", "items\nInvoice # GA-1\nQuestions? Call (555) 010-0555"
    )
    assert [(b.page, b.reason) for b in propose_boundaries(pages)][1:] == [
        (2, "contact details change")
    ]


def test_a_changed_top_date_splits_only_with_corroboration():
    # alone: a dated service log keeps its weeks together
    log = _pages(
        *(f"Pool Service Log\nWeek of 2024-05-{d:02d}\nvisit notes" for d in (6, 13, 20, 27))
    )
    assert [b.page for b in propose_boundaries(log)] == [1]
    # with new contact details too: a new document
    pages = _pages(
        "June 3, 2025\nLetter one",
        "Letter one, page two",
        "July 9, 2025\nLetter two\n(555) 010-0777",
    )
    assert [(b.page, b.reason) for b in propose_boundaries(pages)][1:] == [
        (3, "contact details change")
    ]


def test_a_dated_page_after_an_undated_document_starts_a_new_one():
    pages = _pages(
        "Rivera Tile\nnote",
        "Tuesday, June 4, 2024\nKitchen journal",
        "Friday, June 7, 2024\nKitchen journal",
    )
    assert [(b.page, b.reason) for b in propose_boundaries(pages)][1:] == [
        (2, "a dated page after an undated document")
    ]


def test_step_and_figure_counters_are_not_page_markers():
    # one "Step 3 of 5" inside a manual must not hold every later page in the manual
    manual = [f"Dishwasher guide\nSection {k}\nbody" for k in range(1, 7)]
    manual[3] += "\nInstallation, Step 3 of 5"
    pages = _pages(*manual, "Bayside Electric Co.\nINVOICE\nInvoice # 5521\nitems")
    assert [b.page for b in propose_boundaries(pages)] == [1, 7]
    figures = page_signals(
        ParsedPage(page=1, text="Pump guide\nbody\nWiring diagram, Figure 1 of 3")
    )
    assert figures.marker is None


def test_a_bare_k_of_n_needs_a_consistent_neighbour():
    lone = _pages("Checklist\nbody\n12 GULL LN 04/02/2025 3 of 5", "next page\nbody")
    assert page_signals(lone[0]).marker == (3, 5)
    assert [b.page for b in propose_boundaries(lone)] == [1]  # not trusted: no neighbour
    pair = _pages(
        "intro\nbody", "form\n12 GULL LN 04/02/2025 1 of 2", "12 GULL LN 04/02/2025 2 of 2\nbody"
    )
    assert [(b.page, b.reason) for b in propose_boundaries(pair)][1:] == [(2, "page 1 of 2")]


@pytest.mark.parametrize(("total", "starts"), [(4, [1, 5]), (9, [1])])
def test_the_unfinished_document_guard_expires(total, starts):
    # "Page 2 of N" then unnumbered pages: it absorbs at most N - 2 of them
    pages = _pages(f"Page 1 of {total}\nA", f"Page 2 of {total}\nB", "c", "d", "INVOICE\nitems")
    assert [b.page for b in propose_boundaries(pages)] == starts


def test_a_form_word_followed_by_an_id_is_a_reference_not_a_heading():
    assert (
        page_signals(ParsedPage(page=1, text="Inspection Record - Permit B24-0117\nx")).form is None
    )
    assert (
        page_signals(ParsedPage(page=1, text="BUILDING PERMIT\nPermit No. B24-0117")).form
        == "permit"
    )


def test_a_form_header_where_the_document_had_none_starts_one():
    pages = _pages("Thank you for your business!\nreceipt", "LiftRight Opener\nOWNERS MANUAL\nbody")
    assert [(b.page, b.reason) for b in propose_boundaries(pages)][1:] == [
        (2, "form header OWNERS MANUAL")
    ]


# ------------------------------------------ the KWP1b reviewer's over-fitting probes
@pytest.mark.parametrize(
    "file", [corpus.REVIEW_B1, corpus.REVIEW_B1B, corpus.REVIEW_B2], ids=lambda f: f.name
)
def test_review_bundles_split_with_no_misses_and_at_most_one_false_split(file):
    found = {b.page for b in propose_boundaries(parsed(file).pages)}
    expected = {d.page_start for d in file.documents}
    assert expected - found == set()
    assert len(found - expected) <= 1


def test_review_b3_misses_only_pages_without_any_signal():
    file = corpus.REVIEW_B3
    found = {b.page for b in propose_boundaries(parsed(file).pages)}
    expected = {d.page_start for d in file.documents}
    assert found - expected == set()
    assert expected - found == set(corpus.REVIEW_B3_NO_SIGNAL)  # left to the LLM path


# ----------------------------------------------- heuristic labels (KWP1c)
_ENVELOPE = "Docusign Envelope ID: 5E7C1A90-3B2D-4C6E-9F10-A1B2C3D4E5F6"


def _only(pages):
    documents = heuristic_documents(pages, file_sha=SHA)
    assert len(documents) == 1, _ranges(documents)
    return documents[0]


def test_a_cover_title_wins_over_a_running_address_and_name_header():
    # every page opens with the street glued to the client's name and a vendor/page line
    header = "12 Gull Ln Jordan Reyes\nHarbor Point Inspections Page {k} of 4"
    cover = (
        "HARBOR POINT INSPECTIONS\n(555) 010-0100\ninfo@harborpoint.example\n"
        "https://www.harborpoint.example\n"
        "PROPERTY INSPECTION REPORT\n12 Gull Ln\nPort Alder, CA 90000\nJordan Reyes\n"
        f"JUNE 3, 2025\nInspector\nPat Lindqvist\n{_ENVELOPE}"
    )
    rest = "Roof\nThe roof covering is worn.\nPlumbing\nThe water heater is 13 years old."
    pages = _pages(
        f"{header.format(k=1)}\n{cover}",
        *(f"{header.format(k=k)}\n{rest}\n{_ENVELOPE}" for k in (2, 3, 4)),
    )
    document = _only(pages)
    assert (document.doc_type, document.title) == (
        "inspection_report",
        "Property Inspection Report",
    )


def test_the_title_can_follow_a_cover_page_and_a_table_of_contents():
    pages = _pages(
        "12 Gull Ln\nPort Alder, CA 90000\nOskar Lindqvist\nJune 3, 2025\nPage 1 of 3",
        "TABLE OF CONTENTS\n1: Roof\n2: Plumbing\n3: Electrical\nPage 2 of 3",
        "HOME INSPECTION REPORT\nSummary\nThe roof covering is worn.\nPage 3 of 3",
    )
    document = _only(pages)
    assert (document.doc_type, document.title) == ("inspection_report", "Home Inspection Report")


def test_envelope_lines_names_and_addresses_never_become_the_title():
    pages = _pages(
        f"{_ENVELOPE}\nOskar Lindqvist\n12 Gull Ln\nBackyard drainage plan\nSwale 1 runs east.",
        *(f"{_ENVELOPE}\nOskar Lindqvist\n12 Gull Ln\nSwale {k} runs east." for k in (2, 3)),
    )
    document = _only(pages)
    assert (document.doc_type, document.title) == ("other", "Backyard drainage plan")


def test_an_estimate_that_mentions_an_invoice_is_an_estimate():
    header = _only(_pages("Kestrel Plumbing Inc.\nESTIMATE\nRe-pipe\nAn invoice follows the work."))
    assert (header.doc_type, header.title) == ("estimate", "Estimate")
    # no form header: the estimate word in a heading still beats "invoice" and a total
    heading = _only(
        _pages(
            "Kestrel Plumbing Inc.\nWater heater replacement proposal\n"
            "This is not an invoice.\nTotal: $900.00"
        )
    )
    assert heading.doc_type == "estimate"
    # nor in the prose: no heading at all, so the title is "<Type> – <vendor>"
    prose = _only(
        _pages(
            "Kestrel Plumbing Inc.\nRe-pipe\n"
            "This written estimate holds for 30 days, and no invoice is sent until the work is done."
        )
    )
    assert (prose.doc_type, prose.title) == ("estimate", "Estimate – Kestrel Plumbing Inc.")


def test_a_pest_report_form_is_typed_and_titled_without_its_checkbox_rows():
    pages = _pages(
        "Inspector / License No.\n"
        "FULL REPORT PARTIAL REPORT FOLLOW-UP REPORT REINSPECTION REPORT\n"
        "SITE NOTES: Tag Location:\n"
        "Drywood Termite Findings Decay Findings\n"
        "Ask for copies of prior reports and completion notices\n"
        "Secondary Inspection\n"
        "What Does A Termite Inspection Report Cover?\n"
        "Questions about this report go to the state board that licenses pest companies.\n"
        "BLDG NO. STREET CITY ZIP INSPECTION DATE NUMBER OF PAGES\n"
        "12 GULL LN PORT ALDER 90000 06/16/2025 1 of 2",
        "Section 1 findings\nEvidence of drywood termites at the eaves.\n"
        "12 GULL LN PORT ALDER 90000 06/16/2025 2 of 2",
    )
    document = _only(pages)
    assert (document.doc_type, document.title) == ("pest_report", "Pest report – 2025-06-16")


def test_a_home_inspection_report_that_mentions_termites_is_not_a_pest_report():
    pages = _pages(
        "Licensed termite inspector on staff\nHOME INSPECTION REPORT\nDrywood termite evidence."
    )
    assert _only(pages).doc_type == "inspection_report"


def test_a_priced_letter_from_a_sole_trader_is_an_invoice_not_titled_by_name():
    pages = _pages(
        "Oskar Lindqvist\n88 Kelp Ave.\nPort Alder, CA. 90000\nLic # 000111\n(555) 010-0444\n"
        "June 3, 2025\nRe: 12 Gull Ln\nRepaired the eave boards noted in the pest report.\n"
        "Total: $900"
    )
    document = _only(pages)
    assert (document.doc_type, document.title) == ("invoice", "Invoice – 2025-06-03")


def test_a_running_heading_that_names_the_document_is_its_title():
    pages = _pages(
        *(
            f"HydroMax Variable Speed Pump\nInstallation and User Guide\nChapter {k}: Clean the"
            " filter screen with warm water.\n- {k} -"
            for k in (1, 2, 3)
        )
    )
    document = _only(pages)
    assert (document.doc_type, document.title) == ("manual", "Installation and User Guide")


@pytest.mark.parametrize(
    ("text", "doc_type"),
    [
        ("Sandpiper Pest Control, Inc.\nStandard notice of work completed", "completion_notice"),
        ("Wood destroying organisms findings\nSection 1", "pest_report"),
        ("Signed receipt for reports\nBuyer acknowledges", "disclosure_form"),
        ("Seller disclosure\nKnown defects", "disclosure_form"),
        ("Window cleaning quote\nFront and back", "estimate"),
        ("Bill to Pat Owner\nWindow cleaning", "invoice"),
        ("Window cleaning\nAmount due on receipt", "invoice"),
        ("Thank you for your business!\nPaid in full for gutter cleaning", "invoice"),
        ("LiftRight Opener\nOwner's guide", "manual"),
        ("Voltek Breakers Inc.\nProduct data sheet", "manual"),
        ("Roof coverage terms\nLimited warranty for ten years", "warranty"),
        ("City of Port Alder\nPermit for a patio cover", "permit"),
        ("Coastal Bluffs Homeowners Association\nDear Homeowner", "other"),
        ("Seaview Water District\nStatement date 01/15/2024", "other"),
    ],
)
def test_keyword_rules_type_documents(text, doc_type):
    assert _only(_pages(text)).doc_type == doc_type


@pytest.mark.parametrize(
    "line",
    [
        "Estimate No. E-1001",  # a reference, not a heading
        "Secondary Inspection",  # a title-case checkbox label
        "Why Order A Roof Report?",  # a question
        "Report Date: 06/16/2025",  # a dated label
        "Invoice total $900.00",  # an amount
        "FULL REPORT PARTIAL REPORT",  # a row of checkboxes
        "The inspection report follows.",  # a sentence
        "12 Gull Ln Report Services",  # an address
        "REPORT " + "X" * 120,  # too long
    ],
)
def test_lines_that_are_not_titles(line):
    document = _only(_pages(f"{line}\nHOME INSPECTION REPORT\nbody text here"))
    assert document.title == "Home Inspection Report"


def test_titles_are_at_most_120_characters():
    heading = (
        "Report on Subsurface Drainage Remediation Recommendations for the Northwest "
        "Retaining Wall Improvements"
    )
    assert 80 < len(heading) <= 120
    document = _only(_pages(f"{heading}\nbody"))
    assert document.title == heading
    long_first = "x" * 150
    assert len(_only(_pages(f"{long_first}\nbody")).title) <= 120


@pytest.mark.parametrize("file", [corpus.MIXED, *corpus.REVIEW_FILES], ids=lambda f: f.name)
def test_heuristic_types_on_the_synthetic_bundles(file):
    # the documents the boundaries find are typed like the ground truth, without an LLM
    documents = segment_file(parsed(file).pages, file_sha=SHA, use_llm=False).documents
    expected = {d.page_start: d.doc_type for d in file.documents}
    got = {d.page_start: d.doc_type for d in documents if d.page_start in expected}
    assert got == {start: expected[start] for start in got}
    assert all(len(d.title) <= 120 and "envelope" not in d.title.casefold() for d in documents)
