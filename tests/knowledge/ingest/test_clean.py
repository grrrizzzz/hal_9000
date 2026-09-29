"""Cleaning (plan §5.1 step 5; handoff KWP1 acceptance).

Written before :mod:`hal9000.knowledge.ingest.clean`: running header and
footer removal by frequency with digits normalised, letter-spacing and
hyphenation repair, whitespace, and the offset map back to pages.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hal9000.knowledge.contracts import Trace
from hal9000.knowledge.ingest.clean import (
    PAGE_SEPARATOR,
    PagedText,
    _segment,
    clean_pages,
    normalize_line,
    repair_letter_spacing,
    running_lines,
    source_offset,
    word_costs,
)
from hal9000.knowledge.ingest.parsers import ParsedPage
from hal9000.knowledge.obs.timing import start_trace

SHA = "a" * 64
HEADER = "Docusign Envelope ID: 5E7C1A90-3B2D-4C6E-9F10-A1B2C3D4E5F6"


def _page(number: int, body: str, total: int = 10) -> ParsedPage:
    text = f"{HEADER}\r\nAcme Inspections | 12 Fake St\r\n{body}\r\nPage {number} of {total}"
    return ParsedPage(page=number, text=text)


def _pages(bodies: list[str]) -> list[ParsedPage]:
    return [_page(index, body, len(bodies)) for index, body in enumerate(bodies, start=1)]


# ------------------------------------------------------------- header/footer
def test_normalize_line_folds_digits_case_and_spaces():
    assert normalize_line("  Page 12   of 58 ") == normalize_line("page 3 of 7")
    assert normalize_line("Page 12 of 58") == "page # of #"


def test_normalize_line_folds_uuid_and_hex_ids():
    one = normalize_line("Docusign Envelope ID: 5E7C1A90-3B2D-4C6E-9F10-A1B2C3D4E5F6")
    two = normalize_line("Docusign Envelope ID: 0B99F2DE-71AA-4E0C-8D3B-FF10E2A7C901")
    assert one == two == "docusign envelope id: <id>"
    assert normalize_line("Ref 9F3A2B7C11D0E4") == "ref <id>"
    # ordinary words made of a-f letters are not IDs
    assert normalize_line("Faced a bad decade") == "faced a bad decade"


def test_headers_from_different_envelopes_count_as_one_running_line():
    envelopes = [
        "5E7C1A90-3B2D-4C6E-9F10-A1B2C3D4E5F6",
        "0B99F2DE-71AA-4E0C-8D3B-FF10E2A7C901",
        "C41D0E55-9A0B-4B1C-A2D3-0E1F2A3B4C5D",
    ]
    # each envelope heads only two pages (< 3), together six of seven
    pages = [
        ParsedPage(page=n, text=f"Docusign Envelope ID: {envelopes[(n - 1) // 2]}\nBody {w}.")
        for n, w in enumerate("abcdef", start=1)
    ]
    pages.append(ParsedPage(page=7, text="Body g."))
    result = clean_pages(pages, file_sha=SHA, parser="pdfium")
    assert all("Envelope" not in page.text for page in result.pages)
    assert "docusign envelope id: <id>" in result.removed


def test_running_header_and_footer_are_removed_from_every_page():
    parts = [
        "Roof",
        "Gutter",
        "Chimney",
        "Deck",
        "Stairs",
        "Siding",
        "Window",
        "Door",
        "Fence",
        "Patio",
    ]
    bodies = [f"Finding {n}.1 {part}: loose." for n, part in enumerate(parts, start=1)]
    result = clean_pages(_pages(bodies), file_sha=SHA, parser="pdfium")
    for page in result.pages:
        assert "Envelope" not in page.text
        assert "Acme Inspections" not in page.text
        assert " of 10" not in page.text
        assert page.text.startswith("Finding ")
    assert "page # of #" in result.removed


def test_a_line_repeated_on_two_pages_is_kept():
    repeated = "Recommendation: Recommend a qualified contractor evaluate and repair."
    bodies = [f"Finding: {word} observed." for word in "abcdefghij"]
    bodies[2] += "\n" + repeated
    bodies[6] += "\n" + repeated
    result = clean_pages(_pages(bodies), file_sha=SHA, parser="pdfium")
    assert repeated in result.pages[2].text
    assert repeated in result.pages[6].text


def test_threshold_is_thirty_percent_of_pages_with_text():
    # on 3 of 10 pages (30%) -> removed; on 2 of 10 -> kept
    bodies = [f"Body {word}." for word in "abcdefghij"]
    for index in (0, 4, 8):
        bodies[index] = "Stamp: RECEIVED\n" + bodies[index]
    bodies[1] = "Only twice\n" + bodies[1]
    bodies[5] = "Only twice\n" + bodies[5]
    result = clean_pages(_pages(bodies), file_sha=SHA, parser="pdfium")
    assert all("RECEIVED" not in page.text for page in result.pages)
    assert "Only twice" in result.pages[1].text
    keys = {key for _, _, key in running_lines([page.text for page in _pages(bodies)], 0.5)}
    assert "page # of #" in keys and "stamp: received" not in keys  # 3 < 50% of 10


def test_a_five_page_file_keeps_a_line_repeated_on_two_pages():
    repeated = "Recommendation: Recommend a qualified contractor evaluate and repair."
    bodies = [f"Finding {word} noted." for word in "abcde"]
    bodies[1] += "\n" + repeated
    bodies[3] += "\n" + repeated
    result = clean_pages(_pages(bodies), file_sha=SHA, parser="pdfium")
    assert [p.page for p in result.pages if repeated in p.text] == [2, 4]
    assert all("Envelope" not in p.text for p in result.pages)  # 5 of 5 still goes


def test_running_lines_need_at_least_three_pages():
    two = [_page(1, "Body one.", 2), _page(2, "Body two.", 2)]
    result = clean_pages(two, file_sha=SHA, parser="pdfium")
    assert all(HEADER in page.text for page in result.pages)
    assert result.removed == frozenset()


ESTIMATE_PAGE_1 = """Kestrel Plumbing, Inc.
ESTIMATE
1 Replace angle stop valve 85.00
2 Replace supply line 45.00
3 Replace angle stop valve 255.00
Total 385.00
Page 1 of 2"""
ESTIMATE_PAGE_2 = """Kestrel Plumbing, Inc.
4 Replace angle stop valve 170.00
5 Replace wax ring 60.00
Total 230.00
Page 2 of 2"""


def test_a_two_page_estimate_keeps_its_line_items_and_totals():
    pages = [ParsedPage(page=1, text=ESTIMATE_PAGE_1), ParsedPage(page=2, text=ESTIMATE_PAGE_2)]
    result = clean_pages(pages, file_sha=SHA, parser="pdfium")
    text = "\n".join(page.text for page in result.pages)
    for line in (
        "1 Replace angle stop valve 85.00",
        "3 Replace angle stop valve 255.00",
        "4 Replace angle stop valve 170.00",
        "Total 385.00",
        "Total 230.00",
    ):
        assert line in text


def test_items_that_match_only_after_digit_folding_are_kept():
    # a four-page estimate: letterhead and footer repeat in place; each page's first
    # item and its page total sit at the same position but differ in their numbers
    pages = []
    for n in range(1, 5):
        items = "\n".join(
            f"{3 * n + k} Replace angle stop valve {85 * (n + k)}.00" for k in range(3)
        )
        text = f"Kestrel Plumbing, Inc.\n{items}\nPage total {n * 95 + 40}.00\nPage {n} of 4"
        pages.append(ParsedPage(page=n, text=text))
    result = clean_pages(pages, file_sha=SHA, parser="pdfium")
    for page in result.pages:
        n = page.page
        assert "Kestrel" not in page.text and f"Page {n} of 4" not in page.text
        assert page.text.splitlines() == [
            *(f"{3 * n + k} Replace angle stop valve {85 * (n + k)}.00" for k in range(3)),
            f"Page total {n * 95 + 40}.00",
        ]


def test_varying_lines_that_are_page_numbers_or_counters_are_running():
    pages = [
        ParsedPage(page=n, text=f"- {n} -\nBatch {n + 60} scan\nBody {w}.\nSheet {n + 2}/9")
        for n, w in enumerate("abcd", start=1)
    ]
    result = clean_pages(pages, file_sha=SHA, parser="pdfium")
    assert [page.text for page in result.pages] == [f"Body {w}." for w in "abcd"]


def test_clean_pages_reads_the_repeat_fraction_from_config(tmp_path, monkeypatch):
    (tmp_path / "knowledge.yaml").write_text("ingest:\n  header_repeat_fraction: 0.9\n")
    monkeypatch.setenv("HAL_CONFIG_DIR", str(tmp_path))
    bodies = [f"Body {word}." for word in "abcdefghij"]
    for index in range(8):  # 8 of 10 pages: removed at 30%, kept at 90%
        bodies[index] = "Stamp: RECEIVED\n" + bodies[index]
    result = clean_pages(_pages(bodies), file_sha=SHA, parser="pdfium")
    assert "RECEIVED" in result.pages[0].text
    assert "Envelope" not in result.pages[0].text
    assert (
        "RECEIVED"
        not in clean_pages(_pages(bodies), file_sha=SHA, parser="pdfium", repeat_fraction=0.3)
        .pages[0]
        .text
    )


def test_body_lines_outside_the_edge_zone_are_not_header_candidates():
    body = "\n".join(f"Line {i} of the body text." for i in range(20))
    pages = [ParsedPage(page=n, text=f"Top {n}\n{body}\nBottom") for n in range(1, 6)]
    result = clean_pages(pages, file_sha=SHA, parser="pdfium", zone=3)
    # "line # of the body text." repeats on every page but only its edge copies go
    text = result.pages[0].text
    assert "Line 10 of the body text." in text
    assert "Bottom" not in text


def test_pseudo_pages_never_lose_repeated_lines():
    pages = [
        ParsedPage(page=n, text="Checked the pump basket.\nVisit notes follow.") for n in (1, 2, 3)
    ]
    result = clean_pages(pages, file_sha=SHA, parser="text", pseudo_pages=True)
    assert all(
        page.text == "Checked the pump basket.\nVisit notes follow." for page in result.pages
    )
    assert result.removed == frozenset()


def test_single_page_file_keeps_everything():
    result = clean_pages([_page(1, "Just one page.", 1)], file_sha=SHA, parser="pdfium")
    assert HEADER in result.pages[0].text


def test_page_without_text_layer_stays_empty_and_flagged():
    pages = _pages(["One.", "Two.", "Three."])
    pages.insert(1, ParsedPage(page=2, text="", has_text_layer=False))
    pages = [
        ParsedPage(page=i, text=p.text, has_text_layer=p.has_text_layer)
        for i, p in enumerate(pages, 1)
    ]
    result = clean_pages(pages, file_sha=SHA, parser="pdfium")
    assert [p.has_text_layer for p in result.pages] == [True, False, True, True]
    assert result.pages[1].text == ""
    assert result.pages[1].offset_map == []


# ------------------------------------------------------------ letter spacing
BUNDLE_HEADING = (
    "D ES C R I PT I O N O F T H E P ROJ EC T & D ES C R I PT I O N O F T H E S I G N I F I C A N T "
    "M AT E R I A L S TO B E U S E D & EQ U I P M E N T TO B E I N STA L L E D"
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("D ES C R I PT I O N O F", "DESCRIPTION OF"),
        ("D ES C R I PT I O N O F W O R K", "DESCRIPTION OF WORK"),
        ("T E R M S A N D C O N D I T I O N S", "TERMS AND CONDITIONS"),
        ("E S T I M A T E", "ESTIMATE"),
        ("D ES C R I PT I O N O F W O R K:", "DESCRIPTION OF WORK:"),
        ("Item D ES C R I PT I O N here", "Item DESCRIPTION here"),
        (
            BUNDLE_HEADING,
            "DESCRIPTION OF THE PROJECT & DESCRIPTION OF THE SIGNIFICANT MATERIALS "
            "TO BE USED & EQUIPMENT TO BE INSTALLED",
        ),
        ("S I G N IF I C A N T", "SIGNIFICANT"),
        ("U S E D", "USED"),
        ("P E S T C O N T R O L", "PEST CONTROL"),
        ("TO TA L $1,234.00", "TOTAL $1,234.00"),
        (
            "W O O D D E S T R O Y I N G P E S T S A N D O R G A N I S M S",
            "WOOD DESTROYING PESTS AND ORGANISMS",
        ),
        ("I N V O I C E", "INVOICE"),
    ],
)
def test_letter_spaced_headings_are_rejoined(raw, expected):
    assert repair_letter_spacing(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "A B testing",
        "We ran A B testing on the thermostat schedule.",
        "Options A B C D E are listed.",
        "SEE A OR B IN THE TABLE",
        "U S A",
        "Circuit breakers 1 2 3 4 are labelled.",
        "I O U",
        "IT IS OK TO GO",
        "PART A OF THE FORM",
        "NO. OF UNITS",
        "W X Y Z",
    ],
)
def test_normal_text_is_untouched(text):
    assert repair_letter_spacing(text) == text


@pytest.mark.parametrize(("text", "tokens"), [("SEE A OR B IN THE", 6), ("U S A", 3)])
def test_weak_compression_is_not_a_rewrite(text, tokens):
    # the DP does find words ("SEE A ORB IN THE", "US A"); only the ratio guard keeps the text
    words = _segment(text.split(" "), word_costs())
    assert words is not None and " ".join(words) != text
    assert 3 * len(words) >= 2 * tokens
    assert repair_letter_spacing(text) == text


def test_an_alphabet_run_is_an_option_list():
    assert _segment(["A", "B", "C", "D", "E"], word_costs()) == ["ABCDE"]
    assert repair_letter_spacing("A B C D E") == "A B C D E"


def test_single_letters_no_word_covers_are_joined_whole():
    # a name the word list lacks: joined, never split around short words inside it
    assert repair_letter_spacing("Z O R V E L") == "ZORVEL"
    assert repair_letter_spacing("EQ U I P M E N T Z O R V E L") == "EQUIPMENT ZORVEL"


def test_fragments_no_word_can_absorb_leave_the_run_alone():
    assert repair_letter_spacing("QX Z V K") == "QX Z V K"


def test_extra_vocabulary_is_used():
    # "OF" can't be split off an unknown name without a word for the rest ...
    assert repair_letter_spacing("O F Z O R V E L") == "OFZORVEL"
    # ... but with the name as a word, it can
    assert repair_letter_spacing("O F Z O R V E L", vocabulary=["Zorvel"]) == "OF ZORVEL"
    assert word_costs(["Zorvel"])["zorvel"] == word_costs()["the"] + 0.2


def test_letter_spacing_is_repaired_inside_clean_pages():
    pages = [ParsedPage(page=1, text="D ES C R I PT I O N O F W O R K\r\nReplace valve")]
    result = clean_pages(pages, file_sha=SHA, parser="pdfium")
    assert result.pages[0].text == "DESCRIPTION OF WORK\nReplace valve"


# --------------------------------------------------------------- hyphenation
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "The inspector is not required to deter-\nmine the age.",
            "The inspector is not required to determine the age.",
        ),
        ("hyphen￾ated word", "hyphenated word"),
        ("soft­hyphen", "softhyphen"),
        ("line-end￾\r\njoin", "line-endjoin"),
        ("A well-known fact", "A well-known fact"),
        ("Total -\n$5", "Total -\n$5"),
        ("Item 3-\nB", "Item 3-\nB"),
    ],
)
def test_hyphenation_is_repaired(raw, expected):
    result = clean_pages([ParsedPage(page=1, text=raw)], file_sha=SHA, parser="pdfium")
    assert result.pages[0].text == expected


# ---------------------------------------------------------------- whitespace
def test_whitespace_is_collapsed():
    raw = "  Line\t one    here  \r\n\r\n\r\n\r\n  Line two \x0c\n\n"
    result = clean_pages([ParsedPage(page=1, text=raw)], file_sha=SHA, parser="pdfium")
    assert result.pages[0].text == "Line one here\n\nLine two"


# ---------------------------------------------------------------- offset map
def test_offset_map_points_back_at_the_raw_text():
    raw = "  D ES C R I PT I O N O F W O R K\r\nhyphen￾ated  text"
    result = clean_pages([ParsedPage(page=1, text=raw)], file_sha=SHA, parser="pdfium")
    page = result.pages[0]
    for index, char in enumerate(page.text):
        if not char.isspace():
            assert raw[source_offset(page.offset_map, index)] == char
    assert page.offset_map[0] == (0, 2)


def test_paged_text_maps_offsets_to_pages():
    result = clean_pages(
        _pages(["Alpha one.", "Beta two.", "Gamma three."]), file_sha=SHA, parser="pdfium"
    )
    joined = result.document()
    assert joined.text == PAGE_SEPARATOR.join(["Alpha one.", "Beta two.", "Gamma three."])
    assert joined.page_at(0) == 1
    assert joined.page_at(joined.text.index("Beta")) == 2
    assert joined.page_at(len(joined.text) - 1) == 3
    assert joined.page_span(0, joined.text.index("Gamma") + 2) == (1, 3)
    assert joined.locate(joined.text.index("two")) == (2, "Beta two.".index("two"))
    with pytest.raises(IndexError):
        joined.page_at(len(joined.text))


def test_paged_text_can_be_limited_to_a_page_range_and_skips_empty_pages():
    pages = _pages(["One.", "Two.", "Three.", "Four."])
    pages[1] = ParsedPage(page=2, text="", has_text_layer=False)
    result = clean_pages(pages, file_sha=SHA, parser="pdfium")
    part = result.document(2, 3)
    assert part.text == "Three."
    assert part.page_at(0) == 3
    assert result.document(2, 2).text == ""


def test_source_offset_on_empty_map_is_identity_free():
    assert source_offset([], 0) == 0
    assert source_offset([(0, 5), (3, 20)], 4) == 21


def test_clean_is_timed_as_the_clean_stage():
    with start_trace(Trace()) as trace:
        clean_pages([ParsedPage(page=1, text="x")], file_sha=SHA, parser="pdfium")
    assert [timing.stage for timing in trace.stages] == ["clean"]


# ---- property: for any cleaned offset, the mapped page contains the character
_WORD = st.sampled_from(
    [
        "roof",
        "valve",
        "Leak",
        "3.1.2",
        "$475.00",
        "deter-\nmine",
        "hyphen￾ated",
        "D ES C R I PT I O N",
        "A B",
        "Page 7 of 9",
        "  ",
        "\t",
        "\r\n",
        "\n\n\n",
        " ",
        "x",
        "W O R K",
        "soft­hyphen",
    ]
)
_BODY = st.lists(_WORD, min_size=0, max_size=25).map(" ".join)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(bodies=st.lists(_BODY, min_size=1, max_size=8), header=st.booleans(), data=st.data())
def test_offset_map_property(bodies, header, data):
    raw_pages = []
    for number, body in enumerate(bodies, start=1):
        text = (f"{HEADER}\n{body}\nPage {number} of {len(bodies)}") if header else body
        has_text = data.draw(st.booleans()) if not body.strip() else True
        raw_pages.append(
            ParsedPage(page=number, text=text if has_text else "", has_text_layer=has_text)
        )
    result = clean_pages(raw_pages, file_sha=SHA, parser="pdfium")
    by_number = {page.page: page for page in result.pages}
    joined: PagedText = result.document()
    for offset, char in enumerate(joined.text):
        number = joined.page_at(offset)
        page_text = by_number[number].text
        local = offset - joined.start_of(number)
        if local < len(page_text):
            assert page_text[local] == char  # the mapped page contains the character
        else:
            assert char.isspace() and joined.text[offset:].startswith(
                PAGE_SEPARATOR[local - len(page_text) :]
            )
    # and every non-space cleaned character maps back to the same raw character
    for page, raw in zip(result.pages, raw_pages):
        for index, char in enumerate(page.text):
            if not char.isspace():
                assert raw.text[source_offset(page.offset_map, index)] == char
