"""Verify (plan §7.1): citations must name supplied sources; every figure must be in a cited one.

Written before ``knowledge/answer/verify.py`` (rule K6). Synthetic figures only.
"""

from __future__ import annotations

from datetime import date

import pytest

from hal9000.knowledge.answer.verify import (
    SourceFacts,
    cited_markers,
    haystack,
    strip_markers,
    verify_answer,
)


def src(marker: str = "S1", text: str = "", **meta) -> SourceFacts:
    return SourceFacts(marker=marker, text=text, **meta)


def five() -> list[SourceFacts]:
    return [src(f"S{i}", f"Synthetic passage number {i}.") for i in range(1, 6)]


# ---------------------------------------------------------------- the handoff cases
def test_a_transposed_amount_is_flagged():
    result = verify_answer(
        "The pest repair cost $1,324.00 [S1].", [src(text="Total due: $1,234.00")]
    )
    assert not result.ok
    assert result.grounding == "failed"
    assert [issue.kind for issue in result.issues] == ["numeral"]
    assert "$1,324.00" in result.issues[0].message
    assert "[S1]" in result.issues[0].message


def test_the_same_amount_without_cents_passes():
    result = verify_answer("The repair cost $1,234 [S1].", [src(text="Total due: $1,234.00")])
    assert result.ok and result.grounding == "verified"
    assert result.issues == ()


def test_the_documents_date_backs_a_month_and_year_missing_from_the_text():
    source = src(
        text="The relief valve on the water heater drips.",
        title="Synthetic home inspection",
        doc_date=date(2023, 6, 22),
    )
    answer = "The June 2023 inspection [S1] noted a dripping relief valve."
    assert verify_answer(answer, [source]).ok
    # without the metadata the same answer can't be backed: the date really is what passes it
    undated = src(text="The relief valve on the water heater drips.")
    failed = verify_answer(answer, [undated])
    assert not failed.ok and failed.issues[0].kind == "numeral"


@pytest.mark.parametrize(
    "answer",
    [
        "The valve drips [S1].",
        "The valve drips [S1][S2].",
        "The valve drips [S1, S2].",
        "The valve drips ([S1]) and [S2] agrees.",
    ],
)
def test_source_markers_are_never_figures(answer):
    sources = [src("S1", "The valve drips."), src("S2", "A dripping valve was seen.")]
    result = verify_answer(answer, sources)
    assert result.ok, result.issues
    assert "1" not in strip_markers(answer) and "2" not in strip_markers(answer)


def test_a_citation_beyond_the_supplied_sources_is_flagged():
    result = verify_answer("The furnace is old [S9].", five())
    assert not result.ok
    assert [issue.kind for issue in result.issues] == ["citation"]
    assert "[S9]" in result.issues[0].message and "5 sources" in result.issues[0].message


def test_a_malformed_marker_is_a_citation_issue():
    result = verify_answer("The furnace is old [S0].", five())
    assert [issue.kind for issue in result.issues] == ["citation"]


# --------------------------------------------------------------------- figures
def test_a_figure_only_in_an_uncited_source_is_flagged():
    sources = [src("S1", "The roof is composition shingle."), src("S2", "Total due: $880.00")]
    result = verify_answer("The roof cost $880.00 [S1].", sources)
    assert not result.ok and "$880.00" in result.issues[0].message


def test_a_figure_with_no_citation_at_all_is_flagged():
    result = verify_answer("The roof cost $880.00.", [src(text="Total due: $880.00")])
    assert not result.ok and result.issues[0].kind == "numeral"


def test_figures_may_come_from_any_cited_source():
    sources = [src("S1", "Estimate: $450.00"), src("S2", "Invoice total: $475.50")]
    assert verify_answer("Estimated $450 [S1], invoiced $475.50 [S2].", sources).ok


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("It was inspected on June 22, 2023 [S1].", "Inspected 6/22/23."),
        ("It was inspected on 2023-06-22 [S1].", "Inspection date: June 22, 2023"),
        ("It was inspected on 6/22/2023 [S1].", "Inspected 22 June 2023."),
        ("It was inspected in Jun. 2023 [S1].", "Inspection date: June 22, 2023"),
        ("It was inspected on June 22nd [S1].", "Inspection date: June 22, 2023"),
        ("It was inspected in 2023 [S1].", "Inspection date: June 22, 2023"),
        ("The warranty lasts ten years [S1].", "Warranted for 10 years."),
        ("The warranty lasts 10 years [S1].", "Warranted for ten years."),
        ("Humidity was 45% [S1].", "Humidity 45 percent."),
        ("The filter is 16 x 25 x 1 inches [S1].", "Use a 16 x 25 x 1 inch filter."),
    ],
)
def test_dates_currency_and_words_are_normalized_on_both_sides(answer, source):
    result = verify_answer(answer, [src(text=source)])
    assert result.ok, result.issues


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("It was inspected in July 2023 [S1].", "Inspection date: June 22, 2023"),
        ("It was inspected on June 23, 2023 [S1].", "Inspection date: June 22, 2023"),
        ("It was inspected in 2024 [S1].", "Inspection date: June 22, 2023"),
        ("The warranty lasts twelve years [S1].", "Warranted for 10 years."),
        ("Humidity was 54% [S1].", "Humidity 45 percent."),
    ],
)
def test_a_wrong_date_or_figure_is_flagged(answer, source):
    assert not verify_answer(answer, [src(text=source)]).ok


def test_metadata_backs_vendor_title_and_reference_numbers():
    source = src(
        text="Termite damage repair at the rear deck ledger.",
        title="Invoice 1001 - termite repair",
        vendor="Example Pest Co. 7",
        doc_date=date(2023, 7, 14),
    )
    answer = "Invoice 1001 from Example Pest Co. 7, dated 7/14/23, covers the repair [S1]."
    assert verify_answer(answer, [source]).ok


def test_each_problem_is_listed_once():
    sources = [src("S1", "Total due: $1,234.00")]
    result = verify_answer("It cost $9.99 [S1], then $9.99 again [S7].", sources)
    kinds = sorted(issue.kind for issue in result.issues)
    assert kinds == ["citation", "numeral"]
    assert result.kinds == {"citation", "numeral"}


# ------------------------------------------------------------------ helpers
def test_cited_markers_in_order_without_duplicates():
    assert cited_markers("a [S2] b [S1][S2] c [S3, S1]") == ["S2", "S1", "S3"]


def test_the_haystack_writes_the_date_every_way():
    text = haystack(src(text="body", title="T", vendor="V", doc_date=date(2023, 6, 22)))
    for piece in ("2023-06-22", "June 2023", "6/22/23", "2023", "T", "V", "body"):
        assert piece in text
