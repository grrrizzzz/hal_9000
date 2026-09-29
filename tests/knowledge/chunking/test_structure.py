"""The structure-aware chunker and the unit detector (plan §6.1; handoff KWP2 acceptance)."""

from __future__ import annotations

import re

import pytest

from hal9000.knowledge.chunking import ChunkInput, Unit, chunk, detect_units, overlap_tokens
from hal9000.knowledge.tokens import count_tokens

from .synthetic import inspection_doc

DOC = inspection_doc()
SIZES = [128, 256, 384, 512, 768, 1024, 1536]


def _covering(drafts, start: int, end: int):
    return [d for d in drafts if d.char_start < end and d.char_end > start]


def test_fixture_has_an_oversized_finding_and_many_small_ones() -> None:
    text = DOC.input.text
    small = [f for f in DOC.findings if f is not DOC.oversized]
    assert len(small) == 12
    assert all(count_tokens(text[f.start : f.end]) <= 128 for f in small)
    assert count_tokens(text[DOC.oversized.start : DOC.oversized.end]) > max(SIZES)
    assert len(DOC.input.page_starts) > 3


@pytest.mark.parametrize("size", SIZES)
def test_structure_keeps_every_finding_whole_when_it_fits(size: int) -> None:
    text = DOC.input.text
    drafts = chunk(DOC.input, "structure", size=size)
    checked = 0
    for f in DOC.findings:
        if count_tokens(text[f.start : f.end]) > size:
            continue
        holders = [d for d in drafts if d.char_start <= f.start and d.char_end >= f.end]
        assert holders, f"finding {f.title!r} was split at size {size}"
        checked += 1
    assert checked >= 12


@pytest.mark.parametrize("size", SIZES)
def test_structure_splits_the_oversized_finding_recursively(size: int) -> None:
    text = DOC.input.text
    f = DOC.oversized
    parts = _covering(chunk(DOC.input, "structure", size=size), f.start, f.end)
    assert len(parts) >= 2
    assert all(p.tokens <= size for p in parts)
    assert parts[0].char_start == f.start  # the finding starts its own chunk
    for p in parts[1:]:
        # Every split falls on a paragraph or sentence boundary inside the finding.
        assert f.start < p.char_start < f.end
        before = text[: p.char_start].rstrip()
        assert before.endswith((".", ":")) or "\n\n" in text[len(before) : p.char_start]
    for p in parts[:-1]:
        assert p.char_end <= f.end


def test_structure_chunks_start_at_units_or_sentences() -> None:
    """No structure chunk starts in the middle of a sentence."""
    text = DOC.input.text
    for d in chunk(DOC.input, "structure", size=256):
        before = text[: d.char_start].rstrip()
        assert d.char_start == 0 or before.endswith((".", ":")) or "\n" in text[len(before) :][:3]


def test_tiny_units_merge_with_their_neighbour() -> None:
    """Heading-only units ("3 PLUMBING", "3.1 Water Heater") don't become chunks of their own."""
    drafts = chunk(DOC.input, "structure", size=512)
    for d in drafts:
        assert d.tokens >= 16, d.text
    text = DOC.input.text
    heading = text.index("\n\n3 PLUMBING\n") + 2
    holder = next(d for d in drafts if d.char_start <= heading < d.char_end)
    assert "3.1.1 Plumbing – Water Heater" in holder.text


def test_structure_with_n_as_large_as_the_document_is_one_chunk() -> None:
    small = ChunkInput(document_id=1, text="1 ROOF\n\n1.1.1 Roof – Shingles: Wear\nOld shingles.")
    drafts = chunk(small, "structure", size=512)
    assert [d.text for d in drafts] == [small.text]


def test_explicit_units_override_detection() -> None:
    part_a = "intro line here. " + "alpha beta gamma delta. " * 8
    part_b = "zeta eta theta iota. " * 8
    text = (part_a + part_b).strip()
    cut = len(part_a)
    units = [Unit(start=0, title="Part A", level=1), Unit(start=cut, title="Part B", level=1)]
    inp = ChunkInput(document_id=1, text=text, units=tuple(units))
    drafts = chunk(inp, "structure", size=64, header=True)
    assert [d.text for d in drafts] == [text[:cut].rstrip(), text[cut:]]
    assert [d.section_path for d in drafts] == [("Part A",), ("Part B",)]
    # Without units there is no structure to follow: windows pack sentences to N.
    plain = chunk(ChunkInput(document_id=1, text=text, units=()), "structure", size=64)
    assert plain[0].char_end > cut
    assert all(d.section_path == () for d in plain)


def test_units_with_explicit_ends_nest() -> None:
    text = "SECTION\nlead in text.\nfinding one body. more body.\nafter the finding text."
    f_start, f_end = text.index("finding"), text.index("after") - 1
    units = (
        Unit(start=0, end=len(text), title="Section", level=1),
        Unit(start=f_start, end=f_end, title="Finding", level=2),
    )
    inp = ChunkInput(document_id=1, text=text, units=units)
    drafts = chunk(inp, "structure", size=8)
    holder = next(d for d in drafts if d.char_start <= f_start < d.char_end)
    assert holder.section_path == ("Section", "Finding")
    assert all(d.tokens <= 8 for d in drafts)


# ------------------------------------------------------------------ detector
def test_detector_finds_numbered_findings_and_headings() -> None:
    units = detect_units(DOC.input.text)
    titles = [u.title for u in units]
    for f in DOC.findings:
        assert f.title in titles
        unit = next(u for u in units if u.start == f.start)
        assert unit.title == f.title and unit.level == 3
    assert "3 PLUMBING" in titles and "3.1 Water Heater" in titles
    assert "SUMMARY" in titles and "SYNTHETIC HOME INSPECTION REPORT" in titles
    assert all(not t.startswith("Recommendation") for t in titles)
    levels = {u.title: u.level for u in units}
    assert levels["3 PLUMBING"] == 1 and levels["3.1 Water Heater"] == 2
    assert [u.start for u in units] == sorted(u.start for u in units)


@pytest.mark.parametrize(
    ("line", "title", "level"),
    [
        (
            "3.1.2 Plumbing – Water Heater: Valve Leak",
            "3.1.2 Plumbing – Water Heater: Valve Leak",
            3,
        ),
        ("  7.2 Plumbing – Water Heater: Valve Leak", "7.2 Plumbing – Water Heater: Valve Leak", 2),
        ("12 ELECTRICAL", "12 ELECTRICAL", 1),
        ("4. Heating And Cooling", "4. Heating And Cooling", 1),
        ("## Water Heater", "Water Heater", 2),
        ("# Overview", "Overview", 1),
        ("PLUMBING SYSTEM", "PLUMBING SYSTEM", 1),
    ],
)
def test_detector_heading_forms(line: str, title: str, level: int) -> None:
    units = detect_units(f"{line}\nBody text follows here.")
    assert [(u.title, u.level) for u in units] == [(title, level)]
    assert units[0].start == len(line) - len(line.lstrip())


@pytest.mark.parametrize(
    "line",
    [
        "1. Replace the valve and test the relief line.",  # a numbered list item
        "3.5 gallons were found in the pan.",  # a measurement
        "2023 was the year of the inspection.",
        "Recommendation: Replace the valve.",
        "The WATER HEATER is old.",
        "A",
        "OK",
        "#hashtag",
        "1.2.3",
    ],
)
def test_detector_ignores_ordinary_lines(line: str) -> None:
    assert detect_units(f"Some text.\n{line}\nMore text.") == []


def test_caps_label_inside_a_finding_is_a_sub_unit() -> None:
    text = "3.1.2 Plumbing – Water Heater: Leak\nThe valve drips.\nRECOMMENDATION\nReplace it."
    units = detect_units(text)
    assert [(u.title, u.level) for u in units] == [
        ("3.1.2 Plumbing – Water Heater: Leak", 3),
        ("RECOMMENDATION", 4),
    ]
    drafts = chunk(ChunkInput(document_id=1, text=text), "structure", size=256)
    assert len(drafts) == 1


def test_detector_is_deterministic_and_fast() -> None:
    text = DOC.input.text * 20
    assert detect_units(text) == detect_units(text)
    assert re.search(r"\n\n", text)


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_split_pieces_of_an_oversized_unit_overlap_by_whole_sentences(size: int) -> None:
    text = DOC.input.text
    overlap = overlap_tokens(size, 15)
    drafts = chunk(DOC.input, "structure", size=size, overlap=overlap)
    f = DOC.oversized
    parts = _covering(drafts, f.start, f.end)
    assert len(parts) >= 2
    for a, b in zip(parts, parts[1:]):
        assert b.char_start < a.char_end  # every split carries an overlap
        shared = count_tokens(text[b.char_start : a.char_end])
        assert overlap - 40 <= shared <= overlap  # 40 ≥ the fixture's longest sentence
        assert text[: b.char_start].rstrip().endswith((".", ":"))  # a whole sentence
        assert b.tokens <= size
    # Units that fit are never overlapped into: no chunk starts inside a small finding.
    for small in DOC.findings:
        if small is not f:
            assert not any(small.start < d.char_start < small.end for d in drafts)
