"""The contextual header (plan §6.1): indexed and embedded, never shown in citations."""

from __future__ import annotations

from datetime import date

import pytest

from hal9000.knowledge.chunking import ChunkInput, build_header, chunk
from hal9000.knowledge.contracts import Chunk
from hal9000.knowledge.tokens import count_tokens

from .synthetic import META, inspection_doc

DOC = inspection_doc()
CHUNKERS = ["fixed", "recursive", "structure", "parent_child"]


def _kw(name: str) -> dict:
    return dict(size=256, parent_size=1500) if name == "parent_child" else dict(size=256)


def test_header_format() -> None:
    assert (
        build_header(
            "Synthetic Home Inspection",
            "inspection_report",
            "Example Inspections LLC",
            date(2023, 6, 22),
            ("3 PLUMBING", "3.1.2 Plumbing – Water Heater: Valve Leak"),
        )
        == "Synthetic Home Inspection (inspection report, Example Inspections LLC, 2023-06-22)"
        " › 3 PLUMBING › 3.1.2 Plumbing – Water Heater: Valve Leak"
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("T", "invoice", None, None, ()), "T (invoice)"),
        (("T", "", None, None, ("A",)), "T › A"),
        (("", "", None, None, ("A", "B")), "A › B"),
        (("", "estimate", "V", None, ()), "(estimate, V)"),
        (("", "", None, None, ()), ""),
        (("T\nwith break", "other", " V ", None, ("A\tB",)), "T with break (other, V) › A B"),
    ],
)
def test_header_leaves_out_missing_parts(args, expected: str) -> None:
    assert build_header(*args) == expected


@pytest.mark.parametrize("name", CHUNKERS)
def test_header_is_indexed_but_not_displayed(name: str) -> None:
    drafts = chunk(DOC.input, name, header=True, **_kw(name))
    assert drafts
    for d in drafts:
        assert d.header.startswith(
            "Synthetic Home Inspection (inspection report, Example Inspections LLC, 2023-06-22)"
        )
        assert d.index_text == f"{d.header}\n{d.text}"
        assert d.index_tokens == count_tokens(d.index_text) > d.tokens
        assert d.header not in d.text
        assert d.text == DOC.input.text[d.char_start : d.char_end]
        row = d.to_chunk(profile_id=4, parent_id=None)
        assert isinstance(row, Chunk)
        assert row.text == d.text and row.header == d.header  # citations use .text
        assert row.tokens == d.tokens
        if d.section_path:
            assert d.header.endswith(" › " + " › ".join(d.section_path))
    # Chunks starting inside a finding carry the section › sub-section › finding path.
    deep = [d for d in drafts if len(d.section_path) == 3]
    assert deep
    for d in deep:
        top, sub, leaf = d.section_path
        assert sub.startswith(top.split()[0] + ".") and leaf.startswith(sub.split()[0] + ".")


@pytest.mark.parametrize("name", CHUNKERS)
def test_no_header_means_index_text_is_the_text(name: str) -> None:
    for d in chunk(DOC.input, name, header=False, **_kw(name)):
        assert d.header == ""
        assert d.index_text == d.text
        assert d.index_tokens == d.tokens


def test_chunks_before_any_heading_have_the_document_header_only() -> None:
    inp = ChunkInput(document_id=2, text="Plain words. No headings here.", **META)
    (d,) = chunk(inp, "fixed", size=64)
    assert d.section_path == ()
    assert d.header == (
        "Synthetic Home Inspection (inspection report, Example Inspections LLC, 2023-06-22)"
    )


def test_to_chunk_carries_ids_pages_and_span() -> None:
    drafts = chunk(DOC.input, "parent_child", size=128, parent_size=1500)
    child = next(d for d in drafts if d.role == "child")
    row = child.to_chunk(profile_id=9, parent_id=77)
    assert (row.profile_id, row.document_id, row.parent_id) == (9, 3, 77)
    assert (row.char_start, row.char_end) == (child.char_start, child.char_end)
    assert (row.page_start, row.page_end) == (child.page_start, child.page_end)
    assert Chunk.model_validate_json(row.model_dump_json()) == row
