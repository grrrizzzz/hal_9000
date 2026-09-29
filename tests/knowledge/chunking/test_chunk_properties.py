"""Property tests for every chunker (handoff KWP2 acceptance; plan §6.1, §18).

Each property runs ≥ 300 Hypothesis examples per chunker, measuring tokens with
the real tokenizer from ``hal9000.knowledge.tokens``.
"""

from __future__ import annotations

import re
from bisect import bisect_right

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hal9000.knowledge.chunking import ChunkDraft, ChunkInput, chunk, overlap_tokens
from hal9000.knowledge.tokens import count_tokens

from .synthetic import chunk_inputs, doc_texts, inspection_doc, sizes

PROPERTY = settings(
    max_examples=300,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
CHUNKERS = ["fixed", "recursive", "structure", "parent_child"]


@st.composite
def cases(draw, texts=None):
    """(input, chunker kwargs) with a size, an overlap up to half the size and a parent size."""
    inp = draw(chunk_inputs(texts))
    size = draw(sizes)
    overlap = draw(st.one_of(st.just(0), st.integers(0, size // 2)))
    parent_size = draw(st.integers(size, max(size, 2000)))
    return inp, dict(size=size, overlap=overlap, parent_size=parent_size)


def _run(name: str, inp: ChunkInput, kw: dict) -> list[ChunkDraft]:
    if name != "parent_child":
        kw = {k: v for k, v in kw.items() if k != "parent_size"}
    return chunk(inp, name, header=True, **kw)


def _page_of(inp: ChunkInput, pos: int) -> int:
    offsets = [o for o, _ in inp.page_starts]
    return inp.page_starts[max(0, bisect_right(offsets, pos) - 1)][1]


def _uncovered(text: str, spans) -> list[int]:
    covered = bytearray(len(text))
    for s, e in spans:
        covered[s:e] = b"\x01" * (e - s)
    return [i for i, ch in enumerate(text) if not covered[i] and not ch.isspace()]


# ------------------------------------------------------------- spans are exact
@pytest.mark.parametrize("name", CHUNKERS)
@PROPERTY
@given(case=cases())
def test_spans_reproduce_the_exact_cleaned_text(name: str, case) -> None:
    inp, kw = case
    drafts = _run(name, inp, kw)
    for d in drafts:
        assert 0 <= d.char_start < d.char_end <= len(inp.text)
        assert d.text == inp.text[d.char_start : d.char_end]
        assert d.text.strip() == d.text  # no leading or trailing whitespace
        assert d.document_id == inp.document_id
        assert d.page_start == _page_of(inp, d.char_start)
        assert d.page_end == _page_of(inp, d.char_end - 1)
    assert [d.key for d in drafts] == list(range(len(drafts)))


# --------------------------------------------------------- the union covers all
@pytest.mark.parametrize("name", CHUNKERS)
@PROPERTY
@given(case=cases())
def test_union_covers_every_non_boilerplate_character(name: str, case) -> None:
    """Cleaned text has boilerplate removed already; only whitespace may fall between chunks."""
    inp, kw = case
    drafts = _run(name, inp, kw)
    searchable = [d for d in drafts if d.role != "parent"]
    assert _uncovered(inp.text, [(d.char_start, d.char_end) for d in searchable]) == []
    if name == "parent_child":
        parents = [d for d in drafts if d.role == "parent"]
        assert _uncovered(inp.text, [(d.char_start, d.char_end) for d in parents]) == []
    if not inp.text.strip():
        assert drafts == []


# ------------------------------------------------------------- token ceilings
@pytest.mark.parametrize("name", CHUNKERS)
@PROPERTY
@given(case=cases())
def test_no_chunk_exceeds_n_tokens_header_excluded(name: str, case) -> None:
    inp, kw = case
    drafts = _run(name, inp, kw)
    for d in drafts:
        assert d.tokens == count_tokens(d.text)
        limit = kw["parent_size"] if d.role == "parent" else kw["size"]
        assert d.tokens <= limit
        if d.header:  # the header is extra: it never counts against N
            assert d.index_text.endswith(d.text) and count_tokens(d.index_text) > d.tokens


def _longest_sentence_tokens(text: str, *, lines: bool = True) -> int:
    """Tokens in the longest sentence; with ``lines``, every line break also ends one."""
    pattern = r"(?<=[.!?:])\s+|\n" if lines else r"(?<=[.!?:])\s+|\n\s*\n"
    return max((count_tokens(p) for p in re.split(pattern, text)), default=0)


@PROPERTY
@given(case=cases(doc_texts()))
def test_fixed_snaps_backward_within_the_final_tenth(case) -> None:
    """``fixed`` windows end at most N tokens in and at least ~90% of N (except the last)."""
    inp, kw = case
    drafts = _run("fixed", inp, kw)
    for d in drafts[:-1]:
        assert d.tokens <= kw["size"]
        assert d.tokens >= int(0.9 * kw["size"]) - 3, (d.tokens, kw)


# ---------------------------------------------------------------- overlap
@PROPERTY
@given(case=cases(doc_texts()))
def test_fixed_overlap_equals_the_configured_value_within_one_sentence(case) -> None:
    inp, kw = case
    drafts = _run("fixed", inp, kw)
    tolerance = _longest_sentence_tokens(inp.text)
    for a, b in zip(drafts, drafts[1:]):
        assert a.char_start < b.char_start
        shared = (
            count_tokens(inp.text[b.char_start : a.char_end]) if b.char_start < a.char_end else 0
        )
        assert abs(shared - kw["overlap"]) <= tolerance, (shared, kw["overlap"], tolerance)
        if kw["overlap"] == 0:
            assert b.char_start >= a.char_end


@pytest.mark.parametrize("name", ["recursive", "structure", "parent_child"])
@PROPERTY
@given(case=cases(doc_texts()))
def test_sentence_overlap_never_exceeds_the_configured_value(name: str, case) -> None:
    """Other chunkers carry whole trailing sentences into the next chunk, up to *O* tokens."""
    inp, kw = case
    drafts = [d for d in _run(name, inp, kw) if d.role != "parent"]
    for a, b in zip(drafts, drafts[1:]):
        if b.char_start < a.char_end:
            assert count_tokens(inp.text[b.char_start : a.char_end]) <= kw["overlap"] + 2
    if kw["overlap"] == 0:
        for a, b in zip(drafts, drafts[1:]):
            assert b.char_start >= a.char_end


@pytest.mark.parametrize("size", [256, 512, 768, 1024, 1536])
def test_fixed_windows_end_at_sentence_or_line_ends(size: int) -> None:
    """On wrapped report text every final tenth holds a line end, so no window cuts a line."""
    doc = inspection_doc().input
    drafts = chunk(doc, "fixed", size=size, overlap=overlap_tokens(size, 15))
    assert len(drafts) >= 3
    for d in drafts[:-1]:
        assert d.text.endswith((".", "!", "?", ":")) or doc.text[d.char_end] == "\n", d.text[-60:]
    sentence_ends = sum(d.text.endswith(".") for d in drafts[:-1])
    assert sentence_ends >= len(drafts[:-1]) / 2


@pytest.mark.parametrize("name", ["recursive", "parent_child"])
@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_recursive_neighbours_overlap_by_whole_sentences(name: str, size: int) -> None:
    doc = inspection_doc().input
    overlap = overlap_tokens(size, 15)
    kw = dict(parent_size=1500) if name == "parent_child" else {}
    drafts = chunk(doc, name, size=size, overlap=overlap, **kw)
    tolerance = _longest_sentence_tokens(doc.text, lines=False)
    by_key = {d.key: d for d in drafts}
    children = [d for d in drafts if d.role != "parent"]
    pairs = [(a, b) for a, b in zip(children, children[1:]) if a.parent_key == b.parent_key]
    assert pairs
    for a, b in pairs:
        assert b.char_start < a.char_end, (a.text[-80:], b.text[:80])
        shared = count_tokens(doc.text[b.char_start : a.char_end])
        assert overlap - tolerance <= shared <= overlap
        before = doc.text[: b.char_start].rstrip()  # whole sentences, or whole lines
        assert before.endswith((".", ":")) or "\n" in doc.text[len(before) : b.char_start]
        if b.parent_key is not None:
            assert by_key[b.parent_key].char_start <= b.char_start
