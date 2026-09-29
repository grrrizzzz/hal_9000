"""The parent–child chunker (plan §6.1; handoff KWP2 acceptance)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hal9000.knowledge.chunking import chunk
from hal9000.knowledge.tokens import count_tokens

from .synthetic import chunk_inputs, inspection_doc

DOC = inspection_doc()


def _check_family(drafts, size: int, parent_size: int) -> None:
    by_key = {d.key: d for d in drafts}
    parents = [d for d in drafts if d.role == "parent"]
    children = [d for d in drafts if d.role == "child"]
    assert len(parents) + len(children) == len(drafts)
    for p in parents:
        assert p.parent_key is None
        assert p.tokens == count_tokens(p.text) <= parent_size
    for c in children:
        assert c.parent_key is not None
        parent = by_key[c.parent_key]
        assert parent.role == "parent" and parent.key < c.key
        assert parent.char_start <= c.char_start < c.char_end <= parent.char_end
        assert parent.page_start <= c.page_start <= c.page_end <= parent.page_end
        assert c.tokens <= size
    # every parent has at least one child
    assert {c.parent_key for c in children} == {p.key for p in parents}


@pytest.mark.parametrize("size", [128, 256, 384, 512])
def test_every_child_sits_in_a_parent_of_at_most_1500_tokens(size: int) -> None:
    drafts = chunk(DOC.input, "parent_child", size=size, parent_size=1500)
    _check_family(drafts, size, 1500)
    parents = [d for d in drafts if d.role == "parent"]
    children = [d for d in drafts if d.role == "child"]
    assert len(children) > len(parents) > 1
    # Parents are sections packed up to the parent size, not single small findings.
    assert max(p.tokens for p in parents) > 1000


def test_parents_keep_small_findings_whole() -> None:
    drafts = chunk(DOC.input, "parent_child", size=128, parent_size=1500)
    parents = [d for d in drafts if d.role == "parent"]
    for f in DOC.findings:
        if f is DOC.oversized:
            continue
        assert any(p.char_start <= f.start and p.char_end >= f.end for p in parents), f.title


@settings(
    max_examples=300,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(inp=chunk_inputs(), size=st.integers(8, 512), extra=st.integers(0, 1500))
def test_parent_child_family_property(inp, size: int, extra: int) -> None:
    parent_size = min(1500, size + extra)
    overlap = size // 7
    drafts = chunk(inp, "parent_child", size=size, overlap=overlap, parent_size=parent_size)
    _check_family(drafts, size, parent_size)


def test_parent_child_needs_a_parent_at_least_as_big_as_a_child() -> None:
    with pytest.raises(ValueError, match="parent_size"):
        chunk(DOC.input, "parent_child", size=512, parent_size=256)
    with pytest.raises(ValueError, match="parent_size"):
        chunk(DOC.input, "parent_child", size=512)


def test_other_chunkers_reject_a_parent_size() -> None:
    with pytest.raises(ValueError, match="parent_size"):
        chunk(DOC.input, "fixed", size=512, parent_size=1500)
