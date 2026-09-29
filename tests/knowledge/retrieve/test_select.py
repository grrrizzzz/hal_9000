"""``select`` on hand-made candidates: budget, parent expansion, near-duplicates (plan §7.1)."""

from __future__ import annotations

from hal9000.knowledge.contracts import RetrievedChunk
from hal9000.knowledge.retrieve.select import overlap, prompt_tokens, select, shingles
from hal9000.knowledge.tokens import count_tokens

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()


def chunk(cid: int, tokens: int, text: str | None = None, *, parent: int | None = None, rank=1):
    return RetrievedChunk(
        chunk_id=cid,
        document_id=1,
        text=text if text is not None else f"unique text number {cid} " + " ".join(WORDS[:cid]),
        page_start=1,
        page_end=1,
        tokens=tokens,
        prompt_tokens=tokens,
        rank=rank,
        score=1.0 / rank,
        dense_score=0.9 - rank / 100,
        parent_id=parent,
    )


def test_stops_at_the_first_item_that_would_overflow():
    items = [chunk(1, 60, rank=1), chunk(2, 50, rank=2), chunk(3, 10, rank=3)]
    chosen = select(items, 100)
    # 60 fits, 50 would overflow: selection stops, the later 10 is not squeezed in
    assert [c.chunk_id for c in chosen.items] == [1] and chosen.tokens == 60
    assert [c.chunk_id for c in select(items, 120).items] == [1, 2, 3]
    assert select(items, 110).tokens == 110
    assert select(items, 59).items == []  # the top item alone overflows: nothing is sent


def test_a_budget_boundary_is_inclusive():
    items = [chunk(1, 40), chunk(2, 60)]
    assert select(items, 100).tokens == 100


def test_children_expand_to_their_parent_counted_once():
    parent = chunk(10, 300, "the whole parent section " + " ".join(WORDS))
    children = [
        chunk(11, 50, "first child words here", parent=10, rank=1),
        chunk(20, 40, "an unrelated chunk of text", rank=2),
        chunk(12, 60, "second child words there", parent=10, rank=3),
    ]
    chosen = select(children, 1000, parents={10: parent})
    assert [c.chunk_id for c in chosen.items] == [10, 20]
    assert chosen.tokens == 340  # the parent's 300 once, plus 40
    assert [c.chunk_id for c in chosen.sources] == [11, 20]
    expanded = chosen.items[0]
    assert expanded.rank == 1 and expanded.dense_score == children[0].dense_score
    assert expanded.text == parent.text and expanded.parent_id is None
    # without the parent row the child is sent as itself
    assert [c.chunk_id for c in select(children, 1000).items] == [11, 20, 12]


def test_near_duplicates_are_skipped_and_cost_nothing():
    text = "The temperature and pressure relief valve is dripping at the water heater"
    items = [
        chunk(1, 30, text, rank=1),
        chunk(2, 30, text.upper() + "!", rank=2),  # the same passage, another file
        chunk(3, 30, "Total due: $625.00 for the inspection", rank=3),
        chunk(4, 30, "relief valve is dripping at the water", rank=4),  # contained in 1
    ]
    chosen = select(items, 90)
    assert [c.chunk_id for c in chosen.items] == [1, 3] and chosen.tokens == 60
    assert overlap(shingles(items[0].text), shingles(items[3].text)) == 1.0


def test_overlap_and_shingles():
    assert overlap(frozenset(), shingles("a b c")) == 0.0
    assert shingles("Café au lait") == shingles("cafe AU lait.")
    assert shingles("two words") == frozenset({("two",), ("words",)})
    a = shingles(" ".join(WORDS))
    b = shingles(" ".join(WORDS[:2] + ["zulu"] + WORDS[3:]))
    assert 0.5 < overlap(a, b) < 0.8  # one changed word is not a duplicate of 13 words


def test_prompt_tokens():
    assert prompt_tokens("", "some text", 7, send_headers=True, encoding="cl100k_base") == 7
    assert prompt_tokens("Head", "some text", 7, send_headers=False, encoding="cl100k_base") == 7
    assert prompt_tokens("Head", "some text", 7, send_headers=True, encoding="cl100k_base") == (
        count_tokens("Head\nsome text")
    )
