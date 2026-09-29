"""``chunk_document``: profile dispatch, the ``chunk`` stage (rule K7), inputs and speed."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from hal9000.knowledge.chunking import (
    ChunkInput,
    chunk,
    chunk_document,
    make_profile,
)
from hal9000.knowledge.obs.timing import REGISTRY, start_trace
from hal9000.knowledge.tokens import count_tokens

from .synthetic import META, inspection_doc

DOC = inspection_doc()


def _profile(chunker: str, **kw):
    return make_profile(
        parser="pdfium",
        chunker=chunker,
        size=kw.pop("size", 256),
        embed_model="openai/text-embedding-3-small",
        dimensions=1536,
        **kw,
    )


@pytest.mark.parametrize(
    ("chunker", "extra"),
    [
        ("fixed", {"overlap": 38}),
        ("recursive", {}),
        ("structure", {}),
        ("parent_child", {"parent_size": 1500}),
    ],
)
def test_chunk_document_follows_the_profile(chunker: str, extra: dict) -> None:
    profile = _profile(chunker, **extra)
    drafts = chunk_document(DOC.input, profile)
    expected = chunk(
        DOC.input,
        chunker,
        size=profile.size,
        overlap=profile.overlap,
        parent_size=profile.parent_size,
        header=True,
    )
    assert drafts == expected


def test_header_follows_the_profile() -> None:
    drafts = chunk_document(DOC.input, _profile("structure", header=False))
    assert all(d.header == "" for d in drafts)


def test_chunking_is_timed_as_the_chunk_stage() -> None:
    before = REGISTRY.get_sample_value("kb_stage_seconds_count", {"stage": "chunk"}) or 0.0
    with start_trace(kind="ingest") as trace:
        chunk_document(DOC.input, _profile("fixed"))
    after = REGISTRY.get_sample_value("kb_stage_seconds_count", {"stage": "chunk"})
    assert after == before + 1
    assert [s.stage.value for s in trace.stages] == ["chunk"]


def test_tokens_are_measured_with_the_profiles_encoding() -> None:
    """A chat-family model would use o200k_base; the embedding models use cl100k_base."""
    text = "Überprüfung der Warmwasserbereitung. " * 200
    inp = ChunkInput(document_id=1, text=text)
    small = chunk_document(inp, _profile("fixed", size=64))
    assert all(d.tokens == count_tokens(d.text, "cl100k_base") for d in small)
    o200 = chunk(inp, "fixed", size=64, encoding_name="o200k_base")
    assert all(count_tokens(d.text, "o200k_base") == d.tokens <= 64 for d in o200)
    assert len(o200) != len(small)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (dict(size=4), "size"),
        (dict(size=64, overlap=64), "overlap"),
        (dict(size=64, overlap=-1), "overlap"),
    ],
)
def test_bad_parameters_are_rejected(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        chunk(DOC.input, "fixed", **kw)


def test_unknown_chunker_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunker"):
        chunk(DOC.input, "semantic", size=256)  # type: ignore[arg-type]


# -------------------------------------------------------------------- inputs
def test_from_pages_joins_pages_and_maps_offsets() -> None:
    inp = ChunkInput.from_pages(7, [(4, "Page four text."), (5, ""), (6, "Page six text.")], **META)
    assert inp.text == "Page four text.\n\n\n\nPage six text."
    assert inp.page_starts == ((0, 4), (17, 5), (19, 6))
    assert inp.page_of(0) == 4 and inp.page_of(16) == 4 and inp.page_of(19) == 6
    (d,) = chunk(inp, "fixed", size=64)
    assert (d.page_start, d.page_end) == (4, 6)


@pytest.mark.parametrize(
    "page_starts",
    [(), ((5, 1),), ((0, 1), (0, 2)), ((0, 2), (5, 1)), ((0, 0),), ((0, 1), (99, 2))],
)
def test_bad_page_maps_are_rejected(page_starts) -> None:
    with pytest.raises(ValueError, match="page"):
        ChunkInput(document_id=1, text="short text", page_starts=page_starts)


def test_units_outside_the_text_are_rejected() -> None:
    from hal9000.knowledge.chunking import Unit

    with pytest.raises(ValueError, match="unit"):
        ChunkInput(document_id=1, text="short", units=(Unit(start=9, title="x"),))
    with pytest.raises(ValueError, match="unit"):
        ChunkInput(document_id=1, text="short", units=(Unit(start=3, end=2, title="x"),))


def test_empty_and_blank_documents_give_no_chunks() -> None:
    for text in ["", "   \n\n\t "]:
        inp = ChunkInput(document_id=1, text=text)
        for name in ["fixed", "recursive", "structure"]:
            assert chunk(inp, name, size=64) == []
        assert chunk(inp, "parent_child", size=64, parent_size=128) == []


# ----------------------------------------------------------------------- speed
def test_chunking_a_35k_token_corpus_takes_well_under_a_second() -> None:
    """The evaluation re-chunks the ~35k-token corpus about 60 times (plan §8.4)."""
    docs = [inspection_doc(seed=s, oversized_paragraphs=24).input for s in range(11)]
    total = sum(count_tokens(d.text) for d in docs)
    assert 30_000 <= total <= 45_000
    for name, kw in [
        ("fixed", dict(size=128, overlap=19)),
        ("recursive", dict(size=512)),
        ("structure", dict(size=768)),
        ("parent_child", dict(size=128, parent_size=1500)),
    ]:
        chunk(docs[0], name, **kw)  # warm the encodings
        start = time.perf_counter()
        for d in docs:
            chunk(replace(d, text=d.text + " "), name, **kw)  # not cached
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, (name, elapsed)
