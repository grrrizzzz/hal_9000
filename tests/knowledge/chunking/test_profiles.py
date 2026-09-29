"""Index profile naming and parsing (plan §6.2; handoff KWP2 acceptance)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hal9000.knowledge.chunking import (
    EMBED_ALIASES,
    FAKE_EMBED_DIMENSIONS,
    FAKE_EMBED_MODEL,
    ChunkInput,
    chunk_document,
    make_profile,
    overlap_tokens,
    parse_profile_name,
    profile_name,
)
from hal9000.knowledge.contracts import IndexProfile
from hal9000.knowledge.tokens import count_tokens, encoding_for_model


def test_plan_example_parses_to_its_fields() -> None:
    p = parse_profile_name("pdfium.structure-768.hdr.te3s-1536")
    assert p == IndexProfile(
        name="pdfium.structure-768.hdr.te3s-1536",
        parser="pdfium",
        chunker="structure",
        size=768,
        overlap=0,
        header=True,
        embed_model="openai/text-embedding-3-small",
        dimensions=1536,
    )


def test_percent_overlap_and_no_header() -> None:
    p = parse_profile_name("pdfium.fixed-512-o15.nohdr.te3l-1024")
    assert (p.chunker, p.size, p.overlap, p.header) == ("fixed", 512, 77, False)
    assert p.embed_model == "openai/text-embedding-3-large"
    assert p.dimensions == 1024
    assert profile_name(p) == "pdfium.fixed-512-o15.nohdr.te3l-1024"


def test_parent_child_and_token_overlap() -> None:
    p = parse_profile_name("docling.parent_child-256-p1500-o11t.hdr.te3s-1536")
    assert (p.parser, p.chunker, p.size, p.parent_size, p.overlap) == (
        "docling",
        "parent_child",
        256,
        1500,
        11,
    )


def test_overlap_is_named_as_a_percent_when_exact_else_in_tokens() -> None:
    assert overlap_tokens(512, 15) == 77
    assert overlap_tokens(128, 15) == 19
    fifteen = make_profile(
        parser="pdfium", chunker="fixed", size=128, overlap=19, embed_model=EMBED_ALIASES["te3s"],
        dimensions=1536,
    )  # fmt: skip
    assert fifteen.name == "pdfium.fixed-128-o15.hdr.te3s-1536"
    odd = fifteen.model_copy(update={"size": 1000, "overlap": 151})
    assert profile_name(odd) == "pdfium.fixed-1000-o151t.hdr.te3s-1536"


def test_make_profile_names_itself() -> None:
    p = make_profile(
        parser="pdfium",
        chunker="structure",
        size=768,
        embed_model="openai/text-embedding-3-small",
        dimensions=1536,
    )
    assert p.name == "pdfium.structure-768.hdr.te3s-1536"
    assert p.id is None and p.active is False


@pytest.mark.parametrize(
    "name",
    [
        "",
        "pdfium",
        "pdfium.structure-768.hdr",
        "pdfium.chonky-768.hdr.te3s-1536",  # unknown chunker
        "pdfium.structure-768.hdr.ada-1536",  # unknown embedding alias
        "pdfium.structure-0768.hdr.te3s-1536",  # leading zero: not canonical
        "pdfium.structure-768-o0.hdr.te3s-1536",  # zero overlap is written by omission
        "pdfium.fixed-512-o77t.hdr.te3s-1536",  # 77 tokens is 15%: canonical form is o15
        "pdfium.fixed-512-o100.hdr.te3s-1536",  # overlap >= size
        "pdfium.parent_child-256.hdr.te3s-1536",  # parent_child needs a parent size
        "pdfium.structure-768-p1500.hdr.te3s-1536",  # only parent_child has one
        "pdf.ium.structure-768.hdr.te3s-1536",
        "PDFIUM.structure-768.hdr.te3s-1536",
        "pdfium.structure-768.header.te3s-1536",
        "pdfium.structure-768.hdr.te3s-1536.extra",
        "pdfium.fixed-4.hdr.te3s-1536",  # size below 8 tokens
        "pdfium.parent_child-256-p128.hdr.te3s-1536",  # parent smaller than its children
    ],
)
def test_bad_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        parse_profile_name(name)


def test_unnameable_profiles_are_rejected() -> None:
    base = make_profile(
        parser="pdfium", chunker="fixed", size=512, embed_model=EMBED_ALIASES["te3s"],
        dimensions=1536,
    )  # fmt: skip
    with pytest.raises(ValueError, match="alias"):
        profile_name(base.model_copy(update={"embed_model": "text-embedding-3-small"}))
    with pytest.raises(ValueError, match="parser"):
        profile_name(base.model_copy(update={"parser": "pdf.ium"}))


# -------------------------------------------------------------- round trip
@st.composite
def profiles(draw) -> IndexProfile:
    chunker = draw(st.sampled_from(["fixed", "recursive", "structure", "parent_child"]))
    size = draw(st.integers(8, 5000))
    overlap = draw(
        st.one_of(
            st.just(0),
            st.integers(0, size - 1),
            st.integers(1, 99).map(lambda pct: min(overlap_tokens(size, pct), size - 1)),
        )
    )
    return make_profile(
        parser=draw(st.from_regex(r"[a-z0-9_]{1,12}", fullmatch=True)),
        chunker=chunker,
        size=size,
        overlap=overlap,
        parent_size=draw(st.integers(size, 6000)) if chunker == "parent_child" else None,
        header=draw(st.booleans()),
        embed_model=draw(st.sampled_from(sorted(EMBED_ALIASES.values()))),
        dimensions=draw(st.integers(1, 4096)),
    )


@settings(max_examples=300, deadline=None, database=None)
@given(profiles())
def test_profile_names_round_trip(p: IndexProfile) -> None:
    name = profile_name(p)
    assert name == p.name
    assert parse_profile_name(name) == p
    assert profile_name(parse_profile_name(name)) == name


@settings(
    max_examples=300, deadline=None, database=None, suppress_health_check=[HealthCheck.too_slow]
)
@given(st.text(st.sampled_from("abcdefhinoprstu0123456789_.-"), max_size=48))
def test_parse_accepts_only_canonical_names(name: str) -> None:
    try:
        p = parse_profile_name(name)
    except ValueError:
        return
    assert profile_name(p) == name


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        (dict(chunker="fixed", size=7), "size"),
        (dict(chunker="parent_child", size=512, parent_size=256), "parent_size"),
        (dict(chunker="fixed", size=64, overlap=64), "overlap"),
    ],
)
def test_make_profile_rejects_sizes_the_chunkers_would_refuse(fields: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_profile(parser="pdfium", embed_model=EMBED_ALIASES["te3s"], dimensions=8, **fields)


def test_fake_embedder_alias_counts_tokens_like_the_embedding_models() -> None:
    p = make_profile(
        parser="pdfium",
        chunker="structure",
        size=768,
        embed_model=FAKE_EMBED_MODEL,
        dimensions=FAKE_EMBED_DIMENSIONS,
    )
    assert p.name == "pdfium.structure-768.hdr.fake-64"
    assert parse_profile_name(p.name) == p
    assert encoding_for_model(FAKE_EMBED_MODEL) == "cl100k_base"
    assert encoding_for_model(EMBED_ALIASES["te3s"]) == "cl100k_base"
    text = "Überprüfung der Warmwasserbereitung. " * 40
    (d, *_) = chunk_document(ChunkInput(document_id=1, text=text), p)
    assert d.tokens == count_tokens(d.text, "cl100k_base") != count_tokens(d.text, "o200k_base")
