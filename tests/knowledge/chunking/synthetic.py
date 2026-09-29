"""Synthetic, deterministic test documents for the chunker suite (no house data, rule K2).

``inspection_doc`` builds an inspection-like report: a title, a summary list
that repeats finding titles, three numbered sections of findings written as
"3.1.2 Plumbing – Water Heater: Valve Leak" with a body and a recommendation,
hard-wrapped lines like PDF text, and one oversized finding. It records every
finding's span so tests don't depend on the unit detector.

The Hypothesis strategies generate document-like text (headings, findings,
paragraphs, lists) and wild text (arbitrary Unicode, unbroken runs) for the
property tests.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date

from hypothesis import strategies as st

from hal9000.knowledge.chunking import ChunkInput

WORDS = (
    "the valve water heater pipe drain roof shingle gutter flashing attic vent duct "
    "furnace filter panel breaker outlet wire joist beam slab crawlspace moisture "
    "stain crack leak drip corrosion rust seal caulk grout tile window door frame "
    "deck railing stair fence grade slope soil siding stucco paint trim fascia "
    "was is appears shows needs should be observed noted at near under above along "
    "with from for and or of on in a an minor major active prior evidence damage "
    "repair replace monitor evaluate licensed qualified contractor plumber electrician"
).split()

SYSTEMS = [
    ("ROOF", "Roof", ["Shingles", "Flashing", "Gutters", "Vents"]),
    ("EXTERIOR", "Exterior", ["Siding", "Trim", "Grading", "Deck"]),
    ("PLUMBING", "Plumbing", ["Water Heater", "Supply Lines", "Drains", "Fixtures"]),
]
ISSUES = ["Valve Leak", "Granule Loss", "Loose Fasteners", "Corrosion", "Missing Seal", "Damage"]

META = dict(
    title="Synthetic Home Inspection",
    doc_type="inspection_report",
    vendor="Example Inspections LLC",
    doc_date=date(2023, 6, 22),
)


def _sentence(rng: random.Random, lo: int = 6, hi: int = 18) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(lo, hi))]
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."


def _wrap(text: str, width: int = 78) -> str:
    """Hard-wrap like PDF-extracted text."""
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True)
class Finding:
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class SyntheticDoc:
    input: ChunkInput
    findings: list[Finding]
    oversized: Finding


def inspection_doc(seed: int = 7, oversized_paragraphs: int = 24) -> SyntheticDoc:
    """An inspection-like report with numbered findings and one oversized finding."""
    rng = random.Random(seed)
    parts: list[str] = []
    findings: list[Finding] = []
    pos = 0

    def add(text: str) -> tuple[int, int]:
        nonlocal pos
        start = pos
        parts.append(text)
        pos += len(text)
        return start, pos

    plan: list[tuple[str, str, list[tuple[str, str]]]] = []
    for s, (caps, name, components) in enumerate(SYSTEMS, start=1):
        subs = []
        for c, component in enumerate(components[:2], start=1):
            for f in range(1, 3):
                number = f"{s}.{c}.{f}"
                issue = ISSUES[(s + c + f) % len(ISSUES)]
                subs.append((number, f"{number} {name} – {component}: {issue}"))
        plan.append((f"{s} {caps}", name, subs))
    oversized_title = "3.2.3 Plumbing – Supply Lines: Corrosion Throughout"

    add("SYNTHETIC HOME INSPECTION REPORT\n")
    add(_wrap("This report is synthetic and describes no real property. " + _sentence(rng)))
    add("\n\nSUMMARY\n")
    add("\n".join(title for _, _, subs in plan for _, title in subs) + "\n" + oversized_title)
    for heading, _, subs in plan:
        add(f"\n\n{heading}\n")
        add(_wrap(_sentence(rng)))
        current_sub = ""
        for number, title in subs:
            sub = number.rsplit(".", 1)[0]
            if sub != current_sub:
                current_sub = sub
                add(f"\n\n{sub} {title.split(' – ')[1].split(':')[0]}")
            add("\n\n")
            body = " ".join(_sentence(rng) for _ in range(rng.randint(2, 5)))
            text = f"{title}\n{_wrap(body)}\nRecommendation: {_sentence(rng, 5, 12)}"
            start, end = add(text)
            findings.append(Finding(title, start, end))
    add("\n\n")
    paragraphs = [
        _wrap(" ".join(_sentence(rng) for _ in range(5))) for _ in range(oversized_paragraphs)
    ]
    body = "\n\n".join(paragraphs)
    start, end = add(f"{oversized_title}\n{body}\nRecommendation: {_sentence(rng, 5, 12)}")
    oversized = Finding(oversized_title, start, end)
    findings.append(oversized)
    add("\n\nEND OF REPORT\n")
    text = "".join(parts)

    # Page breaks every ~1,800 characters, at paragraph starts.
    page_starts = [(0, 1)]
    for match in re.finditer(r"\n\n", text):
        if match.end() - page_starts[-1][0] >= 1800:
            page_starts.append((match.end(), page_starts[-1][1] + 1))
    doc = ChunkInput(document_id=3, text=text, page_starts=tuple(page_starts), **META)
    return SyntheticDoc(doc, findings, oversized)


# ----------------------------------------------------------------- strategies
_word = st.sampled_from(WORDS)


@st.composite
def _sentence_st(draw) -> str:
    words = draw(st.lists(_word, min_size=1, max_size=20))
    end = draw(st.sampled_from([".", ".", ".", "!", "?", ":", ""]))
    return words[0].capitalize() + "".join(" " + w for w in words[1:]) + end


@st.composite
def _paragraph(draw) -> str:
    sentences = draw(st.lists(_sentence_st(), min_size=1, max_size=8))
    joiners = draw(st.lists(st.sampled_from([" ", " ", "\n"]), min_size=len(sentences)))
    return "".join(s + j for s, j in zip(sentences, joiners)).rstrip()


@st.composite
def _numbered(draw) -> str:
    depth = draw(st.integers(1, 3))
    number = ".".join(str(draw(st.integers(1, 20))) for _ in range(depth))
    title = " ".join(w.capitalize() for w in draw(st.lists(_word, min_size=1, max_size=6)))
    if depth == 1:
        title = title.upper()
    return f"{number} {title}"


@st.composite
def _finding(draw) -> str:
    return f"{draw(_numbered())}\n{draw(_paragraph())}\nRecommendation: {draw(_sentence_st())}"


_caps = st.lists(_word, min_size=1, max_size=5).map(lambda ws: " ".join(ws).upper())
_markdown = st.tuples(st.integers(1, 4), st.lists(_word, min_size=1, max_size=5)).map(
    lambda t: "#" * t[0] + " " + " ".join(t[1]).capitalize()
)
_bullets = st.lists(_sentence_st(), min_size=1, max_size=6).map(
    lambda items: "\n".join(f"- {i}" for i in items)
)

_block = st.one_of(
    _paragraph(), _paragraph(), _finding(), _finding(), _numbered(), _caps, _markdown, _bullets
)


@st.composite
def doc_texts(draw, max_blocks: int = 30) -> str:
    """Document-like text: headings, findings, paragraphs and lists; sentences stay short."""
    blocks = draw(st.lists(_block, min_size=1, max_size=max_blocks))
    seps = draw(
        st.lists(st.sampled_from(["\n", "\n\n", "\n\n\n", " \n\n", "\n \n"]), min_size=len(blocks))
    )
    lead = draw(st.sampled_from(["", "", "\n", "  "]))
    return lead + "".join(b + s for b, s in zip(blocks, seps))


_no_surrogates = st.characters(blacklist_categories=("Cs",))
_wild = st.one_of(
    st.text(_no_surrogates, max_size=2500),
    st.text(st.sampled_from("abcxyz019.-/ \n"), max_size=4000),
    st.text(st.sampled_from("abcxyz0123456789é中😀"), min_size=1, max_size=6000),
)


@st.composite
def any_texts(draw) -> str:
    """Document-like text, wild text, or both glued together."""
    kind = draw(st.integers(0, 3))
    if kind == 0:
        return draw(doc_texts())
    if kind == 1:
        return draw(_wild)
    return draw(doc_texts(max_blocks=10)) + draw(_wild) + draw(doc_texts(max_blocks=10))


@st.composite
def chunk_inputs(draw, texts=None) -> ChunkInput:
    """A ``ChunkInput`` with random page breaks and header metadata."""
    text = draw(texts if texts is not None else any_texts())
    breaks = sorted(set(draw(st.lists(st.integers(1, max(1, len(text))), max_size=8))))
    first = draw(st.integers(1, 60))
    page_starts = ((0, first), *((b, first + i + 1) for i, b in enumerate(breaks)))
    meta = draw(st.sampled_from([META, {}, dict(title="Synthetic Invoice", doc_type="invoice")]))
    return ChunkInput(document_id=1, text=text, page_starts=page_starts, **meta)


sizes = st.one_of(
    st.sampled_from([16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536]), st.integers(8, 2000)
)
