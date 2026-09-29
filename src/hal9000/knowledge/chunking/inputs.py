"""What the chunkers read and write (plan §6.1).

``ChunkInput`` is one logical document as the chunkers see it: its cleaned
text (boilerplate already removed by KWP1's cleaning), where each page starts
in that text, optional structural units (headings, numbered findings) and the
metadata the contextual header is built from. KWP1's parse/clean/segment
output is adapted to it; :meth:`ChunkInput.from_pages` covers the common case.

``ChunkDraft`` is one chunk before it has a database id. ``text`` is exactly
``input.text[char_start:char_end]`` and is what citations display;
``index_text`` (header + text) is what gets embedded and keyword-indexed.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from hal9000.knowledge.contracts import Chunk

__all__ = ["ChunkDraft", "ChunkInput", "Role", "Unit"]

Role = Literal["chunk", "parent", "child"]


@dataclass(frozen=True, slots=True)
class Unit:
    """A structural unit: a heading or a numbered finding.

    ``start`` is the offset of its first character (the heading line) in the
    cleaned text. With ``end=None`` the unit runs to the next unit of the same
    or a higher level (a smaller ``level`` number), like a heading; an explicit
    ``end`` bounds it exactly. Units nest by containment, which gives each
    chunk its section path.
    """

    start: int
    title: str
    level: int = 1
    end: int | None = None
    kind: Literal["heading", "finding"] = "heading"


@dataclass(frozen=True, slots=True)
class ChunkInput:
    """One document's cleaned text, page map, structure hints and header metadata.

    ``page_starts`` holds ``(offset, page_number)`` pairs: the first at offset
    0, offsets strictly increasing and at most ``len(text)``, page numbers
    ascending and ≥ 1. ``units=None`` means "detect them" (numbered findings,
    Markdown and upper-case headings); ``units=()`` means "this document has
    no structure".
    """

    document_id: int
    text: str
    page_starts: tuple[tuple[int, int], ...] = ((0, 1),)
    units: tuple[Unit, ...] | None = None
    title: str = ""
    doc_type: str = ""
    vendor: str | None = None
    doc_date: date | None = None
    _offsets: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        starts = tuple((int(o), int(p)) for o, p in self.page_starts)
        if not starts or starts[0][0] != 0:
            raise ValueError("page_starts must begin with the page at offset 0")
        for (o1, p1), (o2, p2) in zip(starts, starts[1:]):
            if o2 <= o1 or p2 <= p1:
                raise ValueError("page_starts offsets and page numbers must increase")
        if starts[0][1] < 1:
            raise ValueError("page numbers start at 1")
        if starts[-1][0] > len(self.text):
            raise ValueError("a page starts beyond the end of the text")
        object.__setattr__(self, "page_starts", starts)
        object.__setattr__(self, "_offsets", tuple(o for o, _ in starts))
        if self.units is not None:
            units = tuple(self.units)
            for u in units:
                end = len(self.text) if u.end is None else u.end
                if not 0 <= u.start < end <= len(self.text) or u.level < 1:
                    raise ValueError(f"unit {u.title!r} has an invalid span or level")
            object.__setattr__(self, "units", units)

    @classmethod
    def from_pages(
        cls,
        document_id: int,
        pages: Iterable[tuple[int, str]],
        *,
        joiner: str = "\n\n",
        **meta,
    ) -> ChunkInput:
        """Join cleaned ``(page_number, text)`` pages into one document text."""
        parts: list[str] = []
        starts: list[tuple[int, int]] = []
        pos = 0
        for number, text in pages:
            if parts:
                parts.append(joiner)
                pos += len(joiner)
            starts.append((pos, number))
            parts.append(text)
            pos += len(text)
        return cls(document_id=document_id, text="".join(parts), page_starts=tuple(starts), **meta)

    def page_of(self, offset: int) -> int:
        """The page holding the character at ``offset``."""
        return self.page_starts[max(0, bisect_right(self._offsets, offset) - 1)][1]


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk before storage. ``key`` is its index in the chunker's output."""

    key: int
    document_id: int
    char_start: int
    char_end: int
    text: str
    header: str
    tokens: int
    page_start: int
    page_end: int
    section_path: tuple[str, ...] = ()
    role: Role = "chunk"
    parent_key: int | None = None

    @property
    def index_text(self) -> str:
        """What is embedded and keyword-indexed: the header line, then the text."""
        return f"{self.header}\n{self.text}" if self.header else self.text

    @property
    def searchable(self) -> bool:
        """Parents of ``parent_child`` are expanded into prompts, not searched."""
        return self.role != "parent"

    def to_chunk(self, profile_id: int, parent_id: int | None = None) -> Chunk:
        """The storage contract; ``parent_id`` is the stored id of ``parent_key``'s draft."""
        return Chunk(
            profile_id=profile_id,
            document_id=self.document_id,
            page_start=self.page_start,
            page_end=self.page_end,
            char_start=self.char_start,
            char_end=self.char_end,
            text=self.text,
            header=self.header,
            tokens=self.tokens,
            parent_id=parent_id,
        )
