"""The parser protocol, its output shape and the pseudo-page splitter (plan §2, §5.1 step 4).

A parser turns one stored original into **raw per-page text**. It does no
cleaning (that is :mod:`hal9000.knowledge.ingest.clean`) and ignores images
(plan §2: text only). PDFs have real pages; formats without pages (DOCX, HTML,
Markdown, TXT) are split into *pseudo-pages*: one per top-level heading, or
about 3,000 characters when there are no headings.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_PSEUDO_PAGE_CHARS",
    "ParseError",
    "ParseTimeout",
    "ParsedFile",
    "ParsedPage",
    "Parser",
    "Section",
    "pages_from_sections",
    "split_text",
    "top_level",
]

#: Pseudo-page size when a document has no headings (plan §2; ``ingest.pseudo_page_chars``).
DEFAULT_PSEUDO_PAGE_CHARS = 3000


class ParseError(RuntimeError):
    """A file could not be parsed. The message is meant for the Library's error line."""


class ParseTimeout(ParseError):
    """Parsing ran past its time limit (plan §5.1 step 4)."""


@dataclass(frozen=True)
class ParsedPage:
    """One page (or pseudo-page) of raw extracted text.

    ``has_text_layer`` is ``False`` for a PDF page with no extractable text (a
    scan or a photo page). Such a page keeps its number, with empty ``text``, so
    page numbers stay aligned with the original.
    """

    page: int
    text: str
    has_text_layer: bool = True


@dataclass(frozen=True)
class ParsedFile:
    """What a parser returns: the parser's name and the pages in order (1-based, no gaps)."""

    parser: str
    pages: list[ParsedPage] = field(default_factory=list)
    pseudo_pages: bool = False

    @property
    def page_count(self) -> int:
        return len(self.pages)


@runtime_checkable
class Parser(Protocol):
    """Anything that turns a file into :class:`ParsedFile`. ``name`` is stored with each page."""

    name: str

    def parse(self, path: Path) -> ParsedFile: ...  # pragma: no cover


# ------------------------------------------------------------ pseudo-pages
@dataclass(frozen=True)
class Section:
    """A run of text under one heading (``level`` 0 = text before any heading)."""

    level: int
    text: str


def top_level(levels: Sequence[int]) -> int | None:
    """The heading level that splits a document into pseudo-pages.

    The highest level (smallest number) that occurs at least twice; a single
    top heading is treated as the document's title. With only one heading of
    any level, that level is used.
    """
    present = sorted(set(level for level in levels if level > 0))
    if not present:
        return None
    for level in present:
        if sum(1 for value in levels if value == level) >= 2:
            return level
    return present[0]


def _break_score(text: str, index: int) -> int:
    """How good a place ``index`` is to end a pseudo-page (higher is better)."""
    before = text[:index]
    if before.endswith("\n\n"):
        return 4
    if before.endswith("\n"):
        return 3
    if re.search(r"[.!?][\"')\]]?\s$", before[-3:]):
        return 2
    if before.endswith((" ", "\t")):
        return 1
    return 0


def split_text(text: str, max_chars: int = DEFAULT_PSEUDO_PAGE_CHARS) -> list[str]:
    """Split ``text`` into balanced pieces of at most about ``max_chars``.

    The pieces are exact, contiguous slices: ``"".join(result) == text``. Each
    piece ends at the best nearby break (blank line, line end, sentence end,
    space), searched near an even share of what is left, so 7,000 characters
    become three pieces of about 2,300 rather than 3,000 + 3,000 + 1,000.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    start = 0
    while len(text) - start > max_chars:
        remaining = len(text) - start
        count = math.ceil(remaining / max_chars)
        target = start + remaining // count
        window = max(1, max_chars // 5)
        low = max(start + 1, target - window)
        high = min(start + max_chars, target + window)
        best, best_key = high, (-1, 0)
        for index in range(low, high + 1):
            key = (_break_score(text, index), -abs(index - target))
            if key > best_key:
                best, best_key = index, key
        pieces.append(text[start:best])
        start = best
    pieces.append(text[start:])
    return pieces


def pages_from_sections(
    sections: Sequence[Section], max_chars: int = DEFAULT_PSEUDO_PAGE_CHARS
) -> list[ParsedPage]:
    """Group sections into pseudo-pages: one per top-level heading, or by size.

    Text before the first top-level heading (a title, a preamble) joins the
    first pseudo-page. A top-level section longer than ``2 × max_chars`` is split
    further, so a citation never points at a 20-page "page".
    """
    split_level = top_level([section.level for section in sections])
    # [seen a top-level heading?, texts]; preamble joins the first heading's page
    groups: list[tuple[list[bool], list[str]]] = []
    for section in sections:
        is_top = split_level is not None and section.level == split_level
        if not groups or (is_top and groups[-1][0][0]):
            groups.append(([is_top], [section.text]))
            continue
        if is_top:
            groups[-1][0][0] = True
        groups[-1][1].append(section.text)
    texts: list[str] = []
    if split_level is None:
        texts = split_text("".join("".join(group) for _, group in groups), max_chars)
    else:
        for _, group in groups:
            body = "".join(group)
            texts.extend(split_text(body, max_chars) if len(body) > 2 * max_chars else [body])
    texts = [text for text in texts if text.strip()]
    return [ParsedPage(page=index, text=text) for index, text in enumerate(texts, start=1)]
