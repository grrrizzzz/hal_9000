"""``TextParser``: plain text and Markdown, split into pseudo-pages (plan §2).

Markdown is split at its top-level ATX headings (``#`` … ``######``, outside
fenced code blocks); plain text has no headings and is split by size. The
pseudo-pages are exact slices of the file's text, so nothing is lost.
"""

from __future__ import annotations

import re
from pathlib import Path

from hal9000.knowledge.ingest.parsers.base import (
    DEFAULT_PSEUDO_PAGE_CHARS,
    ParsedFile,
    ParseError,
    Section,
    pages_from_sections,
)

__all__ = ["MARKDOWN_SUFFIXES", "TextParser", "markdown_sections", "read_text"]

MARKDOWN_SUFFIXES = (".md", ".markdown")
_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")


def read_text(path: Path) -> str:
    """Decode a text file (UTF-8 with a BOM tolerated, else Latin-1), newlines normalised."""
    raw = Path(path).read_bytes()
    if b"\x00" in raw[:4096]:
        raise ParseError(f"Could not parse {Path(path).name}: it looks like a binary file.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def markdown_sections(text: str) -> list[Section]:
    """Split Markdown at every ATX heading; the text before the first is level 0."""
    sections: list[Section] = []
    level, start, offset, fenced = 0, 0, 0, False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        heading = None if fenced else _ATX.match(line)
        if heading:
            if offset > start:
                sections.append(Section(level=level, text=text[start:offset]))
            level, start = len(heading.group(1)), offset
        offset += len(line)
    if offset > start:
        sections.append(Section(level=level, text=text[start:offset]))
    return sections


class TextParser:
    """TXT and Markdown. ``markdown=None`` decides by the file suffix."""

    name = "text"

    def __init__(
        self, markdown: bool | None = None, max_chars: int = DEFAULT_PSEUDO_PAGE_CHARS
    ) -> None:
        self.markdown = markdown
        self.max_chars = max_chars

    def parse(self, path: Path) -> ParsedFile:
        text = read_text(path)
        markdown = self.markdown
        if markdown is None:
            markdown = Path(path).suffix.lower() in MARKDOWN_SUFFIXES
        sections = markdown_sections(text) if markdown else [Section(level=0, text=text)]
        return ParsedFile(
            parser=self.name,
            pages=pages_from_sections(sections, self.max_chars),
            pseudo_pages=True,
        )
