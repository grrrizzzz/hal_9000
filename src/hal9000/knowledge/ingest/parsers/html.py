"""``HtmlParser``: HTML via ``selectolax`` (lexbor), split into pseudo-pages (plan §2).

Scripts, styles and other non-content elements are dropped. Block elements
start new lines, table cells are joined with `` | ``, and inline whitespace is
collapsed as a browser would. ``<h1>`` … ``<h6>`` split the text into
sections, and the document becomes one pseudo-page per top-level heading.
"""

from __future__ import annotations

import re
from pathlib import Path

from hal9000.knowledge.ingest.parsers.base import (
    DEFAULT_PSEUDO_PAGE_CHARS,
    ParsedFile,
    Section,
    pages_from_sections,
)
from hal9000.knowledge.ingest.parsers.text import read_text

__all__ = ["HtmlParser", "html_sections"]

_SKIP = ["script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "head"]
_BLOCK = {
    "address", "article", "aside", "blockquote", "dd", "details", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "header", "hr", "li", "main",
    "nav", "ol", "p", "pre", "section", "summary", "table", "tbody", "thead", "tfoot",
    "tr", "ul", "caption",
}  # fmt: skip
_HEADING = re.compile(r"^h([1-6])$")
_SPACES = re.compile(r"[ \t\r\n\f ]+")


class _Collector:
    def __init__(self) -> None:
        self.sections: list[Section] = []
        self.level = 0
        self.lines: list[str] = []
        self.line: list[str] = []

    def text(self, value: str) -> None:
        self.line.append(value)

    def newline(self) -> None:
        joined = _SPACES.sub(" ", "".join(self.line)).strip()
        self.line.clear()
        if joined:
            self.lines.append(joined)

    def heading(self, level: int) -> None:
        self.newline()
        if self.lines:
            self.sections.append(Section(level=self.level, text="\n".join(self.lines) + "\n\n"))
            self.lines.clear()
        self.level = level

    def finish(self) -> list[Section]:
        self.heading(0)
        return self.sections


def _walk(node, out: _Collector, preformatted: bool = False) -> None:
    for child in node.iter(include_text=True):
        tag = child.tag
        if tag == "-text":
            value = child.text_content or ""
            if preformatted:
                for index, part in enumerate(value.split("\n")):
                    if index:
                        out.newline()
                    out.text(part)
            else:
                out.text(value)
            continue
        if tag in _SKIP or tag.startswith("-"):
            continue
        heading = _HEADING.match(tag)
        if heading:
            out.heading(int(heading.group(1)))
            _walk(child, out)
            out.newline()
            continue
        if tag == "br":
            out.newline()
            continue
        if tag in ("td", "th"):
            if out.line and "".join(out.line).strip():
                out.text(" | ")
            _walk(child, out, preformatted)
            continue
        block = tag in _BLOCK
        if block:
            out.newline()
        _walk(child, out, preformatted or tag == "pre")
        if block:
            out.newline()


def html_sections(html: str) -> list[Section]:
    """Visible text of ``html`` as sections split at every heading."""
    from selectolax.lexbor import LexborHTMLParser

    tree = LexborHTMLParser(html)
    tree.strip_tags(_SKIP)
    out = _Collector()
    root = tree.body or tree.root
    if root is not None:
        _walk(root, out)
    return out.finish()


class HtmlParser:
    """HTML → pseudo-pages, one per top-level heading (or ~3,000 characters)."""

    name = "html"

    def __init__(self, max_chars: int = DEFAULT_PSEUDO_PAGE_CHARS) -> None:
        self.max_chars = max_chars

    def parse(self, path: Path) -> ParsedFile:
        sections = html_sections(read_text(path))
        return ParsedFile(
            parser=self.name,
            pages=pages_from_sections(sections, self.max_chars),
            pseudo_pages=True,
        )
