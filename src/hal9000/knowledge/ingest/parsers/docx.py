"""``DocxParser``: Word documents via ``python-docx``, split into pseudo-pages (plan §2).

Paragraphs and tables are read in document order. Headings are recognised by
their style (``Heading 1`` … ``Heading 9``) or an explicit outline level, and
the document is split into one pseudo-page per top-level heading. Table rows
become ``cell | cell`` lines. Images and drawings are ignored.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from hal9000.knowledge.ingest.parsers.base import (
    DEFAULT_PSEUDO_PAGE_CHARS,
    ParsedFile,
    ParseError,
    Section,
    pages_from_sections,
)

__all__ = ["DocxParser"]

_HEADING_STYLE = re.compile(r"^heading\s*([1-9])$", re.IGNORECASE)
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _heading_level(paragraph) -> int:
    style = paragraph.style
    while style is not None:
        match = _HEADING_STYLE.match(style.name or "")
        if match:
            return int(match.group(1))
        style = style.base_style
    ppr = paragraph._p.pPr
    found = ppr.find(f"{_W}outlineLvl") if ppr is not None else None
    value = found.get(f"{_W}val", "") if found is not None else ""
    if value.isdigit() and int(value) < 9:
        return int(value) + 1
    return 0


class DocxParser:
    """DOCX → pseudo-pages, one per top-level heading (or ~3,000 characters)."""

    name = "docx"

    def __init__(self, max_chars: int = DEFAULT_PSEUDO_PAGE_CHARS) -> None:
        self.max_chars = max_chars

    def parse(self, path: Path) -> ParsedFile:
        import docx
        from docx.opc.exceptions import PackageNotFoundError
        from docx.table import Table

        try:
            document = docx.Document(str(path))
        except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as error:
            raise ParseError(
                f"Could not parse {Path(path).name}: the file is not a readable Word document."
            ) from error

        sections: list[Section] = []
        level, buffer = 0, []

        def flush() -> None:
            if buffer:
                sections.append(Section(level=level, text="".join(buffer)))
                buffer.clear()

        for block in document.iter_inner_content():
            if isinstance(block, Table):
                for row in block.rows:
                    cells: list[str] = []
                    for cell in row.cells:
                        value = " ".join(cell.text.split())
                        if not cells or cells[-1] != value:  # merged cells repeat
                            cells.append(value)
                    if any(cells):
                        buffer.append(" | ".join(cells) + "\n")
                buffer.append("\n")
                continue
            heading = _heading_level(block)
            if heading:
                flush()
                level = heading
            text = block.text
            buffer.append(text + ("\n\n" if heading else "\n"))
        flush()
        return ParsedFile(
            parser=self.name,
            pages=pages_from_sections(sections, self.max_chars),
            pseudo_pages=True,
        )
