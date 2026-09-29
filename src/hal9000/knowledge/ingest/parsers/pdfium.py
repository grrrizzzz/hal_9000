"""``PdfiumParser``: the default PDF parser (plan §4), built on ``pypdfium2``.

Per-page text comes from PDFium's text page, which also reports whether the
page has a text layer at all: a page with no characters (a scan, a photo page)
is flagged ``has_text_layer=False`` and kept with empty text, so page numbers
stay aligned with the original. Images are never extracted (plan §2).
"""

from __future__ import annotations

from pathlib import Path

from hal9000.knowledge.ingest.parsers.base import ParsedFile, ParsedPage, ParseError

__all__ = ["PdfiumParser", "open_pdf"]


def open_pdf(path: Path):
    """Open ``path`` with PDFium, turning its errors into a readable :class:`ParseError`."""
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as error:
        detail = str(error)
        if "password" in detail.lower():
            reason = "the PDF is password-protected"
        else:
            reason = f"the file is not a readable PDF ({detail.rstrip('.')})"
        raise ParseError(f"Could not parse {Path(path).name}: {reason}.") from error


class PdfiumParser:
    """Per-page text and a text-layer flag via PDFium."""

    name = "pdfium"

    def parse(self, path: Path) -> ParsedFile:
        import pypdfium2 as pdfium

        document = open_pdf(path)
        pages: list[ParsedPage] = []
        try:
            for index in range(len(document)):
                page = document[index]
                try:
                    textpage = page.get_textpage()
                    try:
                        count = textpage.count_chars()
                        text = textpage.get_text_range() if count > 0 else ""
                    finally:
                        textpage.close()
                except pdfium.PdfiumError as error:
                    raise ParseError(
                        f"Could not parse {Path(path).name}: page {index + 1} is unreadable "
                        f"({error})."
                    ) from error
                finally:
                    page.close()
                has_text = bool(text.strip())
                pages.append(ParsedPage(page=index + 1, text=text, has_text_layer=has_text))
        finally:
            document.close()
        if not pages:
            raise ParseError(f"Could not parse {Path(path).name}: the PDF has no pages.")
        return ParsedFile(parser=self.name, pages=pages)
