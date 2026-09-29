"""``DoclingParser``: the layout-aware candidate parser (plan §4, §8.3).

Docling gives reading order, headings and tables as Markdown, at the cost of
PyTorch and layout models downloaded from Hugging Face. It lives in the
optional ``knowledge-docling`` group and is imported only when a parse runs
(rule K11); its tests carry the ``models`` marker. OCR and picture handling are
switched off (plan §2: text only). The text-layer flag comes from PDFium, so it
means the same thing for both parsers.
"""

from __future__ import annotations

from pathlib import Path

from hal9000.knowledge.ingest.parsers.base import ParsedFile, ParsedPage, ParseError
from hal9000.knowledge.ingest.parsers.pdfium import PdfiumParser

__all__ = ["DoclingParser", "docling_available"]


def docling_available() -> bool:
    """Whether the ``knowledge-docling`` group is installed (it does not check the models)."""
    import importlib.util

    return importlib.util.find_spec("docling") is not None


class DoclingParser:
    """PDF → per-page Markdown via Docling; no OCR, no pictures."""

    name = "docling"

    def __init__(self) -> None:
        self._converter = None

    def _get_converter(self):  # pragma: no cover - needs the models (``models`` marker)
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = False
            options.generate_picture_images = False
            options.generate_page_images = False
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def parse(self, path: Path) -> ParsedFile:
        if not docling_available():
            raise ParseError(
                "The docling parser is not installed. Install it with "
                "`uv sync --group knowledge-docling`, or use the pdfium parser."
            )
        # PDFium supplies the page count and text-layer flags (and rejects a
        # corrupt file with a readable error before Docling loads its models).
        layer = PdfiumParser().parse(path)
        return self._convert(path, layer)  # pragma: no cover - ``models`` marker

    def _convert(self, path: Path, layer: ParsedFile) -> ParsedFile:  # pragma: no cover
        try:
            result = self._get_converter().convert(str(path))
        except Exception as error:  # Docling wraps many backends; report them uniformly
            raise ParseError(f"Could not parse {Path(path).name} with docling: {error}") from error
        document = result.document
        pages: list[ParsedPage] = []
        for raw in layer.pages:
            text = ""
            if raw.has_text_layer:
                text = document.export_to_markdown(page_no=raw.page, image_placeholder="")
            pages.append(ParsedPage(page=raw.page, text=text, has_text_layer=raw.has_text_layer))
        return ParsedFile(parser=self.name, pages=pages)
