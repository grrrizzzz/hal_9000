"""Parsers: stored original → raw per-page text (plan §5.1 step 4).

:func:`parse_file` is the entry point the ingest worker uses. It picks a
parser by file type (or by name, for the evaluation's parser comparison),
runs it **in a child process with a time limit**, and wraps the whole step in
``stage("parse")`` (rule K7). A corrupt or unsupported file, a crash or a
timeout all surface as :class:`ParseError` with a message that can be shown on
the Library's error line.

Parsers:

========== ============================ ==========================
name       formats                      pages
========== ============================ ==========================
pdfium     PDF (default)                real pages
docling    PDF (``knowledge-docling``)  real pages
docx       DOCX                         pseudo-pages
html       HTML                         pseudo-pages
text       TXT, Markdown                pseudo-pages
========== ============================ ==========================
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from hal9000.knowledge.ingest.parsers.base import (
    DEFAULT_PSEUDO_PAGE_CHARS,
    ParsedFile,
    ParsedPage,
    ParseError,
    Parser,
    ParseTimeout,
    Section,
    pages_from_sections,
    split_text,
    top_level,
)
from hal9000.knowledge.ingest.parsers.docling import DoclingParser
from hal9000.knowledge.ingest.parsers.docx import DocxParser
from hal9000.knowledge.ingest.parsers.html import HtmlParser
from hal9000.knowledge.ingest.parsers.pdfium import PdfiumParser
from hal9000.knowledge.ingest.parsers.text import TextParser
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = [
    "DEFAULT_PSEUDO_PAGE_CHARS",
    "PARSERS",
    "SUFFIXES",
    "DoclingParser",
    "DocxParser",
    "HtmlParser",
    "ParseError",
    "ParseTimeout",
    "ParsedFile",
    "ParsedPage",
    "Parser",
    "PdfiumParser",
    "Section",
    "TextParser",
    "get_parser",
    "pages_from_sections",
    "parse_file",
    "parser_for",
    "split_text",
    "top_level",
]

PARSERS: dict[str, type] = {
    "pdfium": PdfiumParser,
    "docling": DoclingParser,
    "docx": DocxParser,
    "html": HtmlParser,
    "text": TextParser,
}

#: File suffix → parser name; ``pdf`` is resolved to the chosen PDF parser.
SUFFIXES: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".md": "text",
    ".markdown": "text",
    ".txt": "text",
}


def get_parser(name: str, *, max_chars: int = DEFAULT_PSEUDO_PAGE_CHARS) -> Parser:
    """A parser instance by name (see the module table)."""
    try:
        cls = PARSERS[name]
    except KeyError:
        raise ParseError(
            f"Unknown parser {name!r}. Choose one of: {', '.join(sorted(PARSERS))}."
        ) from None
    if name in ("docx", "html", "text"):
        return cls(max_chars=max_chars)
    return cls()


def parser_for(
    path: Path | str,
    *,
    pdf_parser: str = "pdfium",
    max_chars: int = DEFAULT_PSEUDO_PAGE_CHARS,
) -> Parser:
    """The parser for ``path``'s type. Unsupported types raise :class:`ParseError`."""
    suffix = Path(path).suffix.lower()
    kind = SUFFIXES.get(suffix)
    if kind is None:
        allowed = ", ".join(sorted(SUFFIXES))
        raise ParseError(
            f"Could not parse {Path(path).name}: {suffix or 'no extension'} files are not "
            f"supported ({allowed})."
        )
    return get_parser(pdf_parser if kind == "pdf" else kind, max_chars=max_chars)


def _check(result: ParsedFile, path: Path) -> ParsedFile:
    if not result.pages:
        raise ParseError(f"Could not parse {path.name}: no text was found in the file.")
    numbers = [page.page for page in result.pages]
    if numbers != list(range(1, len(numbers) + 1)):  # pragma: no cover - parser bug guard
        raise ParseError(f"Parser {result.parser} returned pages out of order for {path.name}.")
    return result


def _child(conn, parser: Parser, path: str) -> None:  # pragma: no cover - runs in a child
    try:
        conn.send(("ok", parser.parse(Path(path))))
    except ParseError as error:
        conn.send(("parse_error", str(error)))
    except BaseException as error:  # noqa: BLE001 - reported to the parent, not swallowed
        conn.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        conn.close()


def _run_isolated(parser: Parser, path: Path, timeout_s: float) -> ParsedFile:
    """Run ``parser.parse(path)`` in a fresh process; kill it after ``timeout_s``.

    A thread can't be stopped once PDFium is stuck inside C code, so the limit is
    enforced on a process. ``spawn`` is used because the server has threads.
    """
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(sender, parser, str(path)), daemon=True)
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_s):
            raise ParseTimeout(
                f"Could not parse {path.name}: parsing took longer than {timeout_s:g} s and "
                "was stopped."
            )
        try:
            kind, payload = receiver.recv()
        except EOFError:
            process.join(5)
            raise ParseError(
                f"Could not parse {path.name}: the {parser.name} parser crashed "
                f"(exit code {process.exitcode})."
            ) from None
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():  # pragma: no cover - terminate is normally enough
                process.kill()
                process.join()
    if kind == "ok":
        return payload
    if kind == "parse_error":
        raise ParseError(payload)
    raise ParseError(f"Could not parse {path.name}: {payload}")


def parse_file(
    path: Path | str,
    *,
    parser: Parser | str | None = None,
    timeout_s: float | None = None,
    isolate: bool = True,
    max_chars: int | None = None,
) -> ParsedFile:
    """Parse ``path`` into raw per-page text, timed as ``stage("parse")``.

    Args:
        parser: a parser, a parser name, or ``None`` to choose by file type.
        timeout_s: the time limit; ``None`` reads ``ingest.parse_timeout_seconds``
            from ``config/knowledge.yaml``.
        isolate: run in a child process with the time limit (the default). ``False``
            runs in-process with no limit, for tests and one-off scripts.
        max_chars: pseudo-page size; ``None`` reads ``ingest.pseudo_page_chars``.
    """
    path = Path(path)
    with stage(Stage.PARSE):
        if max_chars is None or (timeout_s is None and isolate):
            from hal9000.knowledge.config import load_knowledge_file

            ingest = load_knowledge_file().ingest
            max_chars = max_chars if max_chars is not None else ingest.pseudo_page_chars
            if timeout_s is None:
                timeout_s = ingest.parse_timeout_seconds
        if parser is None:
            chosen = parser_for(path, max_chars=max_chars)
        elif isinstance(parser, str):
            chosen = get_parser(parser, max_chars=max_chars)
        else:
            chosen = parser
        if not path.is_file():
            raise ParseError(f"Could not parse {path.name}: the file does not exist.")
        if isolate:
            result = _run_isolated(chosen, path, float(timeout_s))
        else:
            result = chosen.parse(path)
        return _check(result, path)
