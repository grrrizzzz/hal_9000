"""The synthetic fixture corpus: file paths and ground truth for every knowledge test.

Import it from any knowledge-base test::

    from tests.knowledge.fixtures import corpus

    parsed = parse_file(corpus.INSPECTION.path, isolate=False)
    assert parsed.page_count == corpus.INSPECTION.page_count

All content is invented (see :mod:`.build_fixture_corpus`, which generates the
files and holds their text). Page numbers are 1-based. Quotes are verbatim
substrings of the **cleaned** page text once whitespace is collapsed (a quote
never spans a line break in the PDFs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tests.knowledge.fixtures import build_fixture_corpus as _b

__all__ = [
    "ALL_FILES",
    "BUNDLE",
    "CorpusFile",
    "DOCX",
    "ESTIMATE",
    "ExpectedDocument",
    "ExpectedFinding",
    "FILES_DIR",
    "FINDINGS",
    "FINDING_IDS",
    "HTML",
    "INSPECTION",
    "INVOICE",
    "LETTER_SPACED",
    "MARKDOWN",
    "PDF_FILES",
    "QAFact",
    "QA_FACTS",
    "REPEATED_LINE",
    "RUNNING_LINES",
    "TEXT",
    "UNNUMBERED",
]

FILES_DIR: Path = _b.FILES_DIR


@dataclass(frozen=True)
class ExpectedDocument:
    """One logical sub-document inside a file (what segmentation should find)."""

    page_start: int
    page_end: int
    doc_type: str
    vendor: str | None
    doc_date: date | None
    title: str


@dataclass(frozen=True)
class CorpusFile:
    """One fixture file and what parsing, cleaning and segmentation should make of it."""

    name: str
    page_count: int
    documents: tuple[ExpectedDocument, ...]
    no_text_pages: tuple[int, ...] = ()
    pseudo_pages: bool = False
    #: pseudo-page files: the heading each pseudo-page starts with (after any title)
    page_headings: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return FILES_DIR / self.name

    @property
    def mime_type(self) -> str:
        return {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".html": "text/html",
            ".md": "text/markdown",
            ".txt": "text/plain",
        }[self.path.suffix]


@dataclass(frozen=True)
class ExpectedFinding:
    """A numbered inspection finding (``3.1.2 Plumbing – Water Heater: Valve Leak``)."""

    id: str
    heading: str
    severity: str
    recommendation: str
    page: int  # in inspection.pdf, and in bundle.pdf (the inspection comes first)


@dataclass(frozen=True)
class QAFact:
    """A question with its answer and verbatim evidence, for retrieval and answer tests."""

    question: str
    answer: str
    file: str
    page: int
    quote: str
    category: str = "factoid"


_INSPECTION_DOC = ExpectedDocument(
    1, 12, "inspection_report", _b.INSPECTION_VENDOR, date.fromisoformat(_b.INSPECTION_DATE),
    "Home Inspection Report",
)  # fmt: skip
_ESTIMATE_DOC = ExpectedDocument(
    1, 2, "estimate", _b.ESTIMATE_VENDOR, date.fromisoformat(_b.ESTIMATE_DATE), "Estimate TPR-58213"
)
_INVOICE_DOC = ExpectedDocument(
    1, 1, "invoice", _b.INVOICE_VENDOR, date.fromisoformat(_b.INVOICE_DATE), "Invoice BE-10442"
)


def _shift(document: ExpectedDocument, by: int) -> ExpectedDocument:
    return ExpectedDocument(
        document.page_start + by,
        document.page_end + by,
        document.doc_type,
        document.vendor,
        document.doc_date,
        document.title,
    )


INSPECTION = CorpusFile(
    "inspection.pdf", _b.INSPECTION_PAGES, (_INSPECTION_DOC,), no_text_pages=(_b.PHOTO_PAGE,)
)
ESTIMATE = CorpusFile("estimate.pdf", 2, (_ESTIMATE_DOC,))
INVOICE = CorpusFile("invoice.pdf", 1, (_INVOICE_DOC,))
BUNDLE = CorpusFile(
    "bundle.pdf",
    15,
    (_INSPECTION_DOC, _shift(_ESTIMATE_DOC, 12), _shift(_INVOICE_DOC, 14)),
    no_text_pages=(_b.PHOTO_PAGE,),
)
UNNUMBERED = CorpusFile(
    "unnumbered_bundle.pdf",
    5,
    (
        ExpectedDocument(1, 1, "invoice", _b.UNNUMBERED_VENDORS[0],
                         date.fromisoformat(_b.UNNUMBERED_DATES[0]), "Invoice DPS-1001"),
        ExpectedDocument(2, 2, "invoice", _b.UNNUMBERED_VENDORS[0],
                         date.fromisoformat(_b.UNNUMBERED_DATES[1]), "Invoice"),
        ExpectedDocument(3, 4, "estimate", _b.UNNUMBERED_VENDORS[1],
                         date.fromisoformat(_b.UNNUMBERED_DATES[2]), "Proposal HPR-311"),
        ExpectedDocument(5, 5, "completion_notice", _b.UNNUMBERED_VENDORS[2],
                         date.fromisoformat(_b.UNNUMBERED_DATES[3]), "Notice Of Work Completed"),
    ),
)  # fmt: skip
DOCX = CorpusFile(
    "heat_pump_guide.docx",
    3,
    (ExpectedDocument(1, 3, "manual", "Harborline", None, _b.DOCX_TITLE),),
    pseudo_pages=True,
    page_headings=_b.DOCX_HEADINGS,
)
HTML = CorpusFile(
    "irrigation_notes.html",
    3,
    (ExpectedDocument(1, 3, "other", None, None, _b.HTML_TITLE),),
    pseudo_pages=True,
    page_headings=_b.HTML_SECTIONS,
)
MARKDOWN = CorpusFile(
    "garage_door.md",
    3,
    (ExpectedDocument(1, 3, "other", None, None, "Garage Door Opener"),),
    pseudo_pages=True,
    page_headings=tuple(f"## {name}" for name in _b.MARKDOWN_SECTIONS),
)
TEXT = CorpusFile(
    "pool_log.txt",
    3,
    (ExpectedDocument(1, 3, "other", None, None, "pool_log"),),
    pseudo_pages=True,
)

ALL_FILES: tuple[CorpusFile, ...] = (
    INSPECTION,
    ESTIMATE,
    INVOICE,
    BUNDLE,
    UNNUMBERED,
    DOCX,
    HTML,
    MARKDOWN,
    TEXT,
)
PDF_FILES: tuple[CorpusFile, ...] = (INSPECTION, ESTIMATE, INVOICE, BUNDLE, UNNUMBERED)

FINDINGS: tuple[ExpectedFinding, ...] = tuple(
    ExpectedFinding(f.id, f.heading, f.severity, f.recommendation, f.page) for f in _b.FINDINGS
)
FINDING_IDS: tuple[str, ...] = tuple(f.id for f in FINDINGS)

#: Strings from the running header and footer: none may survive cleaning of a
#: multi-page PDF (inspection.pdf, bundle.pdf).
RUNNING_LINES: tuple[str, ...] = (
    "Docusign Envelope ID",
    "5E7C1A90-3B2D-4C6E-9F10-A1B2C3D4E5F6",
    "Inspector: Jordan Avery Reyes | Report",
    "Page 3 of 12",
    "of 12",
)
#: Repeats on exactly two inspection pages (3 and 4), in the footer zone: cleaning keeps it.
REPEATED_LINE: str = _b.REPEATED_LINE
#: Letter-spaced headings as extracted (estimate.pdf) → their repaired form.
LETTER_SPACED: dict[str, str] = dict(_b.LETTER_SPACED)

QA_FACTS: tuple[QAFact, ...] = (
    QAFact("What is the invoice number for the home inspection?", "CHI-INV-20250314",
           "inspection.pdf", 11, "Invoice number: CHI-INV-20250314"),
    QAFact("How much did the home inspection cost in total?", "$625.00",
           "inspection.pdf", 11, "Total due: $625.00", "money"),
    QAFact("What is wrong with the water heater's relief valve?", "It is dripping.",
           "inspection.pdf", 7, "The temperature and pressure relief valve is dripping"),
    QAFact("How old is the water heater?", "13 years old (manufactured in 2012).",
           "inspection.pdf", 7, "manufactured in 2012 and is 13 years old"),
    QAFact("Where is the main water shutoff valve?", "At the front of the house, left of the garage door.",
           "inspection.pdf", 8, "The main water shutoff valve is located at the front of the house"),
    QAFact("What severity was the double-tapped breaker given?", "Safety Hazard",
           "inspection.pdf", 5, "2.1.1 Electrical – Service Panel: Double-Tapped Breaker"),
    QAFact("What is the total of the plumbing estimate?", "$605.00",
           "estimate.pdf", 1, "Total: $605.00", "money"),
    QAFact("How long is the plumbing estimate valid?", "30 days",
           "estimate.pdf", 2, "This estimate is valid for 30 days.", "date"),
    QAFact("How much did the electrician charge?", "$340.00",
           "invoice.pdf", 1, "Total: $340.00", "money"),
    QAFact("How often should the heat pump filter be replaced?", "Every 90 days (60 with pets).",
           "heat_pump_guide.docx", 1, "every 90 days, or every 60 days when pets live in the house"),
    QAFact("How long is the compressor warranty?", "10 years, if registered within 60 days.",
           "heat_pump_guide.docx", 3, "The compressor carries a 10-year limited warranty"),
    QAFact("When do the lawns get watered?", "Monday, Wednesday and Friday at 5:00 AM for 12 minutes.",
           "irrigation_notes.html", 2, "Lawns run Monday, Wednesday and Friday at 5:00 AM for 12 minutes."),
    QAFact("Which breaker feeds the garage door opener?", "Breaker 14 in the garage subpanel.",
           "garage_door.md", 1, "breaker 14 in the garage subpanel"),
)  # fmt: skip
