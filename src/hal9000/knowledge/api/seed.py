"""Seeding a library through the real ingestion pipeline (``kb serve --fake``, tests).

A :class:`SeedFile` is uploaded with :meth:`IngestPipeline.submit`, optionally
given user-labelled documents (page ranges and metadata, which win over
segmentation exactly as a Library edit does), and ingested synchronously.

:func:`fake_corpus` is the synthetic library ``kb serve --fake`` starts with:
the UI's own fixture PDF (``web/src/mocks/fixtures/synthetic-house.pdf``, an
imaginary house at "12 Example Lane") labelled like the Mock Service Worker
data, a furnace manual and a roof warranty, so the web end-to-end suite runs
against the real server. Nothing in it is house data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from hal9000.knowledge.contracts import DocumentMeta, UploadResult

__all__ = ["REPO", "SeedDocument", "SeedFile", "fake_corpus", "seed_library"]

REPO = Path(__file__).resolve().parents[4]
WEB_FIXTURE_PDF = REPO / "web" / "src" / "mocks" / "fixtures" / "synthetic-house.pdf"


@dataclass(frozen=True)
class SeedDocument:
    page_start: int
    page_end: int
    title: str
    doc_type: str = "other"
    vendor: str | None = None
    doc_date: date | None = None


@dataclass(frozen=True)
class SeedFile:
    relative_path: str
    source: bytes | Path
    documents: Sequence[SeedDocument] = field(default_factory=tuple)


def seed_library(
    conn: sqlite3.Connection, pipeline: Any, files: Sequence[SeedFile]
) -> list[UploadResult]:
    """Upload and ingest every file now; returns the upload results."""
    from hal9000.knowledge.index.library import insert_documents

    worker = pipeline.worker(":seed:")
    results: list[UploadResult] = []
    for seed in files:
        result = pipeline.submit(conn, seed.source, seed.relative_path)
        results.append(result)
        if result.status != "added" or result.job_id is None:
            continue
        if seed.documents:
            assert result.sha256 is not None
            with conn:
                insert_documents(
                    conn,
                    [
                        DocumentMeta(
                            file_sha=result.sha256,
                            page_start=d.page_start,
                            page_end=d.page_end,
                            title=d.title,
                            doc_type=d.doc_type,  # type: ignore[arg-type]
                            vendor=d.vendor,
                            doc_date=d.doc_date,
                            meta_source="user",
                        )
                        for d in seed.documents
                    ],
                )
        job = worker.run_job(conn, result.job_id)
        if job.status != "done":
            raise RuntimeError(f"seeding {seed.relative_path} failed: {job.error}")
    return results


_MANUAL = (
    "# Air handler care\n\n"
    "Replace the air filter every 90 days, or monthly during heavy use. "
    "Use a MERV 8 filter sized 16x25x1.\n\n"
    "Keep the area around the unit clear by at least 30 inches.\n\n"
    "## Troubleshooting\n\n"
    "If the burners do not light, check that the gas valve is open and the thermostat "
    "is calling for heat.\n\n"
    "A blinking red status light with three flashes means the pressure switch did not close.\n"
)

_WARRANTY = (
    "Roof workmanship warranty\n\n"
    "Acme Roofing warrants the installation of the composition shingle roof at "
    "12 Example Lane for ten years from completion on March 3, 2021.\n"
)


def fake_corpus() -> list[SeedFile]:
    """The synthetic library of ``kb serve --fake`` (see the module notes)."""
    files: list[SeedFile] = []
    if WEB_FIXTURE_PDF.is_file():
        files.append(
            SeedFile(
                "Inspection/synthetic-house.pdf",
                WEB_FIXTURE_PDF,
                (
                    SeedDocument(
                        1,
                        2,
                        "Home inspection report",
                        "inspection_report",
                        "Sample Inspections Co.",
                        date(2023, 6, 22),
                    ),
                    SeedDocument(
                        3,
                        3,
                        "Invoice 1001 — termite repair",
                        "invoice",
                        "Brightline Pest Co.",
                        date(2023, 7, 14),
                    ),
                ),
            )
        )
    files.append(
        SeedFile(
            "Appliances/Furnace/furnace-manual.md",
            _MANUAL.encode(),
            (SeedDocument(1, 1, "Air handler care and troubleshooting", "manual"),),
        )
    )
    files.append(
        SeedFile(
            "Warranties/roof-warranty.txt",
            _WARRANTY.encode(),
            (
                SeedDocument(
                    1, 1, "Roof workmanship warranty", "warranty", "Acme Roofing", date(2021, 3, 3)
                ),
            ),
        )
    )
    return files
