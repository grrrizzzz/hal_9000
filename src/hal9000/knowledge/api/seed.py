"""Seeding a library through the real ingestion pipeline (``kb serve --fake``, tests).

A :class:`SeedFile` is uploaded with :meth:`IngestPipeline.submit`, optionally
given user-labelled documents (page ranges and metadata, which win over
segmentation exactly as a Library edit does), and ingested synchronously.

:func:`fake_corpus` is the synthetic library ``kb serve --fake`` starts with:
the UI's own fixture PDF (``web/src/mocks/fixtures/synthetic-house.pdf``, an
imaginary house at "12 Example Lane") labelled like the Mock Service Worker
data, an air-handler manual and a roof warranty, so the web end-to-end suite
runs against the real server. :func:`seed_evaluation` adds what the
Evaluation page needs: draft and approved question-set items with evidence on
those pages, judge-calibration answers (one already labelled), sweep runs with
paged per-question results, and a second built profile to switch to. Nothing
in it is house data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from hal9000.knowledge.contracts import DocumentMeta, UploadResult

__all__ = [
    "REPO",
    "SeedDocument",
    "SeedFile",
    "fake_corpus",
    "seed_evaluation",
    "seed_library",
]

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


# ------------------------------------------------------------------ evaluation
def _chunk_for(conn: sqlite3.Connection, needle: str) -> tuple[int, int, int] | None:
    """(chunk id, document id, page) of the active profile's chunk containing ``needle``."""
    row = conn.execute(
        "SELECT c.id, c.document_id, c.page_start FROM chunks c "
        "JOIN profiles p ON p.id = c.profile_id AND p.active = 1 "
        "WHERE instr(c.text, ?) > 0 ORDER BY c.id LIMIT 1",
        (needle,),
    ).fetchone()
    return (row[0], row[1], row[2]) if row else None


def _citation(conn: sqlite3.Connection, needle: str):
    from hal9000.knowledge.contracts import Citation
    from hal9000.knowledge.index.library import _document

    found = _chunk_for(conn, needle)
    if found is None:
        return None
    chunk_id, document_id, page = found
    doc = _document(conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone())
    text = conn.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]
    return Citation(
        marker="S1",
        chunk_id=chunk_id,
        document_id=document_id,
        file_sha=doc.file_sha,
        page=page,
        title=doc.title,
        doc_type=doc.doc_type,
        doc_date=doc.doc_date,
        vendor=doc.vendor,
        snippet=text,
    )


def _metrics(size: int, lift: float, n: int, host: str):
    import math

    from hal9000.knowledge.contracts import EvalMetrics, MetricCI

    def ci(value: float, width: float) -> MetricCI:
        value = round(min(0.99, max(0.05, value)), 3)
        return MetricCI(value=value, low=round(max(0.0, value - width), 3),
                        high=round(min(1.0, value + width * 0.9), 3))  # fmt: skip

    recall = 0.78 + lift - (math.log2(size / 480) ** 2) * 0.05
    latency = 18 + size / 60 + (40 if host == "pi" else 0)
    budgeted = {2000: ci(recall, 0.05), 4000: ci(recall + 0.05, 0.045)}
    if size <= 1000:
        budgeted[1000] = ci(recall - 0.08 - size / 20000, 0.06)
    by_category = {
        c: {"recall@2k": round(min(0.99, recall + bias), 3), "mrr": round(recall - 0.05 + bias, 3)}
        for c, bias in (("factoid", 0.0), ("money", 0.04), ("date", 0.0), ("list", -0.08),
                        ("cross_document", -0.14))
    }  # fmt: skip
    return EvalMetrics(
        n_questions=n,
        budgeted_recall=budgeted,
        full_support_hit={5: ci(recall - 0.1, 0.07)},
        mrr=ci(recall - 0.12, 0.06),
        context_precision={2000: ci(0.4 - size / 5000, 0.04)},
        abstain_auc=ci(0.82, 0.06),
        latency_ms_p50=round(latency, 1),
        latency_ms_p95=round(latency * 1.8, 1),
        cost_per_question_usd=round(0.00002 + size / 1536 * 0.00004, 6),
        by_category=by_category,
    )


def _records(n: int, recall: float) -> list[dict]:
    categories = ["factoid", "money", "date", "list", "cross_document", "unanswerable"]
    subjects = ["furnace", "roof", "water heater", "termite repair", "gas connector", "warranty"]
    out = []
    for i in range(n):
        category = categories[i % len(categories)]
        answerable = category != "unanswerable"
        value = round(min(1.0, max(0.0, recall + ((i * 37) % 11 - 5) / 20)), 3) if answerable else 0
        out.append(
            {
                "gold_item_id": 100 + i,
                "question": f"Synthetic question {i + 1} about the {subjects[i % len(subjects)]}?",
                "category": category,
                "recall": {
                    "1000": max(0.0, value - 0.1),
                    "2000": value,
                    "4000": min(1.0, value + 0.06),
                }
                if answerable
                else {},  # fmt: skip
                "full_support": {"5": answerable and value > 0.8},
                "first_hit_rank": 1 + i % 5 if answerable and value >= 0.2 else None,
                "abstain_score": round(0.3 + (i % 7) / 17, 3),
                "abstained": not answerable and i % 4 != 0,
            }
        )
    return out


def seed_evaluation(
    conn: sqlite3.Connection, eval_dir: Path, *, pipeline: Any = None, extra_profile: str = ""
) -> None:
    """Question-set items, judge labels, sweep runs and (with ``pipeline``) a second profile."""
    from hal9000.knowledge.contracts import EvidenceQuote, GoldItem, RubricLabels
    from hal9000.knowledge.eval.runs import record_run
    from hal9000.knowledge.eval.store import insert_gold, insert_judge_label

    furnace = _citation(conn, "installed in 2009")
    gas = _citation(conn, "flexible gas connector")
    invoice = _citation(conn, "Total due")
    sha = furnace.file_sha if furnace else None

    def quote(text: str, page: int) -> list[EvidenceQuote]:
        return [EvidenceQuote(quote=text, page=page, file_sha=sha)]

    items = insert_gold(
        conn,
        [
            GoldItem(
                question="Does the house have a swimming pool heater?",
                category="unanswerable",
                split="dev",
            ),
            GoldItem(
                question="How much did the termite repair cost?",
                reference_answer="$1,234.00 (Invoice 1001, Brightline Pest Co.).",
                category="money",
                difficulty="easy",
                evidence=quote("Total due: $1,234.00", 3),
                split="test",
            ),
            GoldItem(
                question="Is anything wrong with the gas line near the kitchen?",
                reference_answer="A flexible gas connector is routed through the cabinet wall; "
                "an HVAC evaluation was recommended.",
                category="factoid",
                paraphrased=True,
                evidence=quote("A flexible gas connector is routed through the cabinet wall.", 2),
                split="dev",
            ),
            GoldItem(
                question="How old is the furnace according to the inspection?",
                reference_answer="It was installed in 2009, per the June 2023 inspection.",
                category="date",
                difficulty="easy",
                evidence=quote(
                    "It was installed in 2009 and is near the end of its expected life.", 2
                ),
                split="dev",
            ),
            GoldItem(
                question="What type of roof covering does the house have?",
                reference_answer="Composition shingle.",
                category="factoid",
                evidence=quote("The roof covering is composition shingle in fair condition.", 1),
                status="approved",
                split="dev",
            ),
        ],
    )
    by_question = {item.question: item for item in items}
    correct = RubricLabels(correctness="correct", faithful=True, citations_correct=True)
    wrong = RubricLabels(correctness="wrong", faithful=False, citations_correct=False)
    labels = [
        ("How old is the furnace according to the inspection?",
         "The June 2023 inspection says the furnace was installed in 2009 and is near the end of "
         "its expected life [S1].", furnace, None, correct),
        ("How much did the termite repair cost?",
         "Invoice 1001 from Brightline Pest Co. totals $1,234.00 [S1].", invoice, None, correct),
        ("Is anything wrong with the gas line near the kitchen?",
         "The inspection found no issues with the gas line [S1].", gas, wrong,
         RubricLabels(correctness="wrong", faithful=False, citations_correct=True)),
        ("Does the house have a swimming pool heater?",
         "I couldn't find that in the house documents.", None, None,
         RubricLabels(correctness="correct", faithful=True, citations_correct=True,
                      abstention_correct=True)),
    ]  # fmt: skip
    for question, answer, citation, human, judge in labels:
        insert_judge_label(
            conn,
            by_question[question].id,  # type: ignore[arg-type]
            answer,
            eval_run_id="j-001",
            judge=judge,
            human=human,
            citations=[citation] if citation else [],
        )

    sizes = [128, 256, 384, 512, 768, 1024, 1536]
    runs: list[tuple[str, int, float, str, str, dict]] = [
        (chunker, size, lift, "mac", "S1a", {})
        for chunker, lift in (("fixed", 0.0), ("recursive", 0.025), ("structure", 0.05))
        for size in sizes
    ]
    runs += [("parent_child", size, 0.04, "mac", "S1a", {"parent_size": 1500})
             for size in (128, 256, 384, 512)]  # fmt: skip
    runs += [
        ("structure", 512, 0.05, "pi", "S6", {}),
        ("recursive", 384, 0.025, "pi", "S6", {}),
        ("parent_child", 256, 0.04, "pi", "S6", {"parent_size": 1500}),
    ]
    for number, (chunker, size, lift, host, stage, extra) in enumerate(runs, start=1):
        config = {
            "chunker": chunker,
            "size": size,
            "overlap": round(size * 0.15) if chunker == "fixed" else 0,
            "retrieval": "hybrid",
            "embed_model": "fake/embedding-64",
            **extra,
        }
        record_run(
            conn,
            eval_dir=eval_dir,
            stage=stage,
            config=config,
            metrics=_metrics(size, lift, 58, host),
            records=_records(58, 0.78 + lift),
            run_id=f"{stage.lower()}-{number:03d}",
            host=host,
            cost_usd=0.002,
        )
    if pipeline is not None and extra_profile:
        pipeline.build(conn, extra_profile)
