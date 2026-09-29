"""`eval export` (the leak test) and `eval report` (aggregates only), on the fixture corpus."""

from __future__ import annotations

import json

import pytest

from hal9000.knowledge.contracts import MetricCI
from hal9000.knowledge.eval.export import (
    LEAK_CHARS,
    LeakDetected,
    export_runs,
    find_leaks,
    page_texts,
)
from hal9000.knowledge.eval.report import pareto_front, render_report, write_report
from hal9000.knowledge.eval.runs import question_record, record_run
from hal9000.knowledge.eval.scoring import ScoringConfig, ScoringItem, aggregate, score_question
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem
from tests.knowledge.eval.conftest import store_documents
from tests.knowledge.fixtures import corpus

CFG = ScoringConfig(n_boot=50)


def _page_with(docs, quote):
    for doc in docs:
        for text in doc.pages.values():
            if quote in text:
                return text
    raise AssertionError(quote)


def fixture_run(conn, docs, eval_dir, *, config, stage="S1a", split="dev", hits=None, latency=20.0):
    """Score every QA fact on real page text; records carry question text like real runs."""
    scores, records = [], []
    facts = corpus.QA_FACTS if hits is None else corpus.QA_FACTS[:hits]
    for i, fact in enumerate(corpus.QA_FACTS, 1):
        item = ScoringItem(i, fact.category, (fact.quote,))
        page = _page_with(docs, fact.quote) if fact in facts else "Nothing relevant."
        out = RetrievalOutput([SelectedItem(i, page, 400)], abstain_score=0.7, latency_ms=latency)
        s = score_question(item, out, config=CFG)
        scores.append(s)
        records.append(question_record(s, question=fact.question, selected=[i]))
    unanswerable = ScoringItem(99, "unanswerable", ())
    s = score_question(unanswerable, RetrievalOutput([], abstain_score=0.2), config=CFG)
    scores.append(s)
    records.append(question_record(s, question="Who painted the fence?"))
    metrics = aggregate(scores, config=CFG).model_copy(update={"cost_per_question_usd": 0.00002})
    return record_run(conn, eval_dir=eval_dir, stage=stage, split=split, config=config,
                      metrics=metrics, records=records, host="mac")  # fmt: skip


# --------------------------------------------------------------------- leaks
def test_find_leaks_flags_strings_longer_than_40_characters():
    page = "The temperature and pressure relief valve is dripping at the discharge pipe."
    assert find_leaks("x" + page[:LEAK_CHARS] + "y", [page]) == []
    leak = page[5 : 5 + LEAK_CHARS + 1]
    assert find_leaks(f'{{"a": "{leak}"}}', [page]) == [leak]
    assert find_leaks("prefix " + page + " suffix", [page]) == [page]
    assert find_leaks("short", [page]) == [] and find_leaks(page, []) == []


def test_eval_export_contains_no_page_text(kb_db, tmp_path, fixture_docs):
    store_documents(kb_db, fixture_docs)
    run = fixture_run(
        kb_db, fixture_docs, tmp_path, config={"profile": "p1", "size": 512, "sweep_id": "sw"}
    )
    path = export_runs(kb_db, run.id, eval_dir=tmp_path)
    text = path.read_text()
    pages = page_texts(kb_db)
    assert len(pages) > 30
    assert find_leaks(text, pages) == []  # the leak test (handoff KWP5)
    for fact in corpus.QA_FACTS:
        assert fact.question not in text and fact.quote not in text
    data = json.loads(text)
    (exported,) = data["runs"]
    assert (
        exported["id"] == run.id and exported["metrics"]["budgeted_recall"]["2000"]["value"] == 1.0
    )
    q = exported["questions"][0]
    assert set(q) >= {
        "gold_item_id",
        "category",
        "recall",
        "full_support",
        "first_hit_rank",
        "abstain_score",
    }
    assert "question" not in q and "selected" not in q
    assert path == tmp_path / "runs" / run.id / "export.json"
    # the whole sweep, by its id
    assert len(json.loads(export_runs(kb_db, "sw", eval_dir=tmp_path).read_text())["runs"]) == 1


def test_an_export_that_would_leak_is_refused(kb_db, tmp_path, fixture_docs):
    store_documents(kb_db, fixture_docs)
    page = fixture_docs[0].pages[7]
    run = fixture_run(kb_db, fixture_docs, tmp_path, config={"profile": "p1", "note": page[:80]})
    with pytest.raises(LeakDetected):
        export_runs(kb_db, run.id, eval_dir=tmp_path)
    assert not (tmp_path / "runs" / run.id / "export.json").exists()
    with pytest.raises(KeyError):
        export_runs(kb_db, "missing", eval_dir=tmp_path)


# -------------------------------------------------------------------- report
def _sweep(conn, docs, eval_dir):
    runs = []
    for chunker, size, latency, hits in (
        ("fixed", 256, 12.0, 8),
        ("fixed", 512, 15.0, 11),
        ("structure", 512, 30.0, 13),
        ("structure", 1536, 25.0, 13),
    ):
        config = {"profile": f"pdfium.{chunker}-{size}", "chunker": chunker, "size": size,
                  "parser": "pdfium", "sweep_id": "sw1"}  # fmt: skip
        runs.append(fixture_run(conn, docs, eval_dir, config=config, hits=hits, latency=latency))
    runs.append(
        fixture_run(
            conn,
            docs,
            eval_dir,
            config={"profile": "pdfium.structure-512", "sweep_id": "sw1"},
            stage="S6",
            latency=90.0,
        )  # fmt: skip
    )
    runs.append(
        fixture_run(
            conn,
            docs,
            eval_dir,
            config={"profile": "pdfium.structure-512", "sweep_id": "sw1"},
            split="test",
        )  # fmt: skip
    )
    return runs


def test_eval_report_has_aggregates_only(kb_db, tmp_path, fixture_docs):
    store_documents(kb_db, fixture_docs)
    runs = _sweep(kb_db, fixture_docs, tmp_path)
    path = write_report(kb_db, "sw1", tmp_path / "docs" / "KNOWLEDGE_EVAL_REPORT.md")
    text = path.read_text()
    assert find_leaks(text, page_texts(kb_db)) == []
    for fact in corpus.QA_FACTS:
        assert fact.question not in text
    assert "### T1: Chunk size" in text and "### T3: Chunking strategy" in text
    t5 = text.split("### T5: Parser")[1].split("###")[0]
    assert "Not varied in these runs." in t5
    assert "## Pareto data" in text and "| yes |" in text
    assert "## Chosen configuration" in text and "gate G-K2" in text
    assert "## Test split" in text and runs[-1].id in text
    assert "N/A" in text  # size 1536 has no recall @ 1k
    summary = text.split("## Summary by stage")[1].split("##")[0]
    assert "| S1a | 5 | dev, test |" in summary and "| S6 | 1 |" in summary
    assert "money" in text.split("## Recall @ 2k by category")[1]
    with pytest.raises(KeyError):
        write_report(kb_db, "nope", tmp_path / "r.md")


def test_report_handles_runs_without_metrics_or_latency(kb_db, tmp_path):
    from hal9000.knowledge.eval.runs import start_run

    run = start_run(kb_db, stage="S1a", config={"profile": "p", "size": 256}, host="mac")
    text = render_report([run], source=run.id)
    assert "No runs with both recall @ 2k and latency yet." in text
    assert "No per-category results yet." in text and "Not run yet." in text


def test_pareto_front():
    points = [("a", 0.9, 30.0), ("b", 0.8, 10.0), ("c", 0.7, 20.0), ("d", 0.9, 40.0)]
    assert pareto_front(points) == {"a", "b"}


def test_metric_formatting_in_tables():
    from hal9000.knowledge.eval.report import _fmt

    assert _fmt(MetricCI(value=0.5, low=0.4, high=0.6)) == "0.500 [0.400-0.600]"
    assert _fmt(None) == "N/A"
