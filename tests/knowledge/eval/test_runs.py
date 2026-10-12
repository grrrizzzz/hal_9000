"""eval_runs rows and files, as KWP8's collector and the results API read them."""

from __future__ import annotations

import json

import pytest

from hal9000.knowledge.eval.runs import (
    finish_run,
    get_run,
    list_runs,
    load_results,
    new_run_id,
    question_record,
    record_run,
    results_page,
    run_dir,
    runs_for,
    start_run,
)
from hal9000.knowledge.eval.scoring import ScoringConfig, ScoringItem, aggregate, score_question
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem
from hal9000.knowledge.obs.collector import KnowledgeDbCollector

QUOTE = "The main water shutoff valve is located at the front of the house"
CFG = ScoringConfig(n_boot=50)


def scored(n: int = 4):
    scores, records = [], []
    for i in range(1, n + 1):
        unanswerable = i == n
        item = ScoringItem(
            i, "unanswerable" if unanswerable else "factoid", () if unanswerable else (QUOTE,)
        )
        text = QUOTE if i % 2 else "Nothing relevant in this chunk of text at all."
        out = RetrievalOutput(
            [SelectedItem(10 + i, text, 30)], abstain_score=0.1 * i, abstained=unanswerable
        )
        s = score_question(item, out, config=CFG, count=lambda t: len(t.split()))
        scores.append(s)
        records.append(question_record(s, question=f"Question {i}?", selected=[10 + i]))
    return aggregate(scores, config=CFG), records


def samples(db_path) -> dict:
    out = {}
    for family in KnowledgeDbCollector(db_path).collect():
        for s in family.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def test_a_finished_run_is_what_the_collector_exposes(kb_db, tmp_path):
    metrics, records = scored()
    run = start_run(
        kb_db,
        stage="S1a",
        config={"profile": "pdfium.fixed-512.hdr.te3s-1536", "size": 512},
        host="mac",
    )
    db_path = tmp_path / "knowledge.db"
    assert not [k for k in samples(db_path) if k[0] == "kb_eval_metric"]  # unfinished: hidden
    done = finish_run(
        kb_db, run, eval_dir=tmp_path / "eval", metrics=metrics, records=records, cost_usd=0.01
    )
    got = samples(db_path)
    key = (
        "kb_eval_metric",
        (("metric", "budgeted_recall@2000"), ("profile", "pdfium.fixed-512.hdr.te3s-1536")),
    )
    assert got[key] == pytest.approx(metrics.budgeted_recall[2000].value)
    assert 0.0 <= got[key] <= 1.0
    assert (
        "kb_eval_metric",
        (("metric", "mrr"), ("profile", "pdfium.fixed-512.hdr.te3s-1536")),
    ) in got
    assert done.finished_at and done.cost_usd == 0.01
    folder = run_dir(tmp_path / "eval", run.id)
    assert json.loads((folder / "config.json").read_text())["size"] == 512
    assert json.loads((folder / "metrics.json").read_text())["n_questions"] == 4
    assert get_run(kb_db, run.id) == done


def test_a_later_stage_wins_in_the_collector(kb_db, tmp_path):
    metrics, records = scored()
    low = metrics.model_copy(
        update={"mrr": metrics.mrr.model_copy(update={"value": 0.1, "low": 0.1})}
    )
    record_run(
        kb_db,
        eval_dir=tmp_path,
        stage="S2",
        config={"profile": "p"},
        metrics=metrics,
        records=records,
    )
    record_run(
        kb_db, eval_dir=tmp_path, stage="S1a", config={"profile": "p"}, metrics=low, records=records
    )
    got = samples(tmp_path / "knowledge.db")
    assert got[("kb_eval_metric", (("metric", "mrr"), ("profile", "p")))] == pytest.approx(
        metrics.mrr.value
    )


def test_results_are_written_and_paged_for_the_api(kb_db, tmp_path):
    metrics, records = scored(7)
    run = record_run(
        kb_db,
        eval_dir=tmp_path,
        stage="S1a",
        config={"profile": "p", "sweep_id": "sw1"},
        metrics=metrics,
        records=records,
    )
    assert len(load_results(run)) == 7
    page = results_page(kb_db, run.id, offset=2, limit=3)
    assert page.total == 7 and page.offset == 2 and page.limit == 3
    assert [i.gold_item_id for i in page.items] == [3, 4, 5]
    first = results_page(kb_db, run.id, limit=1).items[0]
    assert first.recall == {"1000": 1.0, "2000": 1.0, "4000": 1.0}
    assert first.full_support is True and first.first_hit_rank == 1
    assert first.abstain_score == pytest.approx(0.1) and first.abstained is False
    last = results_page(kb_db, run.id, offset=6).items[0]
    assert last.category == "unanswerable" and last.recall == {} and last.abstained
    assert not results_page(kb_db, run.id, offset=1, limit=1).items[0].full_support
    with pytest.raises(KeyError):
        results_page(kb_db, "nope")
    with pytest.raises(ValueError):
        results_page(kb_db, run.id, limit=0)
    assert runs_for(kb_db, "sw1") == [run] and runs_for(kb_db, run.id) == [run]
    assert list_runs(kb_db, stage="S1a") == [run] and list_runs(kb_db, stage="S2") == []


def test_full_support_falls_back_to_the_largest_k_and_missing_files(kb_db, tmp_path):
    metrics, records = scored(2)
    for r in records:
        r["full_support"] = {"1": False, "3": True}
    run = record_run(
        kb_db, eval_dir=tmp_path, stage="S1a", config={}, metrics=metrics, records=records
    )
    assert results_page(kb_db, run.id).items[0].full_support is True
    (tmp_path / "runs" / run.id / "results.jsonl").unlink()
    assert results_page(kb_db, run.id).total == 0
    unfinished = start_run(kb_db, stage="S1b", config={}, host="")
    assert load_results(unfinished) == []


def test_results_are_found_after_the_library_is_copied_to_another_machine(
    kb_db, tmp_path, monkeypatch
):
    """``results_path`` is absolute on the machine that ran the sweep (the Mac); on the Pi
    the same ``runs/<id>/results.jsonl`` is under the local eval folder."""
    metrics, records = scored(3)
    run = record_run(
        kb_db, eval_dir=tmp_path, stage="S5", config={}, metrics=metrics, records=records
    )
    kb_db.execute(
        "UPDATE eval_runs SET results_path = ? WHERE id = ?",
        (f"/Users/elsewhere/data/knowledge/eval/runs/{run.id}/results.jsonl", run.id),
    )
    kb_db.commit()
    moved = get_run(kb_db, run.id)
    assert load_results(moved) == []
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path.parent / "data"))
    (tmp_path.parent / "data" / "knowledge").mkdir(parents=True)
    (tmp_path.parent / "data" / "knowledge" / "eval").symlink_to(tmp_path)
    assert len(load_results(moved)) == 3


def test_run_ids_are_safe_directory_names(tmp_path):
    rid = new_run_id("S1a")
    assert rid.startswith("s1a-") and run_dir(tmp_path, rid).parent == tmp_path / "runs"
    for bad in ("", "../x", "a/b", ".hidden"):
        with pytest.raises(ValueError):
            run_dir(tmp_path, bad)
