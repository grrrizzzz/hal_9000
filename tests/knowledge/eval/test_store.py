"""gold_items and judge_labels storage, JSONL exchange and the data/ path rule (rule K2)."""

from __future__ import annotations

import pytest

from hal9000.knowledge.contracts import EvidenceQuote, GoldItem, RubricLabels
from hal9000.knowledge.eval.store import (
    UnsafePath,
    check_data_path,
    export_jsonl,
    get_gold,
    import_jsonl,
    insert_gold,
    insert_judge_label,
    list_gold,
    list_judge_labels,
    set_gold_status,
    set_human_labels,
    set_judge_labels,
)

ITEMS = [
    GoldItem(
        question="How much was the inspection?",
        reference_answer="$625.00",
        category="money",
        paraphrased=True,
        evidence=[EvidenceQuote(quote="Total due: $625.00", page=11, file_sha="a" * 64)],
        split="dev",
    ),
    GoldItem(
        question="Who painted the fence?", category="unanswerable", split="test", reviewer_note="n"
    ),
]


def test_insert_list_get_and_status(kb_db):
    stored = insert_gold(kb_db, ITEMS)
    assert [i.id for i in stored] == [1, 2]
    assert list_gold(kb_db) == stored
    assert get_gold(kb_db, 1) == stored[0] and get_gold(kb_db, 99) is None
    assert list_gold(kb_db, split="test", category="unanswerable") == [stored[1]]
    set_gold_status(kb_db, [1], "approved")
    assert [i.id for i in list_gold(kb_db, status="approved")] == [1]


def test_jsonl_round_trip_updates_known_ids_and_inserts_the_rest(kb_db, tmp_path):
    insert_gold(kb_db, ITEMS)
    path = tmp_path / "gold.jsonl"
    assert export_jsonl(kb_db, path) == 2
    lines = path.read_text().splitlines()
    edited = GoldItem.model_validate_json(lines[0]).model_copy(update={"status": "approved"})
    new = ITEMS[1].model_copy(update={"question": "When was the chimney swept?"})
    path.write_text(edited.model_dump_json() + "\n\n" + new.model_dump_json() + "\n")
    assert import_jsonl(kb_db, path) == (1, 1)
    assert get_gold(kb_db, 1).status == "approved"
    assert [i.question for i in list_gold(kb_db)][-1] == "When was the chimney swept?"


def test_import_validates_every_line_before_writing(kb_db, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(ITEMS[0].model_dump_json() + "\n" + '{"question": "x", "category": "money"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        import_jsonl(kb_db, path)
    assert list_gold(kb_db) == []


def test_data_path_rule(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "data").mkdir()
    data_dir = tmp_path / "elsewhere" / "data"
    assert (
        check_data_path(repo / "data" / "knowledge" / "gold.jsonl", data_dir).name == "gold.jsonl"
    )
    assert check_data_path(data_dir / "gold.jsonl", data_dir).parent == data_dir.resolve()
    assert check_data_path(tmp_path / "outside" / "gold.jsonl", data_dir)
    with pytest.raises(UnsafePath, match="never be committed"):
        check_data_path(repo / "docs" / "gold.jsonl", data_dir)
    with pytest.raises(UnsafePath, match="never be committed"):
        check_data_path(repo / "gold.jsonl", data_dir)
    with pytest.raises(UnsafePath, match=".jsonl"):
        check_data_path(data_dir / "gold.json", data_dir)


LABELS = RubricLabels(
    correctness="correct", faithful=True, citations_correct=True, abstention_correct=None
)


def test_judge_labels(kb_db):
    (item,) = insert_gold(kb_db, ITEMS[:1])
    label_id = insert_judge_label(
        kb_db, item.id, "It was $625.00 [S1].", eval_run_id="run-1", judge=LABELS
    )
    (row,) = list_judge_labels(kb_db, eval_run_id="run-1")
    assert row.question == item.question and row.judge == LABELS and row.human is None
    human = LABELS.model_copy(update={"correctness": "partial"})
    set_human_labels(kb_db, label_id, human)
    set_judge_labels(kb_db, label_id, LABELS)
    (row,) = list_judge_labels(kb_db)
    assert row.human == human
    assert list_judge_labels(kb_db, eval_run_id="other") == []
    with pytest.raises(KeyError):
        set_human_labels(kb_db, 999, human)
