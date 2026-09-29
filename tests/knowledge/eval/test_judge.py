"""The judge: pinned rubric, strict schema, judge-prep, agreement and the S5 gate."""

from __future__ import annotations

import logging

import pytest

from hal9000.cost.metering import BudgetExceeded
from hal9000.knowledge.contracts import EvidenceQuote, GoldItem, JudgeLabel, RubricLabels
from hal9000.knowledge.eval.config import JudgeConfig
from hal9000.knowledge.eval.judge import (
    RUBRIC,
    RUBRIC_VERSION,
    JudgeError,
    JudgeNotCalibrated,
    agreement,
    choose_calibration_items,
    estimate_judge_prep,
    judge_answer,
    judge_prep,
    judge_prompt,
    require_agreement,
)
from hal9000.knowledge.eval.store import (
    insert_gold,
    list_judge_labels,
    set_gold_status,
    set_human_labels,
)
from hal9000.knowledge.eval.types import AnswerOutput, Source
from tests.fakes import FakeLLM, tool_call

GOOD = RubricLabels(
    correctness="correct", faithful=True, citations_correct=True, abstention_correct=True
)


def verdict(**kw) -> str:
    args = {
        "correctness": "correct",
        "faithful": True,
        "citations_correct": True,
        "abstention_correct": True,
    }
    args.update(kw)
    return tool_call("record_judgement", **args)


def item(i: int, category: str = "money", split: str = "dev") -> GoldItem:
    evidence = (
        [] if category == "unanswerable" else [EvidenceQuote(quote=f"Total: ${i}.00", page=1)]
    )
    return GoldItem(
        id=i,
        question=f"How much was bill number {i}?",
        reference_answer=None if category == "unanswerable" else f"${i}.00",
        category=category,
        evidence=evidence,
        split=split,
    )


ANSWER = AnswerOutput("It was $5.00 [S1].", sources=[Source("S1", "Total: $5.00", "Invoice 5")])


def test_the_rubric_is_pinned_and_the_prompt_carries_everything():
    prompt = judge_prompt(item(5), ANSWER)
    assert prompt.messages[0]["content"] == RUBRIC
    assert prompt.metadata["rubric"] == RUBRIC_VERSION
    body = prompt.messages[1]["content"]
    assert (
        "How much was bill number 5?" in body
        and "$5.00" in body
        and "[S1] Invoice 5: Total: $5.00" in body
    )
    schema = prompt.tools[0]["function"]
    assert schema["strict"] and schema["parameters"]["additionalProperties"] is False
    assert schema["parameters"]["properties"]["correctness"]["enum"] == [
        "correct",
        "partial",
        "wrong",
    ]
    unanswerable = judge_prompt(item(6, "unanswerable"), AnswerOutput("I couldn't find that."))
    assert (
        "(none: the documents do not answer this question)" in unanswerable.messages[1]["content"]
    )
    assert "Sources:\n(none)" in unanswerable.messages[1]["content"]


def test_judge_answer_parses_retries_once_then_fails():
    llm = FakeLLM([verdict(correctness="partial", faithful=False)])
    labels = judge_answer(llm, "powerful", item(5), ANSWER)
    assert labels.correctness == "partial" and labels.faithful is False
    assert llm.tiers == ["powerful"]
    llm = FakeLLM([verdict(correctness="great"), verdict()])
    assert judge_answer(llm, "powerful", item(5), ANSWER) == GOOD
    assert "That reply was invalid" in llm.calls[1].messages[-1]["content"]
    with pytest.raises(JudgeError):
        judge_answer(FakeLLM(["nope", "{}"]), "powerful", item(5), ANSWER)


def test_calibration_items_spread_over_categories():
    items = [
        item(i, c) for i, c in enumerate(["money"] * 10 + ["date"] * 3 + ["unanswerable"] * 2, 1)
    ]
    chosen = choose_calibration_items(items, 6)
    cats = [i.category for i in chosen]
    assert cats.count("date") == 2 and cats.count("unanswerable") == 2 and cats.count("money") == 2
    assert choose_calibration_items(items, 6) == chosen
    assert len(choose_calibration_items(items, 50)) == 15


def _seed(conn, n=5):
    stored = insert_gold(conn, [item(i).model_copy(update={"id": None}) for i in range(1, n + 1)])
    stored += insert_gold(conn, [item(99, split="test").model_copy(update={"id": None})])
    set_gold_status(conn, [g.id for g in stored], "approved")
    return stored


def test_judge_prep_answers_dev_items_and_stores_the_judges_labels(kb_db):
    _seed(kb_db)
    asked = []

    def answer(question):
        asked.append(question)
        return AnswerOutput(f"Answer to {question}", sources=[Source("S1", "text")])

    llm = FakeLLM([verdict()] * 3)
    result = judge_prep(kb_db, answer, llm, eval_run_id="s4-1", config=JudgeConfig(n_items=3))
    assert len(result.label_ids) == 3 and len(asked) == 3
    assert "bill number 99" not in " ".join(asked)  # test split is never used
    labels = list_judge_labels(kb_db, eval_run_id="s4-1")
    assert all(label.judge == GOOD and label.human is None for label in labels)
    assert labels[0].answer.startswith("Answer to")
    # resumable: a second run skips what is done
    again = judge_prep(
        kb_db, answer, FakeLLM([verdict()] * 2), eval_run_id="s4-1", config=JudgeConfig(n_items=5)
    )
    assert again.skipped == 3 and len(again.label_ids) == 2


def test_judge_prep_stops_at_the_cap_and_records_errors(kb_db):
    _seed(kb_db)
    calls = {"n": 0}

    def capped(prompt, tier):
        calls["n"] += 1
        if calls["n"] == 3:
            raise BudgetExceeded("cap")
        return verdict()

    result = judge_prep(
        kb_db, lambda q: ANSWER, capped, eval_run_id="r", config=JudgeConfig(n_items=5)
    )
    assert result.stopped == "cap" and len(result.label_ids) == 2

    def broken(question):
        raise RuntimeError("pipeline down")

    result = judge_prep(kb_db, broken, FakeLLM([]), eval_run_id="r2", config=JudgeConfig(n_items=2))
    assert result.label_ids == [] and "pipeline down" in result.errors[0]
    result = judge_prep(
        kb_db,
        lambda q: ANSWER,
        FakeLLM(["x", "y"]),
        eval_run_id="r3",
        config=JudgeConfig(n_items=1),
    )
    assert len(result.label_ids) == 1 and "invalid twice" in result.errors[0]
    assert list_judge_labels(kb_db, eval_run_id="r3")[0].judge is None


def _label(i, human, judge) -> JudgeLabel:
    return JudgeLabel(id=i, gold_item_id=i, question="q", answer="a", human=human, judge=judge)


def test_agreement_counts_every_label():
    partial = GOOD.model_copy(update={"correctness": "partial"})
    na = GOOD.model_copy(update={"abstention_correct": None})
    labels = [
        _label(1, GOOD, GOOD),  # 4/4
        _label(2, GOOD, partial),  # 3/4
        _label(3, na, na),  # 3/3 (abstention n/a on both sides)
        _label(4, na, GOOD),  # 3/4 (None vs a value disagrees)
        _label(5, None, GOOD),  # unlabelled by Mike
        _label(6, GOOD, None),  # the judge failed twice: 0/4, every label a disagreement
    ]
    result = agreement(labels)
    assert (result.items, result.labels, result.agreed, result.unlabelled) == (5, 19, 13, 1)
    assert result.rate == pytest.approx(13 / 19)
    assert result.per_field["correctness"] == 0.6 and result.per_field["abstention_correct"] == 0.5
    assert not result.passes(0.85)
    assert agreement(labels[:4]).passes(0.85)
    assert not agreement(labels[:4] + labels[5:]).passes(0.85)  # 13/19: the failure counts
    failed_na = agreement([_label(7, na, None)])
    assert (failed_na.labels, failed_na.agreed) == (3, 0)
    assert not agreement([]).passes(0.85) and agreement([]).rate == 0.0
    assert "13/19" in result.describe()


def _calibrated(conn, human_labels):
    stored = _seed(conn, len(human_labels))
    llm = FakeLLM([verdict()] * len(human_labels))
    judge_prep(
        conn, lambda q: ANSWER, llm, eval_run_id="s4", config=JudgeConfig(n_items=len(human_labels))
    )
    for label, human in zip(list_judge_labels(conn), human_labels):
        set_human_labels(conn, label.id, human)
    return stored


def test_require_agreement_passes_at_the_gate(kb_db):
    _calibrated(kb_db, [GOOD] * 5)
    result = require_agreement(kb_db, gate=0.85)
    assert result.rate == 1.0 and not result.forced


def test_s5_refuses_below_the_gate_or_without_labels_unless_forced(kb_db, caplog):
    with pytest.raises(JudgeNotCalibrated, match="no judge calibration labels"):
        require_agreement(kb_db)
    wrong = RubricLabels(
        correctness="wrong", faithful=False, citations_correct=False, abstention_correct=False
    )
    _calibrated(kb_db, [wrong, wrong, GOOD, GOOD, GOOD])  # 12/20 = 60%
    with pytest.raises(JudgeNotCalibrated, match="60%"):
        require_agreement(kb_db, gate=0.85)
    with caplog.at_level(logging.WARNING, logger="hal9000.knowledge.eval.judge"):
        forced = require_agreement(kb_db, gate=0.85, force=True)
    assert forced.forced and forced.rate == pytest.approx(0.6)
    assert "--force-judge" in caplog.text
    assert require_agreement(kb_db, gate=0.5).rate == pytest.approx(0.6)
    with pytest.raises(JudgeNotCalibrated):
        require_agreement(kb_db, eval_run_id="another-run")


def test_judge_prep_estimate(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_FAST", "openai/gpt-5-mini")
    monkeypatch.setenv("AGENT_MODEL_POWERFUL", "openai/gpt-5")
    estimate = estimate_judge_prep(20, JudgeConfig(), "fast", "powerful")
    assert estimate.calls == 40
    assert 0.01 < estimate.usd < 0.5  # plan §8.4: about $0.10
    assert "gpt-5-mini" in estimate.model and "openai/gpt-5 (judge)" in estimate.model
