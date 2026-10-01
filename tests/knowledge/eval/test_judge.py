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
    SourcesUnavailable,
    agreement,
    choose_calibration_items,
    estimate_judge_prep,
    judge_answer,
    judge_prep,
    judge_prompt,
    rejudge,
    require_agreement,
)
from hal9000.knowledge.eval.store import (
    insert_gold,
    label_sources,
    list_judge_labels,
    set_gold_status,
    set_human_labels,
    set_judge_labels,
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
    judged = judge_answer(llm, "powerful", item(5), ANSWER)
    assert judged.labels.correctness == "partial" and judged.labels.faithful is False
    assert judged.reasons is None  # a reply without reasons still grades
    assert llm.tiers == ["powerful"]
    llm = FakeLLM([verdict(correctness="great"), verdict()])
    assert judge_answer(llm, "powerful", item(5), ANSWER).labels == GOOD
    assert "That reply was invalid" in llm.calls[1].messages[-1]["content"]
    with pytest.raises(JudgeError):
        judge_answer(FakeLLM(["nope", "{}"]), "powerful", item(5), ANSWER)


REASONS = {
    "correctness": "States $5.00, the reference answer.",
    "faithful": "S1 says Total: $5.00.",
    "citations_correct": "The one claim cites S1, which supports it.",
    "abstention_correct": "Answerable, and it answered.",
}


def test_the_judge_gives_a_reason_for_each_label_before_the_labels():
    schema = judge_prompt(item(5), ANSWER).tools[0]["function"]["parameters"]
    assert schema["required"][0] == "reasons" and next(iter(schema["properties"])) == "reasons"
    reasons = schema["properties"]["reasons"]
    assert reasons["additionalProperties"] is False
    assert reasons["required"] == [
        "correctness",
        "faithful",
        "citations_correct",
        "abstention_correct",
    ]
    assert "reason in one short sentence" in RUBRIC
    judged = judge_answer(FakeLLM([verdict(reasons=REASONS)]), "powerful", item(5), ANSWER)
    assert judged.labels == GOOD and judged.reasons.faithful == "S1 says Total: $5.00."
    with pytest.raises(JudgeError):  # reasons must cover every label, and nothing else
        judge_answer(
            FakeLLM([verdict(reasons={"faithful": "x"}), verdict(reasons={**REASONS, "x": "y"})]),
            "powerful",
            item(5),
            ANSWER,
        )


def test_the_judge_sees_each_source_header_the_answer_saw():
    """A date or vendor taken from a source's header line is supported (calibration 2026-10-01:
    the judge saw only titles and marked such answers unfaithful)."""
    meta = "Invoice 5 · invoice · Acme Plumbing · dated March 3, 2024 · page 1"
    answer = AnswerOutput(
        "Acme Plumbing billed $5.00 on March 3, 2024 [S1].",
        sources=[Source("S1", "Total: $5.00", "Invoice 5", meta=meta)],
    )
    body = judge_prompt(item(5), answer).messages[1]["content"]
    assert f"[S1] {meta}\nTotal: $5.00" in body
    assert "header line" in RUBRIC and "supported by it" in RUBRIC


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


def test_judge_prep_keeps_the_reasons_and_the_sources_the_answer_was_given(kb_db):
    _seed(kb_db)
    meta = "Invoice 1 · invoice · dated March 3, 2024 · page 1"
    sources = [Source("S1", "Total: $1.00", "Invoice 1", meta=meta)]
    judge_prep(
        kb_db,
        lambda q: AnswerOutput("It was $1.00 [S1].", sources=sources),
        FakeLLM([verdict(reasons=REASONS)]),
        eval_run_id="r",
        config=JudgeConfig(n_items=1),
    )
    (label,) = list_judge_labels(kb_db, eval_run_id="r")
    assert (
        label.judge == GOOD
        and label.judge_reasons.citations_correct == REASONS["citations_correct"]
    )
    assert label_sources(kb_db, label.id) == sources
    # replacing the judge's labels without reasons clears the old reasons
    set_judge_labels(kb_db, label.id, GOOD)
    assert list_judge_labels(kb_db, eval_run_id="r")[0].judge_reasons is None


def test_rejudge_replaces_the_judges_labels_and_keeps_mikes(kb_db):
    _seed(kb_db)
    sources = [Source("S1", "Total: $1.00", "Invoice 1", meta="Invoice 1 · invoice · page 1")]
    judge_prep(
        kb_db,
        lambda q: AnswerOutput("It was $1.00 [S1].", sources=sources),
        FakeLLM([verdict(faithful=False)] * 3),
        eval_run_id="r",
        config=JudgeConfig(n_items=3),
    )
    first, second, third = list_judge_labels(kb_db, eval_run_id="r")
    set_human_labels(kb_db, first.id, GOOD)
    with kb_db:  # two answers stored before sources were kept
        kb_db.execute(
            "UPDATE judge_labels SET sources = '[]' WHERE id IN (?, ?)", (second.id, third.id)
        )
    rebuilt = [Source("S1", "Total: $2.00", "Invoice 2", meta="rebuilt header")]
    asked = []

    def rebuild(question, cited):
        asked.append(question)
        if question == third.question:
            raise SourcesUnavailable("retrieval no longer returns S1")
        return rebuilt

    llm = FakeLLM([verdict(reasons=REASONS)] * 2)
    result = rejudge(kb_db, llm, eval_run_id="r", rebuild_sources=rebuild)
    assert result.label_ids == [first.id, second.id] and result.skipped == 1
    assert "retrieval no longer returns S1" in result.errors[0]
    assert asked == [second.question, third.question]  # stored sources are used as they are
    assert "[S1] Invoice 1 · invoice · page 1" in llm.calls[0].messages[1]["content"]
    assert "rebuilt header" in llm.calls[1].messages[1]["content"]
    after = {label.id: label for label in list_judge_labels(kb_db, eval_run_id="r")}
    assert after[first.id].human == GOOD and after[first.id].judge == GOOD
    assert after[first.id].judge_reasons.faithful == REASONS["faithful"]
    assert after[second.id].answer == second.answer  # the answer itself is never regenerated
    assert label_sources(kb_db, second.id) == rebuilt
    assert after[third.id].judge.faithful is False and after[third.id].judge_reasons is None

    def capped(prompt, tier):
        raise BudgetExceeded("cap")

    stopped = rejudge(kb_db, capped, eval_run_id="r", rebuild_sources=rebuild)
    assert stopped.stopped == "cap" and not stopped.label_ids


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
    assert 0.01 < estimate.usd < 0.6  # plan §8.4: about $0.10 metered; the estimate is cautious
    assert "gpt-5-mini" in estimate.model and "openai/gpt-5 (judge)" in estimate.model
