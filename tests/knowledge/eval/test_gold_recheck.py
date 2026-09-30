"""Review round fixes for `kb gold generate` and the new `kb gold recheck`, offline.

Signals that land while a paid reply is being recorded, the surviving
mutations (M1, M6, M7, M9), and the recheck of unanswerable drafts that a
capped run of the old code stored as "could not be checked".
"""

from __future__ import annotations

import os
import re
import signal
import sqlite3
import time
from collections import Counter

import pytest

from hal9000.__main__ import main
from hal9000.cost.metering import BudgetExceeded
from hal9000.knowledge.contracts import GoldItem
from hal9000.knowledge.eval import wiring
from hal9000.knowledge.eval.checkpoint import Checkpoint, CheckpointError
from hal9000.knowledge.eval.config import GoldConfig, GoldEstimate
from hal9000.knowledge.eval.gold import (
    CHECKED_CLEAN,
    SourceDocument,
    generate_gold,
    plan_generation,
    task_labels,
)
from hal9000.knowledge.eval.progress import ledger_basis
from hal9000.knowledge.eval.store import get_gold, insert_gold, list_gold
from hal9000.storage.db import Database
from tests.knowledge.eval.fakes import FakeRetrieve, GoldLLM, RealisticProvider
from tests.knowledge.eval.test_gold_resume import (
    GENERATE,
    QUOTAS,
    checkpoints,
    ledger,
    use,
)

RECHECK = ["kb", "gold", "recheck", "--yes"]
NOT_CHECKED = "FLAGGED: could not be checked (retrieval failed: BudgetExceeded)"


def signal_on_ledger_row(monkeypatch, when, signum=signal.SIGINT):
    """Send ``signum`` from inside ``Database.add_ledger`` once ``when(rows)`` is true."""
    original = Database.add_ledger
    state = {"rows": 0, "sent": False}

    def add_ledger(self, *args, **kwargs):
        original(self, *args, **kwargs)
        state["rows"] += 1
        if not state["sent"] and when(state["rows"]):
            state["sent"] = True
            os.kill(os.getpid(), signum)
            time.sleep(0.05)  # the handler runs here, mid-write of the paid call's record

    add_ledger.__wrapped__ = original
    monkeypatch.setattr(Database, "add_ledger", add_ledger)
    return state


def use_retriever(texts=("A passage about the yard.",)):
    wiring.register_retriever(
        lambda conn, settings, *, meter, run: wiring.RetrievalBinding(FakeRetrieve(texts), 1, "p")
    )


# ------------------------------------------------------- 1. signals and paid replies
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_while_the_ledger_row_is_written_never_costs_a_second_call(
    gold_env, monkeypatch, tmp_path, capsys, signum
):
    state = signal_on_ledger_row(monkeypatch, lambda rows: rows == 5, signum)
    first = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main(GENERATE) == 128 + signum
    assert state["sent"] and first.calls == 5
    assert len(ledger(tmp_path)) == 5
    [manifest] = checkpoints(tmp_path)
    assert manifest["calls"] == 5  # the 5th reply was saved before the stop
    monkeypatch.setattr(Database, "add_ledger", Database.add_ledger.__wrapped__)
    rest = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--resume"]) == 0
    assert rest.calls == 7
    assert len(ledger(tmp_path)) == 12  # exactly the plan's calls
    assert Counter(i.category for i in list_gold(gold_env)) == Counter(QUOTAS)


class BadFirstDrafts(GoldLLM):
    """The drafting call's first reply is invalid (so a repair call follows)."""

    def __init__(self) -> None:
        super().__init__()
        self.bad = True

    def __call__(self, prompt, tier="powerful"):
        if prompt.metadata["purpose"] == "kb-gold-unanswerable" and self.bad:
            self.bad = False
            self.calls.append(prompt)
            self.tiers.append(tier)
            return "oops"
        return super().__call__(prompt, tier)


def test_an_invalid_drafting_reply_is_saved_so_an_interrupted_repair_never_redrafts(
    gold_env, monkeypatch, tmp_path
):
    provider = use(monkeypatch, RealisticProvider(BadFirstDrafts()))
    signal_on_ledger_row(
        monkeypatch, lambda rows: provider.purposes[-1:] == ["kb-gold-unanswerable"]
    )
    assert main(GENERATE) == 130
    assert provider.purposes[-1] == "kb-gold-unanswerable" and provider.calls == 7
    monkeypatch.setattr(Database, "add_ledger", Database.add_ledger.__wrapped__)
    rest = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--resume"]) == 0
    first = rest.llm.calls[0]
    assert first.metadata["purpose"] == "kb-gold-unanswerable" and first.metadata["repair"]
    assert "That reply was invalid" in first.messages[-1]["content"]
    drafting = [p for p in rest.llm.calls if p.metadata["purpose"] == "kb-gold-unanswerable"]
    assert len(drafting) == 1  # only the repair: the paid first reply came from the checkpoint
    assert len(ledger(tmp_path)) == 12 + 1  # the plan's calls plus the one repair
    assert Counter(i.category for i in list_gold(gold_env)) == Counter(QUOTAS)


# ------------------------------------------------------------ 2. surviving mutations
def test_m1_a_paid_reply_with_no_valid_item_is_recorded_even_when_the_repair_hits_the_cap(
    fixture_docs,
):
    config = GoldConfig(quotas={"factoid": 3}, overgenerate=1.0, window_pages=12)
    plan = plan_generation([fixture_docs[0]], config)
    inner = GoldLLM(bad_quotes=3)  # every item of the first reply fails verification

    def capped(prompt, tier):
        if prompt.metadata.get("repair"):
            raise BudgetExceeded("Run cost $3.01 reached the $3.00 hard cap.")
        return inner(prompt, tier)

    checkpoint = Checkpoint.memory(tasks=plan.descriptors(), targets=plan.targets)
    result = generate_gold(plan, capped, retrieve=FakeRetrieve(), checkpoint=checkpoint)
    assert result.items == [] and result.stopped
    [task] = plan.tasks
    assert checkpoint.done == {task.key: []}  # paid for: done, with nothing to keep
    assert len(result.dropped) == 3
    assert all("not regenerated (stopped)" in why for _, why in result.dropped)
    assert not plan.pending(checkpoint)
    again = GoldLLM()
    generate_gold(plan, again, checkpoint=checkpoint)
    assert again.calls == []


def test_m9_a_top_up_balances_the_split_against_stored_items_all_in_dev(
    gold_env, monkeypatch, capsys
):
    stored = [
        GoldItem(question=q, category="unanswerable", split="dev")
        for q in ("Who repaired the quilmet gazebo?", "When was the brunosh cistern drained?")
    ]
    insert_gold(gold_env, stored)
    use_retriever()
    use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--only", "unanswerable", "--count", "3"]) == 0
    items = list_gold(gold_env, category="unanswerable")
    assert len(items) == 5
    assert Counter(i.split for i in items[2:]) == {"dev": 1, "test": 2}
    assert Counter(i.split for i in items) == {"dev": 3, "test": 2}  # 60/40 of all five


def test_m6_the_ledger_basis_ignores_gold_calls_to_another_model(tmp_path):
    db = Database(tmp_path / "hal9000.db")
    try:
        mini = {"model": "openai/gpt-5-mini", "prompt": 1000, "completion": 900, "reasoning": 500}
        for i in range(38):
            db.add_ledger(
                "kb-eval-gold-1", "knowledge-eval", mini, 0.003, at=f"2026-09-29T10:00:{i:02d}"
            )
        basis = ledger_basis(db, "openai/gpt-5", "powerful", GoldEstimate())
        assert basis.calls == 0 and basis.completion_tokens == 3200
        assert "(fewer than 5 earlier calls to this model in the ledger)" in basis.describe()
        assert ledger_basis(db, "openai/gpt-5-mini", "fast", GoldEstimate()).calls == 38
    finally:
        db.close()


def test_m7_a_very_long_document_title_stays_capped_in_progress_labels():
    title = "Comprehensive " * 20 + "Report"
    doc = SourceDocument(1, "b" * 64, title, pages={1: "word " * 100, 2: "text " * 100})
    other = SourceDocument(2, "c" * 64, title, pages={1: "word " * 100})
    plan = plan_generation([doc, other], GoldConfig(quotas={"factoid": 2}, overgenerate=1.0))
    labels = task_labels(plan)
    assert labels
    for label in labels.values():
        assert len(label) <= 60 and "…" in label and title not in label


# ------------------------------------------------------------------ 4. small ones
def test_retired_estimate_keys_are_accepted_with_a_warning():
    with pytest.warns(UserWarning, match="no longer used"):
        allowance = GoldEstimate.model_validate(
            {"output_tokens_per_item": 220, "repair_allowance": 0.25, "seconds_per_call": 50}
        )
    assert allowance.seconds_per_call == 50 and allowance.completion("powerful") == 3200


def test_a_running_checkpoint_cannot_be_resumed_twice(gold_env, monkeypatch, tmp_path, capsys):
    use(monkeypatch, RealisticProvider(GoldLLM(), interrupt_on=3))
    assert main(GENERATE) == 130
    [manifest] = checkpoints(tmp_path)
    other = Checkpoint.load(tmp_path / "data" / "knowledge" / "eval" / "gold", manifest["id"])
    other.acquire()  # as if another process were running it
    idle = use(monkeypatch, RealisticProvider(GoldLLM()))
    try:
        assert main([*GENERATE, "--resume"]) == 1
        assert "being run by another process" in capsys.readouterr().err
        assert idle.calls == 0
        with pytest.raises(CheckpointError):
            Checkpoint.load(other.directory.parent, manifest["id"]).acquire()
    finally:
        other.release()
    assert main([*GENERATE, "--resume"]) == 0 and idle.calls == 10


def test_ctrl_c_at_the_confirmation_prompt_is_a_clean_no(gold_env, monkeypatch, capsys):
    provider = use(monkeypatch, RealisticProvider(GoldLLM()))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("hal9000.knowledge.eval.spend.confirm", interrupted)
    assert main(GENERATE[:-1]) == 3
    assert "Not started." in capsys.readouterr().out and provider.calls == 0


def test_the_cap_inside_retrieval_stops_the_run_and_leaves_the_draft_unchecked(fixture_docs):
    plan = plan_generation(fixture_docs, GoldConfig(quotas={"unanswerable": 2}))
    llm = GoldLLM()

    def capped_retrieve(question):
        raise BudgetExceeded("Run cost $2.00 reached the $2.00 hard cap.")

    checkpoint = Checkpoint.memory(tasks=plan.descriptors(), targets=plan.targets)
    result = generate_gold(plan, llm, retrieve=capped_retrieve, checkpoint=checkpoint)
    assert result.stopped and "hard cap" in result.stopped
    assert result.items == []  # never stored as "could not be checked"
    assert plan.pending(checkpoint).checks == 3


def test_a_top_up_prints_its_own_plan_line(gold_env, monkeypatch, capsys):
    use_retriever()
    use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--only", "unanswerable"]) == 0
    out = capsys.readouterr().out
    assert "Top-up plan: 1 generation tasks, 5 unanswerable drafts to check." in out
    assert "linked document sets" not in out


# ------------------------------------------------------------------- 3. recheck
def _old_capped_set(conn) -> dict[str, GoldItem]:
    """Unanswerable drafts as the old code stored them after a capped run (plus review state)."""

    def item(question, note, status="draft"):
        return GoldItem(
            question=question, category="unanswerable", status=status, reviewer_note=note
        )

    items = {
        "u1": item("When was the quilmet gazebo last painted?", NOT_CHECKED),
        "u2": item("Who installed the zorvak skylight?", NOT_CHECKED),
        "u3": item("How old is the brunosh water heater?", NOT_CHECKED),
        "u4": item("Which company serviced the plixdran pool pump?", NOT_CHECKED),
        "keyword": item(
            "When was the tosh hedge trimmed?",
            "Checked by keyword search only (checked by keyword search only (no index was wired)).",
        ),
        "clean": item("What colour is the metyem front door?", None),
        "answered": item(
            "Who replaced the kovsarn roof?", "FLAGGED: retrieval found it answered: P1 says so"
        ),
        "approved": item("When was the fulvak fence built?", NOT_CHECKED, status="approved"),
        "rejected": item("Who dug the dranful well?", NOT_CHECKED, status="rejected"),
        "edited": item("When was the yemtosh deck stained?", NOT_CHECKED),
    }
    stored = dict(zip(items, insert_gold(conn, list(items.values())), strict=True))
    with conn:  # Mike edited this draft in the review UI later
        conn.execute(
            "UPDATE gold_items SET updated_at = '2099-01-01T00:00:00+00:00' WHERE id = ?",
            (stored["edited"].id,),
        )
    return stored


def _rows(conn) -> dict[int, tuple]:
    return {
        r[0]: tuple(r[1:])
        for r in conn.execute(
            "SELECT id, question, status, reviewer_note, split, updated_at FROM gold_items"
        )
    }


def test_recheck_checks_only_unreviewed_unchecked_drafts_and_saves_each_result(
    gold_env, monkeypatch, tmp_path, capsys
):
    stored = _old_capped_set(gold_env)
    before = _rows(gold_env)
    use_retriever()
    llm = GoldLLM(answered=[stored["u2"].question])
    provider = use(monkeypatch, RealisticProvider(llm, prompt_tokens=1880))
    assert main(RECHECK) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert out.index("5 unanswerable drafts") < out.index("Estimate: about 5 calls")
    assert "(5 unanswerable checks)" in out and "Basis:" in out
    assert out.index("Estimate:") < out.index("Rechecked")
    assert provider.calls == 5 and set(provider.purposes) == {"kb-gold-check"}
    assert {agent for agent, *_ in _ledger_agents(tmp_path)} == {"knowledge-eval"}
    line = re.compile(
        r"^\[(\d)/5\] (unanswerable recheck \d/5): \d clean, \d flagged as answered; "
        r"\$\d+\.\d\d spent; ~\d+ s/call; (<1 min|~\d+ min) left$"
    )
    lines = [m for m in map(line.match, captured.err.splitlines()) if m]
    assert [m.group(2) for m in lines] == [f"unanswerable recheck {i}/5" for i in range(1, 6)]
    assert "1 flagged as answered" in lines[-1].group(0)
    progress = "\n".join(m.group(0) for m in lines)
    assert not any(i.question in progress for i in stored.values())
    notes = {name: get_gold(gold_env, item.id).reviewer_note for name, item in stored.items()}
    for name in ("u1", "u3", "u4", "keyword"):
        assert notes[name] == CHECKED_CLEAN
    assert notes["u2"] == "FLAGGED: retrieval found it answered: P1 states it"
    after = _rows(gold_env)
    for name in ("clean", "answered", "approved", "rejected", "edited"):
        assert after[stored[name].id] == before[stored[name].id]  # review decisions untouched
    for name in ("u1", "u2", "u3", "u4", "keyword"):
        old, new = before[stored[name].id], after[stored[name].id]
        assert (old[0], old[1], old[3], old[4]) == (new[0], new[1], new[3], new[4])
    assert "Rechecked 5 of 5 drafts: 4 clean" in out and "1 flagged as answered" in out
    assert "--only unanswerable` refills exactly what was rejected" in out
    assert "Next: 2 unanswerable drafts are flagged as answered" in out  # u2 and the old one

    idle = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main(RECHECK) == 0
    assert "Nothing to recheck" in capsys.readouterr().out and idle.calls == 0


def _ledger_agents(tmp_path):
    conn = sqlite3.connect(tmp_path / "data" / "hal9000.db")
    try:
        return conn.execute("SELECT agent, run_id FROM ledger").fetchall()
    finally:
        conn.close()


def test_recheck_ctrl_c_loses_nothing_and_a_rerun_skips_what_is_done(
    gold_env, monkeypatch, tmp_path, capsys
):
    stored = _old_capped_set(gold_env)
    use_retriever()
    signal_on_ledger_row(monkeypatch, lambda rows: rows == 2)
    first = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main(RECHECK) == 130
    out = capsys.readouterr().out
    assert first.calls == 2 and "STOPPED EARLY: interrupted (Ctrl-C)" in out
    assert "3 still unchecked: run `uv run hal9000 kb gold recheck` again" in out
    clean = [
        n for n, i in stored.items() if get_gold(gold_env, i.id).reviewer_note == CHECKED_CLEAN
    ]
    assert len(clean) == 2  # the second reply was saved before the stop
    monkeypatch.setattr(Database, "add_ledger", Database.add_ledger.__wrapped__)
    rest = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main(RECHECK) == 0
    assert rest.calls == 3 and len(ledger(tmp_path)) == 5


def test_recheck_stops_at_the_cap_with_its_exit_code_and_continues_later(
    gold_env, monkeypatch, tmp_path, capsys
):
    _old_capped_set(gold_env)
    use_retriever()
    heavy = use(monkeypatch, RealisticProvider(GoldLLM(), completion=20_000, prompt_tokens=1880))
    assert main([*RECHECK, "--max-cost", "0.30"]) == 4
    out = capsys.readouterr().out
    assert heavy.calls == 2 and "STOPPED EARLY" in out and "hard cap" in out
    assert "3 still unchecked" in out
    rest = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main(RECHECK) == 0 and rest.calls == 3


def test_recheck_needs_the_real_index_and_refuses_over_the_cap(gold_env, monkeypatch, capsys):
    _old_capped_set(gold_env)
    provider = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main(RECHECK) == 1
    assert "needs the real index" in capsys.readouterr().err
    use_retriever()
    assert main([*RECHECK, "--max-cost", "0.01"]) == 3
    assert re.search(r"to run it, pass --max-cost \d+\.\d0", capsys.readouterr().err)
    assert provider.calls == 0
