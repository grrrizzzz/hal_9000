"""`kb gold generate`: progress lines, checkpoints and resume, estimates with reasoning, top-ups.

Offline: the synthetic fixture corpus, :class:`GoldLLM` for the replies and
:class:`RealisticProvider` for GPT-5-like usage (3,190 completion tokens per
call, 2,890 of them reasoning).
"""

from __future__ import annotations

import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from hal9000.__main__ import main
from hal9000.knowledge.contracts import EvidenceQuote, GoldItem
from hal9000.knowledge.eval import wiring
from hal9000.knowledge.eval.checkpoint import (
    Checkpoint,
    CheckpointError,
    GenerationInterrupted,
    StopSignals,
)
from hal9000.knowledge.eval.config import GoldConfig, GoldEstimate
from hal9000.knowledge.eval.gold import (
    assign_splits,
    generate_gold,
    plan_from_descriptors,
    plan_generation,
    question_key,
)
from hal9000.knowledge.eval.progress import Progress, UsageBasis, format_duration, ledger_basis
from hal9000.knowledge.eval.spend import suggest_max_cost
from hal9000.knowledge.eval.store import insert_gold, list_gold
from hal9000.storage.db import Database
from tests.knowledge.eval.conftest import GOLD_QUOTAS as QUOTAS
from tests.knowledge.eval.fakes import FakeRetrieve, GoldLLM, RealisticProvider

REPO = Path(__file__).resolve().parents[3]
GENERATE = ["kb", "gold", "generate", "--allow-keyword-check", "--yes"]
PROGRESS = re.compile(
    r"^\[(\d+)/(\d+)\] (.+?): (\d+) items so far; \$(\d+\.\d\d) spent; ~(\d+) s/call; "
    r"(<1 min|~\d+ min|~\d+ h \d\d min) left$"
)


def use(monkeypatch, provider: RealisticProvider) -> RealisticProvider:
    monkeypatch.setattr("hal9000.llm.generate_response", provider)
    return provider


def ledger(tmp_path) -> list[tuple]:
    conn = sqlite3.connect(tmp_path / "data" / "hal9000.db")
    try:
        return conn.execute(
            "SELECT run_id, completion_tokens, usd FROM ledger ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def checkpoints(tmp_path) -> list[dict]:
    return Checkpoint.manifests(tmp_path / "data" / "knowledge" / "eval" / "gold")


def progress_lines(err: str) -> list[re.Match]:
    return [m for m in (PROGRESS.match(line) for line in err.splitlines()) if m]


def assert_no_duplicates(conn) -> None:
    keys = [question_key(i.question) for i in list_gold(conn)]
    assert len(keys) == len(set(keys))


# -------------------------------------------------------------------- progress
def test_one_progress_line_per_call_with_phase_spend_and_eta_and_no_question_text(
    gold_env, monkeypatch, capsys
):
    provider = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main(GENERATE) == 0
    captured = capsys.readouterr()
    lines = progress_lines(captured.err)
    assert len(lines) == provider.calls == 12  # 7 generation tasks + 5 unanswerable checks
    assert [int(m.group(1)) for m in lines] == list(range(1, 13))
    estimated_calls = int(re.search(r"Estimate: about (\d+) calls", captured.out).group(1))
    assert int(lines[0].group(2)) == estimated_calls == 13  # checks and 1 repair call reserved
    assert int(lines[-1].group(2)) == 12  # the unused reserve drops out at the end
    labels = [m.group(3) for m in lines]
    assert any(re.fullmatch(r"window 1/\d+ \(.+, pp?\. \d+(–\d+)?\)", x) for x in labels)
    assert any(x.startswith("cross-document 1/2 (") for x in labels)
    assert "unanswerable drafts" in labels
    assert [x for x in labels if x.startswith("unanswerable check")] == [
        f"unanswerable check {i}/5" for i in range(1, 6)
    ]
    spent = [float(m.group(5)) for m in lines]
    assert spent == sorted(spent) and spent[-1] > 0.3
    items = [int(m.group(4)) for m in lines]
    assert items == sorted(items) and items[-1] > 15
    assert lines[-1].group(7) == "<1 min"
    assert int(lines[0].group(4)) > 0  # counted once the call's reply was verified
    # nothing from the question set or the pages reaches the progress lines
    progress = "\n".join(m.group(0) for m in lines)
    for item in list_gold(gold_env):
        assert item.question not in progress
        assert all(e.quote[:30] not in progress for e in item.evidence)
    assert "Checkpoint gold-" in captured.out and "Ctrl-C stops" in captured.out


def test_progress_eta_and_duration_formatting():
    clock = iter([0.0, 70.0, 140.0])
    out = []

    class Sink:
        def write(self, text):
            out.append(text)

        def flush(self):
            pass

    progress = Progress(spent=lambda: 0.43, out=Sink(), clock=lambda: next(clock))
    progress.start()
    line = progress.tick("window 9/29 (Home report, pp. 20–22)", items=3, remaining=39)
    assert line == (
        "[1/40] window 9/29 (Home report, pp. 20–22): 3 items so far; $0.43 spent; "
        "~70 s/call; ~46 min left"
    )
    assert "".join(out).endswith("left\n")
    assert progress.tick("unanswerable check 1/36", items=3, remaining=70).endswith(
        "~1 h 22 min left"
    )
    assert format_duration(20) == "<1 min" and format_duration(3600) == "1 h 00 min"


# -------------------------------------------------------------- cap and resume
def test_a_cap_stop_mid_check_stores_what_was_validated_and_resume_pays_only_for_the_rest(
    gold_env, monkeypatch, tmp_path, capsys
):
    # each call costs more than the estimate allows for, so the $0.55 cap hits in the checks
    heavy = use(monkeypatch, RealisticProvider(GoldLLM(), completion=6000, prompt_tokens=1880))
    assert main([*GENERATE, "--max-cost", "0.55"]) == 4  # the cap's own exit code
    out = capsys.readouterr().out
    assert heavy.calls == 9 and heavy.purposes.count("kb-gold-check") == 2
    assert "STOPPED EARLY" in out and "hard cap" in out
    first = list_gold(gold_env)
    counts = Counter(i.category for i in first)
    assert counts["unanswerable"] == 2 and sum(counts.values()) == 2 + 12
    [manifest] = checkpoints(tmp_path)
    assert manifest["status"] == "stopped" and len(manifest["stored"]) == 14
    assert "3 unanswerable checks remain" in out
    command = f"uv run hal9000 kb gold generate --resume {manifest['id']} --max-cost "
    assert command in out

    rows_before = len(ledger(tmp_path))
    fresh = use(monkeypatch, RealisticProvider(GoldLLM(), prompt_tokens=1880))
    assert main([*GENERATE, "--resume"]) == 0
    out = capsys.readouterr().out
    assert f"Resuming {manifest['id']}" in out
    assert fresh.calls == 3 and set(fresh.purposes) == {"kb-gold-check"}  # never redrafted
    assert len(ledger(tmp_path)) == rows_before + 3
    after = list_gold(gold_env)
    assert Counter(i.category for i in after) == Counter(QUOTAS)
    assert [i.id for i in after[: len(first)]] == [i.id for i in first]
    assert_no_duplicates(gold_env)
    [manifest] = checkpoints(tmp_path)
    assert manifest["status"] == "complete" and manifest["sessions"] == 2
    assert main([*GENERATE, "--resume"]) == 1
    assert "no unfinished gold generation" in capsys.readouterr().err


def test_ctrl_c_mid_run_stores_what_is_done_and_resume_skips_it(
    gold_env, monkeypatch, tmp_path, capsys
):
    provider = use(monkeypatch, RealisticProvider(GoldLLM(), interrupt_on=5))
    assert main(GENERATE) == 130
    captured = capsys.readouterr()
    assert "interrupted (Ctrl-C): storing what is done" in captured.err
    assert "STOPPED EARLY: interrupted (Ctrl-C)" in captured.out
    assert "--resume gold-" in captured.out
    stored = list_gold(gold_env)
    assert stored and all(i.category != "unanswerable" for i in stored)
    assert provider.calls == 4 and len(ledger(tmp_path)) == 4  # the interrupted call: unpaid
    first_prompts = list(provider.llm.calls)

    provider = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--resume"]) == 0
    assert provider.calls == 12 - 4 and len(ledger(tmp_path)) == 12
    seen = [p.messages[-1]["content"] for p in first_prompts]
    assert not any(p.messages[-1]["content"] in seen for p in provider.llm.calls)
    assert Counter(i.category for i in list_gold(gold_env)) == Counter(QUOTAS)
    assert_no_duplicates(gold_env)


def test_sigterm_saves_and_the_run_resumes(gold_env, monkeypatch, tmp_path, capsys):
    env = {
        **os.environ,
        "HAL_DATA_DIR": str(tmp_path / "data"),
        "HAL_CONFIG_DIR": str(tmp_path / "config"),
        "OPENAI_API_KEY": "test-openai-key",
        "PYTHONPATH": str(REPO),
    }
    child = subprocess.Popen(
        [sys.executable, "-m", "tests.knowledge.eval.gold_child", "0.4"],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = []
    deadline = time.monotonic() + 120
    try:
        for line in child.stderr:
            lines.append(line)
            if line.startswith("[3/"):
                child.send_signal(signal.SIGTERM)
                break
            assert time.monotonic() < deadline, "".join(lines)
        out, err = child.communicate(timeout=60)
    finally:
        child.kill()
    assert child.returncode == 128 + signal.SIGTERM, "".join(lines) + err
    assert "stopped by SIGTERM: storing what is done" in err
    assert "--resume gold-" in out and "STOPPED EARLY: stopped by SIGTERM" in out
    # the call in flight when SIGTERM came was finished and saved, never paid for twice
    paid = len(ledger(tmp_path))
    assert paid in (3, 4)
    [manifest] = checkpoints(tmp_path)
    assert manifest["status"] == "stopped" and manifest["calls"] == paid

    provider = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--resume", manifest["id"]]) == 0
    assert provider.calls == 12 - paid and len(ledger(tmp_path)) == 12
    assert Counter(i.category for i in list_gold(gold_env)) == Counter(QUOTAS)
    assert_no_duplicates(gold_env)


def test_the_store_is_idempotent_per_checkpoint(gold_env, monkeypatch, tmp_path, capsys):
    use(monkeypatch, RealisticProvider(GoldLLM(), interrupt_on=9))
    assert main(GENERATE) == 130
    use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--resume"]) == 0
    [manifest] = checkpoints(tmp_path)
    ids = [i.id for i in list_gold(gold_env)]
    capsys.readouterr()
    for _ in range(2):  # the finished checkpoint again: nothing to call, nothing stored twice
        idle = use(monkeypatch, RealisticProvider(GoldLLM()))
        assert main([*GENERATE, "--resume", manifest["id"]]) == 0
        assert idle.calls == 0
        assert "Stored 0 draft items" in capsys.readouterr().out
        assert [i.id for i in list_gold(gold_env)] == ids


def test_a_checkpoint_survives_a_truncated_last_line_and_rejects_bad_ids(tmp_path, fixture_docs):
    plan = plan_generation(fixture_docs, GoldConfig(quotas={"factoid": 2, "unanswerable": 2}))
    root = tmp_path / "gold"
    checkpoint = Checkpoint.create(root, tasks=plan.descriptors(), targets=plan.targets)
    run = generate_gold(plan, GoldLLM(), retrieve=FakeRetrieve(), checkpoint=checkpoint)
    assert run.items
    events = (checkpoint.directory / "events.jsonl").read_text().splitlines()
    (checkpoint.directory / "events.jsonl").write_text("\n".join(events[:-1]) + '\n{"type": "ch')
    loaded = Checkpoint.load(root, checkpoint.id)
    assert plan.pending(loaded).checks == 1 and not plan.pending(loaded).tasks
    rebuilt = plan_from_descriptors(loaded.tasks, fixture_docs, plan.config, loaded.targets)
    assert [t.key for t in rebuilt.tasks] == [t.key for t in plan.tasks]
    with pytest.raises(ValueError, match="library changed"):
        plan_from_descriptors(loaded.tasks, fixture_docs[1:], plan.config, loaded.targets)
    for bad in ("../x", "nope"):
        with pytest.raises(CheckpointError):
            Checkpoint.load(root, bad)


def test_stop_signals_never_cut_a_paid_reply_short(capsys):
    def kill(signum=signal.SIGINT):
        os.kill(os.getpid(), signum)
        time.sleep(0.05)

    with StopSignals() as signals:
        kill()  # not running yet: only a notice
        assert "saving what is done" in capsys.readouterr().err
        with signals.running():
            # a write in a shield finishes first; the stop comes when it ends
            with pytest.raises(GenerationInterrupted) as caught:
                with signals.shield():
                    kill(signal.SIGTERM)
                    finished = True
            assert finished and caught.value.signum == signal.SIGTERM
            assert caught.value.exit_code == 143
            # a call in flight: the first Ctrl-C waits for its reply...
            signals.before_call()
            kill()
            assert "stopping once the call in flight returns" in capsys.readouterr().err
            signals.after_call(True)  # ...which arrives and is held until it is saved
            kill()
            with pytest.raises(GenerationInterrupted):
                signals.settle()  # saved: now it stops
            # a second Ctrl-C abandons the call in flight
            signals.before_call()
            kill()
            with pytest.raises(GenerationInterrupted):
                kill()
            signals.after_call(False)
            # a stop noted after a reply is raised before the next paid call at the latest
            signals.before_call()
            signals.after_call(True)
            kill()
            with pytest.raises(GenerationInterrupted):
                signals.before_call()
            signals.after_call(False)
            # idle: at once
            with pytest.raises(GenerationInterrupted):
                kill()
        kill()  # saving: ignored
    assert signal.getsignal(signal.SIGTERM) != signals._handle


# ------------------------------------------------------------------- estimates
TOPICS = (
    "chimney flue", "gutter guards", "sprinkler timer", "attic insulation", "sump pump",
    "garage door springs", "pool heater", "termite treatment", "roof shingles", "septic tank",
    "window screens", "deck stain", "fence posts", "water softener", "dryer vent",
    "smoke detectors", "ceiling fan", "garbage disposal", "front door lock", "crawl space",
    "solar inverter", "hot tub cover", "driveway sealant", "fireplace insert", "skylight",
    "kitchen faucet", "bathroom grout", "patio pavers", "irrigation valve", "mailbox post",
    "doorbell camera", "attic fan", "shower valve", "gas meter", "storm drain", "lawn mower",
)  # fmt: skip
ASKS = (
    "When was the {} last replaced?",
    "Who serviced the {} and what did it cost?",
    "Is the {} still under warranty?",
    "How old is the {}?",
    "Which company installed the {}?",
    "What model is the {}?",
)


def test_the_estimate_with_reasoning_is_within_25_percent_of_metered_spend(
    gold_env, fixture_docs, monkeypatch, tmp_path, capsys
):
    (tmp_path / "config" / "knowledge_eval.yaml").write_text("{}\n")  # the default 120-item mix
    pages = [text for doc in fixture_docs for text in doc.pages.values()]
    wiring.register_retriever(  # the real flow: the index checks drafts, within its budget
        lambda conn, settings, *, meter, run: wiring.RetrievalBinding(
            FakeRetrieve(pages[:8]), 1, "fixture"
        )
    )
    drafts = [ask.format(topic) for topic, ask in zip(TOPICS, ASKS * 6, strict=True)]
    provider = use(monkeypatch, RealisticProvider(GoldLLM(extra_drafts=drafts)))
    assert main(["kb", "gold", "generate", "--yes"]) == 0
    out = capsys.readouterr().out
    estimated = float(re.search(r"estimated \$(\d+\.\d\d)", out).group(1))
    metered = sum(usd for _, _, usd in ledger(tmp_path))
    assert provider.purposes.count("kb-gold-check") == 36
    assert abs(estimated - metered) / metered <= 0.25, (estimated, metered)
    # the reasoning tokens are what used to be missing: without them the estimate is far off
    reasoning = provider.calls * 2890 * 10 / 1e6
    assert reasoning > 0.75 * metered


def test_the_estimate_basis_comes_from_the_ledger_when_it_has_enough_calls(tmp_path):
    allowance = GoldEstimate()
    db = Database(tmp_path / "hal9000.db")
    try:
        empty = ledger_basis(db, "openai/gpt-5", "powerful", allowance)
        assert (empty.completion_tokens, empty.seconds, empty.calls) == (3200, 60, 0)
        assert "default allowance of 3,200 completion tokens" in empty.describe()
        assert ledger_basis(db, "openai/gpt-5-mini", "fast", allowance).completion_tokens == 1000
        usage = {"model": "openai/gpt-5", "prompt": 1880, "completion": 3190, "reasoning": 2890}
        for i in range(38):
            at = f"2026-09-29T10:{(i * 70) // 60:02d}:{(i * 70) % 60:02d}-07:00"
            db.add_ledger("kb-eval-gold-1234abcd", "knowledge-eval", usage, 0.034, at=at)
        other = {**usage, "completion": 900}
        db.add_ledger("kb-eval-judge-x", "knowledge-eval", other, 0.01, at="2026-09-29T12:00:00")
        db.add_ledger("kb-eval-gold-1234abcd", "knowledge", other, 0.01, at="2026-09-29T12:00:00")
        basis = ledger_basis(db, "openai/gpt-5", "powerful", allowance)
        assert (basis.calls, basis.completion_tokens, round(basis.seconds), basis.timed) == (
            38,
            3190,
            70,
            37,
        )
        assert basis.describe() == (
            "~3,190 completion tokens per call incl. reasoning, the mean of 38 earlier "
            "gold-generation calls to openai/gpt-5; ~70 s per call from their ledger timestamps"
        )
        few = ledger_basis(db, "openai/gpt-5", "powerful", GoldEstimate(min_ledger_calls=50))
        assert few.calls == 0 and few.completion_tokens == 3200
    finally:
        db.close()
    # with too few gold calls, every evaluation call to the model counts
    db = Database(tmp_path / "other.db")
    try:
        for i in range(6):
            db.add_ledger(f"kb-eval-sweep-{i}", "knowledge-eval", usage, 0.03, at="2026-09-29")
        basis = ledger_basis(db, "openai/gpt-5", "powerful", allowance)
        assert basis.calls == 6 and "earlier evaluation calls" in basis.describe()
        assert basis.timed == 0 and basis.seconds == 60
    finally:
        db.close()


def test_the_cli_prints_the_ledger_basis_and_the_time(gold_env, monkeypatch, tmp_path, capsys):
    db = Database(tmp_path / "data" / "hal9000.db")
    usage = {"model": "openai/gpt-5", "prompt": 1880, "completion": 3190, "reasoning": 2890}
    for i in range(38):
        at = f"2026-08-01T10:{(i * 70) // 60:02d}:{(i * 70) % 60:02d}-07:00"
        db.add_ledger("kb-eval-gold-00000000", "knowledge-eval", usage, 0.034, at=at)
    db.close()
    use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE[:-1], "--max-cost", "0.01"]) == 3
    captured = capsys.readouterr()
    assert (
        "Basis: ~3,190 completion tokens per call incl. reasoning, the mean of 38" in captured.out
    )
    assert "~70 s per call from their ledger timestamps" in captured.out
    assert "(7 generation + 1 repair reserve + 5 unanswerable checks)" in captured.out
    assert "taking about 15 min" in captured.out  # 13 calls x 70 s
    assert re.search(r"to run it, pass --max-cost \d+\.\d0", captured.err)


def test_every_call_is_allowed_completion_incl_reasoning_plus_checks_and_a_reserve(
    fixture_docs, monkeypatch
):
    monkeypatch.setenv("AGENT_MODEL_POWERFUL", "openai/gpt-5")
    plan = plan_generation(fixture_docs, GoldConfig())
    estimate = plan.estimate(count=lambda text: 10)
    reserve = math.ceil(len(plan.tasks) * 0.10)
    assert estimate.reserve_calls == reserve
    assert estimate.check_calls == plan.n_unanswerable == 36
    assert estimate.calls == len(plan.tasks) + reserve + 36
    assert estimate.output_tokens == estimate.calls * 3200  # the default allowance
    assert estimate.seconds == estimate.calls * 60
    expected = (estimate.prompt_tokens * 1.25 + estimate.output_tokens * 10) / 1e6
    assert estimate.usd == pytest.approx(expected, rel=1e-3)
    measured = plan.estimate(UsageBasis(3190, 70, calls=38), count=lambda text: 10)
    assert measured.output_tokens == measured.calls * 3190
    assert measured.seconds == measured.calls * 70
    assert suggest_max_cost(2.58) == 3.5 and suggest_max_cost(0.1) == 0.5


# ---------------------------------------------------------------------- top-up
def _old_gold_set(conn, docs) -> list[GoldItem]:
    """A question set as the old code stored it: no checkpoint, splits already assigned."""
    doc = docs[0]
    quote = EvidenceQuote(quote="a synthetic quote", page=min(doc.pages), file_sha=doc.file_sha)

    def item(question, category, status="draft", note=None):
        answerable = category != "unanswerable"
        return GoldItem(
            question=question,
            category=category,
            difficulty="easy",
            status=status,
            reviewer_note=note,
            reference_answer="x" if answerable else None,
            evidence=[quote] if answerable else [],
        )

    items = [item(f"Which zorvak number {i} is listed?", "factoid", "approved") for i in range(4)]
    items += [
        item("When was the quilmet fountain last cleaned?", "unanswerable"),
        item("Who installed the brunosh awning?", "unanswerable", status="rejected"),
    ]
    return insert_gold(conn, assign_splits(items, 0.6))


def test_only_unanswerable_fills_exactly_the_shortfall_with_consistent_splits(
    gold_env, fixture_docs, monkeypatch, tmp_path, capsys
):
    before = _old_gold_set(gold_env, fixture_docs)
    wiring.register_retriever(
        lambda conn, settings, *, meter, run: wiring.RetrievalBinding(FakeRetrieve(), 1, "p")
    )
    # the drafter repeats a stored question: it is dropped before its (paid) check
    llm = GoldLLM(extra_drafts=["When was the quilmet fountain last cleaned?"])
    provider = use(monkeypatch, RealisticProvider(llm))
    assert main([*GENERATE, "--only", "unanswerable"]) == 0
    out = capsys.readouterr().out
    assert "unanswerable: 2 (quota 3; 1 stored and not rejected)" in out
    assert out.index("Top-up") < out.index("Estimate:")
    assert "1 generation + 1 repair reserve + 3 unanswerable checks" in out
    assert provider.purposes == ["kb-gold-unanswerable", "kb-gold-check", "kb-gold-check"]
    after = list_gold(gold_env)
    new = after[len(before) :]
    assert [i.category for i in new] == ["unanswerable", "unanswerable"]
    assert all(i.status == "draft" for i in new)
    assert_no_duplicates(gold_env)
    unsplit = [i.model_copy(update={"split": None}) for i in new]
    assert [i.split for i in assign_splits(unsplit, 0.6, before)] == [i.split for i in new]
    assert {i.split for i in after if i.category == "unanswerable"} == {"dev", "test"}
    assert [i.id for i in after[: len(before)]] == [i.id for i in before]  # untouched

    idle = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--only", "unanswerable"]) == 0
    assert "Nothing to top up" in capsys.readouterr().out and idle.calls == 0

    fresh = [
        "Who repaired the kovful gate?",
        "Which contractor sealed the skylight above the dransarn stairs?",
    ]
    more = use(monkeypatch, RealisticProvider(GoldLLM(extra_drafts=fresh)))
    assert main([*GENERATE, "--only", "unanswerable", "--count", "1"]) == 0
    assert more.purposes.count("kb-gold-check") == 2  # 1 wanted, drafted with a 50% margin
    assert len(list_gold(gold_env, category="unanswerable")) == len(new) + 2 + 1
    assert_no_duplicates(gold_env)


def test_only_an_answerable_category_plans_just_enough_windows(
    gold_env, fixture_docs, monkeypatch, capsys
):
    before = _old_gold_set(gold_env, fixture_docs)
    config = GoldConfig(quotas=QUOTAS)
    plan = plan_generation(fixture_docs, config, targets={"money": 2}, avoid=before)
    assert {t.kind for t in plan.tasks} == {"window"} and len(plan.tasks) <= 3
    assert sum(sum(t.requested.values()) for t in plan.tasks) == 3  # ceil(2 x 1.5) items
    full = plan_generation(fixture_docs, config)
    assert len(full.tasks) > 3 * len(plan.tasks)
    provider = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--only", "money,list"]) == 0
    out = capsys.readouterr().out
    assert "money: 2 (quota 2; 0 stored" in out and "list: 2 (quota 2; 0 stored" in out
    assert set(provider.purposes) == {"kb-gold-window"}
    new = list_gold(gold_env)[len(before) :]
    assert Counter(i.category for i in new) == {"money": 2, "list": 2}


def test_only_and_count_are_validated(gold_env, monkeypatch, capsys):
    provider = use(monkeypatch, RealisticProvider(GoldLLM()))
    assert main([*GENERATE, "--only", "gossip"]) == 1
    assert "--only takes categories" in capsys.readouterr().err
    assert main([*GENERATE, "--count", "3"]) == 1
    assert "--count needs --only" in capsys.readouterr().err
    assert main([*GENERATE, "--resume", "--only", "unanswerable"]) == 1
    assert "leave out --only" in capsys.readouterr().err
    assert main([*GENERATE, "--resume", "gold-nope"]) == 1
    assert "cannot resume: no checkpoint gold-nope" in capsys.readouterr().err
    assert provider.calls == 0


def test_usage_basis_describe_without_timings():
    basis = UsageBasis(3190.0, 60.0, calls=6, what="earlier evaluation calls to m")
    assert basis.describe().endswith("60 s per call (default)")


def test_a_cap_during_a_repair_keeps_the_paid_reply_and_never_asks_again(fixture_docs):
    from hal9000.cost.metering import BudgetExceeded

    config = GoldConfig(quotas={"factoid": 3}, overgenerate=1.0, window_pages=12)
    plan = plan_generation([fixture_docs[0]], config)
    assert [t.kind for t in plan.tasks] == ["window"]
    inner = GoldLLM(bad_quotes=1)

    def capped(prompt, tier):
        if prompt.metadata.get("repair"):
            raise BudgetExceeded("Run cost $3.01 reached the $3.00 hard cap.")
        return inner(prompt, tier)

    checkpoint = Checkpoint.memory(tasks=plan.descriptors(), targets=plan.targets)
    result = generate_gold(plan, capped, retrieve=FakeRetrieve(), checkpoint=checkpoint)
    assert result.stopped and len(result.items) == 2  # the two good items of the paid reply
    assert result.dropped and "not regenerated (stopped)" in result.dropped[0][1]
    assert not plan.pending(checkpoint)  # recorded as done: a resume won't pay for it again
    again = GoldLLM()
    assert generate_gold(plan, again, checkpoint=checkpoint).items == result.items
    assert again.calls == []
