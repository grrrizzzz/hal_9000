"""The staged sweep S1a-S5 on the fixture corpus, offline (fake embedder, reranker and LLM)."""
# ruff: noqa: F811 - pytest fixtures imported from test_sweep_fixtures are parameters here

from __future__ import annotations

from dataclasses import replace

import pytest

from hal9000.knowledge.eval.runs import list_runs, results_page
from hal9000.knowledge.eval.sweep import (
    STAGES,
    SweepConfig,
    calibrate_threshold,
    local_latency_ms,
    read_manifest,
)
from hal9000.knowledge.obs.collector import eval_profile, flatten_eval_metrics
from tests.knowledge.eval.test_sweep_fixtures import (  # noqa: F401 - fixtures
    ACTIVE,
    TINY_GRID,
    corpus_root,
    env,
    fake_mode,
    fresh_query_cache,
    label_calibration,
    results,
    swept_env,
    swept_root,
)


def sweep_runs(conn, sweep_id, *stages):
    return [
        r
        for r in list_runs(conn)
        if r.config.get("sweep_id") == sweep_id and (not stages or r.stage in stages)
    ]


# ------------------------------------------------------------------ S1a-S4
def test_s1a_to_s4_on_the_fixture_corpus(swept_env):
    env, sweep_id = swept_env
    log = (env.root / "sweep.log").read_text()
    runs = sweep_runs(env.conn, sweep_id)
    by_stage = {s: [r for r in runs if r.stage == s] for s in STAGES}
    # S1a: 2 chunkers x 2 sizes + 1 parent-child, x dense and RRF
    assert len(by_stage["S1a"]) == 10
    # S1b: top 2 chunkings x overlap x header, Docling recorded unavailable and skipped
    assert len(by_stage["S1b"]) == 8
    assert {r.config["parser"] for r in by_stage["S1b"]} == {"pdfium"}
    notes = read_manifest(env.settings.eval_dir, sweep_id)["notes"]
    assert any("docling unavailable" in n for n in notes["S1b"])
    # S2: top 2 profiles x 2 embeddings with one method; S3: 2 x 3 methods x 2 rerankers + T10
    assert len(by_stage["S2"]) == 4
    assert len({r.config["retrieval"] for r in by_stage["S2"]}) == 1
    assert len(by_stage["S3"]) == 2 * 3 * 2 + 1
    assert {r.config.get("reranker") for r in by_stage["S3"]} == {"none", "minilm"}
    assert sum(r.config.get("query_transform") == "multi" for r in by_stage["S3"]) == 1
    # S4: top 2 configurations, each baseline and k=3
    assert len(by_stage["S4"]) == 4
    assert sorted(str(r.config.get("k")) for r in by_stage["S4"]) == ["3", "3", "None", "None"]
    for run in runs:
        # KWP8's collector contract: a profile, dev split, finished, flattened metrics
        assert eval_profile(run.config) == run.config["profile"]
        assert run.split == "dev" and run.finished_at is not None
        flat = flatten_eval_metrics(run.metrics.model_dump(mode="json"))
        assert "budgeted_recall@4000" in flat and "mrr" in flat
        assert run.host in ("mac", "pi", "linux", "windows")
        assert run.metrics.latency_ms_p95 is not None and run.metrics.latency_ms_p95 >= 0
        # per-question results for the drill-down API
        page = results_page(env.conn, run.id, limit=50)
        assert page.total == 15 and page.items[0].question
    # the per-question records carry each question's local latency
    assert all(r["latency_ms"] is not None for r in results(by_stage["S1a"][0]))
    # the T10 run paid for its rewrites, metered as knowledge-eval
    t10 = next(r for r in by_stage["S3"] if r.config.get("query_transform") == "multi")
    assert t10.cost_usd > 0
    assert {agent for agent, _ in env.ledger()} == {"knowledge-eval"}
    # evaluation profiles are built but never activated
    assert env.active() == ACTIVE
    names = {r[0] for r in env.conn.execute("SELECT name FROM profiles WHERE status = 'ready'")}
    assert {r.config["profile"] for r in runs} <= names
    # abstain thresholds calibrated on the S4 baselines and stored for the finalists
    baselines = [r for r in by_stage["S4"] if r.config.get("k") is None]
    assert all("abstain_note" in r.config for r in baselines)
    stored = env.conn.execute("SELECT count(*) FROM abstain_thresholds").fetchone()[0]
    calibrated = {(r.config["profile"], r.config.get("reranker")) for r in baselines
                  if r.config.get("abstain_threshold") is not None}  # fmt: skip
    assert stored == len(calibrated)
    uncalibrated = [r for r in baselines if r.config.get("abstain_threshold") is None]
    for run in uncalibrated:  # the fake embedder's scores don't separate: recorded why
        assert "moves to the generation step" in run.config["abstain_note"]
    if uncalibrated:
        assert any("no abstain threshold" in n for n in notes["S4"])
    assert "Estimate for S1a" in log


def test_s4_stores_the_calibrated_thresholds_of_the_finalists(swept_env):
    env, sweep_id = swept_env
    with env.conn:
        env.conn.execute("DELETE FROM eval_runs WHERE stage = 'S4'")
    grid = (env.settings.config_dir / "knowledge_eval.yaml").read_text()
    grid = grid.replace("abstain_auc_min: 0.5", "abstain_auc_min: 0.0")
    outcome = env.runner(grid=grid).run(["S4"], sweep_id=sweep_id)
    assert outcome.status == "done", env.out.getvalue()
    baselines = [r for r in sweep_runs(env.conn, sweep_id, "S4") if r.config.get("k") is None]
    rows = env.conn.execute(
        "SELECT p.name, t.reranker, t.threshold, t.eval_run_id FROM abstain_thresholds t "
        "JOIN profiles p ON p.id = t.profile_id"
    ).fetchall()
    expected = {(r.config["profile"], "" if r.config["reranker"] == "none" else r.config["reranker"])
                for r in baselines}  # fmt: skip
    assert {(name, reranker) for name, reranker, _, _ in rows} == expected
    by_id = {r.id: r for r in baselines}
    for _, _, threshold, run_id in rows:
        assert threshold == by_id[run_id].config["abstain_threshold"]
    thresholds = read_manifest(env.settings.eval_dir, sweep_id)["thresholds"]
    assert len(thresholds) == len(expected)
    assert "abstain threshold" in env.out.getvalue()


def test_s5_refuses_without_judge_agreement_and_runs_with_it(swept_env):
    env, sweep_id = swept_env
    runner = env.runner()
    outcome = runner.run(["S5"], sweep_id=sweep_id)
    assert outcome.status == "refused" and "no judge calibration labels" in outcome.message
    leader = runner.finalists()[0]
    label_calibration(env.conn, leader.id, agree=False)
    outcome = env.runner().run(["S5"], sweep_id=sweep_id)
    assert outcome.status == "refused" and "judge agreement not recorded" in outcome.message
    assert not sweep_runs(env.conn, sweep_id, "S5")

    with env.conn:
        env.conn.execute("DELETE FROM judge_labels")
    label_calibration(env.conn, leader.id, agree=True)
    outcome = env.runner().run(["S5"], sweep_id=sweep_id)
    assert outcome.status == "done", env.out.getvalue()
    s5 = sweep_runs(env.conn, sweep_id, "S5")
    # the top 2 finalists x 2 tiers, and the top one repeated once per tier
    assert len(s5) == 6
    assert sorted(r.config["chat_tier"] for r in s5) == ["fast"] * 3 + ["powerful"] * 3
    assert sum(r.config.get("repeat", 0) for r in s5) == 2
    for run in s5:
        assert run.config["judge_agreement"] == 1.0 and "judge_forced" not in run.config
        assert run.metrics.correct is not None and run.metrics.faithful is not None
        assert run.metrics.cost_per_question_usd > 0 and run.cost_usd > 0
        assert run.metrics.n_questions == 15
        record = results(run)[0]
        assert record["correctness"] in ("correct", "partial", "wrong")
    assert env.active() == ACTIVE
    out = env.out.getvalue()
    assert out.index("Estimate for S5") < out.index("[1/6]")


def test_force_judge_is_logged_on_the_runs(swept_env, caplog):
    env, sweep_id = swept_env
    grid = (env.settings.config_dir / "knowledge_eval.yaml").read_text()
    grid = grid.replace("    top: 2\n    chat_tiers: [fast, powerful]\n    repeat_top: 1",
                        "    top: 1\n    chat_tiers: [fast]\n    repeat_top: 0")  # fmt: skip
    outcome = env.runner(grid=grid).run(["S5"], sweep_id=sweep_id, force_judge=True)
    assert outcome.status == "done"
    (run,) = sweep_runs(env.conn, sweep_id, "S5")
    assert run.config["judge_forced"] is True
    assert "forced without judge calibration" in caplog.text
    notes = read_manifest(env.settings.eval_dir, sweep_id)["notes"]["S5"]
    assert any("--force-judge" in n for n in notes)


def test_s5_needs_a_sweep_to_resume(env):
    with pytest.raises(ValueError, match="--resume"):
        env.runner().run(["S5"])
    with pytest.raises(ValueError, match="no sweep"):
        env.runner().run(["S2"], sweep_id="sweep-missing")
    with pytest.raises(ValueError, match="unknown stage"):
        env.runner().run(["S7"])


# ------------------------------------------------------------------ caps and resume
def test_a_cap_stops_the_sweep_keeps_partial_results_and_it_resumes(env):
    outcome = env.runner(cap=0.01, grid=TINY_GRID).run(["S1a", "S1b", "S2", "S3"])
    out = env.out.getvalue()
    assert outcome.status == "stopped", out
    assert "STOPPED at the cost cap" in outcome.message
    # the estimate for the paid T10 half was printed before the first LLM call
    assert out.index("Estimate for S3 (T10") < out.index("FIRST LLM CALL")
    spent = sum(usd for _, usd in env.ledger())
    assert 0.01 <= spent < 0.01 + 0.002  # stopped at the first call over the cap
    kept = {r.id for r in sweep_runs(env.conn, outcome.sweep_id)}
    assert kept and all(r.stage != "S3" or r.config.get("query_transform") != "multi"
                        for r in sweep_runs(env.conn, outcome.sweep_id))  # fmt: skip
    assert all(r.finished_at is not None for r in sweep_runs(env.conn, outcome.sweep_id))

    env.out.seek(0)
    env.out.truncate()
    resumed = env.runner(cap=1.0, grid=TINY_GRID).run(["S1a", "S1b", "S2", "S3"],
                                                      sweep_id=outcome.sweep_id)  # fmt: skip
    assert resumed.status == "done", env.out.getvalue()
    after = sweep_runs(env.conn, outcome.sweep_id)
    assert kept < {r.id for r in after}
    assert any(r.config.get("query_transform") == "multi" for r in after)
    log = env.out.getvalue()
    assert "S1a: 2 configuration(s), 2 already done." in log
    assert "building" not in log.split("S3 (T10")[0]  # nothing rebuilt


def test_an_estimate_over_the_cap_is_refused_before_anything_runs(env):
    env.runner(grid=TINY_GRID)  # writes the tiny grid
    outcome = env.runner(cap=0.0001).run(["S1a", "S3"])
    assert outcome.status == "refused"
    # S1a is free with the fake embedder; S3 has no S2 to start from
    assert "no finished S2 runs" in outcome.message
    outcome = env.runner(cap=1e-9, grid=TINY_GRID).run(
        ["S1a", "S1b", "S2", "S3"], sweep_id=outcome.sweep_id
    )
    assert outcome.status == "refused" and "exceeds what is left of the caps" in outcome.message
    assert not any(r.config.get("query_transform") == "multi"
                   for r in sweep_runs(env.conn, outcome.sweep_id))  # fmt: skip


def test_a_stage_that_is_not_confirmed_does_not_run(env):
    outcome = env.runner(grid=TINY_GRID, confirm=lambda q: False).run(["S1a"])
    assert outcome.status == "refused" and "not confirmed" in outcome.message
    assert not sweep_runs(env.conn, outcome.sweep_id)


def test_no_approved_dev_questions_is_refused(env):
    with env.conn:
        env.conn.execute("UPDATE gold_items SET status = 'draft'")
    outcome = env.runner(grid=TINY_GRID).run(["S1a"])
    assert outcome.status == "refused" and "G-K1" in outcome.message


def test_an_unloadable_reranker_is_recorded_and_skipped(env):
    def factory(name):
        raise ImportError("fastembed is not installed")

    grid = TINY_GRID.replace("query_transform_top: 1", "query_transform_top: 0")
    outcome = env.runner(grid=grid, reranker_factory=factory).run(["S1a", "S1b", "S2", "S3"])
    assert outcome.status == "done", env.out.getvalue()
    s3 = sweep_runs(env.conn, outcome.sweep_id, "S3")
    assert {r.config.get("reranker") for r in s3} == {"none"}
    notes = read_manifest(env.settings.eval_dir, outcome.sweep_id)["notes"]["S3"]
    assert any("reranker minilm unavailable" in n for n in notes)


# ------------------------------------------------------------------ units
def test_configs_name_their_profile_and_round_trip():
    config = SweepConfig(chunker="fixed", size=512, overlap=0.15, retrieval="weighted", alpha=0.5,
                         reranker="minilm", k=5, chat_tier="fast", embed_model="fake/embed-64",
                         dimensions=64)  # fmt: skip
    assert config.profile_name == "pdfium.fixed-512-o15.hdr.fake-64"
    assert config.name == "pdfium.fixed-512-o15.hdr.fake-64+weighted0.5+minilm+k5+fast"
    stored = config.run_config(stage="S4", sweep_id="sweep-1")
    assert stored["overlap_tokens"] == 77 and stored["profile"] == config.profile_name
    assert SweepConfig.from_run_config(stored) == config
    with pytest.raises(ValueError):
        SweepConfig(retrieval="weighted")
    with pytest.raises(ValueError):
        SweepConfig(retrieval="rrf", alpha=0.5)


def test_calibration_picks_the_threshold_and_says_when_it_cannot():
    good = calibrate_threshold([0.8, 0.85, 0.9, 0.95] * 5, [0.2, 0.3, 0.4, 0.9])
    assert good.threshold is not None and 0.4 < good.threshold < 0.8
    assert good.decline == 0.75 and good.refuse == 0.0 and not good.met  # 0.9 can't be declined
    perfect = calibrate_threshold([0.8, 0.9], [0.1, 0.2])
    assert perfect.met and perfect.decline == 1.0 and perfect.refuse == 0.0
    weak = calibrate_threshold([0.5, 0.4, 0.6], [0.5, 0.6, 0.4])
    assert weak.threshold is None and "moves to the generation step" in weak.reason
    assert calibrate_threshold([], [0.1]).threshold is None


def test_local_latency_excludes_the_embedding_call():
    from hal9000.knowledge.contracts import StageTiming, Trace

    trace = Trace(
        stages=[
            StageTiming(stage="embed_query", started_ms=0, duration_ms=100.0),
            StageTiming(stage="vector", started_ms=100, duration_ms=2.0),
            StageTiming(stage="bm25", started_ms=102, duration_ms=3.0),
            StageTiming(stage="fuse", started_ms=105, duration_ms=0.5, depth=1, parent="bm25"),
            StageTiming(stage="select", started_ms=106, duration_ms=1.0),
            StageTiming(stage="condense", started_ms=0, duration_ms=50.0),
        ]
    )
    assert local_latency_ms(trace) == 6.0


# ------------------------------------------------------------------ the test split
def record_choice(env, sweep_id, run, tier="fast"):
    """What `kb eval choose` records in the sweep's manifest."""
    from hal9000.knowledge.eval.sweep import write_manifest

    manifest = read_manifest(env.settings.eval_dir, sweep_id)
    manifest["choice"] = {"config": run.config["name"], "run_id": run.id, "chat_tier": tier,
                          "at": "now"}  # fmt: skip
    write_manifest(env.settings.eval_dir, sweep_id, manifest)


def test_the_test_split_runs_once_judged_and_scored_against_the_targets(swept_env, caplog):
    from hal9000.knowledge.eval.decide import render_targets
    from hal9000.knowledge.eval.report import render_report

    env, sweep_id = swept_env
    runner = env.runner()
    runner.sweep_id = sweep_id
    leader = runner.finalists()
    name = leader[0].config["name"]
    record_choice(env, sweep_id, leader[0])
    refused = env.runner().run_test(sweep_id, name)
    assert refused.status == "refused" and "no judge calibration labels" in refused.message

    label_calibration(env.conn, leader[0].id)
    env.out.seek(0)
    env.out.truncate()
    outcome = env.runner().run_test(sweep_id, name)
    out = env.out.getvalue()
    assert outcome.status == "done", (outcome.message, out)
    assert out.index("Estimate for the test split") < out.index("FIRST LLM CALL")
    (run,) = [r for r in list_runs(env.conn) if r.stage == "test"]
    assert run.split == "test" and run.config["sweep_id"] == sweep_id
    assert run.config["chat_tier"] == "fast" and run.config["source_run"] == leader[0].id
    assert run.metrics.n_questions == 4  # the approved test items
    assert 2000 in run.metrics.budgeted_recall and run.metrics.mrr is not None
    assert run.metrics.correct is not None and run.metrics.faithful is not None
    assert run.config["declined"]["low"] <= run.config["declined"]["value"]
    record = results(run)[0]
    assert "recall" in record and record["correctness"] and "answer_latency_ms" in record
    assert "answer_output" not in record and "judged" not in record
    assert run.cost_usd > 0 and {a for a, _ in env.ledger()} == {"knowledge-eval"}
    assert "Evidence recall within a 2,000-token budget:" in out
    assert not list((env.settings.eval_dir / "runs" / run.id).glob("partial-*"))

    again = env.runner().run_test(sweep_id, name)
    assert again.status == "refused" and "already has a test-split run" in again.message
    forced = env.runner().run_test(sweep_id, name, force=True)
    assert forced.status == "done"
    tests = [r for r in list_runs(env.conn) if r.stage == "test"]
    assert len(tests) == 2 and any(r.config.get("forced_rerun") for r in tests)
    assert "test split forced" in caplog.text
    notes = read_manifest(env.settings.eval_dir, sweep_id)["notes"]["test"]
    assert any("--force" in n for n in notes)
    report = render_report([*sweep_runs(env.conn, sweep_id), *tests], source=sweep_id)
    assert "Plan §1 targets" in report and render_targets(tests).split("\n")[0] in report
    assert "Correct counts partial answers as 0.5" in report


def test_the_test_split_is_only_for_the_recorded_choice(swept_env):
    env, sweep_id = swept_env
    runner = env.runner()
    runner.sweep_id = sweep_id
    first, second = runner.finalists()[:2]
    label_calibration(env.conn, first.id)
    none_yet = env.runner().run_test(sweep_id, first.config["name"])
    assert none_yet.status == "refused" and "no configuration has been chosen" in none_yet.message
    record_choice(env, sweep_id, first)
    other = env.runner().run_test(sweep_id, second.config["name"])
    assert other.status == "refused" and "is not the recorded choice" in other.message
    record_choice(env, sweep_id, first, tier="powerful")  # chosen with the other chat tier
    tier = env.runner().run_test(sweep_id, first.config["name"])
    assert tier.status == "refused" and "chat tier powerful" in tier.message
    record_choice(env, sweep_id, first)
    assert not [r for r in list_runs(env.conn) if r.stage == "test"]
    forced = env.runner().run_test(sweep_id, second.config["name"], force=True)
    assert forced.status == "done", forced.message
    (run,) = [r for r in list_runs(env.conn) if r.stage == "test"]
    assert "is not the recorded choice" in run.config["forced_because"]


def test_the_test_split_refuses_what_it_cannot_run(swept_env):
    env, sweep_id = swept_env
    k_variant = next(r for r in sweep_runs(env.conn, sweep_id, "S4") if r.config.get("k"))
    with pytest.raises(ValueError, match="k variant"):
        env.runner().run_test(sweep_id, k_variant.id)
    with pytest.raises(ValueError, match="no configuration"):
        env.runner().run_test(sweep_id, "nope")
    with pytest.raises(ValueError, match="no sweep"):
        env.runner().run_test("sweep-missing", "nope")
    name = k_variant.config["name"].rsplit("+k", 1)[0]
    baseline = next(r for r in sweep_runs(env.conn, sweep_id, "S4") if r.config["name"] == name)
    record_choice(env, sweep_id, baseline)
    over = env.runner(cap=1e-9).run_test(sweep_id, name, force_judge=True)
    assert over.status == "refused" and "exceeds what is left" in over.message
    declined = env.runner(confirm=lambda q: False).run_test(sweep_id, name, force_judge=True)
    assert declined.status == "refused" and declined.message == "not confirmed"
    with env.conn:
        env.conn.execute("UPDATE gold_items SET split = 'dev'")
    assert "no approved test-split" in env.runner().run_test(sweep_id, name).message
    assert not [r for r in list_runs(env.conn) if r.stage == "test"]


# ------------------------------------------------------------------ resume never re-charges
S5_ONE = ("    top: 2\n    chat_tiers: [fast, powerful]\n    repeat_top: 1",
          "    top: 1\n    chat_tiers: [fast]\n    repeat_top: 0")  # fmt: skip


def test_a_stopped_s5_resumes_paying_only_for_the_remaining_answers(swept_env):
    from tests.knowledge.eval.test_sweep_fixtures import FakeProvider, fake_chat

    env, sweep_id = swept_env
    grid = (env.settings.config_dir / "knowledge_eval.yaml").read_text().replace(*S5_ONE)
    runner = env.runner(grid=grid)
    runner.sweep_id = sweep_id
    label_calibration(env.conn, runner.finalists()[0].id)
    per_answer = 360_000 * 0.25e-6  # $0.09 of gpt-5-mini input per answer; judging is free
    base = sum(usd for _, usd in env.ledger())  # the sweep's earlier (T10) spend
    first_chat = fake_chat(prompt_tokens=360_000, completion=0)
    stopped = env.runner(
        cap=0.40,  # above the estimate; the 5th answer crosses it, the 6th never starts
        stream_response=first_chat,
        generate_response=FakeProvider(usage=(0, 0)),
    ).run(["S5"], sweep_id=sweep_id)
    assert stopped.status == "stopped", env.out.getvalue()
    assert len(first_chat.calls) == 5
    paid_before = sum(usd for _, usd in env.ledger())
    assert paid_before - base == pytest.approx(5 * per_answer)
    assert not sweep_runs(env.conn, sweep_id, "S5")

    second_chat = fake_chat(prompt_tokens=360_000, completion=0)
    resumed = env.runner(
        cap=5.0, stream_response=second_chat, generate_response=FakeProvider(usage=(0, 0))
    ).run(["S5"], sweep_id=sweep_id)
    assert resumed.status == "done", env.out.getvalue()
    assert len(second_chat.calls) == 10  # the 5 answered questions were not asked again
    assert sum(usd for _, usd in env.ledger()) - paid_before == pytest.approx(10 * per_answer)
    (run,) = sweep_runs(env.conn, sweep_id, "S5")
    assert run.metrics.n_questions == 15 and all(r["correctness"] for r in results(run))
    assert run.cost_usd == pytest.approx(15 * per_answer)  # both sessions' spend
    assert not list((env.settings.eval_dir / "runs" / run.id).glob("partial-*"))
    assert "answered 10/15" in env.out.getvalue()


def test_a_stopped_t10_run_resumes_without_rewriting_again(env):
    grid = TINY_GRID
    stopped = env.runner(cap=0.01, grid=grid).run(["S1a", "S1b", "S2", "S3"])
    assert stopped.status == "stopped"
    rows_first = env.ledger()
    partial = list((env.settings.eval_dir / "runs").glob("*.s3.*/partial-retrieval.jsonl"))
    assert len(partial) == 1
    saved = len(partial[0].read_text().splitlines())
    assert 0 < saved < 15
    runner = env.runner(cap=1.0, grid=grid)
    resumed = runner.run(["S3"], sweep_id=stopped.sweep_id)
    assert resumed.status == "done"
    assert runner.deps.generate_response.calls == 15 - saved  # only the questions not yet done
    assert len(env.ledger()) == len(rows_first) + 15 - saved


# ------------------------------------------------------------------ estimates >= metered
def test_the_s5_estimate_covers_realistic_metered_answers_and_judgements(swept_env):
    from tests.knowledge.eval.test_sweep_fixtures import FakeProvider, fake_chat

    env, sweep_id = swept_env
    grid = (env.settings.config_dir / "knowledge_eval.yaml").read_text().replace(*S5_ONE)
    probe = env.runner(grid=grid)
    probe.sweep_id = sweep_id
    label_calibration(env.conn, probe.finalists()[0].id)
    config = SweepConfig.from_run_config(probe.finalists()[0].config)
    estimate = probe._estimate("S5", [replace(config, chat_tier="fast")])
    # realistic: a 2k-token context answered with reasoning; a judge reading it with reasoning
    base = sum(usd for _, usd in env.ledger())
    runner = env.runner(
        grid=grid,
        stream_response=fake_chat(prompt_tokens=2800, completion=850, reasoning=600),
        generate_response=FakeProvider(
            usage=(3600, 1300), models={"powerful": "openai/gpt-5", "fast": "openai/gpt-5-mini"}
        ),
    )
    assert runner.run(["S5"], sweep_id=sweep_id).status == "done"
    metered = sum(usd for _, usd in env.ledger()) - base
    assert 0 < metered <= estimate.usd

    # with judge-prep's metered cost recorded, the estimate is that cost times the margin
    from hal9000.knowledge.eval.sweep import metered_answer_cost, record_metered_costs

    record_metered_costs(env.settings.eval_dir, answer_tier="fast", judge_tier="powerful",
                         answers=3, answer_usd=0.03, judgements=3, judge_usd=0.06)  # fmt: skip
    assert metered_answer_cost(env.settings.eval_dir, "fast", "powerful") == pytest.approx(0.03)
    again = probe._estimate("S5", [replace(config, chat_tier="fast")])
    assert again.usd == pytest.approx(15 * 0.03 * 1.25)
    assert "metered" in again.lines[-1]
    assert metered_answer_cost(env.settings.eval_dir, "powerful", "powerful") is None


def test_the_embedding_estimate_covers_a_real_model_metered(env):
    from hal9000.knowledge.index.fake import FakeEmbedder
    from hal9000.knowledge.tokens import count_tokens

    class TokenCounting(FakeEmbedder):  # bills real cl100k token counts, like the provider
        def __call__(self, texts, model=None, dimensions=None, **kw):
            kw["count_tokens"] = count_tokens
            return super().__call__(texts, model, dimensions, **kw)

    grid = TINY_GRID.replace(
        "embedding: {model: fake/embed-64, dimensions: 64}",
        "embedding: {model: openai/text-embedding-3-small, dimensions: 256}",
    ).replace("retrieval: [dense, rrf]", "retrieval: [dense]")
    runner = env.runner(grid=grid, embed_fn=TokenCounting())
    runner.sweep_id = "sweep-estimate"
    configs = [SweepConfig(size=512, embed_model="openai/text-embedding-3-small", dimensions=256,
                           retrieval="dense")]  # fmt: skip
    estimate = runner._estimate("S1a", configs)
    assert (
        estimate.usd > 0
        and "uncached tokens with openai/text-embedding-3-small" in (estimate.lines[0])
    )
    outcome = env.runner(grid=grid, embed_fn=TokenCounting()).run(["S1a"])
    assert outcome.status == "done", env.out.getvalue()
    metered = sum(usd for _, usd in env.ledger())
    assert 0 < metered <= estimate.usd + 1e-12
    assert env.active() == ACTIVE
