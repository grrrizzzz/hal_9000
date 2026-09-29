"""The staged sweep S1a-S5 on the fixture corpus, offline (fake embedder, reranker and LLM)."""
# ruff: noqa: F811 - pytest fixtures imported from test_sweep_fixtures are parameters here

from __future__ import annotations

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
