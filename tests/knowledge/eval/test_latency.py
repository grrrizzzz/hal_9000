"""S6 ``kb eval latency`` in fake mode: the finalists' own configurations, dev questions only,
replay, live end to end, and the timed scratch ingest."""
# ruff: noqa: F811 - pytest fixtures imported from test_sweep_fixtures are parameters here

from __future__ import annotations

import hashlib

from hal9000.knowledge.eval.config import load_eval_config
from hal9000.knowledge.eval.decide import identity, recommend
from hal9000.knowledge.eval.latency import LatencyRunner, percentiles
from hal9000.knowledge.eval.runs import list_runs
from hal9000.knowledge.eval.store import list_gold
from tests.knowledge.eval.test_sweep_fixtures import (  # noqa: F401 - fixtures
    ACTIVE,
    corpus_root,
    deps,
    env,
    fake_mode,
    fresh_query_cache,
    make_meter,
    swept_env,
    swept_root,
)
from tests.knowledge.fixtures import corpus


def latency_runner(env, *, cap: float = 5.0, confirm=None, **kw) -> LatencyRunner:
    meter = make_meter(env.root, cap=cap)
    env.meters.append(meter)
    return LatencyRunner(
        env.conn,
        env.settings,
        eval_config=load_eval_config(env.settings.config_dir / "knowledge_eval.yaml"),
        meter=meter,
        deps=deps(env.out),
        confirm=confirm or (lambda question: True),
        out=env.out,
        segment=False,
        **kw,
    )


def finalists(env, sweep_id):
    runner = env.runner()
    runner.sweep_id = sweep_id
    return runner.finalists()


def estimate_sha() -> str:
    return hashlib.sha256(corpus.ESTIMATE.path.read_bytes()).hexdigest()


def test_s6_measures_each_finalist_with_its_own_configuration(swept_env, monkeypatch):
    env, sweep_id = swept_env
    monkeypatch.setattr("hal9000.knowledge.eval.runs.host_class", lambda: "pi")
    top = finalists(env, sweep_id)
    assert len(top) == 2 and top[0].config["profile"] == top[1].config["profile"]  # same profile
    chunks_before = env.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    runner = latency_runner(env)
    outcome = runner.run(n=25, end_to_end=3, ingest_sha=estimate_sha())  # default: the finalists
    out = env.out.getvalue()
    assert outcome.status == "done", (outcome.message, out)
    assert out.index("Estimate: $") < out.index("local p50/p95/p99")
    assert {r.config["name"] for r in outcome.runs} == {r.config["name"] for r in top}
    for run, finalist in zip(outcome.runs, top):
        measured = run.config["measured"]
        assert identity(run.config) == identity(finalist.config)
        assert run.config["source_run"] == finalist.id and run.config["sweep_id"] == sweep_id
        assert run.stage == "S6" and run.host == "pi"
        assert measured["replayed"] == 25
        assert measured["local_p50_ms"] <= measured["local_p95_ms"] <= measured["local_p99_ms"]
        assert run.metrics.latency_ms_p95 == measured["local_p95_ms"]
        assert measured["end_to_end"] == 3 and measured["ttft_p95_ms"] <= measured["e2e_p95_ms"]
        assert run.metrics.budgeted_recall == finalist.metrics.budgeted_recall
        assert run.metrics.cost_per_question_usd > 0
    ingest = outcome.runs[0].config["measured"]
    assert ingest["ingest_seconds"] > 0 and ingest["ingest_chunks"] > 0
    assert ingest["ingest_file"] == estimate_sha()[:12]
    assert env.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == chunks_before
    assert env.active() == ACTIVE
    assert {agent for agent, _ in env.ledger()} == {"knowledge-eval"}
    # dev questions only: nothing from the test split was replayed or answered
    test_ids = {g.id for g in list_gold(env.conn, split="test")}
    test_questions = {g.question for g in list_gold(env.conn, split="test")}
    for run in outcome.runs:
        assert not test_ids & {r["gold_item_id"] for r in _records(run)}
    prompts = " ".join(
        str(m.get("content")) for p in runner.deps.stream_response.calls for m in p.messages
    )
    assert prompts and not any(q in prompts for q in test_questions)
    # the decision rule matches every finalist to its own Pi row: none is "not measured"
    rec = recommend([r for r in list_runs(env.conn) if r.config.get("sweep_id") == sweep_id])
    assert rec.pi_measured and not rec.excluded
    assert {rec.chosen.name, *(a.name for a in rec.alternatives)} == {r.config["name"] for r in top}


def _records(run):
    import json
    from pathlib import Path

    return [json.loads(line) for line in Path(run.results_path).read_text().splitlines()]


def test_named_configurations_keep_their_transform_and_rewrite_once_per_question(swept_env):
    env, sweep_id = swept_env
    t10 = next(
        r for r in list_runs(env.conn)
        if r.stage == "S3" and r.config.get("query_transform") == "multi"
    )  # fmt: skip
    runner = latency_runner(env)
    outcome = runner.run(sweep_id=sweep_id, configs=[t10.config["name"]], n=30, end_to_end=0)
    assert outcome.status == "done", outcome.message
    (run,) = outcome.runs
    assert run.config["query_transform"] == "multi" and identity(run.config) == identity(t10.config)
    dev = list_gold(env.conn, status="approved", split="dev")
    assert runner.deps.generate_response.calls == len(dev)  # 30 replays, one rewrite each question
    assert run.cost_usd > 0
    assert "multi rewrites once per dev question" in env.out.getvalue()


def test_profiles_mean_the_finalists_on_them(swept_env):
    env, sweep_id = swept_env
    top = finalists(env, sweep_id)
    outcome = latency_runner(env).run(profiles=[top[0].config["profile"]], n=3, end_to_end=0)
    assert {r.config["name"] for r in outcome.runs} == {r.config["name"] for r in top}


def test_a_profile_outside_any_sweep_uses_knowledge_yaml(env):
    outcome = latency_runner(env).run(profiles=[ACTIVE], n=5, end_to_end=0)
    assert outcome.status == "done", outcome.message
    (run,) = outcome.runs
    assert run.config["retrieval"] == "rrf" and "source_run" not in run.config
    assert "sweep_id" not in run.config
    assert run.metrics.n_questions == 5 and run.metrics.budgeted_recall == {}
    assert any("knowledge.yaml" in note for note in run.config["notes"])


def test_latency_refuses_what_it_cannot_do(env):
    runner = latency_runner(env)
    assert "no finalists" in runner.run().message
    assert "not built" in runner.run(profiles=["pdfium.fixed-999.hdr.fake-64"]).message
    assert "no stored file" in runner.run(profiles=[ACTIVE], n=1, end_to_end=0,
                                          ingest_sha="ab" * 32).message  # fmt: skip
    assert "--n must be" in runner.run(profiles=[ACTIVE], n=0).message
    declined = latency_runner(env, confirm=lambda q: False).run(profiles=[ACTIVE], n=1,
                                                                end_to_end=0)  # fmt: skip
    assert declined.status == "refused" and declined.message == "not confirmed"
    over = latency_runner(env, cap=1e-9).run(profiles=[ACTIVE], n=1, end_to_end=2)
    assert over.status == "refused" and "exceeds what is left" in over.message
    assert not [r for r in list_runs(env.conn) if r.stage == "S6"]


def test_latency_refuses_unknown_configurations_and_k_variants(swept_env):
    swept, sweep_id = swept_env
    other = latency_runner(swept)
    assert "no configuration" in other.run(sweep_id=sweep_id, configs=["nope"]).message
    k_variant = next(r for r in list_runs(swept.conn) if r.stage == "S4" and r.config.get("k"))
    assert "k variant" in other.run(configs=[k_variant.config["name"]]).message


def test_a_cap_stops_the_live_answers(env):
    from hal9000.llm import Usage

    chat = deps().stream_response

    def expensive(prompt, tier="fast", *, on_usage=None, **kw):  # each answer costs ~$0.0052
        def bigger(usage):
            on_usage(Usage(model="openai/gpt-5-mini", prompt=20000, cached=0, completion=100,
                           reasoning=0))  # fmt: skip

        return chat(prompt, tier, on_usage=bigger if on_usage else None, **kw)

    runner = latency_runner(env, cap=0.02)  # the estimate for 5 answers fits
    runner.deps.stream_response = expensive
    stopped = runner.run(profiles=[ACTIVE], n=2, end_to_end=5)
    assert stopped.status == "stopped" and "STOPPED at the cost cap" in stopped.message
    assert not stopped.runs
    assert 0.02 <= sum(usd for _, usd in env.ledger()) < 0.03


def test_percentiles():
    assert percentiles([]) == [None, None, None]
    assert percentiles([1.0, 2.0, 3.0, 4.0], (50,)) == [2.5]


def test_replay_cycles_through_the_dev_questions_only(env):
    runner = latency_runner(env)
    replay, distinct = runner._questions(40)
    dev = list_gold(env.conn, status="approved", split="dev")
    assert len(replay) == 40 and [g.id for g in distinct] == [g.id for g in dev]
    assert replay[len(dev)].id == replay[0].id


def test_the_grid_supplies_replay_and_end_to_end_defaults(env):
    grid = (env.settings.config_dir / "knowledge_eval.yaml").read_text()
    grid = grid.replace("  S6:\n", "  S6:\n    replay: 4\n    end_to_end: 0\n")
    (env.settings.config_dir / "knowledge_eval.yaml").write_text(grid)
    outcome = latency_runner(env).run(profiles=[ACTIVE])
    assert outcome.status == "done", outcome.message
    assert outcome.runs[0].config["measured"]["replayed"] == 4
    assert "end_to_end" not in outcome.runs[0].config["measured"]
