"""S6 ``kb eval latency`` in fake mode: replay, live end to end, and the timed scratch ingest."""
# ruff: noqa: F811 - pytest fixtures imported from test_sweep_fixtures are parameters here

from __future__ import annotations

import hashlib

from hal9000.knowledge.eval.config import load_eval_config
from hal9000.knowledge.eval.latency import LatencyRunner, percentiles
from hal9000.knowledge.eval.runs import list_runs
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


def estimate_sha() -> str:
    return hashlib.sha256(corpus.ESTIMATE.path.read_bytes()).hexdigest()


def test_latency_replays_answers_live_and_times_a_scratch_ingest(swept_env):
    env, sweep_id = swept_env
    finalists = [r for r in list_runs(env.conn) if r.stage == "S4" and r.config.get("k") is None]
    profiles = list(dict.fromkeys(r.config["profile"] for r in finalists))
    chunks_before = env.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    outcome = latency_runner(env).run(profiles, n=25, end_to_end=3, ingest_sha=estimate_sha())
    out = env.out.getvalue()
    assert outcome.status == "done", (outcome.message, out)
    assert len(outcome.runs) == len(profiles)
    assert out.index("Estimate: $") < out.index("local p50/p95/p99")
    assert "3 live answers" in out and "scratch ingest of" in out
    first = outcome.runs[0]
    for run in outcome.runs:
        measured = run.config["measured"]
        assert run.stage == "S6" and run.host in ("mac", "pi", "linux", "windows")
        assert measured["replayed"] == 25
        assert measured["local_p50_ms"] <= measured["local_p95_ms"] <= measured["local_p99_ms"]
        assert run.metrics.latency_ms_p95 == measured["local_p95_ms"]
        assert measured["end_to_end"] == 3 and measured["ttft_p95_ms"] <= measured["e2e_p95_ms"]
        # the source run's configuration and dev metrics, with this host's latency
        assert run.config["source_run"] and run.config["sweep_id"] == sweep_id
        assert 2000 in run.metrics.budgeted_recall
        assert run.metrics.cost_per_question_usd > 0
    ingest = first.config["measured"]
    assert ingest["ingest_seconds"] > 0 and ingest["ingest_chunks"] > 0
    assert ingest["ingest_file"] == estimate_sha()[:12]
    # the scratch ingest used a temporary data directory: nothing was added here
    assert env.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == chunks_before
    assert env.active() == ACTIVE
    assert {agent for agent, _ in env.ledger()} == {"knowledge-eval"}


def test_a_profile_without_sweep_runs_uses_knowledge_yaml(env):
    outcome = latency_runner(env).run([ACTIVE], n=5, end_to_end=0)
    assert outcome.status == "done", outcome.message
    (run,) = outcome.runs
    assert run.config["retrieval"] == "rrf" and "source_run" not in run.config
    assert "sweep_id" not in run.config
    assert run.metrics.n_questions == 5 and run.metrics.budgeted_recall == {}
    assert any("knowledge.yaml" in note for note in run.config["notes"])


def test_latency_refuses_what_it_cannot_do(env):
    runner = latency_runner(env)
    assert "not built" in runner.run(["pdfium.fixed-999.hdr.fake-64"]).message
    assert "no stored file" in runner.run([ACTIVE], n=1, end_to_end=0, ingest_sha="ab" * 32).message
    assert "--n must be" in runner.run([ACTIVE], n=0).message
    assert "at least one profile" in runner.run([]).message
    declined = latency_runner(env, confirm=lambda q: False).run([ACTIVE], n=1, end_to_end=0)
    assert declined.status == "refused" and declined.message == "not confirmed"
    over = latency_runner(env, cap=1e-9).run([ACTIVE], n=1, end_to_end=2)
    assert over.status == "refused" and "exceeds what is left" in over.message
    assert not [r for r in list_runs(env.conn) if r.stage == "S6"]


def test_a_cap_stops_the_live_answers(env):
    from hal9000.llm import Usage

    chat = deps().stream_response

    def expensive(prompt, tier="fast", *, on_usage=None, **kw):  # each answer costs ~$0.0052
        def bigger(usage):
            on_usage(Usage(model="openai/gpt-5-mini", prompt=20000, cached=0, completion=100,
                           reasoning=0))  # fmt: skip

        return chat(prompt, tier, on_usage=bigger if on_usage else None, **kw)

    runner = latency_runner(env, cap=0.008)  # the estimate for 5 answers (~$0.0066) fits
    runner.deps.stream_response = expensive
    stopped = runner.run([ACTIVE], n=2, end_to_end=5)
    assert stopped.status == "stopped" and "STOPPED at the cost cap" in stopped.message
    assert not stopped.runs
    assert 0.008 <= sum(usd for _, usd in env.ledger()) < 0.016


def test_percentiles():
    assert percentiles([]) == [None, None, None]
    assert percentiles([1.0, 2.0, 3.0, 4.0], (50,)) == [2.5]


def test_questions_cycle_through_the_approved_set(env):
    runner = latency_runner(env)
    replay, distinct = runner._questions(40)
    assert len(replay) == 40 and len(distinct) == 19
    assert replay[19].id == replay[0].id
