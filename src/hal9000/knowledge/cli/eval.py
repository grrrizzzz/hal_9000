"""``hal9000 kb eval sweep|judge-prep|latency|choose|export|report|regression`` (plan §8, §17).

``sweep``, ``latency`` and ``choose`` are the staged sweep (KWP5 part B):
:mod:`hal9000.knowledge.eval.sweep`, :mod:`~hal9000.knowledge.eval.latency` and
:mod:`~hal9000.knowledge.eval.decide`. Exit codes: 0 done, 1 error, 3 not run
(refused or not confirmed), 4 stopped at a cost cap (partial results kept).
"""

from __future__ import annotations

import argparse
import sys

from hal9000 import config as hal_config

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_RUN = 3
EXIT_STOPPED = 4


def _err(message: str) -> None:
    print(f"hal9000 kb: {message}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    command = args.kb_command
    from hal9000.knowledge.config import settings
    from hal9000.knowledge.storage import open_db

    current = settings()
    conn = open_db(current.db_path)
    try:
        handler = {
            "eval judge-prep": _judge_prep,
            "eval export": _export,
            "eval report": _report,
            "eval regression": _regression,
            "eval sweep": _sweep,
            "eval latency": _latency,
            "eval choose": _choose,
        }[command]
        return handler(args, conn, current)
    finally:
        conn.close()


# ------------------------------------------------------------------ export
def _export(args, conn, current) -> int:
    from hal9000.knowledge.eval.export import LeakDetected, export_runs

    try:
        path = export_runs(conn, args.run, eval_dir=current.eval_dir)
    except KeyError:
        _err(f"no evaluation run or sweep {args.run!r}")
        return EXIT_ERROR
    except LeakDetected as error:
        _err(str(error))
        return EXIT_ERROR
    print(f"Wrote {path} (aggregates and per-question scores; no document text)")
    return EXIT_OK


def _recommendation(conn, current, run_id: str) -> str | None:
    """The plan §8.6 section for the report (None when ``run_id`` is not a sweep)."""
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.decide import recommend, render_recommendation
    from hal9000.knowledge.eval.runs import runs_for
    from hal9000.knowledge.eval.sweep import load_sweep_settings, read_manifest

    runs = runs_for(conn, run_id)
    if not any(r.config.get("sweep_id") == run_id for r in runs):
        return None
    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    limit = load_sweep_settings(cfg).S6.p95_local_retrieval_ms if cfg.sweep else 300.0
    thresholds = _thresholds(conn)
    rec = recommend(
        runs, scoring=cfg.scoring.to_config(), p95_limit_ms=limit, thresholds=thresholds
    )
    choice = read_manifest(current.eval_dir, run_id).get("choice")
    return render_recommendation(rec, choice=choice)


def _thresholds(conn) -> dict[tuple[str, str], float]:
    rows = conn.execute(
        "SELECT p.name, t.reranker, t.threshold FROM abstain_thresholds t "
        "JOIN profiles p ON p.id = t.profile_id"
    ).fetchall()
    return {(name, reranker): float(value) for name, reranker, value in rows}


def _report(args, conn, current) -> int:
    from hal9000.knowledge.eval.export import LeakDetected
    from hal9000.knowledge.eval.report import write_report

    target = current.config_dir.parent / "docs" / "KNOWLEDGE_EVAL_REPORT.md"
    try:
        path = write_report(
            conn, args.run, target, recommendation=_recommendation(conn, current, args.run)
        )
    except KeyError:
        _err(f"no evaluation run or sweep {args.run!r}")
        return EXIT_ERROR
    except LeakDetected as error:
        _err(f"report not written: {error}")
        return EXIT_ERROR
    print(f"Wrote {path} (aggregates only)")
    return EXIT_OK


# --------------------------------------------------------------- judge-prep
def _judge_prep(args, conn, current) -> int:
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.judge import (
        choose_calibration_items,
        estimate_judge_prep,
        judge_prep,
    )
    from hal9000.knowledge.eval.runs import get_run
    from hal9000.knowledge.eval.spend import confirm, open_spend
    from hal9000.knowledge.eval.store import list_gold, list_judge_labels
    from hal9000.knowledge.eval.wiring import NotWired, answerer_for

    run = get_run(conn, args.run)
    if run is None:
        _err(f"no evaluation run {args.run!r} (judge-prep answers with that run's configuration)")
        return EXIT_ERROR
    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml").judge
    pool = list_gold(conn, status="approved", split="dev")
    done = {label.gold_item_id for label in list_judge_labels(conn, eval_run_id=run.id)}
    todo = [i for i in choose_calibration_items(pool, cfg.n_items) if i.id not in done]
    if not todo:
        _err(
            "nothing to answer: no approved dev items"
            if not pool
            else f"all calibration answers for {run.id} already exist"
        )
        return EXIT_ERROR if not pool else EXIT_OK
    estimate = estimate_judge_prep(len(todo), cfg, current.chat_tier, current.judge_tier)
    print(f"Judge calibration: {len(todo)} dev answers with the configuration of {run.id}.")
    print(f"Estimate: {estimate.describe()}.")
    spend = open_spend(current, max_cost=None, label="judge-prep")
    try:
        refusal = spend.refusal(estimate.usd)
        if refusal:
            _err(f"not started: {refusal}")
            return EXIT_NOT_RUN
        if not confirm("Generate and judge the calibration answers?", yes=args.yes):
            return EXIT_NOT_RUN
        try:
            answer = answerer_for(conn, current, meter=spend.meter, run=run)
            generate = spend.generate()
        except NotWired as error:
            _err(str(error))
            return EXIT_ERROR
        except hal_config.ConfigError as error:
            _err(str(error))
            return EXIT_ERROR
        result = judge_prep(
            conn,
            answer,
            generate,
            eval_run_id=run.id,
            config=cfg,
            judge_tier=current.judge_tier,
        )
        print(
            f"Stored {len(result.label_ids)} answers with the judge's labels "
            f"({result.skipped} already done). Label them on the Judge calibration tab. "
            f"Spent ${spend.meter.spent:.4f} (agent knowledge-eval)."
        )
        for error in result.errors[:5]:
            print(f"error: {error}")
        if result.stopped:
            print(f"STOPPED EARLY: {result.stopped} (answers so far kept; re-run to continue)")
        return EXIT_OK
    finally:
        spend.close()


# --------------------------------------------------------------- regression
def _regression(args, conn, current) -> int:
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.regression import run_regression
    from hal9000.knowledge.eval.spend import confirm, open_spend
    from hal9000.knowledge.eval.store import list_gold
    from hal9000.knowledge.eval.wiring import NotWired, retriever_for
    from hal9000.knowledge.tokens import count_tokens

    questions = [
        g.question for g in list_gold(conn, status="approved") if g.category != "unanswerable"
    ]
    if not questions:
        _err("no approved answerable questions: review the question set first (gate G-K1)")
        return EXIT_ERROR
    tokens = sum(count_tokens(q) for q in questions)
    rate_in = hal_config.rates_for(current.embed_model)[0]
    usd = tokens * rate_in / 1e6
    print(
        f"Regression: {len(questions)} approved questions, retrieval only. Estimate: "
        f"~{tokens:,} query-embedding tokens with {current.embed_model}, "
        f"about ${usd:.4f} (cached queries cost nothing)."
    )
    spend = open_spend(current, max_cost=None, label="regression")
    try:
        refusal = spend.refusal(usd)
        if refusal:
            _err(f"not started: {refusal}")
            return EXIT_NOT_RUN
        if not confirm("Run the regression?", yes=args.yes):
            return EXIT_NOT_RUN
        try:
            binding = retriever_for(conn, current, meter=spend.meter)
        except NotWired as error:
            _err(str(error))
            return EXIT_ERROR
        cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
        result = run_regression(
            conn,
            binding.retrieve,
            profile_id=binding.profile_id,
            chunk_size=binding.chunk_size,
            trigger="manual",
            config=cfg,
        )
        print(f"{binding.profile_name}: {result.describe()}")
        return EXIT_OK
    finally:
        spend.close()


# -------------------------------------------------------------------- sweep
def _sweep(args, conn, current) -> int:
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.spend import confirm, open_spend
    from hal9000.knowledge.eval.sweep import SweepRunner

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    try:
        cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    except (hal_config.ConfigError, ValueError) as error:
        _err(str(error))
        return EXIT_ERROR
    spend = open_spend(current, max_cost=args.max_cost, label="sweep")
    try:
        runner = SweepRunner(
            conn,
            current,
            eval_config=cfg,
            meter=spend.meter,
            max_cost=args.max_cost,
            eval_cap=spend.eval_cap,
            confirm=lambda question: confirm(question, yes=args.yes),
        )
        outcome = runner.run(stages, sweep_id=args.resume, force_judge=args.force_judge)
    except (hal_config.ConfigError, ValueError) as error:
        _err(str(error))
        return EXIT_ERROR
    finally:
        spend.close()
    print(
        f"\nSweep {outcome.sweep_id}: {outcome.runs} finished run(s); spent "
        f"${outcome.spent:.4f} this time (agent knowledge-eval)."
    )
    if outcome.status == "refused":
        _err(outcome.message)
        return EXIT_NOT_RUN
    if outcome.status == "stopped":
        print(outcome.message)
        return EXIT_STOPPED
    if "S4" in stages and "S5" not in stages:
        leaders = runner.finalists()
        if leaders:
            print(
                f"Next (gate G-K1b): hal9000 kb eval judge-prep {leaders[0].id}, label the "
                f"answers on the Judge calibration tab, then hal9000 kb eval sweep --stages S5 "
                f"--resume {outcome.sweep_id} --yes"
            )
    print(f"Report: hal9000 kb eval report {outcome.sweep_id}")
    return EXIT_OK


# ------------------------------------------------------------------ latency
def _latency(args, conn, current) -> int:
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.latency import LatencyRunner
    from hal9000.knowledge.eval.spend import confirm, open_spend

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    spend = open_spend(current, max_cost=None, label="latency")
    try:
        runner = LatencyRunner(
            conn,
            current,
            eval_config=cfg,
            meter=spend.meter,
            eval_cap=spend.eval_cap,
            confirm=lambda question: confirm(question, yes=args.yes),
        )
        outcome = runner.run(
            profiles, n=args.n, end_to_end=args.end_to_end, ingest_sha=args.ingest_test
        )
    except (hal_config.ConfigError, ValueError) as error:
        _err(str(error))
        return EXIT_ERROR
    finally:
        spend.close()
    if outcome.status == "refused":
        _err(f"not run: {outcome.message}")
        return EXIT_NOT_RUN
    for run in outcome.runs:
        print(f"Stored {run.id} ({run.config.get('profile')}, host {run.host})")
    print(f"Spent ${spend.meter.spent:.4f} (agent knowledge-eval).")
    if outcome.status == "stopped":
        print(outcome.message)
        return EXIT_STOPPED
    return EXIT_OK


# ------------------------------------------------------------------- choose
def _choose(args, conn, current) -> int:
    from datetime import UTC, datetime

    from hal9000.knowledge.config import EVAL_AGENT
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.decide import (
        Ranker,
        _chat_tier,
        find_run,
        identity,
        write_choice_yaml,
    )
    from hal9000.knowledge.eval.runs import runs_for
    from hal9000.knowledge.eval.spend import confirm, open_spend
    from hal9000.knowledge.eval.sweep import (
        SweepConfig,
        load_sweep_settings,
        profile_texts,
        read_manifest,
        write_manifest,
    )
    from hal9000.knowledge.index.embedding import EmbeddingService, is_fake_model
    from hal9000.knowledge.index.profiles import ProfileError, activate_profile, profile_status
    from hal9000.knowledge.retrieve.thresholds import get_threshold, set_threshold

    runs = [r for r in runs_for(conn, args.run) if r.config.get("sweep_id") == args.run]
    if not runs:
        _err(f"no sweep {args.run!r}")
        return EXIT_ERROR
    run = find_run(runs, args.config)
    if run is None or run.stage not in ("S1a", "S1b", "S2", "S3", "S4", "S5"):
        names = sorted({str(r.config.get("name")) for r in runs if r.stage == "S4"})
        _err(
            f"no configuration {args.config!r} in sweep {args.run} (a run ID or a configuration "
            "name, as the report prints it"
            + (f"; S4 has: {', '.join(names[:6])}" if names else "")
            + ")"
        )
        return EXIT_ERROR
    config = SweepConfig.from_run_config(run.config)
    if config.k is not None:
        _err(
            "a k variant (T9) can't be chosen: config/knowledge.yaml has no k; choose its baseline"
        )
        return EXIT_ERROR
    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    budget = load_sweep_settings(cfg).decision_budget if cfg.sweep else 2000
    tier = config.chat_tier
    if tier is None:
        s5 = [r for r in runs if r.stage == "S5" and r.finished_at is not None]
        tier, _ = _chat_tier(Ranker(cfg.scoring.to_config(), budget=budget), run, s5)
    profile = config.profile()
    spend = open_spend(current, max_cost=None, label="choose")
    try:
        if profile_status(conn, profile.name) != "ready":
            usd = 0.0
            if not is_fake_model(profile.embed_model):
                _, usd = EmbeddingService(conn).estimate(
                    profile_texts(conn, profile), profile.embed_model, profile.dimensions
                )
            print(f"{profile.name} is not built: building it costs about ${usd:.4f} (embeddings).")
            refusal = spend.refusal(usd)
            if refusal:
                _err(f"not built: {refusal}")
                return EXIT_NOT_RUN
            if not confirm("Build the profile?", yes=args.yes):
                return EXIT_NOT_RUN
            from hal9000.knowledge.config import load_knowledge_file
            from hal9000.knowledge.ingest.pipeline import IngestPipeline

            pipeline = IngestPipeline(
                current.files_dir,
                meter=spend.meter,
                agent=EVAL_AGENT,
                use_llm=False,
                bus=None,
                knowledge_file=load_knowledge_file(current.yaml_path),
            )
            try:
                pipeline.build(conn, profile)
            finally:
                pipeline.close()
        try:
            active = activate_profile(conn, profile.name)
        except ProfileError as error:
            _err(str(error))
            return EXIT_ERROR
    finally:
        spend.close()
    threshold = get_threshold(conn, active.id, config.reranker)
    calibrated = [
        r for r in runs
        if r.stage == "S4" and r.config.get("k") is None
        and identity(r.config) == identity(run.config)
        and r.config.get("abstain_threshold") is not None
    ]  # fmt: skip
    if calibrated:
        source = calibrated[0]
        threshold = float(source.config["abstain_threshold"])
        set_threshold(
            conn,
            active.id,
            config.reranker,
            threshold,
            auc=source.config.get("abstain_auc"),
            eval_run_id=source.id,
        )
    when = datetime.now(UTC).strftime("%Y-%m-%d")
    threshold_text = (
        f"{threshold:.4f} (in abstain_thresholds for this profile and reranker)"
        if threshold is not None
        else "none (the generation-side NOT_IN_SOURCES reply decides)"
    )
    comment = [
        f"chosen by `hal9000 kb eval choose {args.run} --config {config.name}` on {when}",
        f"chat tier {tier or 'unchanged'} (set KB_CHAT_TIER in .env); abstain threshold "
        f"{threshold_text}",
    ]
    if config.query_transform != "none":
        comment.append(f"query transform {config.query_transform} has no setting here: not used")
    write_choice_yaml(
        current.yaml_path,
        profile=profile.name,
        reranker=config.reranker,
        method=config.retrieval,
        alpha=config.alpha,
        budget=budget,
        comment=comment,
    )
    manifest = read_manifest(current.eval_dir, args.run)
    manifest["choice"] = {
        "config": config.name,
        "run_id": run.id,
        "profile": profile.name,
        "chat_tier": tier,
        "abstain_threshold": threshold,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_manifest(current.eval_dir, args.run, manifest)
    print(f"Chosen: {config.name}. {profile.name} is now the active profile.")
    print(f"Wrote {current.yaml_path}: profile, reranker {config.reranker}, retrieval "
          f"{config.retrieval}, context budget {budget} tokens.")  # fmt: skip
    print(f"Abstain threshold: {threshold_text}.")
    if tier and tier != current.chat_tier:
        print(f"Set KB_CHAT_TIER={tier} in .env (now {current.chat_tier}).")
    return EXIT_OK
