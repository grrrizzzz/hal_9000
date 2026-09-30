"""``hal9000 kb eval sweep|judge-prep|latency|choose|test|export|report|regression`` (§8, §17).

``sweep``, ``latency``, ``choose`` and ``test`` are the staged sweep (KWP5 part B):
:mod:`hal9000.knowledge.eval.sweep`, :mod:`~hal9000.knowledge.eval.latency` and
:mod:`~hal9000.knowledge.eval.decide`. Exit codes: 0 done, 1 error, 3 not run
(refused or not confirmed), 4 stopped at a cost cap (partial results kept).
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

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
            "eval test": _test,
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


def _sweep_leader(conn, current, sweep_id: str):
    """The top S4 finalist of ``sweep_id`` (what judge-prep and S5 answer with), or None."""
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.sweep import Ranker, finalist_runs, load_sweep_settings

    eval_config = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    grid = load_sweep_settings(eval_config)
    ranker = Ranker(eval_config.scoring.to_config(), budget=grid.decision_budget)
    finalists = finalist_runs(conn, sweep_id, ranker, grid.S5.top)
    return finalists[0] if finalists else None


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

    if not args.run.strip():
        _err(
            "no run ID given (is $SWEEP set in this terminal?). Pass the sweep ID or the "
            "leading run ID that `kb eval sweep` printed."
        )
        return EXIT_ERROR
    run = get_run(conn, args.run)
    if run is None and args.run.startswith("sweep-"):
        run = _sweep_leader(conn, current, args.run)
        if run is not None:
            print(f"Sweep {args.run}: the leading configuration is run {run.id}.")
    if run is None:
        _err(
            f"no evaluation run or finished S4 sweep {args.run!r} (judge-prep answers with "
            "the leading configuration of a sweep, or with the run given)"
        )
        return EXIT_ERROR
    eval_config = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    cfg = eval_config.judge
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
            answer = _run_answerer(conn, current, run, spend.meter)
            if answer is None:  # not a sweep run: KWP6's binding (the run's profile only)
                answer = answerer_for(conn, current, meter=spend.meter, run=run)
            generate = spend.generate()
        except NotWired as error:
            _err(str(error))
            return EXIT_ERROR
        except hal_config.ConfigError as error:
            _err(str(error))
            return EXIT_ERROR
        answer, generate, metered = _metering(spend.meter, answer, generate)
        try:
            result = judge_prep(
                conn,
                answer,
                generate,
                eval_run_id=run.id,
                config=cfg,
                judge_tier=current.judge_tier,
            )
        finally:
            from hal9000.knowledge.eval.sweep import record_metered_costs

            if metered["answers"] and metered["judgements"]:  # S5 estimates prefer these
                record_metered_costs(
                    current.eval_dir,
                    answer_tier=run.config.get("chat_tier") or current.chat_tier,
                    judge_tier=current.judge_tier,
                    **metered,
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


def _metering(meter, answer, generate):
    """Wrap judge-prep's answer and judge calls to measure the metered spend of each."""
    totals = {"answers": 0, "answer_usd": 0.0, "judgements": 0, "judge_usd": 0.0}

    def measured_answer(question):
        before = meter.spent
        output = answer(question)
        totals["answers"] += 1
        totals["answer_usd"] += meter.spent - before
        return output

    def measured_generate(prompt, tier="powerful"):
        before = meter.spent
        reply = generate(prompt, tier)
        totals["judgements"] += 1
        totals["judge_usd"] += meter.spent - before
        return reply

    return measured_answer, measured_generate, totals


def _run_answerer(conn, current, run, meter):
    """``question -> AnswerOutput`` with every setting of a sweep run (profile, method,
    reranker, transform, chat tier), or None for a run that isn't a sweep configuration."""
    from hal9000.knowledge.config import EVAL_AGENT
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.sweep import (
        SweepConfig,
        SweepDeps,
        config_answerer,
        load_sweep_settings,
    )

    if not (run.config.get("sweep_id") and run.config.get("retrieval")):
        return None
    config = SweepConfig.from_run_config(run.config)
    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    budget = load_sweep_settings(cfg).decision_budget if cfg.sweep else 2000
    answer = config_answerer(conn, current, config, meter=meter, deps=SweepDeps(), budget=budget)

    def answer_fn(question: str):
        output, result = answer(question)
        if result.outcome in ("retrieval_only", "error"):
            meter.precheck(EVAL_AGENT)  # a spending cap: stop (judge-prep keeps what it has)
        return output

    return answer_fn


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
            # retrieval only: never a query transform's LLM calls, whatever knowledge.yaml says
            binding = retriever_for(
                conn, current, meter=spend.meter, run=SimpleNamespace(config={"transform": "none"})
            )
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

    def names(value):
        return [p.strip() for p in (value or "").split(",") if p.strip()]

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
            sweep_id=args.sweep,
            profiles=names(args.profiles),
            configs=names(args.configs),
            n=args.n,
            end_to_end=args.end_to_end,
            ingest_sha=args.ingest_test,
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
        print(f"Stored {run.id} ({run.config.get('name')}, host {run.host})")
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
    from hal9000.knowledge.retrieve.config import reranker_key
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
    s4 = [
        r for r in runs
        if r.stage == "S4" and r.config.get("k") is None and r.finished_at is not None
        and identity(r.config) == identity(run.config)
    ]  # fmt: skip
    calibrated = [r for r in s4 if r.config.get("abstain_threshold") is not None]
    auc_min = load_sweep_settings(cfg).S4.abstain_auc_min if cfg.sweep else 0.75
    weak = [r for r in s4 if r.config.get("abstain_threshold") is None]
    moved = ""
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
    elif weak:  # plan §8.5: the score can't separate the groups; abstention moves to generation
        auc = weak[0].config.get("abstain_auc")
        with conn:
            conn.execute(
                "DELETE FROM abstain_thresholds WHERE profile_id = ? AND reranker = ?",
                (active.id, reranker_key(config.reranker)),
            )
        threshold = None
        moved = (
            f" -- the S4 calibration's AUC {auc:.3f} < {auc_min:.2f}"
            if isinstance(auc, int | float)
            else " -- S4 found no usable abstain score"
        ) + ", so abstention moves to the generation step (plan §8.5)"
    when = datetime.now(UTC).strftime("%Y-%m-%d")
    threshold_text = (
        f"{threshold:.4f} (in abstain_thresholds for this profile and reranker)"
        if threshold is not None
        else "none: the generation-side NOT_IN_SOURCES reply decides" + moved
    )
    comment = [
        f"chosen by `hal9000 kb eval choose {args.run} --config {config.name}` on {when}",
        f"abstain threshold {threshold_text}",
    ]
    write_choice_yaml(
        current.yaml_path,
        profile=profile.name,
        reranker=config.reranker,
        method=config.retrieval,
        alpha=config.alpha,
        budget=budget,
        comment=comment,
        transform=config.query_transform,
        chat_tier=tier,
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
          f"{config.retrieval}, transform {config.query_transform}, context budget {budget} "
          f"tokens, chat tier {tier or 'unchanged'}.")  # fmt: skip
    print(f"Abstain threshold: {threshold_text}.")
    import os

    override = os.environ.get("KB_CHAT_TIER", "").strip().lower()
    if tier and override and override != tier:
        print(f"Note: KB_CHAT_TIER={override} in the environment overrides answer.chat_tier.")
    print("Restart hal-kb (the service) so the chat picks up the new profile and settings.")
    if config.reranker != "none":
        print(f"The {config.reranker} reranker needs the Pi image built with WITH_RERANK=1.")
    if config.parser == "docling":
        print("A Docling profile needs the Pi image built with WITH_DOCLING=1.")
    print(f"Then: hal9000 kb eval test {args.run} --config {config.name} --yes (once).")
    return EXIT_OK


# --------------------------------------------------------------------- test
def _test(args, conn, current) -> int:
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.spend import confirm, open_spend
    from hal9000.knowledge.eval.sweep import SweepRunner

    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    spend = open_spend(current, max_cost=None, label="test")
    try:
        runner = SweepRunner(
            conn,
            current,
            eval_config=cfg,
            meter=spend.meter,
            max_cost=spend.meter.hard_cap,
            eval_cap=spend.eval_cap,
            confirm=lambda question: confirm(question, yes=args.yes),
        )
        outcome = runner.run_test(
            args.run, args.config, force=args.force, force_judge=args.force_judge
        )
    except (hal_config.ConfigError, ValueError) as error:
        _err(str(error))
        return EXIT_ERROR
    finally:
        spend.close()
    if outcome.status == "refused":
        _err(f"not run: {outcome.message}")
        return EXIT_NOT_RUN
    if outcome.status == "stopped":
        print(outcome.message)
        return EXIT_STOPPED
    print(f"Stored {outcome.message} (split test); spent ${outcome.spent:.4f} (knowledge-eval).")
    print(f"Report: hal9000 kb eval report {args.run}")
    return EXIT_OK
