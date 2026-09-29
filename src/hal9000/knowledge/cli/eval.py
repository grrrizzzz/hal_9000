"""``hal9000 kb eval judge-prep|export|report|regression`` (plan §8, §17).

``eval sweep`` and ``eval latency`` are the staged sweep runner (KWP5 part B).
"""

from __future__ import annotations

import argparse
import sys

from hal9000 import config as hal_config
from hal9000.knowledge.cli import not_implemented

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_RUN = 3


def _err(message: str) -> None:
    print(f"hal9000 kb: {message}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    command = args.kb_command
    if command in ("eval sweep", "eval latency"):
        return not_implemented(args, "KWP5 part B")
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


def _report(args, conn, current) -> int:
    from hal9000.knowledge.eval.export import LeakDetected
    from hal9000.knowledge.eval.report import write_report

    target = current.config_dir.parent / "docs" / "KNOWLEDGE_EVAL_REPORT.md"
    try:
        path = write_report(conn, args.run, target)
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
