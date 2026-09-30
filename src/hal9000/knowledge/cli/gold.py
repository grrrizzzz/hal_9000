"""``hal9000 kb gold generate|recheck|export|import`` (plan §8.1, §17)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hal9000 import config as hal_config

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_RUN = 3
#: stopped by the --max-cost cap or KB_EVAL_CAP_USD (what was done is stored)
EXIT_STOPPED = 4


def _err(message: str) -> None:
    print(f"hal9000 kb: {message}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    from hal9000.knowledge.config import settings
    from hal9000.knowledge.storage import open_db

    current = settings()
    conn = open_db(current.db_path)
    try:
        if args.kb_command == "gold generate":
            return _generate(args, conn, current)
        if args.kb_command == "gold recheck":
            return _recheck(args, conn, current)
        return _exchange(args, conn, current)
    finally:
        conn.close()


def _exchange(args, conn, current) -> int:
    from hal9000.knowledge.eval.store import UnsafePath, check_data_path, export_jsonl, import_jsonl

    try:
        path = check_data_path(Path(args.file), current.data_dir)
    except UnsafePath as error:
        _err(str(error))
        return EXIT_ERROR
    if args.kb_command == "gold export":
        n = export_jsonl(conn, path)
        print(f"Wrote {n} question-set items to {path}")
        return EXIT_OK
    if not path.exists():
        _err(f"{path} does not exist")
        return EXIT_ERROR
    try:
        inserted, updated = import_jsonl(conn, path)
    except ValueError as error:
        _err(f"nothing imported: {error}")
        return EXIT_ERROR
    print(f"Imported {path}: {inserted} new, {updated} updated")
    return EXIT_OK


LATEST = "latest"
EXIT_INTERRUPTED = 130


def _parse_only(raw: str | None) -> list[str]:
    from hal9000.knowledge.eval.config import CATEGORIES

    if not raw:
        return []
    names = [n.strip() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in CATEGORIES]
    if unknown or not names:
        raise ValueError(f"--only takes categories from {', '.join(CATEGORIES)} (got {raw!r})")
    return list(dict.fromkeys(names))


def _shortfall(cfg, existing, only: list[str], count: int | None) -> tuple[dict, list[str]]:
    """Targets for a top-up of ``only`` and the lines describing them."""
    from collections import Counter

    have = Counter(i.category for i in existing if i.status != "rejected")
    targets: dict[str, int] = {}
    lines = []
    for category in only:
        quota = cfg.quotas.get(category, 0)
        need = count if count is not None else max(0, quota - have[category])
        targets[category] = need
        basis = f"--count {count}" if count is not None else f"quota {quota}"
        lines.append(f"  {category}: {need} ({basis}; {have[category]} stored and not rejected)")
    return targets, lines


def _continue_hint(plan, checkpoint, basis) -> str:
    from hal9000.knowledge.eval.spend import suggest_max_cost

    work = plan.pending(checkpoint)
    rest = plan.estimate(basis, checkpoint=checkpoint)
    parts = []
    if work.tasks:
        parts.append(f"{len(work.tasks)} generation tasks")
    if work.checks:
        parts.append(f"{work.checks} unanswerable checks")
    return (
        f"{' and '.join(parts)} remain (estimated ${rest.usd:.2f}, about "
        f"{_duration(rest.seconds)}). Continue with:\n"
        f"  uv run hal9000 kb gold generate --resume {checkpoint.id} "
        f"--max-cost {suggest_max_cost(rest.usd):.2f}"
    )


def _duration(seconds: float) -> str:
    from hal9000.knowledge.eval.progress import format_duration

    return format_duration(seconds)


def _generate(args, conn, current) -> int:
    from hal9000.knowledge.eval.checkpoint import Checkpoint, CheckpointError
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.gold import load_corpus, plan_from_descriptors, plan_generation
    from hal9000.knowledge.eval.progress import ledger_basis
    from hal9000.knowledge.eval.spend import open_spend
    from hal9000.knowledge.eval.store import UnsafePath, check_data_path, list_gold
    from hal9000.knowledge.eval.wiring import NotWired, retriever_for

    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    try:
        only = _parse_only(args.only)
    except ValueError as error:
        _err(str(error))
        return EXIT_ERROR
    if args.count is not None and (not only or args.count < 0):
        _err("--count needs --only CATEGORY and a number >= 0")
        return EXIT_ERROR
    if args.resume and only:
        _err("--resume continues a checkpoint as it was planned: leave out --only and --count")
        return EXIT_ERROR
    root = current.eval_dir / "gold"
    try:  # rule K2: checkpoints hold questions, so never where git could pick them up
        check_data_path(root / "checkpoint.jsonl", current.data_dir)
    except UnsafePath as error:
        _err(str(error))
        return EXIT_ERROR
    corpus = load_corpus(conn)
    if not corpus:
        _err("no documents in knowledge.db: run `hal9000 kb ingest` first")
        return EXIT_ERROR
    existing = list_gold(conn)
    checkpoint = None
    if args.resume:
        try:
            checkpoint = Checkpoint.load(root, None if args.resume == LATEST else args.resume)
            plan = plan_from_descriptors(checkpoint.tasks, corpus, cfg.gold, checkpoint.targets)
        except (CheckpointError, ValueError) as error:
            _err(f"cannot resume: {error}")
            return EXIT_ERROR
    elif only:
        targets, lines = _shortfall(cfg.gold, existing, only, args.count)
        print("Top-up: the shortfall to fill")
        print("\n".join(lines))
        if not any(targets.values()):
            print("Nothing to top up: those categories are at their quotas.")
            return EXIT_OK
        plan = plan_generation(corpus, cfg.gold, targets=targets, avoid=existing)
    else:
        plan = plan_generation(corpus, cfg.gold)
    spend = open_spend(current, max_cost=args.max_cost, label="gold")
    try:
        try:
            retrieve = retriever_for(conn, current, meter=spend.meter).retrieve
            keyword_note = None
        except NotWired as error:
            if not args.allow_keyword_check:
                _err(
                    f"{error}. Unanswerable drafts must be checked against the real index: "
                    "build one first, or pass --allow-keyword-check to check them with "
                    "keyword search only (they are then marked for review)."
                )
                return EXIT_ERROR
            retrieve = None
            keyword_note = f"Note: {error}; unanswerable drafts are checked by keyword search only."
        model = hal_config.model_for(cfg.gold.tier)
        basis = ledger_basis(spend.meter.db, model, cfg.gold.tier, cfg.gold.estimate)
        estimate = plan.estimate(basis, checkpoint=checkpoint)
        pages = sum(len(d.pages) for d in corpus)
        if checkpoint is not None:
            work = plan.pending(checkpoint)
            print(
                f"Resuming {checkpoint.id}: {checkpoint.n_items()} items collected so far; "
                f"{len(plan.tasks) - len(work.tasks)} of {len(plan.tasks)} generation tasks "
                f"done, {work.checks} unanswerable checks left."
            )
        elif only:
            print(
                f"Top-up plan: {len(plan.tasks)} generation tasks, "
                f"{plan.n_unanswerable} unanswerable drafts to check."
            )
        else:
            print(
                f"Gold generation over {len(corpus)} documents ({pages} pages): "
                f"{len(plan.tasks)} generation tasks, {len(plan.link_sets)} linked document "
                f"sets, {plan.n_unanswerable} unanswerable drafts to check."
            )
        print(f"Estimate: {estimate.describe()} (cap --max-cost ${args.max_cost:.2f}).")
        print(f"Basis: {basis.describe()}.")
        if keyword_note:
            print(keyword_note)
        refusal = spend.refusal(estimate.usd)
        if refusal:
            _err(f"not started: {refusal}")
            return EXIT_NOT_RUN
        if estimate.calls and not _confirm("Generate the draft question set?", args.yes):
            return EXIT_NOT_RUN
        try:
            generate = spend.generate()
        except hal_config.ConfigError as error:
            _err(str(error))
            return EXIT_ERROR
        if checkpoint is None:
            checkpoint = Checkpoint.create(
                root,
                tasks=plan.descriptors(),
                targets=plan.targets,
                only=only,
                keyword_only=retrieve is None,
            )
        try:
            checkpoint.acquire()
        except CheckpointError as error:
            _err(f"cannot resume: {error}")
            return EXIT_ERROR
        try:
            checkpoint.reopen()
            print(
                f"Checkpoint {checkpoint.id}. Progress follows on stderr, one line per LLM "
                "call; Ctrl-C stops (after the call in flight), stores what is done and prints "
                "how to continue.",
                flush=True,
            )
            return _run(plan, checkpoint, generate, retrieve, existing, spend, basis, conn)
        finally:
            checkpoint.release()
    finally:
        spend.close()


def _confirm(question: str, yes: bool) -> bool:
    """``confirm``, with Ctrl-C at the prompt a plain "no" (nothing has started)."""
    from hal9000.knowledge.eval.spend import confirm

    try:
        return confirm(question, yes=yes)
    except (KeyboardInterrupt, EOFError):
        print("\nNot started.")
        return False


def _stop(error: KeyboardInterrupt) -> tuple[str, int]:
    """How an interrupt is described, and the exit code it gives."""
    if getattr(error, "signum", None):
        return error.describe(), error.exit_code  # type: ignore[attr-defined]
    return "interrupted (Ctrl-C)", EXIT_INTERRUPTED


def _run(plan, checkpoint, generate, retrieve, existing, spend, basis, conn) -> int:
    from hal9000.knowledge.eval.checkpoint import StopSignals
    from hal9000.knowledge.eval.gold import GoldRun, question_key
    from hal9000.knowledge.eval.progress import Progress
    from hal9000.knowledge.eval.store import insert_gold, list_gold

    exit_code = EXIT_OK
    with StopSignals() as signals:
        spend.guard_records(signals)
        run = GoldRun(
            plan,
            generate,
            retrieve=retrieve,
            existing=existing,
            checkpoint=checkpoint,
            progress=Progress(spent=lambda: spend.meter.spent),
            guard=signals,
        )
        try:
            with signals.running():
                run.collect()
        except KeyboardInterrupt as error:
            run.interrupted, exit_code = _stop(error)
            print(f"\n{run.interrupted}: storing what is done...", file=sys.stderr, flush=True)
        # outside running(): a signal now only prints a notice, the save always finishes
        result = run.finish(list_gold(conn))
        stored = insert_gold(conn, result.items)
        checkpoint.record_stored([(question_key(i.question), i.id) for i in stored])
        unfinished = bool(plan.pending(checkpoint))
        checkpoint.close_session(
            status="stopped" if unfinished else "complete",
            spent=spend.meter.spent,
            calls=run.runner.calls,
        )
    if exit_code == EXIT_OK and run.runner.stopped:
        exit_code = EXIT_STOPPED
    print(result.summary())
    print(
        f"Stored {len(stored)} draft items for review. Spent ${spend.meter.spent:.4f} "
        f"(agent knowledge-eval); ${checkpoint.manifest['spent_usd']:.2f} over checkpoint "
        f"{checkpoint.id}."
    )
    if unfinished:
        print(f"Saved checkpoint {checkpoint.id}: " + _continue_hint(plan, checkpoint, basis))
    elif result.shortfall:
        categories = ",".join(result.shortfall)
        print(
            "Every planned call is done. To fill the shortfall later:\n"
            f"  uv run hal9000 kb gold generate --only {categories}"
        )
    return exit_code


# ------------------------------------------------------------------- recheck
def _recheck(args, conn, current) -> int:
    from hal9000.knowledge.eval.checkpoint import StopSignals
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.gold import (
        estimate_checks,
        load_corpus,
        needs_recheck,
        recheck_unanswerable,
    )
    from hal9000.knowledge.eval.progress import Progress, ledger_basis
    from hal9000.knowledge.eval.spend import open_spend
    from hal9000.knowledge.eval.store import set_check_note, unreviewed_gold
    from hal9000.knowledge.eval.wiring import NotWired, retriever_for

    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    items = [i for i in unreviewed_gold(conn, category="unanswerable") if needs_recheck(i)]
    if not items:
        print("Nothing to recheck: no unreviewed unanswerable draft says it could not be checked.")
        return EXIT_OK
    corpus = load_corpus(conn)
    if not corpus:
        _err("no documents in knowledge.db: run `hal9000 kb ingest` first")
        return EXIT_ERROR
    spend = open_spend(current, max_cost=args.max_cost, label="gold-recheck")
    try:
        try:
            retrieve = retriever_for(conn, current, meter=spend.meter).retrieve
        except NotWired as error:
            _err(f"{error}. The recheck needs the real index: build one first.")
            return EXIT_ERROR
        model = hal_config.model_for(cfg.gold.tier)
        basis = ledger_basis(spend.meter.db, model, cfg.gold.tier, cfg.gold.estimate)
        estimate = estimate_checks(len(items), corpus, cfg.gold, basis)
        print(
            f"{len(items)} unanswerable drafts were never checked against the library (their "
            "note says they could not be checked); checking each against the whole library now. "
            "Reviewed, edited and rejected items are left alone."
        )
        print(f"Estimate: {estimate.describe()} (cap --max-cost ${args.max_cost:.2f}).")
        print(f"Basis: {basis.describe()}.")
        refusal = spend.refusal(estimate.usd)
        if refusal:
            _err(f"not started: {refusal}")
            return EXIT_NOT_RUN
        if not _confirm("Recheck them?", args.yes):
            return EXIT_NOT_RUN
        try:
            generate = spend.generate()
        except hal_config.ConfigError as error:
            _err(str(error))
            return EXIT_ERROR
        print(
            "Progress follows on stderr, one line per check; each result is saved at once, so "
            "Ctrl-C loses nothing and a rerun skips what is done.",
            flush=True,
        )
        exit_code = EXIT_OK
        interrupted = None
        with StopSignals() as signals:
            spend.guard_records(signals)
            try:
                with signals.running():
                    outcome = recheck_unanswerable(
                        items,
                        generate,
                        retrieve,
                        cfg.gold,
                        lambda item, note: set_check_note(conn, item.id, item.reviewer_note, note),
                        progress=Progress(spent=lambda: spend.meter.spent),
                        guard=signals,
                    )
            except KeyboardInterrupt as error:
                interrupted, exit_code = _stop(error)
                outcome = None
        return _recheck_report(outcome, items, conn, spend, interrupted, exit_code)
    finally:
        spend.close()


def _recheck_report(outcome, items, conn, spend, interrupted, exit_code) -> int:
    from hal9000.knowledge.eval.gold import CHECKED_CLEAN, is_flagged, needs_recheck
    from hal9000.knowledge.eval.store import get_gold, list_gold

    # counted from what is stored, so an interrupted run reports exactly what it saved
    now = [get_gold(conn, i.id) for i in items]
    clean = sum(1 for i in now if i and i.reviewer_note == CHECKED_CLEAN)
    left = sum(1 for i in now if i and needs_recheck(i))
    answered = sum(1 for i in now if i and is_flagged(i) and not needs_recheck(i))
    changed = sum(1 for i in now if i is None or i.status != "draft")
    print(
        f"Rechecked {len(items) - left - changed} of {len(items)} drafts: {clean} clean "
        f"(note: {CHECKED_CLEAN!r}), {answered} flagged as answered by the documents. "
        f"Spent ${spend.meter.spent:.4f} (agent knowledge-eval)."
    )
    stopped = interrupted or (outcome.stopped if outcome else None)
    if stopped:
        print(f"STOPPED EARLY: {stopped} (every finished check is saved)")
        if exit_code == EXIT_OK:
            exit_code = EXIT_STOPPED
    if outcome is not None:
        for error in outcome.errors[:5]:
            print(f"error: {error}")
    if left:
        print(f"{left} still unchecked: run `uv run hal9000 kb gold recheck` again to continue.")
    waiting = [
        i
        for i in list_gold(conn, status="draft", category="unanswerable")
        if is_flagged(i) and not needs_recheck(i)
    ]
    if waiting:
        print(
            f"Next: {len(waiting)} unanswerable drafts are flagged as answered by the documents. "
            "Review, rejecting every one the documents do answer; then "
            "`uv run hal9000 kb gold generate --only unanswerable` refills exactly what was "
            "rejected."
        )
    return exit_code
