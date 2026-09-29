"""``hal9000 kb gold generate|export|import`` (plan §8.1, §17)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hal9000 import config as hal_config

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_RUN = 3


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


def _generate(args, conn, current) -> int:
    from hal9000.knowledge.eval.config import load_eval_config
    from hal9000.knowledge.eval.gold import generate_gold, load_corpus, plan_generation
    from hal9000.knowledge.eval.spend import confirm, open_spend
    from hal9000.knowledge.eval.store import insert_gold, list_gold
    from hal9000.knowledge.eval.wiring import NotWired, retriever_for

    cfg = load_eval_config(current.config_dir / "knowledge_eval.yaml")
    corpus = load_corpus(conn)
    if not corpus:
        _err("no documents in knowledge.db: run `hal9000 kb ingest` first")
        return EXIT_ERROR
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
        plan = plan_generation(corpus, cfg.gold)
        estimate = plan.estimate()
        pages = sum(len(d.pages) for d in corpus)
        print(
            f"Gold generation over {len(corpus)} documents ({pages} pages): {len(plan.tasks)} "
            f"generation tasks, {len(plan.link_sets)} linked document sets, "
            f"{plan.n_unanswerable} unanswerable drafts to check."
        )
        print(f"Estimate: {estimate.describe()} (cap --max-cost ${args.max_cost:.2f}).")
        if keyword_note:
            print(keyword_note)
        refusal = spend.refusal(estimate.usd)
        if refusal:
            _err(f"not started: {refusal}")
            return EXIT_NOT_RUN
        if not confirm("Generate the draft question set?", yes=args.yes):
            return EXIT_NOT_RUN
        try:
            generate = spend.generate()
        except hal_config.ConfigError as error:
            _err(str(error))
            return EXIT_ERROR
        result = generate_gold(plan, generate, retrieve=retrieve, existing=list_gold(conn))
        stored = insert_gold(conn, result.items)
        print(result.summary())
        print(
            f"Stored {len(stored)} draft items for review. Spent ${spend.meter.spent:.4f} "
            f"(agent knowledge-eval)."
        )
        return EXIT_OK
    finally:
        spend.close()
