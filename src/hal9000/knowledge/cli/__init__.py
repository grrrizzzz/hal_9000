"""``hal9000 kb``: the knowledge-base command group (plan §17).

This module only *registers* commands. Each command's implementation lives in
the module its work package owns (rule K4), imported lazily by :func:`dispatch`
so ``hal9000 run`` never pays for knowledge imports:

======================  ==================================  ======
module                  commands                            owner
======================  ==================================  ======
``cli/check.py``        ``check``                           KWP3
``cli/ingest.py``       ``ingest``                          KWP3
``cli/gold.py``         ``gold generate|export|import``     KWP5
``cli/eval.py``         ``eval sweep|judge-prep|latency|    KWP5
                        export|report|regression``
``cli/serve.py``        ``serve``                           KWP6
``cli/ask.py``          ``ask``                             KWP6
``cli/ops.py``          ``profile list|activate``,          KWP10
                        ``backup``, ``restore``, ``stats``
======================  ==================================  ======

Every module exposes ``run(args) -> int``; ``args.kb_command`` names the
command (``"eval sweep"``). Paid commands take ``--yes`` (rule K9).
"""

from __future__ import annotations

import argparse
import importlib
import sys

__all__ = ["EXIT_NOT_IMPLEMENTED", "dispatch", "not_implemented", "register"]

EXIT_NOT_IMPLEMENTED = 2


def not_implemented(args: argparse.Namespace, kwp: str) -> int:
    """What a command prints until its work package lands."""
    print(f"hal9000 kb {args.kb_command}: not implemented yet ({kwp})", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def _command(
    sub: argparse._SubParsersAction, name: str, module: str, help_text: str, full_name: str
) -> argparse.ArgumentParser:
    parser = sub.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(kb_module=f"hal9000.knowledge.cli.{module}", kb_command=full_name)
    return parser


def _yes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--yes", action="store_true", help="skip the cost confirmation (the estimate still prints)"
    )


def register(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the ``kb`` group to the top-level CLI's subparsers."""
    kb = sub.add_parser(
        "kb",
        help="house knowledge base",
        description="House knowledge base: documents, questions, evaluation.",
    )
    kb.set_defaults(kb_parser=kb)
    cmds = kb.add_subparsers(dest="kb_group", metavar="COMMAND")

    check_help = "OpenAI embed + chat reachable, sqlite-vec loads, DB migrated"
    _command(cmds, "check", "check", check_help, "check")

    ingest = _command(cmds, "ingest", "ingest", "add files or folders to the library", "ingest")
    ingest.add_argument("paths", nargs="+", metavar="PATH")
    ingest.add_argument("--collection", metavar="NAME", help="collection tag for these files")
    ingest.add_argument("--no-llm", action="store_true", help="segment with heuristics only")
    _yes(ingest)

    ask = _command(cmds, "ask", "ask", "ask one question from the terminal, with sources", "ask")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true", help="print the stage timings and hits")

    serve = _command(cmds, "serve", "serve", "run the knowledge-base web service", "serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=None, help="default: KB_PORT (8910)")
    serve.add_argument(
        "--fake", action="store_true", help="fake LLM and embedder over the fixture corpus"
    )

    # -- gold ------------------------------------------------------------------
    gold = cmds.add_parser("gold", help="the evaluation question set")
    gold.set_defaults(kb_parser=gold)
    gold_cmds = gold.add_subparsers(dest="kb_gold", metavar="COMMAND")
    generate = _command(
        gold_cmds, "generate", "gold", "generate the draft question set", "gold generate"
    )
    generate.add_argument("--max-cost", type=float, default=2.00, metavar="USD")
    generate.add_argument(
        "--allow-keyword-check",
        action="store_true",
        help="check unanswerable drafts with keyword search when no index is available",
    )
    _yes(generate)
    for name, help_text in (
        ("export", "write the question set to a JSONL file (keep it under data/)"),
        ("import", "read a question set from a JSONL file"),
    ):
        cmd = _command(gold_cmds, name, "gold", help_text, f"gold {name}")
        cmd.add_argument("file", metavar="FILE")

    # -- eval ------------------------------------------------------------------
    ev = cmds.add_parser("eval", help="the setup evaluation")
    ev.set_defaults(kb_parser=ev)
    ev_cmds = ev.add_subparsers(dest="kb_eval", metavar="COMMAND")
    sweep = _command(ev_cmds, "sweep", "eval", "run sweep stages (plan §8.4)", "eval sweep")
    sweep.add_argument("--stages", required=True, help="e.g. S1a,S1b,S2,S3,S4 or S5")
    sweep.add_argument("--max-cost", type=float, default=10.0, metavar="USD")
    sweep.add_argument("--resume", metavar="RUN")
    sweep.add_argument(
        "--force-judge", action="store_true", help="run S5 without judge agreement >= 85%%"
    )
    _yes(sweep)
    judge = _command(
        ev_cmds,
        "judge-prep",
        "eval",
        "answers for 20 dev items, for judge calibration",
        "eval judge-prep",
    )
    judge.add_argument("run", metavar="RUN")
    _yes(judge)
    latency = _command(ev_cmds, "latency", "eval", "S6 latency on the Pi", "eval latency")
    latency.add_argument("--profiles", required=True, help="comma-separated profile names")
    latency.add_argument("--n", type=int, default=200, help="replayed retrieval questions")
    latency.add_argument("--end-to-end", type=int, default=30, help="live end-to-end questions")
    latency.add_argument("--ingest-test", metavar="SHA", help="timed scratch ingest of this file")
    _yes(latency)
    for name, help_text in (
        ("export", "aggregates and per-question scores, no document text"),
        ("report", "write docs/KNOWLEDGE_EVAL_REPORT.md (aggregates only)"),
    ):
        cmd = _command(ev_cmds, name, "eval", help_text, f"eval {name}")
        cmd.add_argument("run", metavar="RUN")
    regression = _command(
        ev_cmds, "regression", "eval", "approved set, retrieval only", "eval regression"
    )
    _yes(regression)

    # -- ops -------------------------------------------------------------------
    profile = cmds.add_parser("profile", help="index profiles")
    profile.set_defaults(kb_parser=profile)
    profile_cmds = profile.add_subparsers(dest="kb_profile", metavar="COMMAND")
    _command(profile_cmds, "list", "ops", "list index profiles", "profile list")
    activate = _command(
        profile_cmds, "activate", "ops", "make a profile serve the chat", "profile activate"
    )
    activate.add_argument("name", metavar="NAME")

    backup = _command(cmds, "backup", "ops", "online backup of knowledge.db and files", "backup")
    backup.add_argument("--to", metavar="DIR", help="parent folder (default: data/backups)")
    restore = _command(
        cmds, "restore", "ops", "restore a backup into an empty data directory", "restore"
    )
    restore.add_argument("source", metavar="FROM", help="a folder written by `kb backup`")
    restore.add_argument(
        "--into", metavar="DIR", help="the data directory to restore into (default: KB_DATA_DIR)"
    )
    restore.add_argument(
        "--force",
        action="store_true",
        help="replace an existing library (it is moved aside, not deleted)",
    )
    _command(cmds, "stats", "ops", "library and index statistics", "stats")
    return kb


def dispatch(args: argparse.Namespace) -> int:
    """Run the selected ``kb`` command; print the group's help when none was given."""
    module = getattr(args, "kb_module", None)
    if module is None:
        args.kb_parser.print_help(sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    return importlib.import_module(module).run(args)
