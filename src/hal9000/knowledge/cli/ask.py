"""``hal9000 kb ask "question" [--trace]``: one question from the terminal (plan §17).

The same pipeline as the chat (metered as ``knowledge``, traced, subject to
the monthly cap): the answer streams to stdout, then the sources, the verify
result, the outcome and the cost. ``--trace`` adds the stage timings and the
retrieved chunks with their scores. A question costs well under a cent on
the fast tier (plan §13), so there is no confirmation prompt.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, TextIO

__all__ = ["print_trace", "run"]

EXIT_OK = 0
EXIT_ERROR = 1


def _date(value: Any) -> str:
    return value.isoformat() if value else "undated"


def print_trace(trace, out: TextIO) -> None:
    print("\nStages (ms):", file=out)
    for timing in trace.stages:
        indent = "  " * (timing.depth + 1)
        error = f"  ! {timing.error}" if timing.error else ""
        print(
            f"{indent}{timing.stage.value:<12} {timing.duration_ms:>9.1f}  "
            f"(at {timing.started_ms:.1f}){error}",
            file=out,
        )
    total, first = trace.total_ms or 0.0, trace.ttft_ms or 0.0
    print(f"  total {total:.1f} ms, first token {first:.1f} ms", file=out)
    print("\nRetrieved (rank, chunk, fused, dense, bm25, in context):", file=out)
    for hit in sorted(trace.hits, key=lambda h: h.rank)[:30]:
        dense = f"{hit.dense_score:.3f}" if hit.dense_score is not None else "-"
        bm25 = f"{hit.bm25_score:.2f}" if hit.bm25_score is not None else "-"
        mark = "*" if hit.selected else ""
        print(
            f"  {hit.rank:>3} {hit.chunk_id:>7} {hit.score:.4f} {dense:>6} {bm25:>7} {mark}",
            file=out,
        )
    print(f"  context: {trace.context_tokens} tokens", file=out)


def run(args: argparse.Namespace, *, pipeline=None, conn=None, out: TextIO | None = None) -> int:
    from hal9000 import config as hal_config
    from hal9000.knowledge.answer.pipeline import AnswerDeps, AnswerPipeline, NoActiveProfile
    from hal9000.knowledge.config import settings as kb_settings
    from hal9000.knowledge.obs.traces import load_trace
    from hal9000.knowledge.storage import open_db

    out = out or sys.stdout
    try:
        if pipeline is None:
            from hal9000.knowledge.api.services import _meter

            settings = kb_settings()
            pipeline = AnswerPipeline(settings, AnswerDeps(meter=_meter(settings)))
        own = conn is None
        conn = conn or open_db(pipeline.settings.db_path)
    except hal_config.ConfigError as error:
        print(f"hal9000 kb ask: {error}", file=sys.stderr)
        return EXIT_ERROR

    state: dict[str, Any] = {"banner": None}

    def emit(event) -> None:
        if event.event == "token":
            out.write(event.text)
            out.flush()
        elif event.event == "reset":
            out.write(f"\n\n[regenerating: {event.reason}]\n\n")
        elif event.event == "sources" and event.banner:
            state["banner"] = event.banner

    try:
        answer = pipeline.ask(conn, args.question, emit=emit)
    except NoActiveProfile as error:
        print(f"hal9000 kb ask: {error}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if own:
            conn.close()
    if state["banner"]:
        print(f"\n\n({state['banner']})", file=out)
    print("\n\nSources:", file=out)
    for c in answer.citations:
        vendor = f", {c.vendor}" if c.vendor else ""
        print(f"  [{c.marker}] {c.title} ({c.doc_type}{vendor}, {_date(c.doc_date)}), "
              f"page {c.page}", file=out)  # fmt: skip
    if not answer.citations:
        print("  (none)", file=out)
    badge = {"verified": "figures verified", "failed": "FIGURES NOT VERIFIED", "skipped": "-"}
    print(f"\nOutcome: {answer.outcome}; grounding: {badge[answer.grounding]}", file=out)
    for issue in answer.issues:
        print(f"  - {issue}", file=out)
    print(f"Cost: ${answer.cost_usd:.4f}   Trace: {answer.trace_id}", file=out)
    if getattr(args, "trace", False):
        reopened = open_db(pipeline.settings.db_path)
        try:
            trace = load_trace(reopened, answer.trace_id)
        finally:
            reopened.close()
        if trace is not None:
            print_trace(trace, out)
    return EXIT_OK if answer.outcome != "error" else EXIT_ERROR
