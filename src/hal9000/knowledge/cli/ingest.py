"""``hal9000 kb ingest PATH... [--collection NAME] [--no-llm] [--yes]`` (plan §17).

Adds files and folders to the library through the same pipeline as the web
upload. Folders are walked recursively; a file's path relative to the folder's
parent (``Appliances/Furnace/manual.pdf``) is its relative path, and its
folder part is its collection unless ``--collection`` names one.

Paid work is announced first (rule K9): the new files are parsed to count
their text, and the embedding and segmentation cost is estimated and printed.
It needs ``--yes`` or a "y" at the prompt. Every call is metered with
``agent="knowledge"``, so ``KB_MONTHLY_CAP_USD`` stops it at the cap. Without an
OpenAI key, segmentation falls back to heuristics; embedding with a real model
needs the key (a ``fake-64`` profile embeds offline, for tests).

Prints one line per file (added, duplicate, skipped, failed) with its
documents, chunks and embedding cost, then the totals. Exits 1 if any file
failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["collect", "estimate_cost", "run"]

#: Rough segmentation prompt size per file (system + schema) and per page (excerpts).
_SEGMENT_BASE_TOKENS = 1500
_SEGMENT_PAGE_TOKENS = 200
_SEGMENT_OUTPUT_TOKENS = 800
#: Headers and overlap make the embedded text somewhat larger than the raw text.
_EMBED_OVERHEAD = 1.3


def collect(paths: Iterable[str]) -> tuple[list[tuple[Path, str]], list[str]]:
    """``(file, relative path)`` pairs for the arguments, and error lines for missing ones."""
    found: list[tuple[Path, str]] = []
    errors: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            root = path.resolve()
            for item in sorted(root.rglob("*")):
                relative = item.relative_to(root.parent)
                if item.is_file() and not any(p.startswith(".") for p in relative.parts):
                    found.append((item, relative.as_posix()))
        elif path.is_file():
            found.append((path, path.name))
        else:
            errors.append(f"no such file or folder: {raw}")
    return found, errors


def _has_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def estimate_cost(
    files: Iterable[Path], *, embed_model: str | None, segment: bool
) -> tuple[float, float, int]:
    """``(embedding usd, segmentation usd, text tokens)`` for new files, parsed to count text."""
    from hal9000 import config as hal_config
    from hal9000.cost.metering import usage_cost
    from hal9000.knowledge.index.embedding import is_fake_model
    from hal9000.knowledge.ingest.parsers import ParseError, parse_file
    from hal9000.knowledge.tokens import count_tokens

    tokens = 0
    segment_usd = 0.0
    for path in files:
        try:
            parsed = parse_file(path)
        except ParseError:
            continue  # it will fail in the pipeline too, at no cost
        tokens += sum(count_tokens(page.text) for page in parsed.pages)
        if segment:
            prompt = _SEGMENT_BASE_TOKENS + _SEGMENT_PAGE_TOKENS * parsed.page_count
            segment_usd += usage_cost(
                {
                    "model": hal_config.model_for("fast"),
                    "prompt": prompt,
                    "completion": _SEGMENT_OUTPUT_TOKENS,
                }
            )
    embed_usd = 0.0
    if embed_model and not is_fake_model(embed_model):
        embed_usd = usage_cost({"model": embed_model, "prompt": int(tokens * _EMBED_OVERHEAD)})
    return embed_usd, segment_usd, tokens


def _confirm(message: str, yes: bool) -> bool:
    print(message)
    if yes:
        return True
    if not sys.stdin.isatty():
        print("Not confirmed: pass --yes to proceed without a prompt.", file=sys.stderr)
        return False
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def _segment_generate() -> Callable:
    """``generate_response`` that also counts ``kb_*{purpose="segment"}`` metrics."""
    from hal9000.knowledge.obs.metrics import record_usage
    from hal9000.llm import generate_response

    def generate(prompt, tier="fast", *, on_usage=None):
        def sink(usage) -> None:
            if on_usage is not None:
                on_usage(usage)
            record_usage(usage, "segment")

        return generate_response(prompt, tier, on_usage=sink)

    return generate


def run(args: argparse.Namespace) -> int:
    from hal9000 import config as hal_config
    from hal9000.cost.metering import Meter
    from hal9000.knowledge.config import AGENT, load_knowledge_file, monthly_caps, settings
    from hal9000.knowledge.index.embedding import is_fake_model
    from hal9000.knowledge.index.library import get_file
    from hal9000.knowledge.index.profiles import active_profile, configured_profile
    from hal9000.knowledge.ingest.pipeline import IngestPipeline
    from hal9000.knowledge.storage import open_db
    from hal9000.storage.db import Database

    targets, errors = collect(args.paths)
    for line in errors:
        print(f"error      {line}", file=sys.stderr)
    if not targets:
        print("Nothing to ingest.", file=sys.stderr)
        return 1

    kb = settings()
    knowledge = load_knowledge_file(kb.yaml_path)
    conn = open_db(kb.db_path)
    ledger: Database | None = None
    try:
        profile = active_profile(conn) or configured_profile(knowledge.active_profile)
        embed_model = profile.embed_model if profile is not None else None
        paid_embedding = embed_model is not None and not is_fake_model(embed_model)
        if profile is None:
            print("No active index profile: files are stored and parsed, not yet searchable.")
        if paid_embedding and not _has_key():
            print(
                f"OPENAI_API_KEY is not set; embedding with {embed_model} needs it.",
                file=sys.stderr,
            )
            return 1
        use_llm = not args.no_llm
        if use_llm and not _has_key():
            print("No OpenAI key: segmenting with heuristics only (as with --no-llm).")
            use_llm = False

        import hashlib

        new_files = [
            path
            for path, _ in targets
            if get_file(conn, hashlib.sha256(path.read_bytes()).hexdigest()) is None
        ]
        meter = generate = None
        if (paid_embedding or use_llm) and new_files:
            embed_usd, segment_usd, tokens = estimate_cost(
                new_files, embed_model=embed_model if paid_embedding else None, segment=use_llm
            )
            caps = monthly_caps(kb)
            if not _confirm(
                f"{len(new_files)} new file(s), about {tokens:,} tokens of text. Estimated cost: "
                f"up to ${embed_usd + segment_usd:.4f} (embeddings ${embed_usd:.4f}, "
                f"segmentation ${segment_usd:.4f}); monthly cap ${caps[AGENT]:.2f}.",
                args.yes,
            ):
                return 1
            ledger = Database(hal_config.settings().data_dir / "hal9000.db")
            run_id = f"kb-ingest-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
            meter = Meter(
                ledger, run_id, soft_cap=float("inf"), hard_cap=float("inf"), monthly_caps=caps
            )
            if use_llm:
                generate = meter.wrap(_segment_generate(), agent=AGENT)

        pipeline = IngestPipeline(
            kb.files_dir,
            meter=meter,
            generate=generate,
            use_llm=use_llm,
            knowledge_file=knowledge,
            bus=None,
        )
        worker = pipeline.worker(kb.db_path)
        counts = {"added": 0, "duplicate": 0, "skipped": 0, "failed": 0}
        total_usd = 0.0
        for path, relative in targets:
            result = pipeline.submit(
                conn, path, relative, collection=args.collection, use_llm=use_llm
            )
            label = f"{result.relative_path}"
            if result.collection:
                label += f"  [{result.collection}]"
            if result.status != "added" or result.job_id is None:
                counts[result.status] += 1
                print(f"{result.status:<10} {label}: {result.reason or ''}".rstrip(": "))
                continue
            job = worker.run_job(conn, result.job_id)
            if job.status == "done":
                counts["added"] += 1
                outcome = pipeline.outcomes.get(result.job_id)
                cost = outcome.embed.cost_usd if outcome else 0.0
                total_usd += cost
                print(f"{'added':<10} {label}: {job.detail}; embedding ${cost:.4f}")
                for warning in outcome.warnings if outcome else []:
                    print(f"{'':<10}   note: {warning}")
            else:
                counts["failed"] += 1
                print(f"{'failed':<10} {label}: {job.error}")
        print(
            f"\n{counts['added']} added, {counts['duplicate']} duplicate, "
            f"{counts['skipped']} skipped, {counts['failed']} failed. "
            f"Embedding cost ${total_usd:.4f}."
            + (f" Total metered spend ${meter.spent:.4f}." if meter is not None else "")
        )
        return 1 if counts["failed"] else 0
    finally:
        conn.close()
        if ledger is not None:
            ledger.close()
