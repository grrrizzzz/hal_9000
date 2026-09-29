"""``kb eval export RUN``: aggregates and per-question scores, safe to share (plan §17).

The export is built from an allow-list of numeric and categorical fields, so
no question, evidence or chunk text can reach it, and it is then checked: no
run of more than :data:`LEAK_CHARS` characters taken from any page's cleaned
text may appear in it (:func:`find_leaks`). A leak refuses the export.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hal9000.knowledge.contracts import EvalRun
from hal9000.knowledge.eval.runs import load_results, run_dir, runs_for

__all__ = [
    "LEAK_CHARS",
    "LeakDetected",
    "export_payload",
    "export_runs",
    "find_leaks",
    "page_texts",
]

#: No string longer than this, taken from page text, may appear in an export.
LEAK_CHARS = 40

#: Per-question fields that are exported (all numbers, booleans or categories).
QUESTION_FIELDS = (
    "gold_item_id",
    "category",
    "answerable",
    "paraphrased",
    "recall",
    "evidence_recall",
    "full_support",
    "first_hit_rank",
    "reciprocal_rank",
    "context_precision",
    "context_tokens",
    "abstain_score",
    "abstained",
    "latency_ms",
    "correctness",
    "faithful",
    "citations_correct",
    "abstention_correct",
)


class LeakDetected(RuntimeError):
    """An export would contain document text."""


def page_texts(conn: sqlite3.Connection) -> list[str]:
    """Every cleaned page text in ``knowledge.db`` (what an export must not contain)."""
    return [row[0] for row in conn.execute("SELECT text FROM pages") if row[0]]


def _fold(text: str) -> str:
    """Case-folded with every run of whitespace as one space (leaks ignore both)."""
    return " ".join(text.casefold().split())


def find_leaks(text: str, pages: Iterable[str], n: int = LEAK_CHARS) -> list[str]:
    """Strings of more than ``n`` characters that ``text`` shares with any page.

    Case- and whitespace-insensitive: both sides are case-folded and their
    whitespace collapsed first. Every (n+1)-character window of ``text`` is
    looked up in the set of all (n+1)-character windows of the pages; the
    leaked windows are returned (folded), merged per run (empty when clean).
    """
    size = n + 1
    text = _fold(text)
    grams: set[str] = set()
    for page in map(_fold, pages):
        grams.update(page[i : i + size] for i in range(len(page) - size + 1))
    leaks: list[str] = []
    i = 0
    while i <= len(text) - size:
        if text[i : i + size] in grams:
            j = i
            while j + 1 <= len(text) - size and text[j + 1 : j + 1 + size] in grams:
                j += 1
            leaks.append(text[i : j + size])
            i = j + size
        else:
            i += 1
    return leaks


def denylist() -> list[tuple[int, str]]:
    """``(number, entry)`` for each ``KB_DENYLIST`` entry, case-folded.

    Entries are separated by ``|`` or newlines and numbered like
    ``scripts/kb_denylist_check.py`` numbers them; entries shorter than three
    characters are skipped (they would match almost anything).
    """
    raw = os.environ.get("KB_DENYLIST", "")
    entries = [p.strip() for p in raw.replace("|", "\n").splitlines() if p.strip()]
    return [(n, e.casefold()) for n, e in enumerate(entries, 1) if len(e) >= 3]


def check_shareable(texts: Iterable[str], pages: Iterable[str]) -> None:
    """Raise :class:`LeakDetected` if any text holds page text or a denylist entry.

    The message gives counts and entry numbers only, never the leaked string.
    """
    texts = list(texts)
    pages = list(pages)
    leaks = [leak for text in texts for leak in find_leaks(text, pages)]
    if leaks:
        raise LeakDetected(f"it would contain {len(leaks)} string(s) of document text; refusing")
    folded = [_fold(t) for t in texts]
    hits = [n for n, entry in denylist() if any(_fold(entry) in t for t in folded)]
    if hits:
        numbers = ", ".join(f"#{n}" for n in hits)
        raise LeakDetected(f"it contains KB_DENYLIST entry {numbers}; refusing")


def _strings(value: Any) -> Iterable[str]:
    """Every string (keys and values) inside a JSON-like value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def _question(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in QUESTION_FIELDS if k in record}


def _run(run: EvalRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "stage": run.stage,
        "split": run.split,
        "config": run.config,
        "metrics": run.metrics.model_dump(mode="json") if run.metrics else None,
        "host": run.host,
        "cost_usd": run.cost_usd,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def export_payload(runs: Sequence[EvalRun]) -> dict[str, Any]:
    return {
        "note": "Aggregates and per-question scores only; no document or question text.",
        "runs": [
            {**_run(run), "questions": [_question(r) for r in load_results(run)]} for run in runs
        ],
    }


def export_runs(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    eval_dir: Path,
    out_path: Path | None = None,
    pages: Iterable[str] | None = None,
) -> Path:
    """Write ``export.json`` for a run (or every run of a sweep) after the leak check.

    Raises KeyError for an unknown run and :class:`LeakDetected` when the
    export would contain page text.
    """
    runs = runs_for(conn, run_id)
    if not runs:
        raise KeyError(run_id)
    payload = export_payload(runs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    pages = list(page_texts(conn) if pages is None else pages)
    # the serialized file, and every decoded string (JSON escapes newlines and quotes)
    check_shareable([text, "\x00".join(_strings(payload))], pages)
    target = out_path or run_dir(eval_dir, run_id) / "export.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n")
    return target
