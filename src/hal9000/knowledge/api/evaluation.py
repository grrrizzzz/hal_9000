"""Evaluation, profile and trace queries behind the API (plan §8, §9.4, §11).

The question set, judge labels and runs are KWP5's tables and functions
(:mod:`hal9000.knowledge.eval`); this module adds what only the API needs:
applying a review edit to a gold item, one judge label by ID, the profile
state and the Traces list.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from hal9000.knowledge.contracts import (
    GoldItem,
    GoldPatch,
    JudgeLabel,
    ProfileState,
    TraceSummary,
)

__all__ = ["judge_label", "patch_gold", "profile_state", "trace_summaries"]


def patch_gold(conn: sqlite3.Connection, item_id: int, patch: GoldPatch) -> GoldItem | None:
    """Apply a review edit; the result must still be a valid gold item (``ValueError`` if not)."""
    from hal9000.knowledge.eval.store import get_gold

    item = get_gold(conn, item_id)
    if item is None:
        return None
    changes = patch.model_dump(exclude_unset=True)
    for key in ("question", "category", "evidence", "status"):
        if key in changes and changes[key] is None:
            changes.pop(key)
    updated = GoldItem.model_validate({**item.model_dump(), **changes})
    with conn:
        conn.execute(
            "UPDATE gold_items SET question = ?, reference_answer = ?, category = ?, "
            "evidence = ?, status = ?, reviewer_note = ?, updated_at = ? WHERE id = ?",
            (
                updated.question,
                updated.reference_answer,
                updated.category,
                json.dumps([e.model_dump(mode="json") for e in updated.evidence]),
                updated.status,
                updated.reviewer_note,
                datetime.now(UTC).isoformat(timespec="seconds"),
                item_id,
            ),
        )
    return get_gold(conn, item_id)


def judge_label(conn: sqlite3.Connection, label_id: int) -> JudgeLabel | None:
    from hal9000.knowledge.eval.store import list_judge_labels

    run = conn.execute("SELECT eval_run_id FROM judge_labels WHERE id = ?", (label_id,)).fetchone()
    if run is None:
        return None
    return next(
        (label for label in list_judge_labels(conn, eval_run_id=run[0]) if label.id == label_id),
        None,
    )


def profile_state(conn: sqlite3.Connection) -> ProfileState:
    from hal9000.knowledge.index.profiles import active_profile, list_profiles
    from hal9000.knowledge.ingest.jobs import get_job

    row = conn.execute(
        "SELECT id FROM jobs WHERE kind = 'reindex' AND status IN ('queued', 'running', 'failed') "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return ProfileState(
        active=active_profile(conn),
        profiles=list_profiles(conn),
        reindex_job=get_job(conn, row[0]) if row else None,
    )


def trace_summaries(conn: sqlite3.Connection, limit: int) -> list[TraceSummary]:
    """The newest question traces (the in-app Traces page)."""
    rows = conn.execute(
        "SELECT id, created_at, outcome, total_ms, ttft_ms, cost_usd, "
        "json_extract(body, '$.question') AS question FROM traces WHERE kind = 'question' "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        TraceSummary(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            question=row["question"],
            outcome=row["outcome"],
            total_ms=row["total_ms"],
            ttft_ms=row["ttft_ms"],
            cost_usd=row["cost_usd"] or 0.0,
        )
        for row in rows
    ]
