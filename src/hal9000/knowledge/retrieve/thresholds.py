"""Abstain thresholds per (index profile, reranker) in ``abstain_thresholds`` (plan §8.5, §12).

The abstain score's scale depends on the embedding model, its dimensions and
the reranker, so each (profile, reranker) pair has its own calibrated
threshold. ``reranker`` is ``''`` for no reranker (:func:`reranker_key`), else
``minilm`` or ``bge-base``. With no stored row the threshold is unknown and
retrieval never abstains (the generation-side ``NOT_IN_SOURCES`` reply still
applies).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from hal9000.knowledge.retrieve.config import reranker_key

__all__ = ["get_threshold", "set_threshold"]


def get_threshold(conn: sqlite3.Connection, profile_id: int, reranker: str | None) -> float | None:
    """The calibrated threshold for (profile, reranker), or ``None`` when not calibrated."""
    row = conn.execute(
        "SELECT threshold FROM abstain_thresholds WHERE profile_id = ? AND reranker = ?",
        (profile_id, reranker_key(reranker)),
    ).fetchone()
    return float(row[0]) if row else None


def set_threshold(
    conn: sqlite3.Connection,
    profile_id: int,
    reranker: str | None,
    threshold: float,
    *,
    auc: float | None = None,
    eval_run_id: str | None = None,
) -> None:
    """Store (or replace) the threshold for (profile, reranker). Commits."""
    with conn:
        conn.execute(
            "INSERT INTO abstain_thresholds (profile_id, reranker, threshold, auc, eval_run_id, "
            "calibrated_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (profile_id, reranker) DO "
            "UPDATE SET threshold = excluded.threshold, auc = excluded.auc, "
            "eval_run_id = excluded.eval_run_id, calibrated_at = excluded.calibrated_at",
            (
                profile_id,
                reranker_key(reranker),
                float(threshold),
                auc,
                eval_run_id,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
