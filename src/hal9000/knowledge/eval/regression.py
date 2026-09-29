"""Post-ingest regression: retrieval-only recall @ 2k on the approved question set (plan §9.2).

After an ingestion (``trigger="ingest"``, called by KWP3's pipeline through
:func:`post_ingest_regression`) or on demand (``kb eval regression``,
``trigger="manual"``), every **approved** answerable gold item is retrieved
against the active profile through an injected retrieve callable, scored with
budgeted recall at the regression budget (2,000 tokens) and a bootstrap 95%
interval, and one ``regression_runs`` row is written. ``hal-kb`` exposes the
latest row as ``kb_regression_recall`` (a fraction, 0-1).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from hal9000.knowledge.contracts import MetricCI
from hal9000.knowledge.eval.config import EvalConfig, load_eval_config
from hal9000.knowledge.eval.scoring import ScoringItem, bootstrap_ci, score_question
from hal9000.knowledge.eval.store import list_gold
from hal9000.knowledge.eval.types import RetrieveFn
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = ["RegressionResult", "post_ingest_regression", "run_regression"]

log = logging.getLogger(__name__)

Trigger = Literal["ingest", "manual"]


@dataclass(frozen=True)
class RegressionResult:
    at: datetime
    trigger: Trigger
    profile_id: int | None
    n_questions: int
    recall: MetricCI | None
    budget: int
    row_id: int | None = None
    errors: int = 0

    def describe(self) -> str:
        if self.recall is None:
            what = "N/A (no approved answerable questions, or chunks larger than the budget)"
        else:
            r = self.recall
            what = f"{r.value:.3f} [{r.low:.3f}-{r.high:.3f}]"
        return (
            f"regression ({self.trigger}): recall @ {self.budget} = {what} over "
            f"{self.n_questions} approved questions"
            + (f"; {self.errors} retrieval errors" if self.errors else "")
        )


def _active_profile(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM profiles WHERE active = 1").fetchone()
    return int(row[0]) if row else None


def run_regression(
    conn: sqlite3.Connection,
    retrieve: RetrieveFn,
    *,
    profile_id: int | None = None,
    chunk_size: int | None = None,
    trigger: Trigger = "manual",
    config: EvalConfig | None = None,
    count=None,
) -> RegressionResult:
    """Score the approved set retrieval-only and write one ``regression_runs`` row.

    ``profile_id`` defaults to the active profile; ``chunk_size`` (the parent
    size for parent-child) makes the budget N/A when larger. Nothing is written
    when there is no approved answerable question.
    """
    cfg = config or load_eval_config()
    budget = cfg.regression.budget
    scoring = replace(cfg.scoring.to_config(), budgets=(budget,), hit_ks=())
    at = datetime.now(UTC)
    with stage(Stage.REGRESSION):
        if profile_id is None:
            profile_id = _active_profile(conn)
        approved = [g for g in list_gold(conn, status="approved") if g.category != "unanswerable"]
        questions = {g.id: g.question for g in approved}
        items = [ScoringItem.from_gold(g) for g in approved]
        if not items:
            log.info("regression skipped: no approved answerable questions")
            return RegressionResult(at, trigger, profile_id, 0, None, budget)
        recalls: list[float] = []
        errors = 0
        for item in items:
            try:
                output = retrieve(questions[item.gold_item_id])
            except Exception as error:  # a failed retrieval scores zero, and is counted
                log.warning(
                    "regression: retrieval failed for item %s: %s", item.gold_item_id, error
                )
                errors += 1
                recalls.append(0.0)
                continue
            score = score_question(item, output, chunk_size=chunk_size, config=scoring, count=count)
            if budget in score.recall:
                recalls.append(score.recall[budget])
        recall = (
            bootstrap_ci(recalls, n_boot=scoring.n_boot, seed=scoring.seed)
            if recalls and not (chunk_size is not None and chunk_size > budget)
            else None
        )
        with conn:
            cur = conn.execute(
                "INSERT INTO regression_runs (at, profile_id, trigger, recall_2k, recall_low, "
                "recall_high, n_questions) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    at.isoformat(timespec="seconds"),
                    profile_id,
                    trigger,
                    recall.value if recall else None,
                    recall.low if recall else None,
                    recall.high if recall else None,
                    len(items),
                ),
            )
    return RegressionResult(
        at, trigger, profile_id, len(items), recall, budget, int(cur.lastrowid), errors
    )


def post_ingest_regression(
    conn: sqlite3.Connection, retrieve: RetrieveFn, **kw
) -> RegressionResult | None:
    """KWP3's hook after an ingestion: like :func:`run_regression`, but never raises.

    An ingestion must not fail because the regression did; the error is logged
    and None returned.
    """
    try:
        return run_regression(conn, retrieve, trigger="ingest", **kw)
    except Exception:
        log.exception("post-ingest regression failed")
        return None
