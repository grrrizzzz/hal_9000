"""``eval_runs`` rows and their files under ``data/knowledge/eval/runs/<id>/`` (plan §8.6).

Each run writes three files, and its ``eval_runs`` row points to them:

- ``results.jsonl``: one line per question (scores, abstain decision, selected
  chunk IDs; the question text for the in-app drill-down, never evidence or
  chunk text),
- ``metrics.json``: the :class:`~hal9000.knowledge.contracts.EvalMetrics`,
- ``config.json``: the run's configuration.

The row matches what KWP8's collector reads (``obs/collector.py``):
``config["profile"]`` names the index profile (a run without one is not
exposed as ``kb_eval_metric``), ``split`` is ``dev`` for sweep stages S1-S5,
``stage`` sorts ``S1a < S1b < S2 ...``, ``finished_at`` is set only when
the metrics are final, and ``metrics`` is the EvalMetrics JSON, which the
collector flattens to ``budgeted_recall@2000``, ``mrr`` and so on.

Well-known ``config`` keys (the report groups by them): ``profile``,
``sweep_id``, ``parser``, ``chunker``, ``size``, ``overlap``,
``parent_size``, ``header``, ``embed_model``, ``dimensions``,
``retrieval``, ``alpha``, ``reranker``, ``budget``, ``k``,
``query_transform``, ``chat_tier``, ``abstain_threshold``.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hal9000.knowledge.contracts import EvalMetrics, EvalRun
from hal9000.knowledge.eval.scoring import QuestionScore
from hal9000.knowledge.eval.types import EvalQuestionResult, EvalResultsPage

__all__ = [
    "RESULTS_FILE",
    "HOST_CLASSES",
    "finish_run",
    "get_run",
    "host_class",
    "list_runs",
    "load_results",
    "new_run_id",
    "question_record",
    "record_run",
    "results_page",
    "run_dir",
    "runs_for",
    "start_run",
]

RESULTS_FILE = "results.jsonl"
#: ``full_support`` in the per-question API is hit @ this k (or the largest k scored).
FULL_SUPPORT_K = 5


#: What ``eval_runs.host`` records: a class of machine, never its name.
HOST_CLASSES = ("mac", "pi", "linux", "windows")


#: Where a Raspberry Pi names itself. ``/proc/device-tree`` links into ``/sys/firmware``,
#: which Docker masks by default, so inside the container only ``/proc/cpuinfo`` (whose
#: ``Model`` line the Raspberry Pi kernel adds) says it is a Pi.
_MODEL_SOURCES = (Path("/proc/device-tree/model"), Path("/proc/cpuinfo"))


def host_class() -> str:
    """``mac``, ``pi`` (a Raspberry Pi), ``linux`` or ``windows`` for this machine."""
    system = platform.system()
    if system == "Darwin":
        return "mac"
    if system == "Windows":
        return "windows"
    for source in _MODEL_SOURCES:
        try:
            text = source.read_text(errors="ignore")
        except OSError:
            continue
        if "raspberry pi" in text.lower():
            return "pi"
    return "linux"


def _now() -> datetime:
    return datetime.now(UTC)


def new_run_id(stage: str) -> str:
    """``s1a-20260928T101500-3f2a``: sortable, readable, unique."""
    stamp = _now().strftime("%Y%m%dT%H%M%S")
    return f"{stage.lower()}-{stamp}-{uuid.uuid4().hex[:4]}"


def run_dir(eval_dir: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise ValueError(f"not a valid run id: {run_id!r}")
    return eval_dir / "runs" / run_id


def question_record(
    score: QuestionScore,
    *,
    question: str = "",
    selected: Iterable[int] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``results.jsonl`` line for a scored question."""
    record = {
        "gold_item_id": score.gold_item_id,
        "question": question,
        "category": score.category,
        "answerable": score.answerable,
        "recall": {str(k): v for k, v in score.recall.items()},
        "evidence_recall": score.evidence_recall,
        "full_support": {str(k): v for k, v in score.full_support.items()},
        "first_hit_rank": score.first_hit_rank,
        "reciprocal_rank": score.reciprocal_rank,
        "context_precision": {str(k): v for k, v in score.context_precision.items()},
        "context_tokens": {str(k): v for k, v in score.context_tokens.items()},
        "abstain_score": score.abstain_score,
        "abstained": score.abstained,
        "latency_ms": score.latency_ms,
        "selected": list(selected),
    }
    if extra:
        record.update(extra)
    return record


def _row_to_run(row: sqlite3.Row) -> EvalRun:
    return EvalRun(
        id=row["id"],
        stage=row["stage"],
        split=row["split"],
        config=json.loads(row["config"] or "{}"),
        metrics=EvalMetrics.model_validate_json(row["metrics"]) if row["metrics"] else None,
        host=row["host"],
        cost_usd=row["cost_usd"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        results_path=row["results_path"],
    )


def start_run(
    conn: sqlite3.Connection,
    *,
    stage: str,
    config: Mapping[str, Any],
    split: str = "dev",
    run_id: str | None = None,
    host: str | None = None,
) -> EvalRun:
    """Insert an unfinished ``eval_runs`` row (the collector ignores it until finished)."""
    run = EvalRun(
        id=run_id or new_run_id(stage),
        stage=stage,
        split=split,
        config=dict(config),
        host=host_class() if host is None else host,
        started_at=_now(),
    )
    with conn:
        conn.execute(
            "INSERT INTO eval_runs (id, stage, split, config, host, cost_usd, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run.id,
                run.stage,
                run.split,
                json.dumps(run.config, sort_keys=True),
                run.host,
                0.0,
                run.started_at.isoformat(),
            ),
        )
    return run


def finish_run(
    conn: sqlite3.Connection,
    run: EvalRun,
    *,
    eval_dir: Path,
    metrics: EvalMetrics,
    records: Sequence[Mapping[str, Any]],
    cost_usd: float = 0.0,
) -> EvalRun:
    """Write the run's files, then mark its row finished with the metrics."""
    folder = run_dir(eval_dir, run.id)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / RESULTS_FILE).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    (folder / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    (folder / "config.json").write_text(json.dumps(run.config, indent=2, sort_keys=True))
    done = run.model_copy(
        update={
            "metrics": metrics,
            "cost_usd": cost_usd,
            "finished_at": _now(),
            "results_path": str(folder / RESULTS_FILE),
        }
    )
    with conn:
        conn.execute(
            "UPDATE eval_runs SET metrics = ?, cost_usd = ?, finished_at = ?, results_path = ? "
            "WHERE id = ?",
            (
                metrics.model_dump_json(),
                cost_usd,
                done.finished_at.isoformat(),
                done.results_path,
                run.id,
            ),
        )
    return done


def record_run(
    conn: sqlite3.Connection,
    *,
    eval_dir: Path,
    stage: str,
    config: Mapping[str, Any],
    metrics: EvalMetrics,
    records: Sequence[Mapping[str, Any]],
    split: str = "dev",
    cost_usd: float = 0.0,
    run_id: str | None = None,
    host: str | None = None,
) -> EvalRun:
    """:func:`start_run` then :func:`finish_run`, for a run computed in one go."""
    run = start_run(conn, stage=stage, config=config, split=split, run_id=run_id, host=host)
    return finish_run(
        conn, run, eval_dir=eval_dir, metrics=metrics, records=records, cost_usd=cost_usd
    )


def get_run(conn: sqlite3.Connection, run_id: str) -> EvalRun | None:
    row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def list_runs(conn: sqlite3.Connection, *, stage: str | None = None) -> list[EvalRun]:
    if stage is None:
        rows = conn.execute("SELECT * FROM eval_runs ORDER BY started_at, id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_runs WHERE stage = ? ORDER BY started_at, id", (stage,)
        ).fetchall()
    return [_row_to_run(r) for r in rows]


def runs_for(conn: sqlite3.Connection, run_id: str) -> list[EvalRun]:
    """The run with this ID, or every run of the sweep with this ``config.sweep_id``."""
    exact = get_run(conn, run_id)
    if exact is not None:
        return [exact]
    return [r for r in list_runs(conn) if r.config.get("sweep_id") == run_id]


def _relocated_results(run: EvalRun) -> Path | None:
    """The run's results file under this machine's own eval folder.

    ``eval_runs.results_path`` is an absolute path written on the machine that ran
    the sweep. After the library is copied to another machine (the Mac to the Pi,
    runbook 4.5) that path doesn't exist, but the same ``runs/<id>/results.jsonl``
    sits under the local ``data/knowledge/eval``.
    """
    try:
        from hal9000.knowledge.config import settings

        candidate = run_dir(settings().eval_dir, run.id) / RESULTS_FILE
    except Exception:  # noqa: BLE001 - settings unavailable or an unusable run ID: no fallback
        return None
    return candidate if candidate.exists() else None


def load_results(run: EvalRun) -> list[dict[str, Any]]:
    """The run's per-question records (empty when it has no results file yet)."""
    if not run.results_path:
        return []
    path = Path(run.results_path)
    if not path.exists():
        path = _relocated_results(run)
        if path is None:
            return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _full_support(record: Mapping[str, Any]) -> bool:
    hits = record.get("full_support") or {}
    if not hits:
        return False
    key = str(FULL_SUPPORT_K) if str(FULL_SUPPORT_K) in hits else max(hits, key=int)
    return bool(hits[key])


def results_page(
    conn: sqlite3.Connection, run_id: str, *, offset: int = 0, limit: int = 25
) -> EvalResultsPage:
    """``GET /api/eval/runs/{id}/results``: paged per-question results (KeyError if no run)."""
    run = get_run(conn, run_id)
    if run is None:
        raise KeyError(run_id)
    if offset < 0 or not 1 <= limit <= 500:
        raise ValueError("offset must be >= 0 and limit between 1 and 500")
    records = load_results(run)
    items = [
        EvalQuestionResult(
            gold_item_id=r["gold_item_id"],
            question=r.get("question", ""),
            category=r["category"],
            recall={str(k): float(v) for k, v in (r.get("recall") or {}).items()},
            full_support=_full_support(r),
            first_hit_rank=r.get("first_hit_rank"),
            abstain_score=r.get("abstain_score"),
            abstained=bool(r.get("abstained")),
        )
        for r in records[offset : offset + limit]
    ]
    return EvalResultsPage(items=items, total=len(records), offset=offset, limit=limit)
