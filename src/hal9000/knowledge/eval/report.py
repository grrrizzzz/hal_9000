"""``kb eval report RUN``: ``docs/KNOWLEDGE_EVAL_REPORT.md`` from ``eval_runs`` (plan §8.6).

Aggregates only: run configurations and metric values. No question, evidence
or document text is read, so none can be written (the report is built from
``eval_runs`` rows alone, never from the per-question files).

Sections: a summary per stage, one table per trade-off T1-T12 that the runs
actually vary (plan §8.3), the Pareto data (recall @ 2k against local
retrieval latency, with cost per question), per-category recall, test-split
runs, and a placeholder for the configuration chosen at gate G-K2.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from hal9000.knowledge.contracts import EvalRun, MetricCI
from hal9000.knowledge.eval.runs import runs_for
from hal9000.knowledge.obs.collector import stage_order

__all__ = ["TRADE_OFFS", "pareto_front", "render_report", "write_report"]

#: plan §8.3: (id, title, the config keys that vary it)
TRADE_OFFS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("T1", "Chunk size", ("size",)),
    ("T2", "Overlap", ("overlap",)),
    ("T3", "Chunking strategy", ("chunker", "parent_size")),
    ("T4", "Contextual header", ("header",)),
    ("T5", "Parser", ("parser",)),
    ("T6", "Embedding model and dimensions", ("embed_model", "dimensions")),
    ("T7", "Retrieval method", ("retrieval", "alpha")),
    ("T8", "Reranking", ("reranker",)),
    ("T9", "Context budget / k", ("budget", "k")),
    ("T10", "Query transformation", ("query_transform",)),
    ("T11", "Chat model tier", ("chat_tier",)),
    ("T12", "Abstain threshold", ("abstain_threshold",)),
)


def _fmt(ci: MetricCI | None) -> str:
    if ci is None:
        return "N/A"
    return f"{ci.value:.3f} [{ci.low:.3f}-{ci.high:.3f}]"


def _num(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _recall(run: EvalRun, budget: int) -> MetricCI | None:
    return run.metrics.budgeted_recall.get(budget) if run.metrics else None


def _cell(value) -> str:
    return "" if value is None else str(value).replace("|", "/")


def _metric_row(run: EvalRun) -> list[str]:
    m = run.metrics
    if m is None:
        return ["unfinished"] + [""] * 7
    hit = m.full_support_hit.get(5) or (
        max(m.full_support_hit.items())[1] if m.full_support_hit else None
    )
    return [
        _fmt(m.budgeted_recall.get(1000)),
        _fmt(m.budgeted_recall.get(2000)),
        _fmt(m.budgeted_recall.get(4000)),
        _fmt(m.context_precision.get(2000)),
        _fmt(m.mrr),
        _fmt(hit),
        _fmt(m.abstain_auc),
        f"{_num(m.latency_ms_p50)} / {_num(m.latency_ms_p95)}",
    ]


METRIC_HEADERS = [
    "recall@1k",
    "recall@2k",
    "recall@4k",
    "precision@2k",
    "MRR",
    "hit@5",
    "abstain AUC",
    "p50 / p95 ms",
]


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _sort_key(run: EvalRun) -> tuple:
    r = _recall(run, 2000)
    return (stage_order(run.stage), -(r.value if r else -1.0), run.id)


def pareto_front(points: Sequence[tuple[str, float, float]]) -> set[str]:
    """IDs not dominated on (higher recall, lower latency)."""
    front: set[str] = set()
    for rid, recall, latency in points:
        dominated = any(
            (r2 >= recall and l2 <= latency) and (r2 > recall or l2 < latency)
            for other, r2, l2 in points
            if other != rid
        )
        if not dominated:
            front.add(rid)
    return front


def render_report(
    runs: Sequence[EvalRun], *, source: str, generated: datetime | None = None
) -> str:
    """The report's Markdown for ``runs`` (aggregates only)."""
    when = (generated or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    runs = sorted(runs, key=_sort_key)
    dev = [r for r in runs if r.split == "dev"]
    lines = [
        "# House knowledge base: evaluation report",
        "",
        f"Generated {when} by `hal9000 kb eval report {source}` from {len(runs)} run(s). "
        "Aggregate numbers only: this file contains no document or question text "
        "(docs/KNOWLEDGE_PLAN.md §8.6). Intervals are bootstrap 95% confidence intervals; "
        "a difference counts only when a paired bootstrap excludes zero.",
        "",
        "## Summary by stage",
        "",
    ]
    stages = sorted({r.stage for r in runs}, key=stage_order)
    rows = []
    for st in stages:
        members = [r for r in runs if r.stage == st]
        best = max(
            (r for r in members if _recall(r, 2000)),
            key=lambda r: _recall(r, 2000).value,
            default=None,
        )
        rows.append(
            [
                st,
                str(len(members)),
                ", ".join(sorted({r.split for r in members})),
                _fmt(_recall(best, 2000)) if best else "N/A",
                _cell(best.config.get("profile")) if best else "",
                f"${sum(r.cost_usd for r in members):.2f}",
            ]
        )
    lines += _table(["stage", "runs", "split", "best recall@2k", "best profile", "cost"], rows)
    lines += ["", "## Trade-offs (dev split)", ""]
    for tid, title, keys in TRADE_OFFS:
        varied = [k for k in keys if len({repr(r.config.get(k)) for r in dev if k in r.config}) > 1]
        lines.append(f"### {tid}: {title}")
        lines.append("")
        if not varied:
            lines.append("Not varied in these runs.")
            lines.append("")
            continue
        members = [r for r in dev if any(k in r.config for k in varied)]
        members.sort(key=lambda r: tuple(repr(r.config.get(k)) for k in varied) + (r.id,))
        table_rows = [
            [r.stage, _cell(r.config.get("profile")), *(_cell(r.config.get(k)) for k in varied)]
            + _metric_row(r)
            for r in members
        ]
        lines += _table(["stage", "profile", *varied, *METRIC_HEADERS], table_rows)
        lines.append("")
    lines += ["## Pareto data: recall @ 2k vs local retrieval latency", ""]
    points = [
        (r.id, _recall(r, 2000).value, r.metrics.latency_ms_p95)
        for r in dev
        if r.metrics and _recall(r, 2000) and r.metrics.latency_ms_p95 is not None
    ]
    front = pareto_front(points)
    by_id = {r.id: r for r in dev}
    pareto_rows = [
        [
            rid,
            _cell(by_id[rid].config.get("profile")),
            by_id[rid].host,
            f"{recall:.3f}",
            f"{latency:.1f}",
            _num(by_id[rid].metrics.cost_per_question_usd, 6),
            "yes" if rid in front else "",
        ]
        for rid, recall, latency in sorted(points, key=lambda p: (-p[1], p[2]))
    ]
    if pareto_rows:
        lines += _table(
            ["run", "profile", "host", "recall@2k", "p95 ms", "USD / question", "Pareto"],
            pareto_rows,
        )
    else:
        lines.append("No runs with both recall @ 2k and latency yet.")
    lines += ["", "## Recall @ 2k by category (dev)", ""]
    categories = sorted({c for r in dev if r.metrics for c in r.metrics.by_category})
    cat_rows = [
        [r.id, _cell(r.config.get("profile"))]
        + [_num(r.metrics.by_category.get(c, {}).get("recall@2k"), 3) for c in categories]
        for r in dev
        if r.metrics and r.metrics.by_category
    ]
    if cat_rows:
        lines += _table(["run", "profile", *categories], cat_rows)
    else:
        lines.append("No per-category results yet.")
    test = [r for r in runs if r.split == "test"]
    lines += ["", "## Test split (reported once, for the chosen configuration)", ""]
    if test:
        lines += _table(
            ["run", "stage", "profile", *METRIC_HEADERS],
            [[r.id, r.stage, _cell(r.config.get("profile")), *_metric_row(r)] for r in test],
        )
    else:
        lines.append("Not run yet.")
    lines += [
        "",
        "## Chosen configuration",
        "",
        "_To be completed at gate G-K2 (plan §8.6): the chosen index profile, reranker and "
        "abstain threshold, the paired-bootstrap comparisons that decided each tie, and the "
        "test-split result._",
        "",
    ]
    return "\n".join(lines)


def write_report(conn: sqlite3.Connection, run_id: str, path: Path) -> Path:
    """Render the report for a run (or a sweep's runs) to ``path``. KeyError if unknown."""
    runs = runs_for(conn, run_id)
    if not runs:
        raise KeyError(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(runs, source=run_id))
    return path
