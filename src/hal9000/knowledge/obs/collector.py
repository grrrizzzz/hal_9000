"""Gauges read from ``knowledge.db`` at scrape time (plan §9.2).

Short-lived CLI processes (``kb eval regression``, a sweep, ``kb ingest``) never
publish metrics themselves. They write rows, and ``hal-kb`` reads them here on
every scrape, so the values appear on ``/metrics`` however they were produced:

- ``kb_regression_recall``: the latest ``regression_runs.recall_2k``.
- ``kb_eval_metric{profile, metric}``: the latest finished ``eval_runs`` row per
  profile, flattened (``budgeted_recall@2000``, ``mrr``, ...). No run label,
  so the number of series stays small.
- ``kb_documents{profile}``, ``kb_chunks{profile}``: per index profile.
- ``kb_db_bytes{profile}``: ``profile="all"`` is the database file plus its
  WAL; a profile's own value is the size of its FTS5 and ``vec0`` tables
  (from ``dbstat``, cached for :data:`DBSTAT_TTL_S`, when SQLite has it).

The database is opened read-only for each scrape. A missing file or a
not-yet-migrated one yields the metric families with no samples rather than
an error, so a scrape never fails because of the database.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

__all__ = ["DBSTAT_TTL_S", "KnowledgeDbCollector", "flatten_eval_metrics", "eval_profile"]

log = logging.getLogger(__name__)

DBSTAT_TTL_S = 300.0
_PROFILE_TABLE = re.compile(r"^(?:fts|vec)_(\d+)(?:_|$)")


def eval_profile(config: Any) -> str | None:
    """The index profile an ``eval_runs.config`` describes.

    ``config["profile"]`` as a name, or as a profile object with a ``name``.
    Returns ``None`` when the run names no profile (it is then not exposed).
    """
    if not isinstance(config, dict):
        return None
    profile = config.get("profile")
    if isinstance(profile, dict):
        profile = profile.get("name")
    if isinstance(profile, str) and profile:
        return profile
    return None


def _value(item: Any) -> float | None:
    if isinstance(item, dict):
        item = item.get("value")
    if isinstance(item, bool) or not isinstance(item, int | float):
        return None
    return float(item)


def flatten_eval_metrics(metrics: Any) -> dict[str, float]:
    """``EvalMetrics`` JSON -> ``{metric: value}``.

    Keyed metrics become ``name@key`` (``budgeted_recall@2000``); a confidence
    interval contributes its point value. ``by_category`` is left out (too many
    series); so is anything that isn't a number.
    """
    out: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for name, item in metrics.items():
        if name == "by_category":
            continue
        if isinstance(item, dict) and "value" not in item:
            for key, sub in item.items():
                value = _value(sub)
                if value is not None:
                    out[f"{name}@{key}"] = value
            continue
        value = _value(item)
        if value is not None:
            out[name] = value
    return out


class KnowledgeDbCollector(Collector):
    """Reads ``knowledge.db`` on each ``collect()`` (one read-only connection per scrape)."""

    def __init__(self, db_path: Path | str, *, dbstat_ttl_s: float = DBSTAT_TTL_S) -> None:
        self.db_path = Path(db_path)
        self.dbstat_ttl_s = dbstat_ttl_s
        self._dbstat: tuple[float, dict[int, float]] | None = None
        self._lock = threading.Lock()

    def describe(self) -> list[Metric]:
        # Nothing to pre-register; collect() is cheap but touches the disk.
        return []

    # ------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path.exists():
            return None
        uri = f"file:{quote(str(self.db_path.resolve()))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        return conn

    def _files_bytes(self) -> float:
        total = 0
        for suffix in ("", "-wal"):
            path = self.db_path.with_name(self.db_path.name + suffix)
            try:
                total += path.stat().st_size
            except OSError:
                pass
        return float(total)

    def _profile_bytes(self, conn: sqlite3.Connection) -> dict[int, float]:
        now = time.monotonic()
        with self._lock:
            if self._dbstat is not None and now - self._dbstat[0] < self.dbstat_ttl_s:
                return self._dbstat[1]
        sizes: dict[int, float] = {}
        try:
            rows = conn.execute("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name").fetchall()
        except sqlite3.Error:  # SQLite built without SQLITE_ENABLE_DBSTAT_VTAB
            rows = []
        for name, size in rows:
            match = _PROFILE_TABLE.match(name or "")
            if match:
                pid = int(match.group(1))
                sizes[pid] = sizes.get(pid, 0.0) + float(size or 0)
        with self._lock:
            self._dbstat = (now, sizes)
        return sizes

    # ------------------------------------------------------------ collect
    def collect(self) -> Iterator[Metric]:
        recall = GaugeMetricFamily(
            "kb_regression_recall",
            "Latest post-ingest budgeted recall @ 2k on the approved question set.",
        )
        evals = GaugeMetricFamily(
            "kb_eval_metric",
            "The latest finished evaluation run's metrics, per index profile.",
            labels=["profile", "metric"],
        )
        documents = GaugeMetricFamily(
            "kb_documents", "Documents with chunks in each index profile.", labels=["profile"]
        )
        chunks = GaugeMetricFamily("kb_chunks", "Chunks per index profile.", labels=["profile"])
        db_bytes = GaugeMetricFamily(
            "kb_db_bytes",
            "Bytes on disk: profile='all' is knowledge.db plus its WAL; "
            "a profile's value is its own FTS5 and vec0 tables.",
            labels=["profile"],
        )
        families = (recall, evals, documents, chunks, db_bytes)
        try:
            conn = self._connect()
        except sqlite3.Error as exc:
            log.warning("knowledge.db metrics: cannot open %s: %s", self.db_path, exc)
            conn = None
        if conn is not None:
            try:
                self._fill(conn, recall, evals, documents, chunks, db_bytes)
            except sqlite3.Error as exc:
                log.warning("knowledge.db metrics: read failed: %s", exc)
            finally:
                conn.close()
        yield from families

    def _fill(
        self,
        conn: sqlite3.Connection,
        recall: GaugeMetricFamily,
        evals: GaugeMetricFamily,
        documents: GaugeMetricFamily,
        chunks: GaugeMetricFamily,
        db_bytes: GaugeMetricFamily,
    ) -> None:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        db_bytes.add_metric(["all"], self._files_bytes())
        if "regression_runs" in tables:
            row = conn.execute(
                "SELECT recall_2k FROM regression_runs WHERE recall_2k IS NOT NULL "
                "ORDER BY at DESC, id DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                recall.add_metric([], float(row[0]))
        if "eval_runs" in tables:
            latest: dict[str, dict[str, float]] = {}
            rows = conn.execute(
                "SELECT config, metrics FROM eval_runs "
                "WHERE metrics IS NOT NULL AND finished_at IS NOT NULL "
                "ORDER BY finished_at, started_at, id"
            )
            for config_json, metrics_json in rows:
                try:
                    profile = eval_profile(json.loads(config_json or "{}"))
                    values = flatten_eval_metrics(json.loads(metrics_json))
                except (TypeError, ValueError):
                    continue
                if profile is not None and values:
                    latest[profile] = values  # rows ascend, so the last one wins
            for profile in sorted(latest):
                for metric, value in sorted(latest[profile].items()):
                    evals.add_metric([profile, metric], value)
        if {"profiles", "chunks"} <= tables:
            names = dict(conn.execute("SELECT id, name FROM profiles").fetchall())
            counts = {
                pid: (n_docs, n_chunks)
                for pid, n_docs, n_chunks in conn.execute(
                    "SELECT profile_id, COUNT(DISTINCT document_id), COUNT(*) "
                    "FROM chunks GROUP BY profile_id"
                )
            }
            sizes = self._profile_bytes(conn) if names else {}
            for pid, name in sorted(names.items()):
                n_docs, n_chunks = counts.get(pid, (0, 0))
                documents.add_metric([name], float(n_docs))
                chunks.add_metric([name], float(n_chunks))
                if pid in sizes:
                    db_bytes.add_metric([name], sizes[pid])
