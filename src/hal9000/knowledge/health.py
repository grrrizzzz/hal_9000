"""``data/knowledge_health.json``: what HAL's watchdog reads about the knowledge base (plan §16).

The file carries the last successful question, the last finished ingestion and
the month-to-date spend (with its cap), and never any question text. The
service (``hal-kb``, KWP6's :meth:`Services.write_health`) writes it on every
``/health`` request, after every answered question and, through
:func:`watch_ingestions`, whenever an ingestion job finishes, so "last
ingestion" is fresh without waiting for the next health probe. ``kb stats``
reads the same two timestamps with :func:`last_question_at` and
:func:`last_ingestion_at`.

It is written atomically (a temporary file, then a rename), so a reader never
sees half a file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "HEALTH_FILE",
    "health_body",
    "last_ingestion_at",
    "last_question_at",
    "watch_ingestions",
    "write_health",
    "write_health_file",
]

log = logging.getLogger(__name__)

HEALTH_FILE = "knowledge_health.json"


def last_question_at(conn: sqlite3.Connection) -> str | None:
    """When the last question was answered (``traces``), ISO 8601, or ``None``."""
    return conn.execute(
        "SELECT MAX(created_at) FROM traces WHERE kind = 'question' AND outcome = 'answered'"
    ).fetchone()[0]


def last_ingestion_at(conn: sqlite3.Connection) -> str | None:
    """When the last ingestion job finished successfully (``jobs``), or ``None``."""
    return conn.execute(
        "SELECT MAX(updated_at) FROM jobs WHERE kind = 'ingest' AND status = 'done'"
    ).fetchone()[0]


def health_body(
    conn: sqlite3.Connection, *, month_spend: float | None, month_cap: float
) -> dict[str, Any]:
    """The health document (plan §16). ``month_spend`` is ``None`` when the ledger is unreadable."""
    return {
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_successful_question_at": last_question_at(conn),
        "last_ingestion_at": last_ingestion_at(conn),
        "month_spend_usd": month_spend,
        "month_cap_usd": month_cap,
    }


def write_health_file(data_dir: Path, body: dict[str, Any]) -> Path:
    """Write ``body`` to ``data_dir/knowledge_health.json`` atomically; returns the path."""
    target = Path(data_dir) / HEALTH_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    tmp.replace(target)
    return target


def write_health(
    conn: sqlite3.Connection, data_dir: Path, *, month_spend: float | None, month_cap: float
) -> dict[str, Any]:
    """Build and write the health file; an unwritable data dir is logged, never raised."""
    body = health_body(conn, month_spend=month_spend, month_cap=month_cap)
    try:
        write_health_file(data_dir, body)
    except OSError:  # a read-only data dir must not break /health or an answer
        log.exception("could not write %s", Path(data_dir) / HEALTH_FILE)
    return body


def watch_ingestions(bus: Any, refresh: Callable[[], Any]) -> Callable[[], None]:
    """Call ``refresh()`` whenever an ingestion job finishes; returns the unsubscribe function.

    ``bus`` is KWP3's :class:`~hal9000.knowledge.ingest.jobs.ProgressBus`; the
    callback runs on the worker's thread, so ``refresh`` opens its own connection.
    """

    def on_job(job: Any) -> None:
        if getattr(job, "kind", None) == "ingest" and getattr(job, "status", None) == "done":
            try:
                refresh()
            except Exception:  # a health write never breaks a job
                log.exception("could not refresh %s after an ingestion", HEALTH_FILE)

    return bus.subscribe(on_job)
