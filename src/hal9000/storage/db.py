"""SQLite persistence for HAL: records, hourly energy, weather cache, alerts, ledger, runs.

One file (``data/hal9000.db``), WAL mode, a lock per database so threads in the
API server don't interleave writes. Every write goes through a method here; no
other module writes SQL.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, date, datetime
from pathlib import Path

from hal9000.contracts import NightlyRecord

__all__ = ["Database"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nightly_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    kind TEXT NOT NULL,
    written_at TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_records_date ON nightly_records(date);
CREATE TABLE IF NOT EXISTS hourly_energy (
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    solar_kwh REAL,
    import_kwh REAL,
    export_kwh REAL,
    complete INTEGER NOT NULL DEFAULT 0,
    flags TEXT,
    PRIMARY KEY (date, hour)
);
CREATE TABLE IF NOT EXISTS weather_hourly (
    date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    poa_kwh_m2 REAL,
    ghi_kwh_m2 REAL,
    cloud_cover REAL,
    temp_c REAL,
    source TEXT,
    PRIMARY KEY (date, hour)
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_key TEXT PRIMARY KEY,
    severity TEXT NOT NULL,
    text TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    agent TEXT,
    model TEXT NOT NULL,
    prompt_tokens INTEGER, cached_tokens INTEGER, completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ledger_run ON ledger(run_id);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,
    body TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_LOCKS: dict[str, threading.Lock] = {}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = _LOCKS.setdefault(self.path, threading.Lock())
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ records
    def append_record(self, record: NightlyRecord) -> int:
        body = NightlyRecord.model_validate(record.model_dump()).model_dump_json()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO nightly_records (date, kind, written_at, body) VALUES (?,?,?,?)",
                (record.date, record.kind, _now(), body),
            )
        return int(cur.lastrowid)

    def latest_record(self, day: date | str) -> NightlyRecord | None:
        row = self._conn.execute(
            "SELECT body FROM nightly_records WHERE date = ? ORDER BY id DESC LIMIT 1",
            (str(day),),
        ).fetchone()
        return NightlyRecord.model_validate_json(row["body"]) if row else None

    def record_versions(self, day: date | str) -> list[NightlyRecord]:
        rows = self._conn.execute(
            "SELECT body FROM nightly_records WHERE date = ? ORDER BY id", (str(day),)
        ).fetchall()
        return [NightlyRecord.model_validate_json(r["body"]) for r in rows]

    def recent_records(self, before: date, days: int) -> list[NightlyRecord]:
        """Latest version of each record in the ``days`` days before ``before``."""
        rows = self._conn.execute(
            """SELECT body FROM nightly_records r
               WHERE id = (SELECT MAX(id) FROM nightly_records WHERE date = r.date)
                 AND date < ? ORDER BY date DESC LIMIT ?""",
            (before.isoformat(), days),
        ).fetchall()
        return [NightlyRecord.model_validate_json(r["body"]) for r in rows]

    def all_latest_records(self) -> list[NightlyRecord]:
        rows = self._conn.execute(
            """SELECT body FROM nightly_records r
               WHERE id = (SELECT MAX(id) FROM nightly_records WHERE date = r.date)
               ORDER BY date"""
        ).fetchall()
        return [NightlyRecord.model_validate_json(r["body"]) for r in rows]

    # -------------------------------------------------------- hourly energy
    def upsert_hourly(self, day: date, rows: list[dict], complete: bool) -> None:
        with self._lock, self._conn:
            for hour, row in enumerate(rows):
                self._conn.execute(
                    """INSERT INTO hourly_energy (date, hour, solar_kwh, import_kwh, export_kwh,
                       complete, flags) VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(date, hour) DO UPDATE SET solar_kwh=excluded.solar_kwh,
                       import_kwh=excluded.import_kwh, export_kwh=excluded.export_kwh,
                       complete=excluded.complete, flags=excluded.flags""",
                    (
                        day.isoformat(),
                        hour,
                        row.get("solar"),
                        row.get("import"),
                        row.get("export"),
                        int(complete),
                        json.dumps(row.get("flags", [])),
                    ),
                )

    def hourly(self, day: date) -> list[dict] | None:
        rows = self._conn.execute(
            "SELECT * FROM hourly_energy WHERE date = ? ORDER BY hour", (day.isoformat(),)
        ).fetchall()
        if len(rows) != 24:
            return None
        return [
            {
                "solar": r["solar_kwh"],
                "import": r["import_kwh"],
                "export": r["export_kwh"],
                "complete": bool(r["complete"]),
                "flags": json.loads(r["flags"] or "[]"),
            }
            for r in rows
        ]

    def hourly_days(self, start: date, end: date) -> list[str]:
        rows = self._conn.execute(
            """SELECT date FROM hourly_energy WHERE date BETWEEN ? AND ? AND complete = 1
               GROUP BY date HAVING COUNT(*) = 24 ORDER BY date""",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [r["date"] for r in rows]

    # -------------------------------------------------------------- weather
    def upsert_weather(self, day: date, rows: list[dict], source: str) -> None:
        with self._lock, self._conn:
            for hour, row in enumerate(rows):
                self._conn.execute(
                    """INSERT INTO weather_hourly (date, hour, poa_kwh_m2, ghi_kwh_m2, cloud_cover,
                       temp_c, source) VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(date, hour) DO UPDATE SET poa_kwh_m2=excluded.poa_kwh_m2,
                       ghi_kwh_m2=excluded.ghi_kwh_m2, cloud_cover=excluded.cloud_cover,
                       temp_c=excluded.temp_c, source=excluded.source""",
                    (
                        day.isoformat(),
                        hour,
                        row.get("poa"),
                        row.get("ghi"),
                        row.get("cloud"),
                        row.get("temp"),
                        source,
                    ),
                )

    def weather(self, day: date) -> list[dict] | None:
        rows = self._conn.execute(
            "SELECT * FROM weather_hourly WHERE date = ? ORDER BY hour", (day.isoformat(),)
        ).fetchall()
        if len(rows) != 24:
            return None
        return [
            {
                "poa": r["poa_kwh_m2"],
                "ghi": r["ghi_kwh_m2"],
                "cloud": r["cloud_cover"],
                "temp": r["temp_c"],
                "source": r["source"],
            }
            for r in rows
        ]

    # --------------------------------------------------------------- alerts
    def alert(self, key: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM alerts WHERE alert_key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def open_alerts(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM alerts WHERE status = 'open' ORDER BY opened_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def open_alert(self, key: str, severity: str, text: str) -> None:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO alerts (alert_key, severity, text, opened_at, updated_at, status)
                   VALUES (?,?,?,?,?,'open')
                   ON CONFLICT(alert_key) DO UPDATE SET severity=excluded.severity,
                   text=excluded.text, updated_at=excluded.updated_at, status='open',
                   opened_at=CASE WHEN alerts.status='open' THEN alerts.opened_at
                                  ELSE excluded.opened_at END""",
                (key, severity, text, now, now),
            )

    def close_alert(self, key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE alerts SET status='closed', updated_at=? WHERE alert_key=?",
                (_now(), key),
            )

    # --------------------------------------------------------------- ledger
    def add_ledger(
        self, run_id: str, agent: str, usage: dict, usd: float, at: str | None = None
    ) -> None:
        """Record one LLM call. ``at`` should be local time (the month is read from it)."""
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO ledger (at, run_id, agent, model, prompt_tokens, cached_tokens,
                   completion_tokens, reasoning_tokens, usd) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    at or _now(),
                    run_id,
                    agent,
                    usage.get("model", "?"),
                    usage.get("prompt", 0),
                    usage.get("cached", 0),
                    usage.get("completion", 0),
                    usage.get("reasoning", 0),
                    usd,
                ),
            )

    def run_cost(self, run_id: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0) AS s FROM ledger WHERE run_id = ?", (run_id,)
        ).fetchone()
        return float(row["s"])

    def month_cost(self, year: int, month: int) -> float:
        prefix = f"{year:04d}-{month:02d}"
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0) AS s FROM ledger WHERE substr(at, 1, 7) = ?", (prefix,)
        ).fetchone()
        return float(row["s"])

    def cost_breakdown(self, year: int, month: int) -> list[dict]:
        prefix = f"{year:04d}-{month:02d}"
        rows = self._conn.execute(
            """SELECT agent, model, COUNT(*) AS calls, SUM(usd) AS usd,
                      SUM(prompt_tokens) AS prompt, SUM(cached_tokens) AS cached,
                      SUM(completion_tokens) AS completion
               FROM ledger WHERE substr(at, 1, 7) = ? GROUP BY agent, model""",
            (prefix,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- runs
    def start_run(self, run_id: str, day: date) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, date, started_at) VALUES (?,?,?)",
                (run_id, day.isoformat(), _now()),
            )

    def finish_run(self, run_id: str, status: str, body: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at=?, status=?, body=? WHERE run_id=?",
                (_now(), status, json.dumps(body, default=str), run_id),
            )

    def last_runs(self, n: int = 5) -> list[dict]:
        rows = self._conn.execute(
            "SELECT run_id, date, started_at, finished_at, status FROM runs "
            "ORDER BY started_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]

    def last_success(self) -> str | None:
        row = self._conn.execute(
            "SELECT finished_at FROM runs WHERE status IN ('ok','degraded') "
            "ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return row["finished_at"] if row else None

    # ------------------------------------------------------------------- kv
    def get_kv(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
