"""The briefing seeded as the solar agent's first memory: what it always needs."""

from __future__ import annotations

import json

from hal9000.solar.domain import fmt
from hal9000.solar.run_context import RunContext

__all__ = ["build_briefing"]


def build_briefing(run: RunContext) -> str:
    yesterday = run.db.recent_records(run.day, 1)
    week = run.db.recent_records(run.day, 7)
    briefing: dict = {
        "run_date": run.day.isoformat(),
        "weekday": run.day.strftime("%A"),
        "run_time": run.now.strftime("%H:%M"),
        "open_alerts": [
            {"alert_key": a["alert_key"], "severity": a["severity"], "since": a["opened_at"][:10]}
            for a in run.db.open_alerts()
        ],
    }
    if yesterday:
        y = yesterday[0]
        briefing["yesterday"] = {
            "date": y.date,
            "production_kwh": run.store.put("yesterday_production_kwh", fmt.kwh(y.production_kwh))
            if y.production_kwh is not None
            else None,
            "band": y.band,
            "verdict": y.verdict,
        }
    produced = [r.production_kwh for r in week if r.production_kwh is not None]
    if produced:
        briefing["last_7_days_avg_production_kwh"] = run.store.put(
            "last_7_days_avg_production_kwh", fmt.kwh(sum(produced) / len(produced))
        )
        briefing["last_7_days_bands"] = [r.band for r in week]
    return (
        f"Evening solar run for {run.day.isoformat()}. Briefing:\n"
        + json.dumps(briefing, indent=2)
        + "\nBegin with get_home_energy."
    )
