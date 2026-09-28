"""Hourly weather for a day: from the cache when complete, otherwise from Open-Meteo."""

from __future__ import annotations

from datetime import date, timedelta

from hal9000.solar.clients.openmeteo import WeatherProtocol, hourly_for_day
from hal9000.storage.db import Database

__all__ = ["day_weather", "backfill_weather"]


def day_weather(
    client: WeatherProtocol, db: Database, day: date, today: date, tz: str | None = None
) -> list[dict]:
    """24 hourly buckets for ``day``. Past days are cached once fetched."""
    if day < today - timedelta(days=1):
        cached = db.weather(day)
        if cached and all(row["poa"] is not None for row in cached):
            return cached
    payload = client.fetch(day, day + timedelta(days=1))
    rows = hourly_for_day(payload, day, tz)
    source = "forecast" if day >= today - timedelta(days=1) else "archive"
    db.upsert_weather(day, rows, source)
    return rows


def backfill_weather(
    client: WeatherProtocol, db: Database, start: date, end: date, tz: str | None = None
) -> int:
    """Fetch and cache every day in ``start``..``end`` in one request. Returns days stored."""
    payload = client.fetch(start, end + timedelta(days=1))
    stored = 0
    day = start
    while day <= end:
        try:
            db.upsert_weather(day, hourly_for_day(payload, day, tz), "archive")
            stored += 1
        except Exception:
            pass
        day += timedelta(days=1)
    return stored
