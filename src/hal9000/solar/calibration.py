"""Backfill history from Home Assistant and Open-Meteo, and fit the expectation model."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from hal9000.config import Settings
from hal9000.solar.clients.homeassistant import HAProtocol
from hal9000.solar.clients.openmeteo import WeatherProtocol
from hal9000.solar.domain.expectation import MIN_FIT_DAYS, ExpectationModel, fit_model
from hal9000.solar.domain.sitecfg import Site
from hal9000.solar.energy import fold_statistics
from hal9000.solar.pipeline import MODEL_KEY
from hal9000.solar.weather import backfill_weather
from hal9000.storage.db import Database

__all__ = ["backfill_energy", "calibrate", "fit_from_db", "maybe_refit"]

CHUNK_DAYS = 30
REFIT_EVERY_DAYS = 7


def backfill_energy(
    ha: HAProtocol, db: Database, settings: Settings, site: Site, start: date, end: date
) -> int:
    """Store complete hourly energy for every day in ``start``..``end``. Returns days stored."""
    tz = ZoneInfo(site.timezone)
    entities = [settings.entity_solar, settings.entity_grid_import, settings.entity_grid_export]
    stored = 0
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end)
        window_start = datetime(chunk_start.year, chunk_start.month, chunk_start.day, tzinfo=tz)
        window_end = datetime(
            chunk_end.year, chunk_end.month, chunk_end.day, tzinfo=tz
        ) + timedelta(days=1)
        points = ha.statistics(entities, window_start, window_end, "hour")
        day = chunk_start
        while day <= chunk_end:
            energy = fold_statistics(points, day, settings, site, "hour")
            if energy.hours_with_data >= 20:
                flags = []
                if energy.sensor_fault:
                    flags.append("sensor_fault")
                if energy.transition_day:
                    flags.append("transition_day")
                rows = [
                    {"solar": s, "import": i, "export": e, "flags": flags}
                    for s, i, e in zip(energy.solar, energy.grid_import, energy.grid_export)
                ]
                db.upsert_hourly(day, rows, complete=True)
                stored += 1
            day += timedelta(days=1)
        chunk_start = chunk_end + timedelta(days=1)
    return stored


def fit_from_db(
    db: Database, site: Site, start: date, end: date, today: date
) -> ExpectationModel | None:
    days = []
    for iso in db.hourly_days(start, end):
        day = date.fromisoformat(iso)
        rows = db.hourly(day)
        weather = db.weather(day)
        if rows is None or weather is None:
            continue
        flags = rows[0]["flags"]
        if "sensor_fault" in flags or "transition_day" in flags:
            continue
        days.append(
            (
                day,
                [w["poa"] or 0.0 for w in weather],
                [w["temp"] if w["temp"] is not None else 20.0 for w in weather],
                [r["solar"] or 0.0 for r in rows],
            )
        )
    model = fit_model(days, site, min_days=MIN_FIT_DAYS, today=today)
    if model is not None:
        db.set_kv(MODEL_KEY, json.dumps(model.to_dict()))
    return model


def maybe_refit(db: Database, site: Site, today: date) -> ExpectationModel | None:
    """Refit from stored data about once a week. Cheap: no network."""
    raw = db.get_kv(MODEL_KEY)
    if raw:
        current = json.loads(raw)
        calibrated = current.get("calibrated_on")
        if calibrated and (today - date.fromisoformat(calibrated)).days < REFIT_EVERY_DAYS:
            return None
    return fit_from_db(db, site, today - timedelta(days=365), today - timedelta(days=1), today)


def calibrate(
    ha: HAProtocol,
    weather: WeatherProtocol,
    db: Database,
    settings: Settings,
    site: Site,
    days: int,
    today: date,
) -> dict:
    end = today - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    energy_days = backfill_energy(ha, db, settings, site, start, end)
    weather_days = 0
    chunk = start
    while chunk <= end:
        chunk_end = min(chunk + timedelta(days=89), end)
        weather_days += backfill_weather(weather, db, chunk, chunk_end)
        chunk = chunk_end + timedelta(days=1)
    model = fit_from_db(db, site, start, end, today)
    return {
        "energy_days": energy_days,
        "weather_days": weather_days,
        "model": model.to_dict() if model else None,
        "note": None
        if model
        else f"Fewer than {MIN_FIT_DAYS} usable days; still on the bootstrap model.",
    }
