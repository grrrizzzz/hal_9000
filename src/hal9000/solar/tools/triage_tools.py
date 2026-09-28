"""The triage sub-agent's tools: read-only diagnostics, and the verdict.

Triage can look at anything about today's production but cannot notify, write
records or deliver the report. Everything it may want spoken must already be a
figure a tool stored; ``submit_verdict`` only accepts figure keys that exist.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from hal9000.contracts import VERDICT_JSON_SCHEMA, Verdict
from hal9000.solar import pipeline
from hal9000.solar.domain import fmt
from hal9000.solar.domain.clearsky import clearsky_poa_hourly
from hal9000.solar.run_context import RunContext
from hal9000.solar.tools.solar_tools import ToolInputError
from hal9000.tools.decorators import register_tool

__all__ = [
    "compare_to_clearsky",
    "find_similar_days",
    "get_intraday_curve",
    "get_irradiance_curve",
    "get_sensor_history",
    "read_handover",
    "submit_verdict",
]

_HANDOVER_KEYS = (
    "production_kwh",
    "expected_kwh",
    "performance_pct",
    "peak_hour",
    "peak_hour_kwh",
    "grid_export_kwh",
    "grid_import_kwh",
    "insolation_kwh_m2",
    "clearsky_insolation_kwh_m2",
    "sky_clearness_pct",
    "daytime_cloud_cover_pct",
    "anomaly_hour",
)


@register_tool(tags=["triage"])
def read_handover(_run: RunContext = None) -> dict:
    """Start here: what the solar agent has already established, and why you were called."""
    run = _run
    energy = run.store.data("energy")
    return {
        "date": run.day.isoformat(),
        "reason": run.store.data("triage_reason"),
        "band": run.store.data("band"),
        "anomaly_profile": run.store.data("anomaly"),
        "figures": {k: run.store.figure(k) for k in _HANDOVER_KEYS if run.store.has(k)},
        "sensor_fault_hours": energy.sensor_fault_hours if energy else None,
        "data_artifacts": energy.artifacts if energy else None,
        "site_notes": (
            "2.47 kW DC array, ~11 years old, transformerless Delta SOLIVIA inverter "
            "(~2014, out of warranty, in its typical failure window). Panels rarely fail as "
            "a group; zero or flat-lined output points at the inverter."
        ),
    }


@register_tool(tags=["telemetry"])
def get_intraday_curve(resolution_min: int = 15, _run: RunContext = None) -> dict:
    """Today's solar power through the day, to spot dropouts, cutoffs or flat-lining.

    Args:
        resolution_min: 5, 15 or 60 minutes per point.
    """
    run = _run
    if resolution_min not in (5, 15, 60):
        raise ToolInputError("resolution_min must be 5, 15 or 60")
    tz = ZoneInfo(run.site.timezone)
    start = datetime(run.day.year, run.day.month, run.day.day, tzinfo=tz)
    entity = run.settings.entity_solar
    series = run.ha.statistics([entity], start, start + timedelta(days=1), "5minute").get(
        entity, []
    )
    step = 5
    note = None
    if not series:
        # Home Assistant purges 5-minute statistics after about 10 days
        series = run.ha.statistics([entity], start, start + timedelta(days=1), "hour").get(
            entity, []
        )
        step = 60
        resolution_min = 60
        note = "5-minute data has been purged for this date; hourly points only."
    if not series:
        return {
            "curve": {},
            "note": "No solar statistics exist for this date. That is missing data, "
            "not zero output.",
        }
    fixed = run.settings.solar_fixed_at
    ceiling = run.site.dc_kw * 1.2 * step / 60
    buckets: dict[str, float] = {}
    zero_minutes = 0
    for moment, change in series:
        local = moment.astimezone(tz)
        if fixed is not None and local < fixed:
            change *= run.settings.solar_scale
        if change > ceiling or change < 0:
            continue
        if 7 <= local.hour < 17 and change <= 0.001:
            zero_minutes += step
        slot = (
            local.replace(minute=(local.minute // resolution_min) * resolution_min)
            if resolution_min < 60
            else local.replace(minute=0)
        )
        key = slot.strftime("%H:%M")
        buckets[key] = buckets.get(key, 0.0) + change
    curve = {k: round(v * 60 / resolution_min, 2) for k, v in sorted(buckets.items()) if v > 0.0005}
    max_kw = max(curve.values(), default=0.0)
    run.store.put("triage_max_kw", fmt.plain(round(max_kw, 1)))
    run.store.put("triage_zero_minutes_7_to_17", fmt.plain(zero_minutes))
    return {
        **({"note": note} if note else {}),
        "unit": "kW average per interval",
        "curve": curve,
        "max_kw": run.store.figure("triage_max_kw"),
        "minutes_with_zero_output_between_7_and_17": run.store.figure(
            "triage_zero_minutes_7_to_17"
        ),
    }


@register_tool(tags=["telemetry"])
def get_sensor_history(sensor: str, days: int = 14, _run: RunContext = None) -> dict:
    """Daily totals for one of the energy sensors over recent days (from Home Assistant).

    Args:
        sensor: "solar", "grid_import" or "grid_export".
        days: How many days back (1-60).
    """
    run = _run
    entity = {
        "solar": run.settings.entity_solar,
        "grid_import": run.settings.entity_grid_import,
        "grid_export": run.settings.entity_grid_export,
    }.get(sensor)
    if entity is None:
        raise ToolInputError('sensor must be "solar", "grid_import" or "grid_export"')
    days = max(1, min(int(days), 60))
    tz = ZoneInfo(run.site.timezone)
    end = datetime(run.day.year, run.day.month, run.day.day, tzinfo=tz)
    start = end - timedelta(days=days)
    points = run.ha.statistics([entity], start, end, "hour").get(entity, [])
    daily: dict[str, float] = {}
    fixed = run.settings.solar_fixed_at
    for moment, change in points:
        local = moment.astimezone(tz)
        if sensor == "solar" and fixed is not None and local < fixed:
            change *= run.settings.solar_scale
        if change < 0 or (sensor == "solar" and change > run.site.dc_kw * 1.2):
            continue
        key = local.date().isoformat()
        daily[key] = daily.get(key, 0.0) + change
    out = {}
    for key, value in sorted(daily.items()):
        out[key] = run.store.put(f"triage_{sensor}_{key}_kwh", fmt.kwh(value))
    return {"sensor": sensor, "daily_kwh": out}


@register_tool(tags=["weather"])
def get_irradiance_curve(_run: RunContext = None) -> dict:
    """Hour-by-hour sunlight on the panels today versus a clear sky, with cloud cover."""
    run = _run
    weather = run.store.data("weather")
    if weather is None:
        weather = pipeline.gather_weather(run)
    clear = clearsky_poa_hourly(run.site, run.day)
    hours = {}
    for h in range(24):
        if clear[h] > 0.01 or weather[h]["poa"] > 0.01:
            hours[f"{h}:00"] = {
                "poa_kwh_m2": round(weather[h]["poa"], 3),
                "clearsky_kwh_m2": round(clear[h], 3),
                "cloud_cover_pct": weather[h]["cloud"],
            }
    return {"hours": hours}


@register_tool(tags=["triage"])
def compare_to_clearsky(_run: RunContext = None) -> dict:
    """Actual production per hour against what a clear sky would have given.

    Hours where actual matches clear-sky but earlier or later hours are near
    zero point to shading or an inverter cutoff; uniformly low hours with
    cloud point to weather.
    """
    run = _run
    energy = run.store.data("energy")
    if energy is None:
        raise ToolInputError("No production data in the handover.")
    model = pipeline.expectation_model(run)
    clear = clearsky_poa_hourly(run.site, run.day)
    clear_expected = model.predict_hourly(clear, [20.0] * 24)
    hours = {}
    for h in range(24):
        if clear_expected[h] > 0.02:
            ratio = energy.solar[h] / clear_expected[h]
            hours[f"{h}:00"] = {
                "actual_kwh": fmt.kwh(energy.solar[h]),
                "clear_sky_kwh": fmt.kwh(clear_expected[h]),
                "ratio": round(ratio, 2),
            }
    total = sum(clear_expected)
    run.store.put("triage_clearsky_expected_kwh", fmt.kwh(total))
    return {
        "clear_sky_expected_kwh": run.store.figure("triage_clearsky_expected_kwh"),
        "hours": hours,
    }


@register_tool(tags=["triage"])
def find_similar_days(n: int = 5, _run: RunContext = None) -> dict:
    """Past days with similar sunlight, and how the array did on them.

    If similar days produced far more, today's shortfall is not the weather.

    Args:
        n: How many similar days to return (1-10).
    """
    run = _run
    today_insolation = sum(r["poa"] for r in (run.store.data("weather") or []))
    if today_insolation <= 0:
        return {"similar_days": [], "note": "No insolation data for today."}
    candidates = [
        r
        for r in run.db.all_latest_records()
        if r.date != run.day.isoformat()
        and r.insolation_kwh_m2
        and r.production_kwh is not None
        and not r.sensor_fault
    ]
    candidates.sort(key=lambda r: abs(r.insolation_kwh_m2 - today_insolation))
    out = []
    for record in candidates[: max(1, min(int(n), 10))]:
        key = f"similar_{record.date}"
        out.append(
            {
                "date": record.date,
                "insolation_kwh_m2": run.store.put(
                    f"{key}_insolation", fmt.kwh(record.insolation_kwh_m2)
                ),
                "production_kwh": run.store.put(
                    f"{key}_production_kwh", fmt.kwh(record.production_kwh)
                ),
                "band": record.band,
            }
        )
    return {"today_insolation_kwh_m2": fmt.kwh(today_insolation), "similar_days": out}


@register_tool(tags=["verdict"], parameters_override=VERDICT_JSON_SCHEMA)
def submit_verdict(
    verdict: str,
    confidence: str,
    evidence: list,
    recommended_alert: dict,
    figure_keys: list | None = None,
    _run: RunContext = None,
) -> dict:
    """Finish triage with your conclusion. This ends your investigation.

    Choose "inconclusive" rather than guessing. Every evidence item names the
    tool it came from. ``figure_keys`` lists figures the parent may speak; only
    keys your tools actually returned are accepted.
    """
    run = _run
    try:
        parsed = Verdict.model_validate(
            {
                "verdict": verdict,
                "confidence": confidence,
                "evidence": evidence,
                "recommended_alert": recommended_alert,
                "figure_keys": figure_keys or [],
            }
        )
    except ValidationError as error:
        raise ToolInputError(f"Verdict is malformed: {error.errors()[:3]}") from None
    unknown = [k for k in parsed.figure_keys if not run.store.has(k)]
    if unknown:
        raise ToolInputError(
            f"These figure keys were not returned by any tool: {unknown}. Remove them or "
            f"use keys from tool results."
        )
    result = parsed.model_dump()
    result["figures"] = {k: run.store.figure(k) for k in parsed.figure_keys}
    run.store.put_data("triage_verdict", result)
    return {"accepted": True}
