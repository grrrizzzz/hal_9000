"""The deterministic steps of an evening run.

Agent tools call these, and so does HAL's fallback path. Each step writes its
figures to the run store (formatted for speech) and returns a compact summary.
Nothing here calls an LLM, and every number that can reach the household is
computed here.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from hal9000.contracts import NightlyRecord
from hal9000.solar.domain import fmt
from hal9000.solar.domain.anomaly import band_for, classify_profile
from hal9000.solar.domain.clearsky import clearsky_poa_hourly
from hal9000.solar.domain.expectation import ExpectationModel, bootstrap_model
from hal9000.solar.domain.net_position import day_net_position, window_net_position
from hal9000.solar.domain.savings import compute_savings
from hal9000.solar.domain.tariff import TariffNotConfigured
from hal9000.solar.domain.tou_calendar import PERIODS
from hal9000.solar.energy import DayEnergy, fetch_day_energy
from hal9000.solar.run_context import RunContext
from hal9000.solar.weather import day_weather

__all__ = [
    "anomaly_profile",
    "assess_performance",
    "build_record",
    "ensure_core_steps",
    "expectation_model",
    "gather_energy",
    "gather_weather",
    "net_position",
    "recent_history",
    "savings",
]

MODEL_KEY = "expectation_model"
PERIOD_LABEL = {"on_peak": "on-peak", "off_peak": "off-peak", "super_off_peak": "super off-peak"}


def expectation_model(run: RunContext) -> ExpectationModel:
    raw = run.db.get_kv(MODEL_KEY)
    if raw:
        return ExpectationModel.from_dict(json.loads(raw))
    return bootstrap_model(run.site)


# ---------------------------------------------------------------- energy
def gather_energy(run: RunContext, day: date | None = None, store: bool = True) -> DayEnergy:
    """Read the day's solar, import and export from Home Assistant and store figures."""
    day = day or run.day
    if run.ha is None:
        raise RuntimeError("Home Assistant client is not configured")
    energy = fetch_day_energy(run.ha, day, run.settings, run.site, run.now)
    rows = [
        {"solar": s, "import": i, "export": e, "flags": []}
        for s, i, e in zip(energy.solar, energy.grid_import, energy.grid_export)
    ]
    if energy.hours_with_data > 0:
        run.db.upsert_hourly(day, rows, complete=energy.complete)
    if not store:
        return energy
    totals = energy.totals()
    s = run.store
    s.put_data("energy", energy)
    s.put("production_kwh", fmt.kwh(totals["solar"]))
    s.put("grid_import_kwh", fmt.kwh(totals["import"]))
    s.put("grid_export_kwh", fmt.kwh(totals["export"]))
    s.put("household_kwh", fmt.kwh(totals["household"]))
    peak_hour = max(range(24), key=lambda h: energy.solar[h])
    s.put("peak_hour_kwh", fmt.kwh(energy.solar[peak_hour]))
    s.put("peak_hour", f"{peak_hour}:00")
    if energy.hours_with_data == 0:
        run.flags.production_missing_reason = "Home Assistant returned no solar data for today"
    return energy


def energy_summary(run: RunContext) -> dict:
    energy: DayEnergy = run.store.data("energy")
    s = run.store
    summary = {
        "date": run.day.isoformat(),
        "production_kwh": s.figure("production_kwh"),
        "grid_import_kwh": s.figure("grid_import_kwh"),
        "grid_export_kwh": s.figure("grid_export_kwh"),
        "household_kwh": s.figure("household_kwh"),
        "peak_hour": s.figure("peak_hour"),
        "peak_hour_kwh": s.figure("peak_hour_kwh"),
        "data_through": energy.through.strftime("%H:%M"),
        "consumption_complete": energy.complete,
        "sensor_fault": energy.sensor_fault,
        "sensor_fault_hours": energy.sensor_fault_hours,
        "data_artifacts": energy.artifacts,
        "transition_day": energy.transition_day,
    }
    if energy.sensor_fault:
        summary["note"] = (
            "Solar reads below grid export in some hours, which is impossible. The solar "
            "sensor is faulty; report the sensor problem, not a performance shortfall, "
            "and do not run triage."
        )
    if run.flags.production_missing_reason:
        summary["note"] = run.flags.production_missing_reason
    return summary


# --------------------------------------------------------------- weather
def gather_weather(run: RunContext, day: date | None = None, store: bool = True) -> list[dict]:
    day = day or run.day
    if run.weather is None:
        raise RuntimeError("Open-Meteo client is not configured")
    rows = day_weather(run.weather, run.db, day, run.now.date())
    if store:
        insolation = sum(r["poa"] for r in rows)
        clear = sum(clearsky_poa_hourly(run.site, day))
        clouds = [r["cloud"] for r in rows[8:17] if r["cloud"] is not None]
        mean_cloud = sum(clouds) / len(clouds) if clouds else None
        run.store.put_data("weather", rows)
        run.store.put("insolation_kwh_m2", fmt.kwh(insolation))
        run.store.put("clearsky_insolation_kwh_m2", fmt.kwh(clear))
        run.store.put("sky_clearness_pct", fmt.percent(insolation / clear if clear else 0))
        if mean_cloud is not None:
            run.store.put("daytime_cloud_cover_pct", f"{round(mean_cloud)}%")
    return rows


def weather_summary(run: RunContext) -> dict:
    s = run.store
    clearness = s.figure("sky_clearness_pct") or "0%"
    value = int(clearness.rstrip("%"))
    sky = "clear" if value >= 85 else "partly cloudy" if value >= 55 else "overcast"
    return {
        "insolation_kwh_m2": s.figure("insolation_kwh_m2"),
        "clearsky_insolation_kwh_m2": s.figure("clearsky_insolation_kwh_m2"),
        "sky_clearness_pct": clearness,
        "daytime_cloud_cover_pct": s.figure("daytime_cloud_cover_pct"),
        "sky": sky,
    }


# ----------------------------------------------------------- performance
def assess_performance(run: RunContext) -> dict:
    energy: DayEnergy | None = run.store.data("energy")
    weather = run.store.data("weather")
    if energy is None:
        raise RuntimeError("Call get_home_energy first.")
    if energy.hours_with_data == 0:
        run.store.put_data("band", "no_data")
        return {
            "band": "no_data",
            "note": "There is no production data, so performance can't be assessed. "
            "Missing data is not zero production.",
        }
    if energy.sensor_fault:
        run.store.put_data("band", "sensor_fault")
        return {
            "band": "sensor_fault",
            "note": "Solar sensor reads below export; performance can't be assessed.",
        }
    if weather is None:
        run.store.put_data("band", "unassessed")
        return {
            "band": "unassessed",
            "note": run.flags.weather_missing_reason or "No weather data.",
        }
    model = expectation_model(run)
    through_hour = energy.through.hour if not energy.complete else 24
    poa = [r["poa"] for r in weather]
    temp = [r["temp"] for r in weather]
    expected_hourly = model.predict_hourly(poa, temp)
    # compare like with like: only hours the production data covers
    expected_hourly = [e if h < through_hour else 0.0 for h, e in enumerate(expected_hourly)]
    expected = sum(expected_hourly)
    actual = sum(energy.solar)
    band, ratio = band_for(actual, expected, run.thresholds)
    s = run.store
    s.put_data("expected_hourly", expected_hourly)
    s.put_data("band", band)
    s.put_data("ratio", ratio)
    s.put("expected_kwh", fmt.kwh(expected))
    if band != "unassessed":
        s.put("performance_pct", fmt.percent(ratio))
    return {
        "production_kwh": s.figure("production_kwh"),
        "expected_kwh": s.figure("expected_kwh"),
        "performance_pct": s.figure("performance_pct"),
        "band": band,
        "model": model.kind,
        "model_calibrated_on": model.calibrated_on,
        "band_meaning": {
            "normal": "output matched what the weather allowed",
            "low": "materially below expectation: look at the hourly profile and history",
            "severe": "far below expectation: investigate, triage if the cause is unclear",
            "zero": "no meaningful output: the inverter is the leading suspect",
            "unassessed": "too little sun expected to judge",
        }.get(band, ""),
    }


# ---------------------------------------------------------------- savings
def savings(run: RunContext) -> dict:
    energy: DayEnergy | None = run.store.data("energy")
    if energy is None:
        raise RuntimeError("Call get_home_energy first.")
    try:
        result = compute_savings(run.day, energy.solar, run.tariff)
    except TariffNotConfigured as error:
        run.flags.savings_absent_reason = "rates are not configured for this date"
        run.store.put_data("savings", None)
        raise error
    s = run.store
    s.put_data("savings", result)
    s.put("savings_usd", fmt.dollars(result.total_usd))
    for period in PERIODS:
        s.put(f"savings_{period}_usd", fmt.dollars(result.by_period_usd[period]))
        s.put(f"production_{period}_kwh", fmt.kwh(result.kwh_by_period[period]))
        s.put(f"rate_{period}_usd_per_kwh", fmt.dollars(result.rates[period]))
    return {
        "savings_usd": s.figure("savings_usd"),
        "by_period": {
            PERIOD_LABEL[p]: {
                "kwh": s.figure(f"production_{p}_kwh"),
                "usd": s.figure(f"savings_{p}_usd"),
                "rate": s.figure(f"rate_{p}_usd_per_kwh"),
            }
            for p in PERIODS
        },
        "schedule_version": result.schedule_version,
        "provisional_rates": result.provisional,
    }


# ----------------------------------------------------------- net position
def net_position(run: RunContext) -> dict:
    """Yesterday's finalized day, plus a rolling 30-day view for trend warnings."""
    yesterday = run.day - timedelta(days=1)
    rows = run.db.hourly(yesterday)
    if rows is None or not all(r["complete"] for r in rows):
        return {"available": False, "note": "Yesterday's full-day data is not available yet."}
    cal = run.tariff.calendar
    warn = run.thresholds.net_position_warn_fraction
    imp = [r["import"] or 0.0 for r in rows]
    exp = [r["export"] or 0.0 for r in rows]
    daily = day_net_position(yesterday, imp, exp, cal, warn)
    window = []
    for back in range(30):
        d = yesterday - timedelta(days=back)
        r = run.db.hourly(d)
        if r is not None and all(x["complete"] for x in r):
            window.append((d, [x["import"] or 0.0 for x in r], [x["export"] or 0.0 for x in r]))
    rolling = window_net_position(window, cal, warn)
    s = run.store
    for p in PERIODS:
        s.put(f"net_{p}_kwh_yesterday", fmt.kwh(daily.net_kwh[p]))
        s.put(f"net_{p}_kwh_30d", fmt.kwh(rolling.net_kwh[p]))
    s.put_data("net_position_yesterday", daily.net_kwh)
    s.put_data(
        "net_trending",
        [PERIOD_LABEL[p] for p in rolling.trending_export + rolling.net_export_periods],
    )
    return {
        "available": True,
        "yesterday": yesterday.isoformat(),
        "net_import_kwh_yesterday": {
            PERIOD_LABEL[p]: s.figure(f"net_{p}_kwh_yesterday") for p in PERIODS
        },
        "net_import_kwh_last_30_days": {
            PERIOD_LABEL[p]: s.figure(f"net_{p}_kwh_30d") for p in PERIODS
        },
        "days_in_window": len(window),
        "net_export_periods_30d": [PERIOD_LABEL[p] for p in rolling.net_export_periods],
        "trending_toward_export_30d": [PERIOD_LABEL[p] for p in rolling.trending_export],
        "note": (
            "Positive = net import. The savings model needs every period to stay net import "
            "over a billing cycle; a single net-export day is normal."
        ),
    }


# ---------------------------------------------------------------- anomaly
def anomaly_profile(run: RunContext) -> dict:
    energy: DayEnergy | None = run.store.data("energy")
    expected_hourly = run.store.data("expected_hourly")
    if energy is None or expected_hourly is None:
        raise RuntimeError("Call get_home_energy and assess_performance first.")
    profile = classify_profile(energy.solar, expected_hourly)
    run.store.put_data("anomaly", profile.kind)
    if profile.worst_hour is not None:
        run.store.put("anomaly_hour", f"{profile.worst_hour}:00")
    hours = {
        f"{h}:00": {
            "actual_kwh": fmt.kwh(energy.solar[h]),
            "expected_kwh": fmt.kwh(expected_hourly[h]),
        }
        for h in range(24)
        if expected_hourly[h] > 0.02 or energy.solar[h] > 0.02
    }
    for h in range(24):
        if expected_hourly[h] > 0.02 or energy.solar[h] > 0.02:
            run.store.put(f"hour_{h}_actual_kwh", fmt.kwh(energy.solar[h]))
            run.store.put(f"hour_{h}_expected_kwh", fmt.kwh(expected_hourly[h]))
    return {"profile": profile.kind, "description": profile.description, "hours": hours}


# ---------------------------------------------------------------- history
def recent_history(run: RunContext, days: int = 7) -> dict:
    days = max(1, min(int(days), 60))
    records = run.db.recent_records(run.day, days)
    rows = []
    produced = []
    for record in records:
        key = f"history_{record.date}"
        if record.production_kwh is not None:
            run.store.put(f"{key}_production_kwh", fmt.kwh(record.production_kwh))
            produced.append(record.production_kwh)
        if record.ratio is not None and record.band not in ("unassessed", "sensor_fault"):
            run.store.put(f"{key}_performance_pct", fmt.percent(record.ratio))
        rows.append(
            {
                "date": record.date,
                "kind": record.kind,
                "production_kwh": run.store.figure(f"{key}_production_kwh"),
                "performance_pct": run.store.figure(f"{key}_performance_pct"),
                "band": record.band,
                "verdict": record.verdict,
            }
        )
    if produced:
        run.store.put(f"history_{days}d_avg_production_kwh", fmt.kwh(sum(produced) / len(produced)))
    low_days = sum(1 for r in rows if r["band"] in ("low", "severe", "zero"))
    return {
        "days_requested": days,
        "days_found": len(rows),
        "average_production_kwh": run.store.figure(f"history_{days}d_avg_production_kwh"),
        "low_or_worse_days": low_days,
        "days": rows,
    }


# ---------------------------------------------------------------- records
def build_record(run: RunContext, kind: str = "provisional") -> NightlyRecord:
    s = run.store
    energy: DayEnergy | None = s.data("energy")
    weather = s.data("weather")
    savings_result = s.data("savings")
    model = expectation_model(run)
    totals = energy.totals() if energy else {}
    band = s.data("band") or ("unassessed" if energy else "no_data")
    verdict = run.flags.verdict or {}
    return NightlyRecord(
        date=run.day.isoformat(),
        kind=kind,
        production_kwh=totals.get("solar") if energy and energy.hours_with_data else None,
        expected_kwh=sum(s.data("expected_hourly")) if s.data("expected_hourly") else None,
        ratio=s.data("ratio"),
        band=band,
        model_kind=model.kind,
        calibration_date=model.calibrated_on,
        savings_usd=savings_result.total_usd if savings_result else None,
        savings_by_period=savings_result.by_period_usd if savings_result else None,
        savings_absent_reason=None if savings_result else run.flags.savings_absent_reason,
        tariff_schedule_version=savings_result.schedule_version if savings_result else None,
        import_kwh=totals.get("import"),
        export_kwh=totals.get("export"),
        household_kwh=totals.get("household"),
        net_position=s.data("net_position_yesterday"),
        insolation_kwh_m2=sum(r["poa"] for r in weather) if weather else None,
        anomaly=s.data("anomaly"),
        verdict=verdict.get("verdict"),
        alerts_raised=list(run.flags.alerts_raised),
        alerts_cleared=list(run.flags.alerts_cleared),
        renderer=run.flags.renderer,
        degraded=bool(run.flags.degraded_reasons),
        sensor_fault=bool(energy and energy.sensor_fault),
        transition_day=bool(energy and energy.transition_day),
    )


# --------------------------------------------------- deterministic fill-in
def ensure_core_steps(run: RunContext) -> None:
    """Run whatever core step hasn't run yet. Used by the fallback and --no-llm paths."""
    s = run.store
    if s.data("energy") is None and run.flags.production_missing_reason is None:
        try:
            gather_energy(run)
            run.note("get_home_energy", f"{s.figure('production_kwh')} kWh")
        except Exception as error:
            run.flags.production_missing_reason = "Home Assistant could not be reached"
            run.degrade(f"production data unavailable: {error}")
            run.note("get_home_energy", "failed")
            return
    if s.data("weather") is None and run.flags.weather_missing_reason is None:
        try:
            gather_weather(run)
            run.note("get_weather", f"{s.figure('insolation_kwh_m2')} kWh/m2")
        except Exception as error:
            run.flags.weather_missing_reason = f"weather unavailable ({type(error).__name__})"
            run.degrade("weather unavailable")
            run.note("get_weather", "failed")
    if s.data("band") is None:
        try:
            result = assess_performance(run)
            run.note("assess_performance", result.get("band", ""))
        except Exception as error:
            run.degrade(f"performance not assessed ({error})")
    if s.data("savings") is None and run.flags.savings_absent_reason is None:
        try:
            savings(run)
            run.note("compute_savings", s.figure("savings_usd") or "")
        except TariffNotConfigured:
            run.note("compute_savings", "tariff not configured")
        except Exception as error:
            run.flags.savings_absent_reason = f"savings could not be computed ({error})"
    if s.data("net_position_yesterday") is None:
        try:
            net_position(run)
        except Exception:
            pass
    if s.data("anomaly") is None and s.data("expected_hourly") is not None:
        try:
            anomaly_profile(run)
        except Exception:
            pass
