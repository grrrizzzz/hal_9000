"""HAL's evening run: precheck, finalize yesterday, brief, dispatch, complete, announce.

This is the coordinator. It is plain Python on purpose: the order never changes,
so there is nothing for a model to decide here. The one thing that branches --
investigating a shortfall -- lives in the solar agent and its triage sub-agent.

Whatever happens inside the agent, the run ends with a delivered report (the
agent's, or the deterministic template) and a written record. A model that
answers in prose, a cost cap, a missing API key or an exception all land in the
same fallback.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from hal9000 import config
from hal9000.config import HalFeatures, Settings
from hal9000.contracts import RunResult
from hal9000.coordinator.delivery import Delivery
from hal9000.core.memory import Memory
from hal9000.cost.metering import BudgetExceeded, Meter, monthly_precheck
from hal9000.solar import pipeline
from hal9000.solar.agents.briefing import build_briefing
from hal9000.solar.agents.solar import build_solar_agent
from hal9000.solar.agents.triage import make_triage_runner
from hal9000.solar.calibration import maybe_refit
from hal9000.solar.clients.homeassistant import HAClient
from hal9000.solar.clients.openmeteo import OpenMeteoClient
from hal9000.solar.clients.replay import (
    FixtureHA,
    FixtureWeather,
    RecordingHA,
    RecordingWeather,
    load_fixture,
)
from hal9000.solar.clients.telegram import TelegramClient
from hal9000.solar.domain.sitecfg import load_site, load_thresholds
from hal9000.solar.domain.tariff import load_tariff
from hal9000.solar.reporting.fallback import render_fallback
from hal9000.solar.run_context import RunContext, RunFlags
from hal9000.storage.db import Database
from hal9000.storage.run_store import RunStore

__all__ = ["RunOptions", "open_database", "run_evening"]

_LOG = logging.getLogger(__name__)


@dataclass
class RunOptions:
    day: date | None = None
    now: datetime | None = None
    shadow: bool = False
    no_llm: bool = False
    announce: bool = True
    speaker: str | None = None
    replay: str | None = None
    generate_response: Callable | None = None  # tests inject a scripted fake
    clients: dict | None = None  # tests inject {"ha": ..., "weather": ..., "telegram": ...}
    settings: Settings | None = None


def open_database(settings: Settings) -> Database:
    return Database(settings.data_dir / "hal9000.db")


def _clients(opts: RunOptions, settings: Settings, site) -> tuple:
    if opts.clients is not None:
        return (
            opts.clients.get("ha"),
            opts.clients.get("weather"),
            opts.clients.get("telegram"),
            None,
        )
    if opts.replay:
        fixture = load_fixture(opts.replay)
        return FixtureHA(fixture), FixtureWeather(fixture), None, fixture
    ha = (
        RecordingHA(HAClient(settings.ha_base_url, settings.ha_token))
        if settings.ha_configured
        else None
    )
    weather = RecordingWeather(
        OpenMeteoClient(
            site.latitude, site.longitude, site.tilt_deg, site.openmeteo_azimuth, site.timezone
        )
    )
    telegram = TelegramClient(settings.telegram_bot_token) if settings.telegram_configured else None
    return ha, weather, telegram, None


def _finalize_yesterday(run: RunContext) -> None:
    """Re-read yesterday's full day and append its final record (plan §5 step 3)."""
    yesterday = run.day - timedelta(days=1)
    latest = run.db.latest_record(yesterday)
    if latest is not None and latest.kind == "final":
        return
    if run.ha is None:
        return
    sub = replace(run, day=yesterday, store=RunStore(), flags=RunFlags(), trace=[])
    pipeline.ensure_core_steps(sub)
    energy = sub.store.data("energy")
    if energy is None or energy.hours_with_data == 0:
        return
    if latest is not None:
        sub.flags.renderer = latest.renderer
        sub.flags.verdict = {"verdict": latest.verdict} if latest.verdict else None
        sub.flags.alerts_raised = list(latest.alerts_raised)
    record = pipeline.build_record(sub, "final")
    run.db.append_record(record)
    run.note("finalize_yesterday", f"{yesterday.isoformat()} final record saved")


_PRODUCTION_ALERT = ("production", "inverter", "output", "shading", "degradation", "dropout")


def _alert_class(key: str) -> str | None:
    """Which kind of evidence can clear an alert: "sensor", "production" or None."""
    key = key.lower()
    if "net" in key.split("_") or key.startswith("net"):
        return None  # net position is a monthly trend: a good day doesn't clear it
    if "sensor" in key:
        return "sensor"
    if any(word in key for word in _PRODUCTION_ALERT):
        return "production"
    return None  # unknown keys stay open until a person or the agent closes them


def _recovery_notices(run: RunContext) -> None:
    """Close alerts that today's data shows are resolved, and say so.

    Production and inverter alerts close only on a ``normal`` day. Sensor alerts
    close when the sensor reads sanely all day. Net-position and unrecognised
    alerts are never auto-closed.
    """
    band = run.store.data("band")
    energy = run.store.data("energy")
    sensor_ok = (
        energy is not None
        and energy.hours_with_data > 0
        and not energy.sensor_fault
        and not energy.missing_solar_hours
    )
    production_ok = band == "normal" and sensor_ok
    for alert in run.db.open_alerts():
        key = alert["alert_key"]
        if key in run.flags.alerts_raised:
            continue
        kind = _alert_class(key)
        if kind == "production" and production_ok:
            message = f"Resolved: {key.replace('_', ' ')}. Production is back to normal."
        elif kind == "sensor" and sensor_ok:
            message = f"Resolved: {key.replace('_', ' ')}. The solar sensor is reading normally."
        else:
            continue
        run.db.close_alert(key)
        run.flags.alerts_cleared.append(key)
        try:
            run.delivery.send_alert(message, "info")
        except Exception:
            pass


def _save_transcript(run: RunContext, memory: Memory | None, extra: dict, recorders: tuple) -> str:
    runs_dir = run.settings.data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run.run_id}.json"
    body = {
        "run_id": run.run_id,
        "date": run.day.isoformat(),
        "trace": run.trace,
        "figures": run.store.figures(),
        "flags": {k: v for k, v in vars(run.flags).items()},
        "outbox": [vars(item) for item in run.delivery.outbox],
        "memory": memory.items if memory is not None else [],
        "triage_memory": run.store.data("triage_memory", []),
        **extra,
    }
    path.write_text(json.dumps(body, indent=2, default=str))
    ha, weather = recorders
    if isinstance(ha, RecordingHA) or isinstance(weather, RecordingWeather):
        fixture = {
            "date": run.day.isoformat(),
            "now": run.now.isoformat(),
            "ha": ha.log if isinstance(ha, RecordingHA) else [],
            "weather": weather.log if isinstance(weather, RecordingWeather) else [],
        }
        (runs_dir / f"{run.run_id}.fixture.json").write_text(json.dumps(fixture, default=str))
    return str(path)


def _write_health(settings: Settings, db: Database, result: RunResult) -> None:
    today = datetime.now(ZoneInfo(settings.timezone)).date()
    health = {
        "last_run_id": result.run_id,
        "last_status": result.status,
        "last_success": db.last_success(),
        "degraded": result.status == "degraded",
        "month_to_date_usd": round(db.month_cost(today.year, today.month), 4),
    }
    (settings.data_dir / "health.json").write_text(json.dumps(health, indent=2))


def run_evening(opts: RunOptions | None = None) -> RunResult:
    opts = opts or RunOptions()
    settings = opts.settings or config.settings()
    site = load_site(settings.config_dir)
    thresholds = load_thresholds(settings.config_dir)
    tariff = load_tariff(settings.config_dir)
    tz = ZoneInfo(site.timezone)

    ha, weather, telegram, fixture = _clients(opts, settings, site)
    if fixture is not None and opts.now is None and fixture.get("now"):
        now = datetime.fromisoformat(fixture["now"])
    else:
        now = (opts.now or datetime.now(tz)).astimezone(tz)
    day = opts.day or (date.fromisoformat(fixture["date"]) if fixture else now.date())

    features = HalFeatures.from_env()
    if opts.shadow:
        features.shadow_mode = True
    db = open_database(settings)
    run_id = f"{day.isoformat()}-{now.strftime('%H%M%S')}"
    delivery = Delivery(
        settings=settings,
        ha=ha,
        telegram=telegram,
        shadow=features.shadow_mode,
        speaker=opts.speaker,
        announce_enabled=opts.announce,
    )
    policy_path = settings.config_dir / "report_policy.txt"
    run = RunContext(
        day=day,
        now=now,
        run_id=run_id,
        settings=settings,
        site=site,
        thresholds=thresholds,
        tariff=tariff,
        db=db,
        ha=ha,
        weather=weather,
        delivery=delivery,
        features=features,
        report_policy=policy_path.read_text() if policy_path.exists() else "",
    )
    db.start_run(run_id, day)
    if ha is None:
        run.flags.production_missing_reason = "Home Assistant is not configured"
        run.degrade("HA_BASE_URL / HA_TOKEN not set")

    # 1. monthly budget
    monthly = monthly_precheck(db, now.date(), settings.monthly_cap_usd)
    if not monthly.triage_enabled:
        features.triage_enabled = False
    use_agent = not opts.no_llm and not monthly.skip_agent
    if monthly.skip_agent and not opts.no_llm:
        run.degrade(
            f"monthly cap reached (${monthly.spent:.2f} of ${monthly.cap:.2f}); agent skipped"
        )

    # 2. finalize yesterday, and refit the expectation model weekly
    try:
        _finalize_yesterday(run)
    except Exception as error:
        run.note("finalize_yesterday", f"failed: {type(error).__name__}")
    try:
        if maybe_refit(db, site, day):
            run.note("expectation_model", "refit from stored history")
    except Exception as error:
        run.note("expectation_model", f"refit failed: {type(error).__name__}")

    # 3. the agent
    memory: Memory | None = None
    meter = Meter(
        db, run_id, settings.run_soft_cap_usd, settings.run_hard_cap_usd, run.flags, tz=tz
    )
    iterations = 0
    if use_agent:
        generate = opts.generate_response
        if generate is None:
            try:
                config.require_llm_key()
                from hal9000.llm import generate_response as generate
            except config.ConfigError as error:
                run.degrade(f"no LLM available ({error}); deterministic report used")
                generate = None
        if generate is not None:
            props = {"run": run}
            run.triage_runner = make_triage_runner(run, meter.wrap(generate, "triage"), props)
            agent = build_solar_agent(run, meter.wrap(generate, "solar"))
            memory = Memory()
            try:
                agent.run(build_briefing(run), memory=memory, action_context_props=props)
            except BudgetExceeded as error:
                run.degrade(str(error))
            except Exception as error:
                _LOG.exception("solar agent failed")
                run.degrade(f"agent error: {type(error).__name__}: {error}")
            iterations = sum(1 for item in memory.items if item.get("type") == "assistant")
            if not run.flags.delivered:
                run.degrade("agent finished without a guarded report")

    # 4. completeness: always a report, always a record
    if not run.flags.delivered:
        pipeline.ensure_core_steps(run)
        run.flags.report_text = render_fallback(run)
        run.flags.renderer = "fallback"
        run.flags.delivered = True
        run.note("report", "deterministic template")
    if run.store.data("energy") is None:
        pipeline.ensure_core_steps(run)
    _recovery_notices(run)
    try:
        run.db.append_record(pipeline.build_record(run, "provisional"))
        run.flags.record_written = True
    except Exception as error:
        run.degrade(f"record not written: {error}")

    # 5. announce
    announce_status = run.delivery.announce(run.flags.report_text)
    run.note("announce", announce_status)
    intentionally_silent = announce_status == "logged"  # shadow mode or --no-announce
    announced = announce_status in ("played", "telegram")
    if announce_status == "telegram":
        run.degrade("speaker unavailable; report sent to Telegram instead")
    elif not announced and not intentionally_silent:
        run.degrade(f"report was not delivered anywhere (announce: {announce_status})")

    # 6. books
    try:
        cost = db.run_cost(run_id)
    except Exception:
        cost = 0.0
    status = "ok" if not run.flags.degraded_reasons else "degraded"
    try:
        transcript = _save_transcript(
            run, memory, {"cost_usd": cost, "announce": announce_status}, (ha, weather)
        )
    except Exception as error:
        _LOG.warning("transcript not saved: %s", error)
        transcript = None
    result = RunResult(
        run_id=run_id,
        date=day.isoformat(),
        status=status,
        report_delivered=run.flags.delivered,
        record_written=run.flags.record_written,
        announced=announced or intentionally_silent,
        renderer=run.flags.renderer,
        report_text=run.flags.report_text,
        band=run.store.data("band"),
        cost_usd=round(cost, 5),
        iterations=iterations,
        degraded_reasons=run.flags.degraded_reasons,
        transcript_path=transcript,
        trace=run.trace,
    )
    try:
        db.finish_run(run_id, status, result.model_dump())
    except Exception as error:
        _LOG.warning("run not recorded as finished: %s", error)
    try:
        _write_health(settings, db, result)
    except Exception:
        pass
    if settings.deadman_ping_url and status in ("ok", "degraded"):
        try:
            httpx.get(settings.deadman_ping_url, timeout=10)
        except Exception:
            pass
    return result
