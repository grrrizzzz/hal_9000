"""The solar agent's tools. Docstrings are prompts: they are all the model reads.

Every tool gets the run through the hidden ``_run`` parameter. None of them
accepts a number from the model that is later reported: figures are computed
here and stored in the run store, and the report is checked against that store.
"""

from __future__ import annotations

from hal9000.solar import pipeline
from hal9000.solar.reporting.fallback import render_fallback
from hal9000.solar.reporting.guard import completeness_problems, find_unbacked_numerals
from hal9000.solar.run_context import RunContext
from hal9000.tools.decorators import register_tool

__all__ = [
    "assess_performance",
    "check_alert_state",
    "check_net_position",
    "compute_savings",
    "deliver_report",
    "get_home_energy",
    "get_hourly_anomaly_profile",
    "get_recent_history",
    "get_weather",
    "record_alert",
    "run_fault_triage",
    "send_telegram",
    "write_nightly_record",
]

SEVERITIES = ("info", "warn", "critical")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


class ToolInputError(ValueError):
    error_type = "InvalidArguments"
    recoverable = True
    guidance = "Fix the arguments as the error describes and call the tool again."


def _check_date(run: RunContext, date: str) -> None:
    if date != run.day.isoformat():
        raise ToolInputError(f"This run is for {run.day.isoformat()}. Call again with that date.")


# ------------------------------------------------------------------ sources
@register_tool(tags=["source"])
def get_home_energy(date: str, _run: RunContext = None) -> dict:
    """Read today's solar production, grid import and grid export from Home Assistant.

    Call this first. It returns production so far today (kWh), the peak hour,
    grid import/export, household use, and data-quality flags. If
    ``sensor_fault`` is true the solar sensor is reading impossibly low: report
    the sensor problem and do not run triage. Missing data is never zero
    production.

    Args:
        date: The run date, YYYY-MM-DD (given in your briefing).
    """
    _check_date(_run, date)
    try:
        pipeline.gather_energy(_run)
    except Exception as error:
        _run.flags.production_missing_reason = "Home Assistant could not be reached"
        _run.store.put_data("band", "no_data")
        _run.degrade(f"production data unavailable: {error}")
        _run.note("get_home_energy", "failed")
        raise
    summary = pipeline.energy_summary(_run)
    _run.note("get_home_energy", f"{summary['production_kwh']} kWh")
    return summary


MAX_REPORT_WORDS = 140


@register_tool(tags=["source"])
def get_weather(date: str, _run: RunContext = None) -> dict:
    """Read today's sunlight on the panels and cloud cover from Open-Meteo.

    Call this second. It returns insolation on the plane of the array
    (kWh/m2), what a clear sky would have given, and a sky description. If it
    fails, performance can't be judged: say so in the report.

    Args:
        date: The run date, YYYY-MM-DD.
    """
    _check_date(_run, date)
    try:
        pipeline.gather_weather(_run)
    except Exception as error:
        _run.flags.weather_missing_reason = "weather data was unavailable"
        _run.note("get_weather", "failed")
        raise error
    summary = pipeline.weather_summary(_run)
    _run.note("get_weather", summary["sky"])
    return summary


# ----------------------------------------------------------------- analysis
@register_tool(tags=["analysis"])
def assess_performance(_run: RunContext = None) -> dict:
    """Compare today's production with what today's weather should have produced.

    Needs get_home_energy and get_weather first. Returns expected kWh, the
    performance percentage and a band: normal, low, severe, zero, unassessed,
    sensor_fault or no_data. The band decides what happens next.
    """
    result = pipeline.assess_performance(_run)
    _run.note("assess_performance", result.get("band", ""))
    return result


@register_tool(tags=["analysis"])
def compute_savings(_run: RunContext = None) -> dict:
    """Value today's production at the time-of-use rate in force for each hour.

    Never compute savings yourself. Returns the total in dollars and a split by
    TOU period. If rates aren't configured for this date you get an error that
    says so: deliver the report without savings and mention it.
    """
    result = pipeline.savings(_run)
    _run.note("compute_savings", result["savings_usd"])
    return result


@register_tool(tags=["analysis"])
def check_net_position(_run: RunContext = None) -> dict:
    """Check that the household is still a net importer in every TOU period.

    Uses yesterday's complete day and the last 30 days. The savings model
    depends on every period staying net import over a billing cycle. If a
    period is trending toward export, mention it as a warning.
    """
    result = pipeline.net_position(_run)
    _run.note("check_net_position", "ok" if result.get("available") else "unavailable")
    return result


@register_tool(tags=["analysis"])
def get_hourly_anomaly_profile(_run: RunContext = None) -> dict:
    """Show where in the day production departed from expectation.

    Call this when the band is low, severe or zero. Returns a profile (late
    start, early cutoff, midday dropout, uniform shortfall, flat zero, or
    normal) and actual vs expected kWh per hour.
    """
    result = pipeline.anomaly_profile(_run)
    _run.note("get_hourly_anomaly_profile", result["profile"])
    return result


@register_tool(tags=["memory_read"])
def get_recent_history(days: int = 7, _run: RunContext = None) -> dict:
    """Recent nightly records, to tell a one-day blip from a trend.

    Args:
        days: How many days back to look (1-60). Default 7.
    """
    result = pipeline.recent_history(_run, days)
    _run.note("get_recent_history", f"{result['days_found']} days")
    return result


# --------------------------------------------------------------- delegation
@register_tool(tags=["delegate"])
def run_fault_triage(reason: str, _run: RunContext = None) -> dict:
    """Ask the fault-triage specialist to find the cause of a shortfall.

    Call this only when the band is severe or zero, or low with no explanation
    from the anomaly profile and history. The specialist already sees
    everything you have fetched, so just state why you are calling. It returns
    a verdict (inverter_suspected, weather_explained, shading,
    degradation_trend, sensor_fault or inconclusive) and figures you may speak.
    Inconclusive is an acceptable answer: report the shortfall without a cause.

    Args:
        reason: One sentence on why triage is needed.
    """
    run = _run
    if run.flags.verdict is not None:
        return {"note": "Triage already ran this evening.", **run.flags.verdict}
    if not run.features.triage_enabled or run.triage_runner is None:
        return {
            "ran": False,
            "verdict": "inconclusive",
            "note": "Triage is disabled this month to stay within budget. Report without a cause.",
        }
    energy = run.store.data("energy")
    if energy is not None and energy.sensor_fault:
        return {
            "ran": False,
            "verdict": "sensor_fault",
            "note": "The solar sensor is faulty today; triage would only diagnose the sensor.",
        }
    verdict = run.triage_runner(reason)
    run.flags.triage_ran = True
    run.flags.verdict = verdict
    run.note("run_fault_triage", verdict.get("verdict", "?"))
    return verdict


# ------------------------------------------------------------------- alerts
@register_tool(tags=["notify"])
def check_alert_state(alert_key: str, _run: RunContext = None) -> dict:
    """Look up an alert before raising it: is it already open, and at what severity?

    Args:
        alert_key: A short stable key such as "production_low", "inverter_suspected",
            "sensor_fault" or "net_position".
    """
    alert = _run.db.alert(alert_key)
    if not alert or alert["status"] != "open":
        return {"alert_key": alert_key, "open": False}
    return {
        "alert_key": alert_key,
        "open": True,
        "severity": alert["severity"],
        "since": alert["opened_at"],
        "text": alert["text"],
    }


@register_tool(tags=["notify"])
def record_alert(alert_key: str, severity: str, text: str, _run: RunContext = None) -> dict:
    """Raise an alert. New or escalated alerts are sent to Telegram; repeats are not.

    Numbers in the text must come from tool results. At most 3 Telegram
    messages go out per run.

    Args:
        alert_key: Stable key (same key tomorrow means the same problem).
        severity: "info", "warn" or "critical".
        text: One or two sentences for the household.
    """
    run = _run
    if severity not in SEVERITIES:
        raise ToolInputError(f"severity must be one of {SEVERITIES}")
    bad = find_unbacked_numerals(text, run.store, run.day)
    if bad:
        raise ToolInputError(
            f"These numbers are not from tool results: {bad}. Use the exact figures tools returned."
        )
    existing = run.db.alert(alert_key)
    if existing and existing["status"] == "open" and _RANK[severity] <= _RANK[existing["severity"]]:
        return {"alert_key": alert_key, "status": "already_open", "sent": False}
    status = "escalated" if existing and existing["status"] == "open" else "opened"
    run.db.open_alert(alert_key, severity, text)
    run.flags.alerts_raised.append(alert_key)
    sent = run.delivery.send_alert(text, severity)
    if sent:
        run.flags.telegram_sent += 1
    run.note("record_alert", f"{alert_key} {status}")
    return {"alert_key": alert_key, "status": status, "sent": sent}


@register_tool(tags=["notify"])
def send_telegram(message: str, severity: str = "info", _run: RunContext = None) -> dict:
    """Send a short Telegram message that is not an alert. Use sparingly.

    Numbers must come from tool results. At most 3 Telegram messages per run.

    Args:
        message: The text to send.
        severity: "info", "warn" or "critical".
    """
    run = _run
    bad = find_unbacked_numerals(message, run.store, run.day)
    if bad:
        raise ToolInputError(f"These numbers are not from tool results: {bad}.")
    sent = run.delivery.send_alert(message, severity)
    if sent:
        run.flags.telegram_sent += 1
    return {"sent": sent}


# ------------------------------------------------------------------ records
@register_tool(tags=["memory_write"])
def write_nightly_record(_run: RunContext = None) -> dict:
    """Save tonight's record so tomorrow's run has trend context.

    Takes no arguments: the record is built from tool results. Call it once,
    before deliver_report.
    """
    run = _run
    record = pipeline.build_record(run, "provisional")
    run.db.append_record(record)
    run.flags.record_written = True
    run.note("write_nightly_record", "saved")
    return {"saved": True, "date": record.date, "band": record.band}


# ------------------------------------------------------------------- report
@register_tool(tags=["report"])
def deliver_report(narrative: str, _run: RunContext = None) -> dict:
    """Deliver the spoken evening report. This finishes the run.

    Keep it under about 45 seconds of speech (four or five short sentences).
    Speak every number exactly as a tool returned it, for example "8.2
    kilowatt-hours" or "$1.37". If a number isn't backed by a tool result the
    report is rejected and you get one chance to correct it; a second rejection
    sends the standard template instead.

    Args:
        narrative: The report text, as it should be spoken.
    """
    run = _run

    def fallback(reason: str) -> dict:
        pipeline.ensure_core_steps(run)
        run.flags.report_text = render_fallback(run)
        run.flags.renderer = "fallback"
        run.flags.delivered = True
        run.degrade(reason)
        run.note("deliver_report", f"fallback: {reason}")
        return {"delivered": True, "renderer": "fallback", "note": "Standard report sent."}

    # Each check gets one correction, counted separately, so a completeness
    # bounce doesn't use up the numeric guard's second chance.
    problems = completeness_problems(run.store, run.flags)
    if problems:
        run.flags.completeness_bounces += 1
        if run.flags.completeness_bounces >= 2:
            return fallback("report still incomplete after one correction")
        return {"delivered": False, "error": "Report is incomplete: " + "; ".join(problems)}
    bad = find_unbacked_numerals(narrative, run.store, run.day)
    if bad:
        run.flags.guard_failures += 1
        if run.flags.guard_failures >= 2:
            return fallback("numeric guard rejected the narrative twice")
        return {
            "delivered": False,
            "error": (
                f"These numbers are not backed by tool results: {bad}. Use only these "
                f"figures, spoken exactly and with their own units: "
                f"{sorted(set(run.store.figures().values()))[:60]}"
            ),
        }
    words = len(narrative.split())
    if words > MAX_REPORT_WORDS:
        run.flags.length_bounces += 1
        if run.flags.length_bounces >= 2:
            return fallback(f"narrative too long twice ({words} words)")
        return {"delivered": False, "error": f"Too long ({words} words). Keep it under 110."}
    run.flags.report_text = narrative.strip()
    run.flags.renderer = "llm"
    run.flags.delivered = True
    run.note("deliver_report", "accepted")
    return {"delivered": True, "renderer": "llm"}
