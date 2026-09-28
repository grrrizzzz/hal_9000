"""The deterministic report: a Jinja2 template over whatever the run store holds.

This is the floor under the agent. It is used whenever the agent can't deliver:
an exception, the iteration or time cap, a cost cap, two guard rejections, the
monthly cap, ``--no-llm``, or a failed completeness check. It is deliberately
dumb -- no branching beyond present/absent fields -- and every number it speaks
comes straight from the run store.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from hal9000.solar.run_context import RunContext

__all__ = ["render_fallback"]

_TEMPLATES = Path(__file__).parent / "templates"
_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATES)),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

_ANOMALY_TEXT = {
    "late_start": "production started late",
    "early_cutoff": "production stopped early",
    "midday_dropout": "there was a midday dropout",
    "uniform_shortfall": "output was low all day",
    "flat_zero": "output stayed near zero",
}

_VERDICT_TEXT = {
    "inverter_suspected": "The likely cause is the inverter; check it first.",
    "weather_explained": "The weather explains the shortfall.",
    "shading": "The pattern looks like shading.",
    "degradation_trend": "The pattern looks like a gradual decline worth watching.",
    "sensor_fault": "The likely cause is a sensor problem rather than the panels.",
    "inconclusive": "The cause could not be determined.",
}


class _Figures(dict):
    def __getattr__(self, name):
        return self.get(name)


def render_fallback(run: RunContext) -> str:
    store = run.store
    net = store.data("net_trending") or []
    band = store.data("band") or "unassessed"
    verdict = (run.flags.verdict or {}).get("verdict", "")
    if band == "zero" and verdict == "inverter_suspected":
        verdict = ""  # the zero-band sentence already says "check the inverter first"
    context = {
        "day_name": run.day.strftime("%A"),
        "f": _Figures(store.figures()),
        "band": band,
        "production_missing": run.flags.production_missing_reason,
        "weather_missing": run.flags.weather_missing_reason,
        "savings_absent": None if store.has("savings_usd") else run.flags.savings_absent_reason,
        "anomaly_text": _ANOMALY_TEXT.get(store.data("anomaly") or ""),
        "verdict_text": _VERDICT_TEXT.get(verdict),
        "trending": ", ".join(net) if net else None,
    }
    text = _ENV.get_template("report.j2").render(**context)
    return " ".join(text.split())
