"""The solar agent: the evening report, on the fast model tier, at most 12 steps."""

from __future__ import annotations

from collections.abc import Callable

from hal9000.core.actions import PythonActionRegistry
from hal9000.core.agent import Agent
from hal9000.core.agent_language import AgentFunctionCallingActionLanguage
from hal9000.core.goals import Goal
from hal9000.solar.environments.solar import RunCapability, SolarEnvironment
from hal9000.solar.run_context import RunContext
from hal9000.solar.tools import ensure_registered

__all__ = ["SOLAR_TAGS", "build_solar_agent", "solar_goals", "solar_registry"]

SOLAR_TAGS = ["source", "analysis", "memory_read", "memory_write", "notify", "delegate", "report"]
SOLAR_MAX_ITERATIONS = 12

_DATA_INTEGRITY = """\
Never report on data you have not fetched. Every figure comes from a tool result \
in this run. A tool error or partial data is stated, never estimated. Missing \
production data is the headline, and a missing sensor is never treated as zero. \
If get_home_energy reports sensor_fault, the sensor is the headline: do not call \
it a performance shortfall and do not run triage."""

_METHOD = """\
You are HAL's solar agent. Each evening you report how the home's 2.47 kW solar \
array did today. Work in this order, one tool per turn:
1. get_home_energy(date)  2. get_weather(date)  3. assess_performance  \
4. compute_savings  5. check_net_position
6. Only if the band is low, severe or zero: get_hourly_anomaly_profile and \
get_recent_history(7); if the band is severe or zero, or the cause is still \
unclear, run_fault_triage.
7. Alerts: for anything needing attention, check_alert_state then record_alert. \
Skip alerts on a normal day.
8. write_nightly_record
9. deliver_report -- this finishes the run.
Never call a tool that isn't listed. Don't repeat a call that already succeeded."""

_PERFORMANCE = """\
Judge the day against expected production for today's weather, never against the \
2.47 kW nameplate. The array is about 11 years old; a nameplate comparison reads \
normal ageing as a permanent fault."""

_SAVINGS = """\
Use only compute_savings for money. Never compute savings yourself or use a blended \
rate. If rates aren't configured, deliver the report without savings and say so."""

_TRIAGE = """\
An inconclusive verdict is acceptable: report the shortfall without a cause rather \
than guessing. For zero or near-zero output, say "check the inverter" first -- the \
Delta SOLIVIA inverter is about 12 years old and out of warranty, and panels rarely \
fail as a group."""

_VALIDITY = """\
If check_net_position shows a period trending toward export or net export, mention \
it as a warning: the savings model depends on every period staying net import."""

_ALERTS = """\
Respect check_alert_state: an alert already open at the same severity is not re-sent. \
Alert keys are short and stable (production_low, production_zero, inverter_suspected, \
sensor_fault, net_position). Write the nightly record before delivering."""

_DELIVER = """\
Finish with deliver_report. The narrative is spoken aloud on a speaker: four or five \
short sentences, under about 45 seconds, no lists or symbols other than $ and %. \
Speak every number exactly as a tool returned it ("8.2 kilowatt-hours", "$1.37", \
"74%"). Do not round, convert, add up or invent numbers. If the report is rejected, \
fix exactly what the error names."""


def solar_goals(report_policy: str = "") -> list[Goal]:
    goals = [
        Goal(1, "data_integrity", _DATA_INTEGRITY),
        Goal(2, "method", _METHOD),
        Goal(3, "assess_performance", _PERFORMANCE),
        Goal(4, "quantify_savings", _SAVINGS),
        Goal(5, "triage_faults", _TRIAGE),
        Goal(6, "watch_model_validity", _VALIDITY),
        Goal(7, "persist_and_notify", _ALERTS),
        Goal(8, "deliver", _DELIVER),
    ]
    if report_policy.strip():
        goals.append(Goal(9, "house_style", report_policy.strip()))
    return goals


def solar_registry() -> PythonActionRegistry:
    ensure_registered()
    return PythonActionRegistry(tags=SOLAR_TAGS)


def build_solar_agent(run: RunContext, generate_response: Callable) -> Agent:
    return Agent(
        goals=solar_goals(run.report_policy),
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=solar_registry(),
        generate_response=generate_response,
        environment=SolarEnvironment(),
        capabilities=[RunCapability(run, lambda: run.flags.delivered, "solar")],
        max_iterations=SOLAR_MAX_ITERATIONS,
        max_duration_seconds=900,
        action_tier="fast",
        cache_key="hal-solar",
    )
