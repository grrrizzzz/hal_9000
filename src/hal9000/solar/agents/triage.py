"""The fault-triage sub-agent: read-only diagnosis on the powerful tier, at most 24 steps.

It is called through the solar agent's ``run_fault_triage`` tool. It starts with
a fresh memory; ``read_handover`` gives it everything the solar agent already
established, straight from the run store, so nothing is paraphrased between the
two agents. It has no tools that notify, write records or deliver.
"""

from __future__ import annotations

from collections.abc import Callable

from hal9000.core.actions import PythonActionRegistry
from hal9000.core.agent import Agent
from hal9000.core.agent_language import AgentFunctionCallingActionLanguage
from hal9000.core.goals import Goal
from hal9000.core.memory import Memory
from hal9000.solar.environments.solar import RunCapability, SolarEnvironment
from hal9000.solar.run_context import RunContext
from hal9000.solar.tools import ensure_registered

__all__ = [
    "FORBIDDEN_TRIAGE_TAGS",
    "TRIAGE_TAGS",
    "build_triage_agent",
    "make_triage_runner",
    "triage_goals",
    "triage_props",
    "triage_registry",
]

TRIAGE_TAGS = ["triage", "telemetry", "weather", "memory_read", "verdict", "inverter_diagnostics"]
FORBIDDEN_TRIAGE_TAGS = ("source", "notify", "memory_write", "delegate", "report")
TRIAGE_MAX_ITERATIONS = 24
#: Context properties a triage run must never receive (plan F4).
NOT_FOR_TRIAGE = frozenset(
    {"delivery", "record_store", "alert_store", "ledger_admin", "agent_registry"}
)

_EVIDENCE = """\
Every claim cites the tool result it came from. Figures you want the parent to speak \
go in figure_keys, and only keys your tools returned are accepted."""

_DIAGNOSE = """\
Start with read_handover. Then separate the causes, in this order:
- inverter fault: zero or flat-lined output, or abrupt cutoffs, with sun available
- weather: irradiance and cloud cover explain the shortfall (compare_to_clearsky, \
get_irradiance_curve, find_similar_days)
- shading: a repeatable notch at the same time of day on clear days
- degradation: a slow multi-week slide on clear days (get_sensor_history, get_recent_history)
- sensor fault: Emporia unavailable, or implausible values
Use the fewest tools that settle it."""

_INVERTER = """\
For zero or near-zero output, the leading hypothesis is the inverter: a ~2014 \
transformerless Delta SOLIVIA, out of warranty. Panels very rarely fail as a group."""

_FINISH = """\
End with submit_verdict. Choose inconclusive rather than guessing. The recommended \
alert is a recommendation; the parent decides what is sent."""


def triage_goals() -> list[Goal]:
    return [
        Goal(1, "evidence_only", _EVIDENCE),
        Goal(2, "diagnose", _DIAGNOSE),
        Goal(3, "inverter_first", _INVERTER),
        Goal(4, "finish", _FINISH),
    ]


def triage_registry() -> PythonActionRegistry:
    ensure_registered()
    return PythonActionRegistry(tags=TRIAGE_TAGS)


def triage_props(parent_props: dict) -> dict:
    """The parent's context properties minus everything triage must not hold (F4)."""
    return {k: v for k, v in parent_props.items() if k not in NOT_FOR_TRIAGE}


def build_triage_agent(run: RunContext, generate_response: Callable) -> Agent:
    return Agent(
        goals=triage_goals(),
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=triage_registry(),
        generate_response=generate_response,
        environment=SolarEnvironment(),
        capabilities=[
            RunCapability(run, lambda: run.store.data("triage_verdict") is not None, "triage")
        ],
        max_iterations=TRIAGE_MAX_ITERATIONS,
        max_duration_seconds=900,
        action_tier="powerful",
        cache_key="hal-triage",
    )


def make_triage_runner(
    run: RunContext, generate_response: Callable, parent_props: dict
) -> Callable[[str], dict]:
    """A callable for ``run.triage_runner``: runs triage and returns the verdict dict."""

    def runner(reason: str) -> dict:
        run.store.put_data("triage_reason", reason)
        agent = build_triage_agent(run, generate_response)
        memory = Memory()
        run.store.put_data("triage_memory", memory.items)
        try:
            agent.run(
                f"Diagnose today's ({run.day.isoformat()}) solar shortfall. Reason: {reason}",
                memory=memory,
                action_context_props=triage_props(parent_props),
            )
        except Exception as error:
            run.degrade(f"triage failed: {type(error).__name__}")
            return {
                "verdict": "inconclusive",
                "confidence": "low",
                "evidence": [
                    {
                        "observation": f"Triage stopped early: {error}",
                        "source_tool": "run_fault_triage",
                        "supports": "inconclusive",
                    }
                ],
                "recommended_alert": {
                    "severity": "warn",
                    "text": "Production was low; cause unknown.",
                },
                "figures": {},
            }
        verdict = run.store.data("triage_verdict")
        if verdict is None:
            return {
                "verdict": "inconclusive",
                "confidence": "low",
                "evidence": [
                    {
                        "observation": "Triage ran out of steps without a verdict.",
                        "source_tool": "run_fault_triage",
                        "supports": "inconclusive",
                    }
                ],
                "recommended_alert": {
                    "severity": "warn",
                    "text": "Production was low; cause unknown.",
                },
                "figures": {},
            }
        return verdict

    return runner
