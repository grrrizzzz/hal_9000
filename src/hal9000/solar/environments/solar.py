"""Where the solar agents' actions run, and the loop rules that go with them.

:class:`SolarEnvironment` turns every tool failure into a structured result the
model can act on (``error_type``, ``recoverable``, ``guidance``), and enforces
per-run side-effect budgets. :class:`RunCapability` watches the loop: it ends it
once the report is delivered (or the verdict submitted), stops at the hard cost
cap, nudges on the soft cap and on repeated identical calls, and keeps the trace.
"""

from __future__ import annotations

import json
import time
import traceback

from hal9000.core.action_context import ActionContext
from hal9000.core.actions import Action
from hal9000.core.capabilities import Capability
from hal9000.core.environment import PythonEnvironment
from hal9000.core.prompt import Prompt

__all__ = ["MAX_STRIKES", "RunCapability", "SolarEnvironment", "TOOL_BUDGETS"]

MAX_STRIKES = 2

#: Maximum *calls* per run for side-effecting tools.
TOOL_BUDGETS = {
    "send_telegram": 3,
    "record_alert": 5,
    "deliver_report": 3,
    "write_nightly_record": 2,
    "run_fault_triage": 2,
    "submit_verdict": 3,
}


class SolarEnvironment(PythonEnvironment):
    def __init__(self, budgets: dict[str, int] | None = None):
        self.budgets = dict(TOOL_BUDGETS if budgets is None else budgets)
        self.calls: dict[str, int] = {}

    def execute_action(
        self, agent, action_context: ActionContext, action: Action, args: dict
    ) -> dict:
        used = self.calls.get(action.name, 0)
        limit = self.budgets.get(action.name)
        if limit is not None and used >= limit:
            return {
                "tool_executed": False,
                "error": (
                    f"{action.name} has already been called {used} times this run (limit {limit})."
                ),
                "error_type": "BudgetLimit",
                "recoverable": True,
                "guidance": "Do not call it again. Move on to delivering the report.",
            }
        self.calls[action.name] = used + 1
        try:
            args_copy = self.inject_dependencies(agent, action_context, action, args)
            return self.format_result(action.execute(**args_copy))
        except TypeError as error:
            return {
                "tool_executed": False,
                "error": f"Bad arguments for {action.name}: {error}",
                "error_type": "InvalidArguments",
                "recoverable": True,
                "guidance": "Check the tool's parameters and call it again.",
            }
        except Exception as error:
            return {
                "tool_executed": False,
                "error": str(error),
                "error_type": getattr(error, "error_type", type(error).__name__),
                "recoverable": getattr(error, "recoverable", True),
                "guidance": getattr(
                    error, "guidance", "Read the error, fix what it names, or continue without it."
                ),
                "trace": traceback.format_exc(limit=2)[-600:],
            }

    def format_result(self, result) -> dict:
        return {"tool_executed": True, "result": result, "timestamp": time.strftime("%H:%M:%S")}


class RunCapability(Capability):
    """Loop rules for one agent in one run.

    ``done`` is a callable returning True once this agent's job is finished
    (report delivered, or verdict submitted).
    """

    def __init__(self, run, done, label: str):
        super().__init__(name=f"Run:{label}", description="Termination, budgets, nudges, trace.")
        self.run = run
        self.done = done
        self.label = label
        self.iterations = 0
        self.last_calls: list[str] = []
        self.nudge: str | None = None
        self.strikes = 0  # consecutive replies that weren't a known tool call

    def init(self, agent, action_context) -> None:
        self.iterations = 0
        self.last_calls = []
        self.strikes = 0

    def start_agent_loop(self, agent, action_context) -> bool:
        return not self.run.flags.hard_cap_hit

    def process_prompt(self, agent, action_context, prompt: Prompt) -> Prompt:
        notes = []
        if self.run.flags.soft_cap_hit:
            notes.append(
                "Budget note: this run is near its cost limit. Finish now: deliver the report "
                "with what you have."
            )
        if self.nudge:
            notes.append(self.nudge)
            self.nudge = None
        remaining = agent.max_iterations - self.iterations
        if remaining <= 2:
            notes.append(f"Only {remaining} steps remain. Finish now.")
        if not notes:
            return prompt
        messages = list(prompt.messages) + [{"role": "system", "content": " ".join(notes)}]
        return Prompt(messages=messages, tools=prompt.tools, metadata=prompt.metadata)

    def process_action(self, agent, action_context, action: dict) -> dict:
        signature = json.dumps(action, sort_keys=True, default=str)
        self.last_calls.append(signature)
        if self.last_calls[-3:].count(signature) >= 2:
            self.nudge = (
                "You just repeated the same call with the same arguments. Don't repeat it; "
                "use the result you already have and move to the next step."
            )
        return action

    def process_result(self, agent, action_context, response, action_def, action, result):
        name = action.get("tool", "?")
        # Prose, or a tool that doesn't exist: one is feedback, two in a row means
        # the model has stopped working the problem, so the loop ends and the
        # coordinator's completeness step takes over.
        self.strikes = self.strikes + 1 if action_def is None else 0
        if isinstance(result, dict) and not result.get("tool_executed", True):
            self.run.note(f"{self.label}:{name}", f"error: {str(result.get('error'))[:80]}")
        return result

    def end_agent_loop(self, agent, action_context) -> None:
        self.iterations += 1

    def should_terminate(self, agent, action_context, response: str) -> bool:
        if self.strikes >= MAX_STRIKES:
            self.run.note(f"{self.label}:loop", "stopped after two replies with no tool call")
            return True
        return bool(self.done())
