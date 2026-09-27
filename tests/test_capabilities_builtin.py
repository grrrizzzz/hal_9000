"""The four shipped capabilities, and the planning tools two of them drive.

``test_capabilities.py`` proves the *mechanism* -- ten hooks, firing in order,
in the default loop. This file proves the mechanism is worth having: that a real
capability slots into those hooks and changes behaviour with no change to
:mod:`hello_agent.core.agent`.

Every test scripts its LLM exactly. That is deliberate accounting, not
pedantry: ``PlanFirstCapability`` spends one extra call in ``init`` and
``ProgressTrackingCapability`` one per tracked iteration, so the length of each
``FakeLLM`` script *is* the cost assertion.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from hello_agent.capabilities import (
    LoggingCapability,
    PlanFirstCapability,
    ProgressTrackingCapability,
    TimeAwareCapability,
)
from hello_agent.capabilities import time_aware as time_aware_module
from hello_agent.core.action_context import ActionContext, default_properties
from hello_agent.core.actions import Action, ActionRegistry, PythonActionRegistry
from hello_agent.core.agent import Agent
from hello_agent.core.agent_language import AgentFunctionCallingActionLanguage
from hello_agent.core.capabilities import Capability
from hello_agent.core.environment import PythonEnvironment
from hello_agent.core.goals import Goal
from hello_agent.core.memory import Memory
from hello_agent.core.prompt import Prompt
from hello_agent.tools.planning import create_plan, track_progress
from tests.fakes import FakeLLM, tool_call

SCHEMA = {"type": "object", "properties": {}, "required": []}
GOALS = [Goal(priority=1, name="greet", description="Greet the visitor, then stop.")]


def build_registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(Action("echo", lambda text="": text, "Echo text.", SCHEMA))
    registry.register(Action("terminate", lambda message: message, "Stop.", SCHEMA, True))
    return registry


def build_agent(llm: FakeLLM, capabilities: list[Capability], **kwargs) -> Agent:
    return Agent(
        goals=GOALS,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=build_registry(),
        generate_response=llm,
        environment=PythonEnvironment(),
        capabilities=capabilities,
        **kwargs,
    )


def build_context(llm: FakeLLM, memory: Memory | None = None) -> ActionContext:
    """An ActionContext shaped exactly like the one ``Agent.run`` builds."""
    memory = memory if memory is not None else Memory()
    return ActionContext(default_properties(llm, memory, build_registry()))


def system_memories(memory: Memory) -> list[str]:
    return [item["content"] for item in memory.get_memories() if item["type"] == "system"]


# ============================================================ TimeAwareCapability


def test_time_aware_seeds_memory_in_init():
    llm = FakeLLM([tool_call("terminate", message="done")])
    memory = build_agent(llm, [TimeAwareCapability()]).run("greet Ada")

    seeded = system_memories(memory)[0]
    assert "Right now, it is" in seeded
    assert "America/Chicago timezone" in seeded
    assert "(ISO: " in seeded
    # and it is in front of the model on the very first turn
    assert any(seeded in message["content"] for message in llm.calls[0].messages)


def test_time_aware_refreshes_the_time_in_every_process_prompt(monkeypatch):
    """Not 'the time is in the prompt' -- 'the time is *re-read* each iteration'."""
    base = datetime(2026, 9, 21, 16, 58, tzinfo=ZoneInfo("America/Chicago"))
    ticks = iter([base, base, base + timedelta(minutes=5), base + timedelta(minutes=11)])

    class FrozenDatetime:
        @staticmethod
        def now(tz=None):
            return next(ticks)

    monkeypatch.setattr(time_aware_module, "datetime", FrozenDatetime)

    llm = FakeLLM([tool_call("echo", text="one"), tool_call("terminate", message="done")])
    build_agent(llm, [TimeAwareCapability()]).run("greet Ada")

    headers = [prompt.messages[0]["content"].splitlines()[0] for prompt in llm.calls]
    assert headers == [
        "Current time: 16:58 Monday, September 21, 2026 (America/Chicago)",
        "Current time: 17:03 Monday, September 21, 2026 (America/Chicago)",
    ]
    # the header is prepended fresh each time, never accumulated
    assert all(prompt.messages[0]["content"].count("Current time:") == 1 for prompt in llm.calls)


def test_time_aware_takes_the_timezone_from_the_action_context():
    llm = FakeLLM([tool_call("terminate", message="done")])
    memory = build_agent(llm, [TimeAwareCapability()]).run(
        "greet Ada", action_context_props={"time_zone": "Asia/Tokyo"}
    )

    assert "Asia/Tokyo timezone" in system_memories(memory)[0]
    assert "(Asia/Tokyo)" in llm.calls[0].messages[0]["content"]


def test_time_aware_defaults_to_america_chicago_and_honours_agent_timezone(monkeypatch):
    context = build_context(FakeLLM([]))
    assert time_aware_module.resolve_time_zone(context)[0] == "America/Chicago"

    monkeypatch.setenv("AGENT_TIMEZONE", "Europe/Paris")
    assert time_aware_module.resolve_time_zone(context)[0] == "Europe/Paris"


def test_time_aware_falls_back_to_utc_on_an_unknown_zone(monkeypatch):
    monkeypatch.setenv("AGENT_TIMEZONE", "Mars/Olympus_Mons")
    name, zone = time_aware_module.resolve_time_zone(build_context(FakeLLM([])))

    assert name == "UTC" and zone == ZoneInfo("UTC")


def test_time_aware_preserves_tools_and_does_not_mutate_the_prompt_it_is_given():
    """The note's version returns ``Prompt(messages=messages)`` and drops tools."""
    original = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[{"name": "echo"}])
    before = list(original.messages)

    result = TimeAwareCapability().process_prompt(None, build_context(FakeLLM([])), original)

    assert result is not original
    assert result.tools == [{"name": "echo"}]
    assert original.messages == before  # untouched
    assert result.messages[0]["role"] == "system"


# ============================================================ PlanFirstCapability


def test_plan_first_plans_once_and_stores_it_in_memory():
    llm = FakeLLM(["1. Greet Ada\n2. Terminate", tool_call("terminate", message="done")])
    capability = PlanFirstCapability()

    memory = build_agent(llm, [capability]).run("greet Ada")

    stored = system_memories(memory)
    assert stored == [
        "You must follow these instructions carefully to complete the task:\n"
        "1. Greet Ada\n2. Terminate"
    ]
    assert capability.first_call is False
    # one planning call plus one loop call -- the plan's entire cost
    assert len(llm.calls) == 2
    planning_prompt = llm.calls[0].messages[0]["content"]
    assert "- echo: Echo text." in planning_prompt  # tools reached the planner
    assert "user: greet Ada" in planning_prompt  # so did the task


def test_plan_first_does_not_re_plan_on_a_second_init():
    llm = FakeLLM(["THE PLAN"])
    capability = PlanFirstCapability()
    context = build_context(llm)

    capability.init(None, context)
    capability.init(None, context)  # a second run, or a nested agent

    assert len(llm.calls) == 1  # no second planning call...
    assert len(system_memories(context.get_memory())) == 1  # ...and no second plan


def test_plan_first_honours_the_configured_memory_type():
    llm = FakeLLM(["THE PLAN"])
    context = build_context(llm)

    PlanFirstCapability(plan_memory_type="user").init(None, context)

    assert context.get_memory().get_memories()[0]["type"] == "user"


def test_plan_first_keeps_the_notes_track_progress_argument():
    """``PlanFirstCapability(track_progress=True)`` from the note must still construct."""
    assert PlanFirstCapability(track_progress=True).track_progress is True


# ====================================================== ProgressTrackingCapability


def test_progress_tracking_honours_track_frequency():
    """frequency=2 over 4 iterations => exactly 2 reports."""
    llm = FakeLLM(
        [
            tool_call("echo", text="1"),
            tool_call("echo", text="2"),
            "REPORT AFTER TWO",
            tool_call("echo", text="3"),
            tool_call("echo", text="4"),
            "REPORT AFTER FOUR",
        ]
    )
    capability = ProgressTrackingCapability(track_frequency=2)

    memory = build_agent(llm, [capability], max_iterations=4).run("greet Ada")

    reports = system_memories(memory)
    assert len(reports) == 2
    assert reports[0].startswith("Progress Report (Iteration 2):\nREPORT AFTER TWO")
    assert reports[1].startswith("Progress Report (Iteration 4):\nREPORT AFTER FOUR")
    assert capability.iteration_count == 4
    assert len(llm.calls) == 6  # 4 loop calls + 2 reports: the documented cost


def test_progress_tracking_reports_every_iteration_by_default():
    llm = FakeLLM([tool_call("terminate", message="done"), "REPORT"])

    memory = build_agent(llm, [ProgressTrackingCapability()]).run("greet Ada")

    assert system_memories(memory) == ["Progress Report (Iteration 1):\nREPORT"]
    assert len(llm.calls) == 2  # one loop call doubled -- exactly what the note warns


def test_progress_tracking_rejects_a_frequency_below_one():
    with pytest.raises(ValueError, match="track_frequency must be at least 1"):
        ProgressTrackingCapability(track_frequency=0)


# ============================================================== LoggingCapability


def test_logging_mutates_nothing_at_any_hook():
    """Byte-identical pass-through: the observer must not be part of the bug."""
    lines: list[str] = []
    capability = LoggingCapability(sink=lines.append)
    context = build_context(FakeLLM([]))

    prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=[{"name": "echo"}])
    response = '{"tool": "echo", "args": {"text": "hi"}}'
    action = {"tool": "echo", "args": {"text": "hi"}}
    result = {"tool_executed": True, "result": "hi"}
    memories = [{"type": "assistant", "content": response}]
    snapshots = (list(prompt.messages), dict(action), dict(result), list(memories))

    assert capability.process_prompt(None, context, prompt) is prompt
    assert capability.process_response(None, context, response) is response
    assert capability.process_action(None, context, action) is action
    assert capability.process_result(None, context, response, None, action, result) is result
    assert (
        capability.process_new_memories(None, context, Memory(), response, result, memories)
        is memories
    )

    assert (prompt.messages, action, result, memories) == snapshots
    assert len(lines) == 5


def test_logging_writes_a_structured_line_for_every_loop_event():
    lines: list[str] = []
    llm = FakeLLM([tool_call("echo", text="hi"), tool_call("terminate", message="done")])

    build_agent(llm, [LoggingCapability(sink=lines.append)]).run("greet Ada")

    events = [line.split(" ", 1)[0] for line in lines]
    assert events[0] == "init" and events[-1] == "terminate"
    assert events.count("start_agent_loop") == 2
    assert events.count("process_result") == 2
    assert "tool='echo'" in next(line for line in lines if line.startswith("process_action"))
    assert "terminal=True" in lines[-2] or any("terminal=True" in line for line in lines)


def test_logging_leaves_the_run_identical():
    """Same script, with and without the logger: same memory, same prompts."""

    def run(capabilities):
        llm = FakeLLM([tool_call("echo", text="hi"), tool_call("terminate", message="done")])
        memory = build_agent(llm, capabilities).run("greet Ada")
        items = json.loads(json.dumps(memory.get_memories(), default=str))
        for item in items:  # the environment stamps results with a wall clock
            item["content"] = item["content"].split('"timestamp"')[0]
        return items, [prompt.messages for prompt in llm.calls]

    assert run([LoggingCapability(sink=lambda line: None)]) == run([])


def test_logging_defaults_to_the_standard_library_logger(caplog):
    with caplog.at_level("INFO", logger="hello_agent.capabilities.logging"):
        LoggingCapability().log("process_response", iteration=1)

    assert "process_response iteration=1" in caplog.text


def test_logging_truncates_a_very_long_value():
    lines: list[str] = []
    LoggingCapability(sink=lines.append).process_response(
        None, build_context(FakeLLM([])), "x" * 5000
    )

    assert "...(+4700 chars)" in lines[0]


# =========================================================== composing two of them


def logged_prompt_line(capabilities: list[Capability], sink: list[str]) -> tuple[str, str]:
    """Run one scripted iteration and return (what Logging saw, what the LLM got)."""
    llm = FakeLLM([tool_call("terminate", message="done")])
    build_agent(llm, capabilities).run("greet Ada")
    seen = next(line for line in sink if line.startswith("process_prompt"))
    return seen, llm.calls[0].messages[0]["content"]


def test_logging_after_time_aware_sees_the_time_enhanced_prompt():
    """``reduce`` composes left to right, so the *last* capability sees the rest."""
    after: list[str] = []
    seen, sent = logged_prompt_line(
        [TimeAwareCapability(), LoggingCapability(sink=after.append)], after
    )

    assert "first_message='Current time:" in seen
    assert sent.startswith("Current time:")


def test_time_aware_after_logging_sees_only_what_logging_passed_through():
    """Reverse the list and the logger records the prompt *before* the time header."""
    before: list[str] = []
    seen, sent = logged_prompt_line(
        [LoggingCapability(sink=before.append), TimeAwareCapability()], before
    )

    assert "first_message='Current time:" not in seen  # Logging ran first...
    assert sent.startswith("Current time:")  # ...but TimeAware still reached the LLM


def test_plan_first_then_progress_tracking_compose_over_a_whole_run():
    llm = FakeLLM(
        [
            "1. Echo\n2. Terminate",  # PlanFirst, in init
            tool_call("echo", text="hi"),  # iteration 1
            "REPORT 1",  # ProgressTracking, end of iteration 1
            tool_call("terminate", message="done"),  # iteration 2
            "REPORT 2",  # ProgressTracking, end of iteration 2
        ]
    )

    memory = build_agent(llm, [PlanFirstCapability(), ProgressTrackingCapability()]).run(
        "greet Ada"
    )

    assert [content.splitlines()[0] for content in system_memories(memory)] == [
        "You must follow these instructions carefully to complete the task:",
        "Progress Report (Iteration 1):",
        "Progress Report (Iteration 2):",
    ]
    assert len(llm.calls) == 5  # 2 loop calls, 3 capability calls


# =============================================================== the planning tools


def test_planning_tools_are_registered_under_the_planning_tag(registered_tools):
    assert set(registered_tools.tools_by_tag["planning"]) == {"create_plan", "track_progress"}


def test_planning_tools_are_absent_from_the_default_registry(registered_tools):
    default = PythonActionRegistry(tags=["greeting", "guestbook", "system"])

    assert default.get_action("create_plan") is None
    assert default.get_action("track_progress") is None
    assert PythonActionRegistry(tags=["planning"]).get_action("create_plan") is not None


def test_planning_tools_hide_their_injected_parameters_from_the_model(registered_tools):
    for name in ("create_plan", "track_progress"):
        schema = registered_tools.tools[name]["parameters"]
        assert schema == {"type": "object", "properties": {}, "required": []}, name


def test_create_plan_and_track_progress_build_prompts_from_tools_and_memory():
    llm = FakeLLM(["A PLAN", "A REPORT"])
    memory = Memory(
        [
            {"type": "user", "content": "greet Ada"},
            {"type": "assistant", "content": "internal chatter"},
        ]
    )
    context = build_context(llm, memory)
    registry = context.get_action_registry()

    assert create_plan(context, memory, registry) == "A PLAN"
    assert track_progress(context, memory, registry) == "A REPORT"

    plan_prompt, report_prompt = (call.messages[0]["content"] for call in llm.calls)
    assert plan_prompt.startswith("Given the task in memory and the available tools")
    assert report_prompt.startswith("Given the current task and available tools")
    for text in (plan_prompt, report_prompt):
        assert "- terminate: Stop." in text
        assert "user: greet Ada" in text
        assert "internal chatter" not in text  # assistant turns are not plan context
