"""The agent loop: sequencing, limits, termination, and error containment.

Everything here runs on :class:`~tests.fakes.FakeLLM`, so a test is a script of
decisions the model "makes". Note that FakeLLM raises when its script runs out,
which is a useful assertion in itself: budget exactly one response per expected
iteration and an over-running loop fails loudly.
"""

from __future__ import annotations

import json

import pytest

from hal9000.core import agent as agent_module
from hal9000.core.action_context import ActionContext
from hal9000.core.actions import Action, ActionRegistry
from hal9000.core.agent import Agent
from hal9000.core.agent_language import AgentFunctionCallingActionLanguage
from hal9000.core.environment import PythonEnvironment
from hal9000.core.goals import Goal
from hal9000.core.memory import Memory
from tests.fakes import FakeLLM, tool_call

SCHEMA = {"type": "object", "properties": {}, "required": []}
GOALS = [Goal(priority=1, name="greet", description="Greet the visitor, then stop.")]


def build_registry(captured: dict | None = None) -> ActionRegistry:
    """A small toolkit: one harmless tool, one that explodes, one that stops."""

    def echo(text: str) -> str:
        return text

    def boom() -> str:
        raise RuntimeError("the guestbook is on fire")

    def inspect_context(action_context: ActionContext) -> str:
        if captured is not None:
            captured["context"] = action_context
        return "captured"

    registry = ActionRegistry()
    registry.register(Action("echo", echo, "Echo text.", SCHEMA))
    registry.register(Action("boom", boom, "Always fails.", SCHEMA))
    registry.register(Action("inspect_context", inspect_context, "Peek.", SCHEMA))
    registry.register(Action("terminate", lambda message: message, "Stop.", SCHEMA, True))
    return registry


def build_agent(llm: FakeLLM, registry: ActionRegistry | None = None, **kwargs) -> Agent:
    return Agent(
        goals=GOALS,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=registry or build_registry(),
        generate_response=llm,
        environment=PythonEnvironment(),
        **kwargs,
    )


def last_result(memory: Memory) -> dict:
    """The result dict the environment produced on the final iteration."""
    return json.loads(memory.get_memories()[-1]["content"])


# ------------------------------------------------------------------ sequencing


def test_the_task_becomes_the_first_memory():
    llm = FakeLLM([tool_call("terminate", message="done")])
    memory = build_agent(llm).run("greet Ada")

    assert memory.get_memories()[0] == {"type": "user", "content": "greet Ada"}


def test_each_iteration_records_the_decision_and_the_result():
    llm = FakeLLM([tool_call("echo", text="hello"), tool_call("terminate", message="bye")])
    memory = build_agent(llm).run("greet Ada")

    types = [item["type"] for item in memory.get_memories()]
    assert types == ["user", "assistant", "user", "assistant", "user"]
    assert json.loads(memory.get_memories()[2]["content"])["result"] == "hello"


def test_memory_from_the_previous_iteration_reaches_the_next_prompt():
    """This is the whole point of Memory: the LLM is stateless, the loop is not."""
    llm = FakeLLM([tool_call("echo", text="hello"), tool_call("terminate", message="bye")])
    build_agent(llm).run("greet Ada")

    assert len(llm.calls[0].messages) == 2  # goals + task
    assert len(llm.calls[1].messages) == 4  # + decision + result


def test_run_returns_the_memory_it_was_given():
    memory = Memory(items=[{"type": "assistant", "content": "earlier run"}])
    llm = FakeLLM([tool_call("terminate", message="done")])

    returned = build_agent(llm).run("greet Ada", memory=memory)

    assert returned is memory
    assert memory.get_memories()[0]["content"] == "earlier run"


# ----------------------------------------------------------------- termination


def test_a_terminal_action_ends_the_loop():
    llm = FakeLLM([tool_call("terminate", message="done"), tool_call("echo", text="never")])
    build_agent(llm).run("greet Ada")

    assert len(llm.calls) == 1
    assert llm.responses == [tool_call("echo", text="never")]


def test_max_iterations_is_a_hard_ceiling():
    llm = FakeLLM([tool_call("echo", text="again")] * 3)
    memory = build_agent(llm, max_iterations=3).run("greet Ada")

    assert len(llm.calls) == 3
    assert len(memory) == 1 + 3 * 2


def test_max_duration_seconds_stops_the_loop(monkeypatch):
    """A fake clock: the deadline is set at t=0, the second check lands past it."""
    clock = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(agent_module.time, "time", lambda: next(clock))

    llm = FakeLLM([tool_call("echo", text="again")] * 5)
    build_agent(llm, max_iterations=5, max_duration_seconds=10).run("greet Ada")

    assert len(llm.calls) == 1


def test_a_zero_second_budget_runs_nothing():
    llm = FakeLLM([tool_call("echo", text="never")])
    build_agent(llm, max_duration_seconds=0).run("greet Ada")

    assert llm.calls == []


# ----------------------------------------------------------- error containment


def test_a_raising_tool_does_not_crash_the_loop():
    llm = FakeLLM([tool_call("boom"), tool_call("terminate", message="recovered")])
    memory = build_agent(llm).run("greet Ada")

    failure = json.loads(memory.get_memories()[2]["content"])
    assert failure["tool_executed"] is False
    assert failure["error"] == "the guestbook is on fire"
    assert "RuntimeError" in failure["traceback"]
    assert last_result(memory)["result"] == "recovered"


def test_an_unknown_tool_returns_instructive_feedback():
    llm = FakeLLM([tool_call("fly_to_mars"), tool_call("terminate", message="ok")])
    memory = build_agent(llm).run("greet Ada")

    failure = json.loads(memory.get_memories()[2]["content"])
    assert failure["tool_executed"] is False
    assert "fly_to_mars" in failure["error"]
    assert "echo" in failure["error"]  # it is told what it may use instead


def test_prose_instead_of_a_tool_call_terminates_cleanly():
    llm = FakeLLM(["Welcome back, Dr. Chen!"])
    memory = build_agent(llm).run("greet Maria")

    assert len(llm.calls) == 1
    assert last_result(memory)["result"] == "Welcome back, Dr. Chen!"


def test_a_non_serialisable_result_still_reaches_memory():
    registry = build_registry()
    registry.register(Action("opaque", lambda: object(), "Returns junk.", SCHEMA))
    llm = FakeLLM([tool_call("opaque"), tool_call("terminate", message="ok")])

    memory = build_agent(llm, registry).run("greet Ada")

    assert "object at 0x" in memory.get_memories()[2]["content"]


# -------------------------------------------------------------- ActionContext


def test_the_action_context_carries_the_standard_resources():
    captured = {}
    llm = FakeLLM([tool_call("inspect_context"), tool_call("terminate", message="ok")])
    memory = build_agent(llm, build_registry(captured)).run("greet Ada")

    context = captured["context"]
    assert context.get_memory() is memory
    assert context.get("llm") is llm
    assert context.get("powerful_llm") is llm  # R1: an alias, same object
    assert context.get_action_registry() is not None
    assert context.get_agent_registry() is None  # delegation is off by default


def test_fast_llm_pins_the_cheap_tier():
    """MATE model efficiency: a tool can ask for the cheap model explicitly."""
    captured = {}
    llm = FakeLLM(
        [tool_call("inspect_context"), tool_call("terminate", message="ok"), "cheap answer"]
    )
    build_agent(llm, build_registry(captured)).run("greet Ada")

    answer = captured["context"].get("fast_llm")(llm.calls[0])

    assert answer == "cheap answer"
    assert llm.tiers == ["powerful", "powerful", "fast"]


def test_caller_supplied_props_reach_tools_and_win_over_defaults():
    captured = {}
    llm = FakeLLM([tool_call("inspect_context"), tool_call("terminate", message="ok")])
    agent = build_agent(llm, build_registry(captured))

    agent.run("greet Ada", action_context_props={"auth_token": "secret", "llm": "override"})

    assert captured["context"].get("auth_token") == "secret"
    assert captured["context"].get("llm") == "override"


def test_each_run_gets_its_own_context_id():
    captured = {}
    script = [tool_call("inspect_context"), tool_call("terminate", message="ok")]
    agent = build_agent(FakeLLM(script * 2), build_registry(captured))

    agent.run("first")
    first_id = captured["context"].context_id
    agent.run("second")

    assert captured["context"].context_id != first_id


def test_the_running_llm_is_always_the_powerful_tier():
    llm = FakeLLM([tool_call("terminate", message="ok")])
    build_agent(llm).run("greet Ada")

    assert llm.tiers == ["powerful"]


def test_fakellm_raises_when_the_loop_over_runs():
    """A guard rail for every other test in the suite."""
    llm = FakeLLM([tool_call("echo", text="again")])
    with pytest.raises(AssertionError, match="ran out of scripted responses"):
        build_agent(llm, max_iterations=2).run("greet Ada")
