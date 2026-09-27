"""Delegation: the four memory-sharing patterns, and the limits on a sub-agent.

The interesting assertions here are about *memory*, because that is the only
thing the four tools actually differ in, and about what a callee is **not**
given -- no agent registry, therefore no way to delegate back.
"""

from __future__ import annotations

import json

import pytest

from hello_agent.agents import TRANSLATOR_AGENT_NAME, build_translator_agent
from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import PythonActionRegistry
from hello_agent.core.agent import Agent
from hello_agent.core.agent_language import AgentFunctionCallingActionLanguage
from hello_agent.core.environment import PythonEnvironment
from hello_agent.core.goals import Goal
from hello_agent.core.memory import Memory
from hello_agent.core.registry import AgentRegistry
from hello_agent.tools.delegation import (
    NOT_INHERITED,
    call_agent,
    call_agent_with_reflection,
    call_agent_with_selected_context,
    hand_off_to_agent,
)
from tests.fakes import FakeLLM, tool_call

CALLER_MEMORIES = [
    {"type": "user", "content": "Dr. Chen is here, she would like Japanese."},
    {"type": "assistant", "content": "Extracted: name=Dr. Chen, language=Japanese"},
    {"type": "system", "content": "Right now it is 09:00."},
]


class ScriptedAgent:
    """A stand-in sub-agent: the ``run`` signature, and a record of every call.

    A real :class:`Agent` is used where the *loop* is under test; this is for the
    tests that are only about what memory crossed the boundary.
    """

    def __init__(self, *replies: str):
        self.replies = replies or ("done",)
        self.calls: list[dict] = []

    def run(self, user_input: str, memory=None, action_context_props: dict = None) -> Memory:
        memory = memory if memory is not None else Memory()
        self.calls.append(
            {
                "user_input": user_input,
                "memory": memory,
                "incoming": [dict(item) for item in memory.items],
                "props": dict(action_context_props or {}),
            }
        )
        memory.add_memory({"type": "user", "content": user_input})
        for reply in self.replies:
            memory.add_memory({"type": "assistant", "content": reply})
        return memory

    @property
    def last_call(self) -> dict:
        return self.calls[-1]


@pytest.fixture
def delegating_context():
    """An ActionContext shaped the way a delegating run's context is shaped.

    Returns ``(context, agent_registry, memory)``. The registry starts empty; a
    test registers whichever callee it needs.
    """

    def _make(memory_items=CALLER_MEMORIES, **properties) -> tuple:
        memory = Memory(items=[dict(item) for item in memory_items])
        agent_registry = AgentRegistry()
        context = ActionContext({"memory": memory, "agent_registry": agent_registry, **properties})
        return context, agent_registry, memory

    return _make


# --- message passing --------------------------------------------------------


def test_call_agent_gives_the_callee_a_fresh_memory(delegating_context):
    context, agent_registry, caller_memory = delegating_context()
    callee = ScriptedAgent("konnichiwa")
    agent_registry.register_agent("translator", callee.run)

    call_agent(context, "translator", "Translate the greeting into Japanese.")

    assert callee.last_call["incoming"] == []
    assert callee.last_call["memory"] is not caller_memory


def test_call_agent_returns_only_the_last_memory_item(delegating_context):
    context, agent_registry, caller_memory = delegating_context()
    agent_registry.register_agent("translator", ScriptedAgent("thinking", "konnichiwa").run)

    result = call_agent(context, "translator", "Translate it.")

    assert result == {"success": True, "agent": "translator", "result": "konnichiwa"}
    # Message passing leaves the caller's memory exactly as it found it.
    assert caller_memory.items == CALLER_MEMORIES


# --- reflection -------------------------------------------------------------


def test_reflection_copies_callee_memories_back_tagged_with_their_source(delegating_context):
    context, agent_registry, caller_memory = delegating_context()
    agent_registry.register_agent("translator", ScriptedAgent("thinking", "konnichiwa").run)

    result = call_agent_with_reflection(context, "translator", "Translate it.")

    added = caller_memory.items[len(CALLER_MEMORIES) :]
    assert [item["type"] for item in added] == ["translator_thought"] * 3
    assert [item["content"] for item in added] == ["Translate it.", "thinking", "konnichiwa"]
    assert result["memories_added"] == 3
    assert result["result"] == "konnichiwa"


def test_reflection_still_starts_the_callee_from_a_clean_slate(delegating_context):
    context, agent_registry, _ = delegating_context()
    callee = ScriptedAgent("konnichiwa")
    agent_registry.register_agent("translator", callee.run)

    call_agent_with_reflection(context, "translator", "Translate it.")

    assert callee.last_call["incoming"] == []


# --- handoff ----------------------------------------------------------------


def test_handoff_passes_the_callers_own_memory_object(delegating_context):
    context, agent_registry, caller_memory = delegating_context()
    callee = ScriptedAgent("konnichiwa")
    agent_registry.register_agent("translator", callee.run)

    result = hand_off_to_agent(context, "translator", "Take it from here.")

    assert callee.last_call["memory"] is caller_memory
    assert result["memory_id"] == id(caller_memory)
    # The callee wrote into the caller's memory directly -- no copying back.
    assert [item["content"] for item in caller_memory.items[len(CALLER_MEMORIES) :]] == [
        "Take it from here.",
        "konnichiwa",
    ]


def test_handoff_shows_the_callee_the_whole_history(delegating_context):
    context, agent_registry, _ = delegating_context()
    callee = ScriptedAgent()
    agent_registry.register_agent("translator", callee.run)

    hand_off_to_agent(context, "translator", "Take it from here.")

    assert callee.last_call["incoming"] == CALLER_MEMORIES


# --- selective memory sharing -----------------------------------------------

SELECTION = {
    "selected_memories": ["mem_0", "mem_1"],
    "reasoning": "Both name the visitor and her language; the clock is irrelevant.",
}


def test_selective_shares_only_the_memories_the_llm_chose(delegating_context):
    llm = FakeLLM([json.dumps(SELECTION)])
    context, agent_registry, _ = delegating_context(llm=llm)
    callee = ScriptedAgent("konnichiwa")
    agent_registry.register_agent("translator", callee.run)

    result = call_agent_with_selected_context(context, "translator", "Translate it.")

    assert callee.last_call["incoming"] == CALLER_MEMORIES[:2]
    assert result["shared_memories"] == 2


def test_selective_shows_the_llm_every_memory_with_an_id(delegating_context):
    llm = FakeLLM([json.dumps(SELECTION)])
    context, agent_registry, _ = delegating_context(llm=llm)
    agent_registry.register_agent("translator", ScriptedAgent().run)

    call_agent_with_selected_context(context, "translator", "Translate it.")

    asked = llm.last_prompt.messages[-1]["content"]
    for index, item in enumerate(CALLER_MEMORIES):
        assert f"Memory mem_{index}: {item['content']}" in asked


def test_selective_preserves_the_selection_reasoning_in_caller_memory(delegating_context):
    llm = FakeLLM([json.dumps(SELECTION)])
    context, agent_registry, caller_memory = delegating_context(llm=llm)
    agent_registry.register_agent("translator", ScriptedAgent("konnichiwa").run)

    result = call_agent_with_selected_context(context, "translator", "Translate it.")

    added = caller_memory.items[len(CALLER_MEMORIES) :]
    assert added[0] == {
        "type": "system",
        "content": f"Memory selection reasoning: {SELECTION['reasoning']}",
    }
    assert result["selection_reasoning"] == SELECTION["reasoning"]
    # Then the sub-agent's whole memory, shared items included. The note is
    # explicit about accepting that duplication; asserting it keeps the
    # behaviour visible rather than surprising.
    assert [item["content"] for item in added[1:]] == [
        CALLER_MEMORIES[0]["content"],
        CALLER_MEMORIES[1]["content"],
        "Translate it.",
        "konnichiwa",
    ]


def test_selective_does_not_pay_for_a_selection_when_the_agent_is_unknown(delegating_context):
    llm = FakeLLM([])  # any LLM call at all would raise AssertionError
    context, _, _ = delegating_context(llm=llm)

    result = call_agent_with_selected_context(context, "nobody", "Translate it.")

    assert result["success"] is False
    assert llm.calls == []


# --- the four patterns really are four --------------------------------------


def test_the_four_patterns_differ_only_in_what_memory_crosses(delegating_context):
    """One table, so a reader can see the whole design at a glance."""
    observed = {}
    for tool in (call_agent, call_agent_with_reflection, hand_off_to_agent):
        llm = FakeLLM([json.dumps(SELECTION)])
        context, agent_registry, caller_memory = delegating_context(llm=llm)
        callee = ScriptedAgent("konnichiwa")
        agent_registry.register_agent("translator", callee.run)
        tool(context, "translator", "Translate it.")
        observed[tool.__name__] = (
            len(callee.last_call["incoming"]),
            len(caller_memory.items) - len(CALLER_MEMORIES),
            callee.last_call["memory"] is caller_memory,
        )

    assert observed == {
        # tool: (memories sent down, memories gained by caller, shared object?)
        "call_agent": (0, 0, False),
        "call_agent_with_reflection": (0, 2, False),
        "hand_off_to_agent": (3, 2, True),
    }


# --- what the callee is not given -------------------------------------------


def test_call_agent_does_not_pass_the_agent_registry_down(delegating_context):
    context, agent_registry, _ = delegating_context(auth_token="secret", greeting_policy="Be warm.")
    callee = ScriptedAgent()
    agent_registry.register_agent("translator", callee.run)

    call_agent(context, "translator", "Translate it.")

    props = callee.last_call["props"]
    assert "agent_registry" not in props
    # Request-scoped values still travel, which is the point of forwarding at all.
    assert props == {"auth_token": "secret", "greeting_policy": "Be warm."}


@pytest.mark.parametrize("tool", [call_agent, call_agent_with_reflection, hand_off_to_agent])
def test_no_delegation_tool_passes_the_registry_down(delegating_context, tool):
    context, agent_registry, _ = delegating_context()
    callee = ScriptedAgent()
    agent_registry.register_agent("translator", callee.run)

    tool(context, "translator", "Translate it.")

    assert set(callee.last_call["props"]) & NOT_INHERITED == set()


def _loop_agent(llm: FakeLLM) -> Agent:
    """A real agent whose only tools are `call_agent` and `terminate`."""
    return Agent(
        goals=[Goal(priority=1, name="delegate", description="Delegate, then terminate.")],
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=PythonActionRegistry(tool_names=["call_agent", "terminate"]),
        generate_response=llm,
        environment=PythonEnvironment(),
        max_iterations=4,
    )


def test_a_delegated_agent_cannot_delegate_back(registered_tools):
    """Infinite recursion is not merely discouraged: the callee has no registry."""
    outer = _loop_agent(
        FakeLLM(
            [
                tool_call("call_agent", agent_name="inner", task="your turn"),
                tool_call("terminate", message="finished"),
            ]
        )
    )
    inner = _loop_agent(
        FakeLLM(
            [
                tool_call("call_agent", agent_name="outer", task="back to you"),
                tool_call("terminate", message="I could not delegate"),
            ]
        )
    )

    seen: dict = {}

    def inner_run(user_input, memory=None, action_context_props=None):
        seen["props"] = dict(action_context_props or {})
        seen["memory"] = inner.run(user_input, memory, action_context_props)
        return seen["memory"]

    agent_registry = AgentRegistry()
    agent_registry.register_agent("outer", outer.run)
    agent_registry.register_agent("inner", inner_run)

    outer_memory = outer.run("Greet her.", action_context_props={"agent_registry": agent_registry})

    assert "agent_registry" not in seen["props"]
    inner_transcript = json.dumps(seen["memory"].items)
    assert "No agent registry is available" in inner_transcript
    # Both loops ended on their own terms rather than running away.
    assert "finished" in json.dumps(outer_memory.items)
    assert "I could not delegate" in inner_transcript


# --- failures are results, not exceptions ------------------------------------


def test_unknown_agent_returns_a_structured_error(delegating_context):
    context, agent_registry, _ = delegating_context()
    agent_registry.register_agent("translator", ScriptedAgent().run)

    result = call_agent(context, "sommelier", "Pick a wine.")

    assert result["success"] is False
    assert result["agent"] == "sommelier"
    assert "no agent named 'sommelier'" in result["error"]
    # The error names who *is* available -- feedback the agent can act on.
    assert "translator" in result["error"]


def test_missing_agent_registry_returns_a_structured_error():
    context = ActionContext({"memory": Memory()})

    result = call_agent(context, "translator", "Translate it.")

    assert result["success"] is False
    assert "No agent registry" in result["error"]


def test_a_sub_agent_that_raises_is_reported_not_propagated(delegating_context):
    context, agent_registry, _ = delegating_context()

    def exploding_run(user_input, memory=None, action_context_props=None):
        raise RuntimeError("the translator fell over")

    agent_registry.register_agent("translator", exploding_run)

    result = call_agent(context, "translator", "Translate it.")

    assert result == {
        "success": False,
        "agent": "translator",
        "error": "RuntimeError: the translator fell over",
    }


def test_an_unknown_agent_does_not_escape_the_agent_loop(registered_tools):
    """Through the real environment: a bad agent name is a result the loop reads."""
    llm = FakeLLM(
        [
            tool_call("call_agent", agent_name="sommelier", task="Pick a wine."),
            tool_call("terminate", message="No such agent; I handled it myself."),
        ]
    )
    agent = _loop_agent(llm)

    memory = agent.run("Greet her.", action_context_props={"agent_registry": AgentRegistry()})

    delegation_result = json.loads(memory.items[2]["content"])
    assert delegation_result["tool_executed"] is True  # the tool returned, it did not raise
    assert delegation_result["result"]["success"] is False
    assert "No such agent" in memory.items[-1]["content"]


# --- the translator sub-agent -----------------------------------------------


def test_translator_runs_standalone_and_returns_a_translation(registered_tools):
    llm = FakeLLM([tool_call("terminate", message="ようこそ、チェン博士。")])

    memory = build_translator_agent(generate_response=llm).run(
        "Translate into Japanese: Welcome, Dr. Chen."
    )

    assert json.loads(memory.items[-1]["content"])["result"] == "ようこそ、チェン博士。"


def test_translator_carries_its_persona_in_its_goals(registered_tools):
    llm = FakeLLM([tool_call("terminate", message="ようこそ")])

    build_translator_agent(generate_response=llm).run("Translate into Japanese: Welcome.")

    system_message = llm.last_prompt.messages[0]["content"]
    assert "conference interpreter" in system_message
    assert "register, warmth and formality" in system_message


def test_translator_is_a_leaf_agent(registered_tools):
    """One tool, and none of them is a delegation tool: it answers and stops."""
    agent = build_translator_agent(generate_response=FakeLLM([]))

    assert [action.name for action in agent.actions.get_actions()] == ["terminate"]


def test_translator_can_be_registered_and_called(registered_tools):
    """The end-to-end shape T8 will assemble: greeter -> registry -> translator."""
    translator = build_translator_agent(
        generate_response=FakeLLM([tool_call("terminate", message="ようこそ、チェン博士。")])
    )
    agent_registry = AgentRegistry()
    agent_registry.register_agent(TRANSLATOR_AGENT_NAME, translator.run)
    context = ActionContext({"memory": Memory(), "agent_registry": agent_registry})

    result = call_agent(context, TRANSLATOR_AGENT_NAME, "Translate into Japanese: Welcome.")

    assert result["success"] is True
    # The translation itself, not the framework's result envelope around it.
    assert result["result"] == "ようこそ、チェン博士。"


# --- the lean default --------------------------------------------------------


def test_delegation_tools_are_tagged_delegation_and_absent_by_default(registered_tools):
    delegation_tools = set(registered_tools.tools_by_tag["delegation"])

    assert delegation_tools == {
        "call_agent",
        "call_agent_with_reflection",
        "hand_off_to_agent",
        "call_agent_with_selected_context",
    }
    default_registry = PythonActionRegistry(tags=["greeting", "guestbook", "system"])
    assert delegation_tools & {action.name for action in default_registry.get_actions()} == set()


def test_delegation_tool_schemas_hide_the_action_context(registered_tools):
    schema = registered_tools.tools["call_agent"]["parameters"]

    assert schema["properties"] == {"agent_name": {"type": "string"}, "task": {"type": "string"}}
    assert schema["required"] == ["agent_name", "task"]


@pytest.mark.parametrize(
    "tool",
    [call_agent, call_agent_with_reflection, hand_off_to_agent, call_agent_with_selected_context],
)
def test_every_delegation_tool_contains_a_sub_agent_failure(delegating_context, tool):
    """Whichever pattern you choose, a failing callee is a result, not a crash."""
    context, agent_registry, caller_memory = delegating_context(
        llm=FakeLLM([json.dumps(SELECTION)])
    )

    def exploding_run(user_input, memory=None, action_context_props=None):
        raise RuntimeError("the translator fell over")

    agent_registry.register_agent("translator", exploding_run)

    result = tool(context, "translator", "Translate it.")

    assert result["success"] is False
    assert "the translator fell over" in result["error"]
    # Nothing is written back from a run that did not happen.
    assert caller_memory.items == CALLER_MEMORIES


def test_an_agent_that_produces_no_memories_is_reported(delegating_context):
    context, agent_registry, _ = delegating_context()
    agent_registry.register_agent(
        "mute", lambda user_input, memory=None, action_context_props=None: Memory()
    )

    result = call_agent(context, "mute", "Say something.")

    assert result["success"] is False
    assert "produced no memories" in result["error"]


def test_a_structured_answer_survives_the_result_envelope(delegating_context):
    """A sub-agent whose answer is an object, not prose, comes back as JSON."""
    context, agent_registry, _ = delegating_context()
    envelope = json.dumps({"tool_executed": True, "result": {"entry_id": 7}})
    agent_registry.register_agent("clerk", ScriptedAgent(envelope).run)

    result = call_agent(context, "clerk", "Sign the guestbook.")

    assert json.loads(result["result"]) == {"entry_id": 7}
