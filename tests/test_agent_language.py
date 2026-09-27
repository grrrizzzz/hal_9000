"""Both agent languages: prompt construction and response parsing."""

from __future__ import annotations

import json

import pytest

from hello_agent.core.actions import Action
from hello_agent.core.agent_language import (
    AgentFunctionCallingActionLanguage,
    AgentJsonActionLanguage,
    AgentLanguage,
)
from hello_agent.core.goals import Goal
from hello_agent.core.memory import Memory

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
ACTIONS = [
    Action("sign_guestbook", lambda name: name, "Record a visit.", SCHEMA),
    Action("terminate", lambda message: message, "Stop.", SCHEMA, terminal=True),
]
GOALS = [
    Goal(priority=2, name="sign", description="Sign the guestbook."),
    Goal(priority=1, name="greet", description="Greet the visitor."),
]


# ------------------------------------------------------- the shared base class


def test_base_class_demands_an_implementation():
    language = AgentLanguage()
    with pytest.raises(NotImplementedError):
        language.construct_prompt([], None, [], Memory())
    with pytest.raises(NotImplementedError):
        language.parse_response("{}")


def test_goals_render_as_one_system_message_highest_priority_first():
    messages = AgentLanguage().format_goals(GOALS)

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].index("greet:") < messages[0]["content"].index("sign:")
    assert "Greet the visitor." in messages[0]["content"]


def test_memory_types_map_onto_chatml_roles():
    memory = Memory(
        items=[
            {"type": "system", "content": "framing"},
            {"type": "user", "content": "greet Ada"},
            {"type": "assistant", "content": "on it"},
            {"type": "environment", "content": "unknown types become user turns"},
        ]
    )

    roles = [m["role"] for m in AgentLanguage().format_memory(memory)]

    assert roles == ["system", "user", "assistant", "user"]


def test_non_string_memory_content_is_json_encoded():
    memory = Memory(items=[{"type": "user", "content": {"tool_executed": True}}])

    content = AgentLanguage().format_memory(memory)[0]["content"]

    assert json.loads(content) == {"tool_executed": True}


# ------------------------------------------ AgentFunctionCallingActionLanguage


def test_function_calling_puts_schemas_in_the_tools_field():
    prompt = AgentFunctionCallingActionLanguage().construct_prompt(
        actions=ACTIONS, environment=None, goals=GOALS, memory=Memory()
    )

    assert [t["function"]["name"] for t in prompt.tools] == ["sign_guestbook", "terminate"]
    assert prompt.tools[0]["type"] == "function"
    assert prompt.tools[0]["function"]["parameters"] == SCHEMA
    assert [m["role"] for m in prompt.messages] == ["system"]


def test_function_calling_truncates_overlong_descriptions():
    """Providers cap tool descriptions; truncate rather than 400."""
    long_action = Action("verbose", lambda: None, "x" * 5000, SCHEMA)
    prompt = AgentFunctionCallingActionLanguage().construct_prompt(
        actions=[long_action], environment=None, goals=GOALS, memory=Memory()
    )

    assert len(prompt.tools[0]["function"]["description"]) == 1024


def test_function_calling_includes_memory_as_conversation():
    memory = Memory(items=[{"type": "user", "content": "greet Ada"}])
    prompt = AgentFunctionCallingActionLanguage().construct_prompt(
        actions=ACTIONS, environment=None, goals=GOALS, memory=memory
    )

    assert prompt.messages[-1] == {"role": "user", "content": "greet Ada"}


def test_function_calling_parses_a_tool_call():
    response = json.dumps({"tool": "sign_guestbook", "args": {"name": "Ada"}})

    assert AgentFunctionCallingActionLanguage().parse_response(response) == {
        "tool": "sign_guestbook",
        "args": {"name": "Ada"},
    }


def test_unparseable_output_falls_back_to_terminate():
    """Prose means the model thinks it is done. Route it to terminate, don't error."""
    response = "Welcome back, Dr. Chen!"

    assert AgentFunctionCallingActionLanguage().parse_response(response) == {
        "tool": "terminate",
        "args": {"message": "Welcome back, Dr. Chen!"},
    }


# ------------------------------------------------------ AgentJsonActionLanguage


def test_json_language_describes_tools_in_a_system_message():
    prompt = AgentJsonActionLanguage().construct_prompt(
        actions=ACTIONS, environment=None, goals=GOALS, memory=Memory()
    )

    catalogue = prompt.messages[1]["content"]

    assert prompt.tools is None
    assert prompt.messages[1]["role"] == "system"
    assert "sign_guestbook" in catalogue and "Record a visit." in catalogue
    assert "```action" in catalogue


def test_json_language_round_trips_a_fenced_action_block():
    invocation = {"tool": "sign_guestbook", "args": {"name": "Ada Lovelace"}}
    response = (
        "Let me record the visit.\n\n```action\n" + json.dumps(invocation, indent=4) + "\n```"
    )

    assert AgentJsonActionLanguage().parse_response(response) == invocation


def test_json_language_ignores_prose_around_the_block():
    response = '\n  thinking...\n```action\n{"tool": "terminate", "args": {}}\n```\n'

    assert AgentJsonActionLanguage().parse_response(response) == {"tool": "terminate", "args": {}}


def test_json_language_raises_when_the_block_is_missing():
    with pytest.raises(ValueError, match="No ```action block"):
        AgentJsonActionLanguage().parse_response("I have decided to stop.")


def test_json_language_raises_when_the_block_is_unterminated():
    with pytest.raises(ValueError, match="Unterminated"):
        AgentJsonActionLanguage().parse_response('```action\n{"tool": "terminate"}')


def test_json_language_raises_on_malformed_json():
    with pytest.raises(ValueError, match="Malformed JSON"):
        AgentJsonActionLanguage().parse_response("```action\n{not json}\n```")
