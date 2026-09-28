"""Self-prompting tools: retries, fence stripping, and where the persona goes."""

from __future__ import annotations

import json

import pytest

from hal9000.core.action_context import ActionContext
from hal9000.core.actions import PythonActionRegistry
from hal9000.core.environment import PythonEnvironment
from hal9000.tools.self_prompting import (
    create_and_consult_expert,
    prompt_expert,
    prompt_llm,
    prompt_llm_for_json,
)
from tests.fakes import FakeLLM

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def context_with(llm: FakeLLM) -> ActionContext:
    """An ActionContext carrying only what these tools read: the powerful tier."""
    return ActionContext({"llm": llm})


# --- prompt_llm -------------------------------------------------------------


def test_prompt_llm_sends_one_user_message_and_returns_the_text():
    llm = FakeLLM(["a considered answer"])
    result = prompt_llm(context_with(llm), "What rhymes with Osaka?")

    assert result == "a considered answer"
    assert llm.last_prompt.messages == [{"role": "user", "content": "What rhymes with Osaka?"}]
    assert llm.last_prompt.tools is None  # plain prompting, not function calling


def test_missing_llm_raises_an_error_that_tells_the_agent_what_to_do():
    with pytest.raises(ValueError) as excinfo:
        prompt_llm(ActionContext({}), "hello")
    assert "'llm'" in str(excinfo.value)
    assert "terminate" in str(excinfo.value)


# --- prompt_llm_for_json ----------------------------------------------------


def test_json_tool_puts_the_schema_in_a_system_message():
    llm = FakeLLM(['{"name": "Ada"}'])
    result = prompt_llm_for_json(context_with(llm), SCHEMA, "Who is this?")

    assert result == {"name": "Ada"}
    system_message, user_message = llm.last_prompt.messages
    assert system_message["role"] == "system"
    assert json.dumps(SCHEMA) in system_message["content"]
    assert user_message == {"role": "user", "content": "Who is this?"}


@pytest.mark.parametrize(
    "response",
    [
        '{"name": "Ada"}',
        '```json\n{"name": "Ada"}\n```',
        'Here you go:\n```json\n{"name": "Ada"}\n```\nHope that helps.',
        '```\n{"name": "Ada"}\n```',
    ],
)
def test_json_tool_strips_code_fences(response):
    llm = FakeLLM([response])
    assert prompt_llm_for_json(context_with(llm), SCHEMA, "extract") == {"name": "Ada"}


def test_json_tool_retries_three_times_then_succeeds():
    """Garbage, garbage, then valid JSON -- the whole point of the retry loop."""
    llm = FakeLLM(["not json at all", "still ```not``` json", '{"name": "Ada"}'])

    assert prompt_llm_for_json(context_with(llm), SCHEMA, "extract") == {"name": "Ada"}
    assert len(llm.calls) == 3


def test_json_tool_raises_after_three_failures_with_a_recovery_instruction():
    llm = FakeLLM(["nope", "still nope", "nope again"])

    with pytest.raises(ValueError) as excinfo:
        prompt_llm_for_json(context_with(llm), SCHEMA, "extract")

    message = str(excinfo.value)
    assert len(llm.calls) == 3  # exactly three, not four
    assert "3 attempts" in message
    assert "simpler, flatter schema" in message  # the instruction the agent can act on
    assert excinfo.value.__cause__ is not None


def test_json_tool_does_not_retry_after_success():
    llm = FakeLLM(['{"name": "Ada"}', "unused"])
    prompt_llm_for_json(context_with(llm), SCHEMA, "extract")
    assert len(llm.calls) == 1


# --- prompt_expert ----------------------------------------------------------


def test_prompt_expert_puts_persona_in_system_and_task_in_user():
    llm = FakeLLM(["Welcome back, Dr. Chen."])
    persona = "A hotel concierge of thirty years, warm but never cloying."

    result = prompt_expert(context_with(llm), persona, "Greet a returning marine biologist.")

    assert result == "Welcome back, Dr. Chen."
    messages = llm.last_prompt.messages
    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert persona in messages[0]["content"]
    assert messages[0]["content"].startswith("Act as the following expert")

    assert messages[1] == {
        "role": "user",
        "content": "Greet a returning marine biologist.",
    }
    # The task must NOT leak into the system message, nor the persona into the user one.
    assert "Greet a returning" not in messages[0]["content"]
    assert persona not in messages[1]["content"]


# --- create_and_consult_expert ---------------------------------------------


def test_create_and_consult_expert_writes_persona_then_prompt_then_consults():
    llm = FakeLLM(["INVENTED PERSONA", "INVENTED PROMPT", "THE ADVICE"])

    result = create_and_consult_expert(
        context_with(llm),
        expertise_domain="conference interpreting",
        problem_description="Greet a visitor in Japanese without sounding stiff.",
    )

    assert result == "THE ADVICE"
    assert len(llm.calls) == 3

    persona_call, prompt_call, consult_call = llm.calls
    assert "conference interpreting" in persona_call.messages[0]["content"]
    assert "background and experience" in persona_call.messages[0]["content"]
    assert "consultation prompt" in prompt_call.messages[0]["content"]

    # The third call is prompt_expert, wearing the persona the first call invented.
    assert consult_call.messages[0]["role"] == "system"
    assert "INVENTED PERSONA" in consult_call.messages[0]["content"]
    assert consult_call.messages[1] == {"role": "user", "content": "INVENTED PROMPT"}


# --- as registered tools, end to end ---------------------------------------


def test_tools_run_through_the_environment_with_the_context_injected(registered_tools):
    """The model supplies `prompt`; the environment supplies the LLM."""
    llm = FakeLLM(["an answer"])
    action = PythonActionRegistry(tags=["prompts"]).get_action("prompt_llm")

    result = PythonEnvironment().execute_action(
        None, context_with(llm), action, {"prompt": "think about it"}
    )

    assert result == {
        "tool_executed": True,
        "result": "an answer",
        "timestamp": result["timestamp"],
    }
    assert llm.last_prompt.messages[0]["content"] == "think about it"


def test_tool_failure_reaches_the_agent_as_feedback_not_a_crash(registered_tools):
    llm = FakeLLM(["nope", "nope", "nope"])
    action = PythonActionRegistry(tags=["prompts"]).get_action("prompt_llm_for_json")

    result = PythonEnvironment().execute_action(
        None, context_with(llm), action, {"schema": SCHEMA, "prompt": "extract"}
    )

    assert result["tool_executed"] is False
    assert "simpler, flatter schema" in result["error"]
