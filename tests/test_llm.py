"""The keystone contract: ``generate_response(Prompt, tier) -> str``.

No network: ``litellm.completion`` is monkeypatched with a stub that returns the
shapes a real provider would.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hal9000 import llm
from hal9000.core.prompt import Prompt
from tests.fakes import FakeLLM, tool_call

TOOLS = [{"type": "function", "function": {"name": "sign_guestbook", "parameters": {}}}]


def _message(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


@pytest.fixture
def captured(monkeypatch):
    """Stub litellm.completion; returns the recorded kwargs and lets tests set a reply."""
    import litellm

    box = {"kwargs": None, "reply": _message(content="hello")}

    def fake_completion(**kwargs):
        box["kwargs"] = kwargs
        return box["reply"]

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return box


def test_plain_text_response_is_returned_verbatim(captured, openai_env):
    assert llm.generate_response(Prompt(messages=[{"role": "user", "content": "hi"}])) == "hello"


def test_tool_call_becomes_tool_args_json(captured, openai_env):
    captured["reply"] = _message(
        tool_calls=[_tool_call("sign_guestbook", '{"name": "Ada", "greeting": "Welcome"}')]
    )
    result = llm.generate_response(Prompt(messages=[], tools=TOOLS))
    assert json.loads(result) == {
        "tool": "sign_guestbook",
        "args": {"name": "Ada", "greeting": "Welcome"},
    }


def test_tool_calls_are_ignored_when_the_prompt_offered_no_tools(captured, openai_env):
    captured["reply"] = _message(content="just talking", tool_calls=[_tool_call("x", "{}")])
    assert llm.generate_response(Prompt(messages=[])) == "just talking"


def test_empty_content_becomes_empty_string(captured, openai_env):
    captured["reply"] = _message(content=None)
    assert llm.generate_response(Prompt(messages=[])) == ""


def test_tier_selects_the_model_and_tools_are_forwarded(captured, anthropic_env):
    prompt = Prompt(messages=[{"role": "user", "content": "hi"}], tools=TOOLS)
    llm.generate_response(prompt, tier="fast")
    assert captured["kwargs"]["model"] == "anthropic/claude-haiku-4-5"
    assert captured["kwargs"]["messages"] is prompt.messages
    assert captured["kwargs"]["tools"] == TOOLS
    assert captured["kwargs"]["max_tokens"] == 4096
    assert captured["kwargs"]["parallel_tool_calls"] is False

    llm.generate_response(prompt)
    assert captured["kwargs"]["model"] == "anthropic/claude-sonnet-5"


def test_no_tools_is_passed_as_none_not_empty_list(captured, openai_env):
    llm.generate_response(Prompt(messages=[], tools=[]))
    assert captured["kwargs"]["tools"] is None


# --- the fake that stands in for all of the above ---------------------------


def test_tool_call_helper_matches_the_real_shape():
    assert json.loads(tool_call("terminate", message="done")) == {
        "tool": "terminate",
        "args": {"message": "done"},
    }
    assert json.loads(tool_call("read_guestbook")) == {"tool": "read_guestbook", "args": {}}


def test_fake_llm_is_scripted_and_records_prompts_and_tiers():
    fake = FakeLLM([tool_call("read_guestbook"), "plain text"])
    first, second = Prompt(messages=[{"role": "user", "content": "a"}]), Prompt(messages=[])

    assert fake(first, tier="fast") == tool_call("read_guestbook")
    assert fake(second) == "plain text"

    assert fake.calls == [first, second]
    assert fake.tiers == ["fast", "powerful"]
    assert fake.last_prompt is second


def test_fake_llm_raises_a_helpful_error_when_the_script_runs_out():
    fake = FakeLLM([])
    with pytest.raises(AssertionError, match="ran out of scripted responses"):
        fake(Prompt())


def test_fake_llm_is_substitutable_for_generate_response():
    import inspect

    real = inspect.signature(llm.generate_response).parameters
    fake = inspect.signature(FakeLLM([]).__call__).parameters
    assert list(real)[:2] == list(fake) == ["prompt", "tier"]
    assert real["on_usage"].kind is inspect.Parameter.KEYWORD_ONLY
    assert real["tier"].default == fake["tier"].default == "powerful"
