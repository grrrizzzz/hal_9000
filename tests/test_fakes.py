"""The test double itself, tested.

``tests/fakes.py`` is load-bearing: 392 other tests describe the model's
behaviour through it, so a bug here would be a bug everywhere at once and
visible nowhere in particular. Two things are worth pinning down.

**The positional-only marker on ``tool_call`` (ruling R18).** Its first
parameter is ``name``, and ``sign_guestbook`` -- the one tool in the template
that writes -- takes an argument also called ``name``. Without ``/`` the two
collide and scripting the template's central write raises ``TypeError``.

**Running out of script is an explicit failure.** A ``FakeLLM`` asked for one
more response than it has must say so, because the alternative -- returning
``None`` or an empty string -- surfaces much later as an unparseable response
that the function-calling language quietly turns into ``terminate``.
"""

from __future__ import annotations

import inspect
import json

import pytest

from hello_agent.core.prompt import Prompt
from tests.fakes import FakeLLM, tool_call


def test_tool_call_accepts_a_tool_argument_named_name():
    """Ruling R18: ``name`` is the tool's name *and* a legal argument name."""
    scripted = tool_call("sign_guestbook", name="Ada", greeting="Welcome, Ada!")

    assert json.loads(scripted) == {
        "tool": "sign_guestbook",
        "args": {"name": "Ada", "greeting": "Welcome, Ada!"},
    }


def test_tool_call_name_is_positional_only():
    """The marker is the mechanism, so assert the marker and not just its effect."""
    parameter = inspect.signature(tool_call).parameters["name"]

    assert parameter.kind is inspect.Parameter.POSITIONAL_ONLY


def test_tool_call_with_no_arguments_still_produces_an_empty_args_object():
    assert json.loads(tool_call("terminate")) == {"tool": "terminate", "args": {}}


def test_fake_llm_replays_in_order_and_records_prompts_and_tiers():
    llm = FakeLLM([tool_call("terminate", message="first"), "second"])
    prompt = Prompt(messages=[{"role": "user", "content": "hi"}])

    assert json.loads(llm(prompt))["args"] == {"message": "first"}
    assert llm(prompt, tier="fast") == "second"
    assert llm.calls == [prompt, prompt]
    assert llm.tiers == ["powerful", "fast"]
    assert llm.last_prompt is prompt


def test_fake_llm_running_out_of_script_fails_loudly():
    llm = FakeLLM([])

    with pytest.raises(AssertionError, match="ran out of scripted responses"):
        llm(Prompt(messages=[]))
