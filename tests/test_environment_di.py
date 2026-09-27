"""Environment execution and dependency injection.

Two guarantees are tested here, and the agent loop rests on both:

1. A tool receives exactly the dependencies it declares -- no more.
2. A tool that raises produces a *result*, never an exception at the loop.
"""

from __future__ import annotations

from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import Action
from hello_agent.core.environment import Environment, PythonEnvironment, has_named_parameter

SCHEMA = {"type": "object", "properties": {}, "required": []}


def make_action(function, name="tool", terminal=False) -> Action:
    return Action(name, function, "A tool.", SCHEMA, terminal=terminal)


# ----------------------------------------------------------- base Environment


def test_successful_result_is_wrapped_with_metadata():
    result = Environment().execute_action(
        None, ActionContext(), make_action(lambda name: f"hi {name}"), {"name": "Ada"}
    )

    assert result["tool_executed"] is True
    assert result["result"] == "hi Ada"
    assert "timestamp" in result


def test_tool_exception_becomes_a_result_not_a_crash():
    def broken():
        raise ValueError("no guestbook configured")

    result = Environment().execute_action(None, ActionContext(), make_action(broken), {})

    assert result["tool_executed"] is False
    assert result["error"] == "no guestbook configured"
    assert "ValueError: no guestbook configured" in result["traceback"]


def test_wrong_arguments_are_caught_too():
    """A model that invents a parameter gets feedback, not a dead agent."""
    result = Environment().execute_action(
        None, ActionContext(), make_action(lambda name: name), {"nmae": "typo"}
    )

    assert result["tool_executed"] is False
    assert "nmae" in result["error"] or "name" in result["error"]


# --------------------------------------------------------- PythonEnvironment


def test_action_context_is_injected_when_declared():
    seen = {}

    def tool(action_context: ActionContext, name: str) -> str:
        seen["context"] = action_context
        return name

    context = ActionContext({"memory": "m"})
    result = PythonEnvironment().execute_action(None, context, make_action(tool), {"name": "Ada"})

    assert result["tool_executed"] is True
    assert seen["context"] is context


def test_underscore_parameters_receive_matching_context_properties():
    captured = {}

    def tool(name: str, _guestbook, _auth_token: str) -> str:
        captured.update(guestbook=_guestbook, token=_auth_token)
        return name

    context = ActionContext({"guestbook": ["entry"], "auth_token": "secret"})
    PythonEnvironment().execute_action(None, context, make_action(tool), {"name": "Ada"})

    assert captured == {"guestbook": ["entry"], "token": "secret"}


def test_a_tool_that_declares_nothing_receives_nothing():
    """The security half of the contract: no ambient access to the context."""
    captured = {}

    def tool(**kwargs) -> str:
        captured.update(kwargs)
        return "ok"

    context = ActionContext({"auth_token": "secret", "memory": "m"})
    PythonEnvironment().execute_action(None, context, make_action(tool), {"name": "Ada"})

    assert captured == {"name": "Ada"}


def test_action_agent_is_injected_when_declared():
    captured = {}

    def tool(action_agent) -> str:
        captured["agent"] = action_agent
        return "ok"

    agent = object()
    PythonEnvironment().execute_action(agent, ActionContext(), make_action(tool), {})

    assert captured["agent"] is agent


def test_a_missing_dependency_surfaces_as_a_tool_error():
    """No ``_storage`` in the context means the call fails -- as a result dict."""

    def tool(_storage) -> str:
        return "unreachable"

    result = PythonEnvironment().execute_action(None, ActionContext(), make_action(tool), {})

    assert result["tool_executed"] is False
    assert "_storage" in result["error"]


def test_injection_does_not_mutate_the_caller_s_arguments():
    def tool(action_context: ActionContext, name: str) -> str:
        return name

    args = {"name": "Ada"}
    PythonEnvironment().execute_action(None, ActionContext(), make_action(tool), args)

    assert args == {"name": "Ada"}


def test_inject_dependencies_is_inspectable_on_its_own():
    def tool(action_context: ActionContext, name: str, _auth_token: str) -> str:
        return name

    context = ActionContext({"auth_token": "secret", "unused": "ignored"})
    injected = PythonEnvironment().inject_dependencies(
        None, context, make_action(tool), {"name": "Ada"}
    )

    assert set(injected) == {"name", "action_context", "_auth_token"}


# ------------------------------------------------------- has_named_parameter


def test_has_named_parameter():
    def tool(action_context, name):
        return name

    assert has_named_parameter(tool, "action_context") is True
    assert has_named_parameter(tool, "_auth_token") is False
    # Builtins cannot always be introspected; that must not raise.
    assert has_named_parameter(len, "obj") in (True, False)
