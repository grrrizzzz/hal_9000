"""Actions, action registries, and the agent registry."""

from __future__ import annotations

import pytest

from hal9000.core import registry
from hal9000.core.actions import Action, ActionRegistry, PythonActionRegistry
from hal9000.core.registry import AgentRegistry

# --------------------------------------------------------------------- Action


def test_action_execute_passes_keyword_arguments():
    action = Action("add", lambda a, b: a + b, "Add two numbers.", {})
    assert action.execute(a=2, b=3) == 5


def test_action_keeps_its_contract():
    schema = {"type": "object", "properties": {"message": {"type": "string"}}}
    action = Action("terminate", lambda message: message, "Stop.", schema, terminal=True)

    assert action.name == "terminate"
    assert action.description == "Stop."
    assert action.parameters == schema
    assert action.terminal is True


def test_action_does_not_swallow_errors():
    """Error handling belongs to the environment, not the action."""

    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        Action("boom", boom, "Explode.", {}).execute()


# ------------------------------------------------------------- ActionRegistry


def test_registry_registers_and_looks_up():
    reg = ActionRegistry()
    action = Action("noop", lambda: None, "Nothing.", {})
    reg.register(action)

    assert reg.get_action("noop") is action
    assert reg.get_actions() == [action]
    assert len(reg) == 1


def test_unknown_action_is_none_not_an_exception():
    """The model invents tool names; that is feedback, not a crash."""
    assert ActionRegistry().get_action("hallucinated") is None


def test_registering_the_same_name_twice_replaces():
    reg = ActionRegistry()
    first = Action("noop", lambda: 1, "One.", {})
    second = Action("noop", lambda: 2, "Two.", {})
    reg.register(first)
    reg.register(second)

    assert reg.get_action("noop") is second
    assert len(reg) == 1


# ------------------------------------------------------- PythonActionRegistry


@pytest.fixture
def catalogue():
    """Populate the global tool catalogue by hand.

    T3's ``@register_tool`` writes these same entries; T2 only has to consume
    them, so the tests fill the dicts directly and leave them empty afterwards.
    """
    registry.clear_tools()
    entries = {
        "sign_guestbook": (["guestbook"], False),
        "read_guestbook": (["guestbook", "read"], False),
        "compose_greeting": (["greeting"], False),
        "terminate": (["system"], True),
    }
    for name, (tags, terminal) in entries.items():
        registry.tools[name] = {
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "function": lambda **kwargs: kwargs,
            "terminal": terminal,
            "tags": tags,
        }
        for tag in tags:
            registry.tools_by_tag.setdefault(tag, []).append(name)
    yield registry.tools
    registry.clear_tools()


def test_python_registry_selects_by_tag(catalogue):
    reg = PythonActionRegistry(tags=["guestbook"])
    assert sorted(a.name for a in reg.get_actions()) == ["read_guestbook", "sign_guestbook"]


def test_python_registry_selects_by_several_tags(catalogue):
    reg = PythonActionRegistry(tags=["greeting", "system"])
    assert sorted(a.name for a in reg.get_actions()) == ["compose_greeting", "terminate"]


def test_python_registry_selects_by_name(catalogue):
    reg = PythonActionRegistry(tool_names=["compose_greeting"])
    assert [a.name for a in reg.get_actions()] == ["compose_greeting"]


def test_python_registry_combines_filters_with_and(catalogue):
    reg = PythonActionRegistry(tags=["guestbook"], tool_names=["compose_greeting"])
    assert reg.get_actions() == []


def test_unknown_tag_yields_an_empty_registry_not_an_error(catalogue):
    """A feature flag that is off must not explode the assembly."""
    assert PythonActionRegistry(tags=["no_such_tag"]).get_actions() == []


def test_no_filters_takes_the_whole_catalogue(catalogue):
    assert len(PythonActionRegistry()) == len(catalogue)


def test_python_registry_carries_metadata_onto_the_action(catalogue):
    action = PythonActionRegistry(tags=["system"]).get_action("terminate")

    assert action.terminal is True
    assert action.description == "terminate description"
    assert action.parameters == {"type": "object", "properties": {}, "required": []}
    assert action.function is catalogue["terminate"]["function"]


def test_empty_catalogue_yields_an_empty_registry():
    """T2 ships with nothing registered; T3's decorator fills the dicts."""
    registry.clear_tools()
    assert PythonActionRegistry(tags=["guestbook"]).get_actions() == []


# -------------------------------------------------------------- AgentRegistry


def test_agent_registry_stores_run_functions():
    agents = AgentRegistry()
    agents.register_agent("echo", lambda user_input, **kwargs: user_input)

    assert agents.get_agent("echo")("hello") == "hello"
    assert agents.list_agents() == ["echo"]


def test_unknown_agent_is_none():
    assert AgentRegistry().get_agent("nobody") is None
