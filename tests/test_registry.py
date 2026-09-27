"""The global catalogue: the tag index, tag-filtered registries, and idempotent import."""

from __future__ import annotations

import importlib

from hello_agent.core import registry
from hello_agent.core.actions import PythonActionRegistry
from hello_agent.tools import self_prompting, system
from hello_agent.tools.decorators import register_tool

#: Everything T3 owns. Later waves add tools; these must always be present.
T3_TOOLS = {
    "terminate": ["system"],
    "prompt_llm": ["prompts"],
    "prompt_llm_for_json": ["prompts"],
    "prompt_expert": ["prompts"],
    "create_and_consult_expert": ["prompts"],
}


def test_importing_the_package_registers_every_tool(registered_tools):
    for name, tags in T3_TOOLS.items():
        assert name in registry.tools, f"{name} was not registered"
        assert registry.tools[name]["tags"] == tags


def test_tag_index_matches_the_tools_it_points_at(registered_tools):
    assert set(registry.tools_by_tag["prompts"]) == {
        "prompt_llm",
        "prompt_llm_for_json",
        "prompt_expert",
        "create_and_consult_expert",
    }
    assert registry.tools_by_tag["system"] == ["terminate"]

    # The index is an index, not a second source of truth: every name it holds
    # must exist in `tools`, and carry the tag that pointed at it.
    for tag, names in registry.tools_by_tag.items():
        for name in names:
            assert tag in registry.tools[name]["tags"]


def test_terminate_is_terminal_and_returns_its_message(registered_tools):
    assert registry.tools["terminate"]["terminal"] is True
    assert "message" in registry.tools["terminate"]["parameters"]["properties"]
    assert registry.tools["terminate"]["parameters"]["required"] == ["message"]
    assert system.terminate("all done") == "all done"


def test_self_prompting_tools_are_not_terminal(registered_tools):
    for name in ("prompt_llm", "prompt_llm_for_json", "prompt_expert"):
        assert registry.tools[name]["terminal"] is False


def test_double_import_does_not_duplicate(registered_tools):
    """Re-executing a tool module must not grow the catalogue or the tag index.

    `tools[name] = ...` de-duplicates by construction; `tools_by_tag[tag]` is a
    list and does not, which is why the decorator drops stale entries first.
    This test deliberately does *not* use the clean_tool_registry fixture -- it
    reloads into the live catalogue, so the modules and the registry stay in
    agreement afterwards.
    """
    before_tools = dict(registry.tools)
    before_tags = {tag: list(names) for tag, names in registry.tools_by_tag.items()}

    importlib.reload(system)
    importlib.reload(self_prompting)

    assert set(registry.tools) == set(before_tools)
    assert {tag: sorted(names) for tag, names in registry.tools_by_tag.items()} == {
        tag: sorted(names) for tag, names in before_tags.items()
    }
    assert registry.tools_by_tag["prompts"].count("prompt_expert") == 1
    assert registry.tools_by_tag["system"] == ["terminate"]


def test_retagging_a_tool_drops_it_from_the_old_tag(clean_tool_registry):
    @register_tool(tags=["old"])
    def renamed() -> None:
        """First registration."""

    @register_tool(tool_name="renamed", tags=["new"])
    def renamed_again() -> None:
        """Second registration under the same tool name."""

    assert clean_tool_registry.tools_by_tag.get("old") == []
    assert clean_tool_registry.tools_by_tag["new"] == ["renamed"]


# --- PythonActionRegistry: slicing the catalogue by tag --------------------


def test_registry_selects_only_the_requested_tags(registered_tools):
    actions = PythonActionRegistry(tags=["system"])
    assert [a.name for a in actions.get_actions()] == ["terminate"]
    assert actions.get_action("prompt_llm") is None
    assert actions.get_action("terminate").terminal is True


def test_registry_unions_multiple_tags(registered_tools):
    names = {a.name for a in PythonActionRegistry(tags=["system", "prompts"]).get_actions()}
    assert names == set(T3_TOOLS)


def test_unknown_tag_yields_an_empty_registry_not_an_error(registered_tools):
    actions = PythonActionRegistry(tags=["no_such_tag"])
    assert actions.get_actions() == []
    assert actions.get_action("terminate") is None


def test_default_tag_set_excludes_the_flag_gated_prompt_tools(registered_tools):
    """The lean default is a feature: `prompts` is opt-in, not shipped."""
    default = PythonActionRegistry(tags=["greeting", "guestbook", "system"])
    names = {a.name for a in default.get_actions()}
    assert "terminate" in names
    assert names.isdisjoint(registry.tools_by_tag["prompts"])


def test_registry_carries_the_schema_the_model_will_see(registered_tools):
    action = PythonActionRegistry(tags=["prompts"]).get_action("prompt_expert")
    assert set(action.parameters["properties"]) == {"description_of_expert", "prompt"}
    assert "action_context" not in action.parameters["properties"]
    assert action.description.startswith("Consult a named expert persona")
