"""What ``@register_tool`` derives from a function -- and what it hides."""

from __future__ import annotations

from typing import Any

import pytest

from hal9000.core.action_context import ActionContext
from hal9000.core.actions import PythonActionRegistry
from hal9000.core.environment import PythonEnvironment
from hal9000.tools.decorators import get_json_type, get_tool_metadata, register_tool


def test_name_and_description_come_from_the_function(clean_tool_registry):
    @register_tool(tags=["demo"])
    def greet_visitor(name: str) -> str:
        """Greet a visitor by name."""
        return f"Hello, {name}"

    entry = clean_tool_registry.tools["greet_visitor"]
    assert entry["description"] == "Greet a visitor by name."
    assert entry["function"] is greet_visitor
    assert entry["terminal"] is False
    assert entry["tags"] == ["demo"]


def test_explicit_name_description_and_terminal_win(clean_tool_registry):
    @register_tool(tool_name="stop", description="Halt everything.", terminal=True)
    def some_badly_named_function() -> None:
        """This docstring is overridden."""

    assert "some_badly_named_function" not in clean_tool_registry.tools
    assert clean_tool_registry.tools["stop"]["description"] == "Halt everything."
    assert clean_tool_registry.tools["stop"]["terminal"] is True


def test_undocumented_function_still_registers(clean_tool_registry):
    @register_tool()
    def undocumented(value: str) -> str:
        return value

    assert clean_tool_registry.tools["undocumented"]["description"] == "No description provided."


def test_schema_maps_types_and_marks_defaults_optional(clean_tool_registry):
    @register_tool()
    def every_type(
        a: str,
        b: int,
        c: float,
        d: bool,
        e: list[str],
        f: dict[str, Any],
        g: object,
        h,
        i: str = "default",
    ) -> None:
        """One parameter per JSON Schema type."""

    schema = clean_tool_registry.tools["every_type"]["parameters"]
    assert schema["type"] == "object"
    assert {name: spec["type"] for name, spec in schema["properties"].items()} == {
        "a": "string",
        "b": "integer",
        "c": "number",
        "d": "boolean",
        "e": "array",
        "f": "object",
        "g": "string",  # unknown type falls back to string
        "h": "string",  # missing hint defaults to str
        "i": "string",
    }
    assert schema["required"] == ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert "i" not in schema["required"]


@pytest.mark.parametrize(
    ("python_type", "expected"),
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (list, "array"),
        (list[int], "array"),
        (dict, "object"),
        (dict[str, int], "object"),
        (ActionContext, "string"),
        (None, "string"),
    ],
)
def test_get_json_type(python_type, expected):
    assert get_json_type(python_type) == expected


def test_parameters_override_wins(clean_tool_registry):
    override = {
        "type": "object",
        "properties": {"anything": {"type": "string", "description": "hand-written"}},
        "required": [],
    }

    @register_tool(parameters_override=override)
    def ignore_my_signature(a: int, b: int) -> int:
        """The override replaces introspection entirely."""
        return a + b

    assert clean_tool_registry.tools["ignore_my_signature"]["parameters"] == override


# --- The contract that makes dependency injection invisible ----------------


def test_schema_excludes_injected_dependencies(clean_tool_registry):
    """action_context, action_agent and every _-prefixed parameter are hidden.

    This is the single most important assertion in the tool layer: it is what
    lets a tool take an LLM, a storage handle, or the agent itself without the
    model ever knowing those parameters exist.
    """

    @register_tool(tags=["demo"])
    def sign_something(
        action_context: ActionContext,
        action_agent,
        name: str,
        note: str = "",
        _guestbook: object = None,
        _auth_token: str = "",
    ) -> dict:
        """Only `name` and `note` belong to the model."""
        return {"name": name}

    schema = clean_tool_registry.tools["sign_something"]["parameters"]

    assert "action_context" not in schema["properties"]
    assert "action_agent" not in schema["properties"]
    assert "_guestbook" not in schema["properties"]
    assert "_auth_token" not in schema["properties"]
    assert set(schema["properties"]) == {"name", "note"}
    assert schema["required"] == ["name"]

    for hidden in ("action_context", "action_agent", "_guestbook", "_auth_token"):
        assert hidden not in schema["required"]


def test_hidden_parameters_are_still_injected_at_call_time(clean_tool_registry):
    """Hidden in the schema, present at execution -- both halves of the contract."""
    seen = {}

    @register_tool(tags=["demo"])
    def records_what_it_got(action_context: ActionContext, name: str, _token: str) -> str:
        """Capture the injected dependencies."""
        seen["context"] = action_context
        seen["token"] = _token
        return name

    context = ActionContext({"token": "s3cret"})
    action = PythonActionRegistry(tags=["demo"]).get_action("records_what_it_got")

    # The model supplies only what the schema showed it.
    result = PythonEnvironment().execute_action(None, context, action, {"name": "Ada"})

    assert result["tool_executed"] is True
    assert result["result"] == "Ada"
    assert seen["context"] is context
    assert seen["token"] == "s3cret"


def test_varargs_are_not_offered_to_the_model(clean_tool_registry):
    @register_tool()
    def variadic(first: str, *rest, **options) -> str:
        """The model cannot fill *args or **kwargs."""
        return first

    assert set(clean_tool_registry.tools["variadic"]["parameters"]["properties"]) == {"first"}


# --- get_tool_metadata directly --------------------------------------------


def test_get_tool_metadata_returns_the_documented_keys():
    def sample(action_context: ActionContext, text: str) -> str:
        """Do a thing."""
        return text

    metadata = get_tool_metadata(sample, tags=["a", "b"], terminal=True)

    assert set(metadata) == {
        "tool_name",
        "description",
        "parameters",
        "function",
        "terminal",
        "tags",
    }
    assert metadata["tool_name"] == "sample"  # note: `tool_name`, not `name`
    assert metadata["description"] == "Do a thing."
    assert metadata["tags"] == ["a", "b"]
    assert metadata["terminal"] is True
    assert metadata["function"] is sample
    assert "action_context" not in metadata["parameters"]["properties"]


def test_get_tool_metadata_does_not_alias_the_caller_tags_list():
    tags = ["mutable"]
    metadata = get_tool_metadata(lambda: None, tool_name="x", tags=tags)
    tags.append("added-later")
    assert metadata["tags"] == ["mutable"]


def test_multiline_docstring_is_dedented():
    def sample() -> None:
        """First line.

        Indented continuation.
        """

    assert get_tool_metadata(sample)["description"] == "First line.\n\nIndented continuation."


def test_decorated_function_is_unchanged(clean_tool_registry):
    @register_tool(tags=["demo"])
    def double(value: int) -> int:
        """Double a number."""
        return value * 2

    assert double(21) == 42
    assert double.__name__ == "double"
