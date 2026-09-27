"""The guestbook tools: validation, injection, and errors that carry instructions."""

from __future__ import annotations

import pytest

from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import PythonActionRegistry
from hello_agent.core.environment import PythonEnvironment
from hello_agent.storage.guestbook import GuestbookStore
from hello_agent.tools.guestbook import (
    MAX_NAME_LENGTH,
    MAX_NOTE_LENGTH,
    read_guestbook,
    sign_guestbook,
)


@pytest.fixture
def store(tmp_path) -> GuestbookStore:
    return GuestbookStore("guestbook.json", root=tmp_path)


# --- sign_guestbook ---------------------------------------------------------


def test_signing_returns_a_monotonic_entry_id(store):
    first = sign_guestbook("Ada Lovelace", "Welcome, Ada.", _guestbook=store)
    second = sign_guestbook("Grace Hopper", "Welcome, Grace.", _guestbook=store)

    assert first == {
        "status": "signed",
        "entry_id": 1,
        "name": "Ada Lovelace",
        "signed_at": first["signed_at"],
    }
    assert second["entry_id"] == 2
    assert store.count() == 2


def test_signing_stores_the_greeting_and_the_note(store):
    sign_guestbook("Ada", "Welcome back.", note="Second visit this year.", _guestbook=store)

    (entry,) = store.entries()
    assert entry["greeting"] == "Welcome back."
    assert entry["note"] == "Second visit this year."


def test_the_note_is_optional(store):
    sign_guestbook("Ada", "Welcome.", _guestbook=store)
    assert store.entries()[0]["note"] == ""


def test_surrounding_whitespace_is_trimmed(store):
    sign_guestbook("  Ada  ", "  Welcome.  ", _guestbook=store)
    assert store.entries()[0]["name"] == "Ada"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"name": "", "greeting": "Welcome."}, "'name'"),
        ({"name": "   ", "greeting": "Welcome."}, "'name'"),
        ({"name": "Ada", "greeting": ""}, "'greeting'"),
        ({"name": 42, "greeting": "Welcome."}, "must be a string"),
    ],
)
def test_invalid_input_raises_with_an_instruction(store, kwargs, expected):
    """Bad arguments are the agent's mistake, so they raise -- with the fix attached."""
    with pytest.raises(ValueError) as excinfo:
        sign_guestbook(_guestbook=store, **kwargs)

    assert expected in str(excinfo.value)
    assert not store.exists()  # nothing was written


def test_over_long_fields_are_refused_and_the_limit_is_named(store):
    with pytest.raises(ValueError) as excinfo:
        sign_guestbook("A" * (MAX_NAME_LENGTH + 1), "Welcome.", _guestbook=store)
    assert str(MAX_NAME_LENGTH) in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        sign_guestbook("Ada", "Welcome.", note="n" * (MAX_NOTE_LENGTH + 1), _guestbook=store)
    assert "Shorten the" in str(excinfo.value)


def test_the_store_is_not_a_parameter_the_model_can_see(registered_tools):
    """Action specificity: the model supplies three fields, never a path."""
    schema = registered_tools.tools["sign_guestbook"]["parameters"]

    assert sorted(schema["properties"]) == ["greeting", "name", "note"]
    assert sorted(schema["required"]) == ["greeting", "name"]
    assert "_guestbook" not in str(schema)


def test_without_an_injected_store_it_uses_the_configured_path(guestbook_path):
    """Falling back to config.guestbook_path() keeps the tool usable on its own."""
    result = sign_guestbook("Ada", "Welcome.", _guestbook=None)

    assert result["entry_id"] == 1
    assert guestbook_path.exists()


# --- read_guestbook ---------------------------------------------------------


def test_reading_an_empty_book_instructs_the_agent_what_to_do_next(store):
    """The injected-instruction pattern: an error that is also a next step."""
    result = read_guestbook(_guestbook=store)

    assert result["entry_count"] == 0
    assert result["entries"] == []
    message = result["error"]
    assert "empty" in message
    assert "not a failure" in message
    assert "sign_guestbook" in message  # names the tool to call instead
    assert "Do not call read_guestbook again" in message  # and forecloses the loop


def test_reading_returns_the_most_recent_entries_newest_last(store):
    for name in ("Ada", "Grace", "Alan"):
        sign_guestbook(name, f"Welcome, {name}.", _guestbook=store)

    result = read_guestbook(limit=2, _guestbook=store)

    assert "error" not in result
    assert result["entry_count"] == 3
    assert result["showing"] == 2
    assert [entry["name"] for entry in result["entries"]] == ["Grace", "Alan"]


@pytest.mark.parametrize("limit", [0, -3, "nonsense", None])
def test_a_nonsense_limit_does_not_break_the_tool(store, limit):
    sign_guestbook("Ada", "Welcome.", _guestbook=store)
    assert read_guestbook(limit=limit, _guestbook=store)["entry_count"] == 1


# --- through the environment ------------------------------------------------


def test_the_environment_injects_the_store_by_parameter_name(store, registered_tools):
    """End to end: model args in, dependency injected, result dict out."""
    registry = PythonActionRegistry(tool_names=["sign_guestbook", "read_guestbook"])
    context = ActionContext({"guestbook": store})
    environment = PythonEnvironment()

    signed = environment.execute_action(
        agent=None,
        action_context=context,
        action=registry.get_action("sign_guestbook"),
        args={"name": "Ada", "greeting": "Welcome."},
    )

    assert signed["tool_executed"] is True
    assert signed["result"]["entry_id"] == 1

    read = environment.execute_action(
        agent=None,
        action_context=context,
        action=registry.get_action("read_guestbook"),
        args={},
    )
    assert read["result"]["entries"][0]["name"] == "Ada"


def test_a_validation_failure_reaches_the_agent_as_feedback_not_a_crash(store, registered_tools):
    registry = PythonActionRegistry(tool_names=["sign_guestbook"])
    result = PythonEnvironment().execute_action(
        agent=None,
        action_context=ActionContext({"guestbook": store}),
        action=registry.get_action("sign_guestbook"),
        args={"name": "", "greeting": "Welcome."},
    )

    assert result["tool_executed"] is False
    assert "'name'" in result["error"]


def test_both_tools_carry_the_guestbook_tag_and_nothing_else(registered_tools):
    assert sorted(registered_tools.tools_by_tag["guestbook"]) == [
        "read_guestbook",
        "sign_guestbook",
    ]


def test_every_tool_description_is_its_docstring(registered_tools):
    for name in ("sign_guestbook", "read_guestbook"):
        description = registered_tools.tools[name]["description"]
        assert len(description) > 200  # enough to guide the model on its own
        assert "Args:" in description and "Returns:" in description
