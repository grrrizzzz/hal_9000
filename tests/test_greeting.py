"""The greeting tools: fixed-schema extraction, model tiers, and policy as prompt."""

from __future__ import annotations

import json

import pytest

from hello_agent.core.action_context import ActionContext, default_properties
from hello_agent.core.actions import PythonActionRegistry
from hello_agent.core.environment import PythonEnvironment
from hello_agent.tools import greeting as greeting_module
from hello_agent.tools.greeting import (
    DEFAULT_POLICY,
    VISITOR_SCHEMA,
    compose_greeting,
    extract_visitor_details,
    read_greeting_policy,
)
from tests.fakes import FakeLLM

VISITOR_TEXT = (
    "Dr. Maria Chen is here -- marine biologist, flew in from Osaka, second visit this year."
)
EXTRACTED = {
    "name": "Dr. Maria Chen",
    "role": "marine biologist",
    "origin": "Osaka",
    "previous_visits": "second this year",
}


@pytest.fixture
def tiered_context():
    """An ActionContext built the way :meth:`Agent.run` builds it: one fake, two tiers.

    ``default_properties`` is what puts ``llm``, ``powerful_llm`` and ``fast_llm``
    in the context, so using it here means these tests assert against the real
    wiring rather than a convenient imitation.
    """

    def _make(*responses: str) -> tuple[ActionContext, FakeLLM]:
        llm = FakeLLM(list(responses))
        properties = default_properties(llm, memory=None, action_registry=None)
        return ActionContext(properties), llm

    return _make


# --- extract_visitor_details ------------------------------------------------


def test_extraction_returns_the_fixed_schema_fields(tiered_context):
    context, _ = tiered_context(json.dumps(EXTRACTED))
    assert extract_visitor_details(context, VISITOR_TEXT) == EXTRACTED


def test_extraction_sends_the_fixed_schema_and_the_text(tiered_context):
    context, llm = tiered_context(json.dumps(EXTRACTED))
    extract_visitor_details(context, VISITOR_TEXT)

    system_message, user_message = llm.last_prompt.messages
    assert json.dumps(VISITOR_SCHEMA) in system_message["content"]
    assert VISITOR_TEXT in user_message["content"]
    assert "never invent a name" in user_message["content"]  # the anti-hallucination rule


def test_extraction_runs_on_the_fast_tier(tiered_context):
    """MATE, model efficiency: pawn work goes to the pawn."""
    context, llm = tiered_context(json.dumps(EXTRACTED))
    extract_visitor_details(context, VISITOR_TEXT)

    assert llm.tiers == ["fast"]


def test_extraction_falls_back_to_the_powerful_tier_when_no_fast_one_exists(tiered_context):
    llm = FakeLLM([json.dumps(EXTRACTED)])
    extract_visitor_details(ActionContext({"llm": llm}), VISITOR_TEXT)

    assert llm.tiers == ["powerful"]


def test_extraction_drops_fields_the_text_did_not_mention(tiered_context):
    context, _ = tiered_context(json.dumps({"name": "Ada", "role": "", "origin": None}))
    assert extract_visitor_details(context, "Ada is here.") == {"name": "Ada"}


def test_extraction_without_a_name_raises_and_says_what_to_do(tiered_context):
    """A guestbook entry needs a name, so a nameless extraction is a hard stop."""
    context, _ = tiered_context(json.dumps({"role": "marine biologist"}))

    with pytest.raises(ValueError) as excinfo:
        extract_visitor_details(context, "A marine biologist is here.")

    message = str(excinfo.value)
    assert "does not name the visitor" in message
    assert "Ask the user" in message


def test_extraction_inherits_the_json_retry_loop(tiered_context):
    """Reusing prompt_llm_for_json means reusing its three attempts and fence stripping."""
    context, llm = tiered_context("not json", f"```json\n{json.dumps(EXTRACTED)}\n```")

    assert extract_visitor_details(context, VISITOR_TEXT) == EXTRACTED
    assert llm.tiers == ["fast", "fast"]


# --- compose_greeting -------------------------------------------------------


def test_composition_returns_the_experts_words_verbatim(tiered_context):
    context, _ = tiered_context("Welcome back, Dr. Chen.")
    assert compose_greeting(context, EXTRACTED) == "Welcome back, Dr. Chen."


def test_composition_runs_on_the_powerful_tier(tiered_context):
    """The other half of the MATE demo: judgement goes to the queen."""
    context, llm = tiered_context("Welcome back, Dr. Chen.")
    compose_greeting(context, EXTRACTED)

    assert llm.tiers == ["powerful"]


def test_composition_puts_the_concierge_persona_in_a_system_message(tiered_context):
    context, llm = tiered_context("Welcome.")
    compose_greeting(context, EXTRACTED)

    system_message, user_message = llm.last_prompt.messages
    assert system_message["role"] == "system"
    assert "concierge" in system_message["content"]
    assert "Dr. Maria Chen" in user_message["content"]


def test_an_injected_policy_wins_over_the_file(tiered_context):
    context, llm = tiered_context("Welcome.")
    compose_greeting(context, EXTRACTED, _greeting_policy="Rule one: speak in haiku.")

    assert "Rule one: speak in haiku." in llm.last_prompt.messages[1]["content"]


# --- policy as prompt -------------------------------------------------------


def write_policy(tmp_path, text: str) -> None:
    """Put a policy file where the working-directory candidate will find it."""
    policy_file = tmp_path / "config" / "greeting_policy.txt"
    policy_file.parent.mkdir(parents=True, exist_ok=True)
    policy_file.write_text(text, encoding="utf-8")


def test_the_shipped_policy_file_is_found_and_read():
    """The checkout's own config/greeting_policy.txt, read from wherever pytest ran."""
    policy = read_greeting_policy()

    assert "Greeting policy" in policy
    assert policy != DEFAULT_POLICY


def test_editing_the_policy_file_changes_the_prompt(tmp_path, monkeypatch, tiered_context):
    """The headline claim: behaviour changes with a text edit and no code change."""
    monkeypatch.chdir(tmp_path)
    write_policy(tmp_path, "Rule one: greet every visitor in exactly four words.")

    context, llm = tiered_context("Good to see you.", "Welcome home, Dr. Chen.")
    compose_greeting(context, EXTRACTED)
    assert "greet every visitor in exactly four words" in llm.last_prompt.messages[1]["content"]

    # Edit the file mid-run -- the next call must pick it up, with no reload.
    write_policy(tmp_path, "Rule one: never mention the weather.")

    compose_greeting(context, EXTRACTED)
    second_prompt = llm.last_prompt.messages[1]["content"]
    assert "never mention the weather" in second_prompt
    assert "exactly four words" not in second_prompt


def test_a_missing_policy_file_degrades_gracefully(tmp_path, monkeypatch, tiered_context):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(greeting_module, "_REPO_ROOT", tmp_path)

    context, llm = tiered_context("Welcome.")
    assert compose_greeting(context, EXTRACTED) == "Welcome."
    assert DEFAULT_POLICY in llm.last_prompt.messages[1]["content"]


def test_an_empty_policy_file_falls_back_to_the_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(greeting_module, "_REPO_ROOT", tmp_path)
    write_policy(tmp_path, "   \n\n")

    assert read_greeting_policy() == DEFAULT_POLICY


def test_the_working_directory_policy_wins_over_the_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_policy(tmp_path, "Local rules only.")

    assert read_greeting_policy() == "Local rules only."


# --- what the model sees ----------------------------------------------------


def test_neither_tool_exposes_its_dependencies_to_the_model(registered_tools):
    extract = registered_tools.tools["extract_visitor_details"]["parameters"]
    compose = registered_tools.tools["compose_greeting"]["parameters"]

    assert list(extract["properties"]) == ["document_text"]
    assert list(compose["properties"]) == ["visitor"]
    assert compose["properties"]["visitor"]["type"] == "object"
    assert "action_context" not in str(extract) + str(compose)
    assert "_greeting_policy" not in str(compose)


def test_both_tools_carry_the_greeting_tag(registered_tools):
    assert sorted(registered_tools.tools_by_tag["greeting"]) == [
        "compose_greeting",
        "extract_visitor_details",
    ]


def test_every_tool_description_is_its_docstring(registered_tools):
    for name in ("extract_visitor_details", "compose_greeting"):
        description = registered_tools.tools[name]["description"]
        assert len(description) > 200
        assert "Args:" in description and "Returns:" in description


def test_the_pair_runs_end_to_end_through_the_environment(registered_tools, tiered_context):
    """Extract then compose, as the agent loop would, with the tiers as MATE asks."""
    context, llm = tiered_context(json.dumps(EXTRACTED), "Welcome back, Dr. Chen.")
    registry = PythonActionRegistry(tags=["greeting"])
    environment = PythonEnvironment()

    extracted = environment.execute_action(
        None,
        context,
        registry.get_action("extract_visitor_details"),
        {"document_text": VISITOR_TEXT},
    )
    composed = environment.execute_action(
        None, context, registry.get_action("compose_greeting"), {"visitor": extracted["result"]}
    )

    assert extracted["result"]["name"] == "Dr. Maria Chen"
    assert composed["result"] == "Welcome back, Dr. Chen."
    assert llm.tiers == ["fast", "powerful"]
