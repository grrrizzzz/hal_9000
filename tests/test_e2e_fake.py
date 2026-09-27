"""The assembled agent, end to end, on a scripted LLM.

Three things are proved here and nowhere else:

1. **The lean default is what it claims.** Five tools, no capabilities, a plain
   environment, function calling. If a later change quietly adds a sixth tool to
   the default run, :func:`test_default_agent_is_lean` is what fails.
2. **Every flag does something, and the agent still works with it on.** Each of
   the seven is asserted twice -- once for the structural change it makes to the
   assembled agent, once by running a full scripted visit with it enabled.
3. **The whole path runs.** ``extract -> compose -> sign -> terminate``, through
   the real CLI, writing a real (temporary) guestbook file, with the greeting
   landing on stdout.

No network, no key, no mocking library: ``FakeLLM`` is a list of strings, because
``generate_response`` is ``(Prompt, tier) -> str`` and nothing more.

**Budgeting the fake** is the only fiddly part. One response per loop iteration,
plus one per LLM call a tool makes inside itself, plus one per tracked iteration
when ``progress_tracking`` is on, plus one for the plan when ``plan_first`` is.
:func:`script_for` builds that script from a ``Features``, so the arithmetic
lives in one place instead of in nine tests.
"""

from __future__ import annotations

import io
import json

import pytest

from hello_agent.__main__ import build_parser, final_message, main, tool_trace
from hello_agent.agents import TRANSLATOR_AGENT_NAME
from hello_agent.agents.greeter import (
    build_greeter_agent,
    greeter_context_props,
    greeter_goals,
)
from hello_agent.capabilities import (
    LoggingCapability,
    PlanFirstCapability,
    ProgressTrackingCapability,
    TimeAwareCapability,
)
from hello_agent.config import Features
from hello_agent.core.agent_language import AgentFunctionCallingActionLanguage
from hello_agent.core.environment import PythonEnvironment
from hello_agent.safety import ReviewDecision, StagedActionEnvironment
from tests.fakes import FakeLLM, tool_call

VISITOR_TEXT = (
    "Dr. Maria Chen is here -- marine biologist, flew in from Osaka, second visit this year."
)

VISITOR = {
    "name": "Dr. Maria Chen",
    "role": "marine biologist",
    "origin": "Osaka",
    "previous_visits": "second visit this year",
}

GREETING = (
    "Welcome back, Dr. Chen -- Osaka's loss is our gain twice over this year. "
    "The reef samples you mentioned last spring are still the talk of the lab."
)

TRANSLATED = "チェン博士、おかえりなさい。"

NOTE = "Marine biologist, travelled from Osaka; second visit this year."

FINAL = f"{GREETING}\n\nSigned the guestbook as entry #1."

PLAN = "1. Extract the details. 2. Compose a greeting. 3. Sign the book. 4. Terminate."
PROGRESS = "Progress: the visit is being handled in order; nothing is blocked."

#: Every flag, and the structural change turning it on must make.
ALL_FLAGS = [
    "time_aware",
    "plan_first",
    "progress_tracking",
    "verbose_logging",
    "delegation",
    "staged_review",
    "dynamic_experts",
]


def script_for(features: Features, translate: bool = False) -> list[str]:
    """The exact sequence of LLM responses one scripted visit needs.

    Written as turns rather than as a flat list, because that is the order the
    loop consumes them in: a capability that spends a call at ``init`` goes
    first, and one that spends a call at ``end_agent_loop`` goes after the tool.
    """
    responses: list[str] = []
    if features.plan_first:
        responses.append(PLAN)  # PlanFirstCapability.init, before iteration 1

    def turn(*items: str) -> None:
        responses.extend(items)
        if features.progress_tracking:
            responses.append(PROGRESS)  # ProgressTrackingCapability.end_agent_loop

    # Each turn: the agent's own decision, then any call the chosen tool makes.
    turn(tool_call("extract_visitor_details", document_text=VISITOR_TEXT), json.dumps(VISITOR))
    turn(tool_call("compose_greeting", visitor=VISITOR), GREETING)
    if translate:
        turn(
            tool_call("call_agent", agent_name=TRANSLATOR_AGENT_NAME, task="Japanese: " + GREETING),
            # the translator's own single iteration, sharing this fake
            tool_call("terminate", message=TRANSLATED),
        )
    # Built by hand rather than with tool_call(): the guestbook's first argument
    # is called `name`, which collides with tool_call's own first parameter.
    turn(
        json.dumps(
            {
                "tool": "sign_guestbook",
                "args": {"name": VISITOR["name"], "greeting": GREETING, "note": NOTE},
            }
        )
    )
    turn(tool_call("terminate", message=FINAL))
    return responses


def run_visit(features: Features, translate: bool = False, guestbook=None):
    """Assemble, wire and run the greeter on a scripted visit. Returns (memory, llm)."""
    llm = FakeLLM(script_for(features, translate=translate))
    agent = build_greeter_agent(features, llm)
    if features.staged_review:
        # A deterministic reviewer: the default asks a human on stdin, and pytest
        # has no terminal, so it would fail closed and refuse every write.
        agent.environment.reviewer = lambda request: ReviewDecision(True, "test", "approved")
    memory = agent.run(
        VISITOR_TEXT,
        action_context_props=greeter_context_props(features, llm, guestbook=guestbook),
    )
    return memory, llm


# ---------------------------------------------------------------- the default


def test_default_agent_is_lean(registered_tools):
    """No arguments means five tools, nothing else. This is the headline shape."""
    agent = build_greeter_agent(generate_response=FakeLLM([]))

    assert sorted(action.name for action in agent.actions.get_actions()) == [
        "compose_greeting",
        "extract_visitor_details",
        "read_guestbook",
        "sign_guestbook",
        "terminate",
    ]
    assert agent.capabilities == []
    assert type(agent.environment) is PythonEnvironment
    assert isinstance(agent.agent_language, AgentFunctionCallingActionLanguage)


def test_default_features_are_all_off():
    """The lean default is a decision, not an accident of construction order."""
    assert Features().enabled() == []
    assert build_greeter_agent(generate_response=FakeLLM([])).capabilities == []


def test_default_goals_say_nothing_about_delegation():
    """A goal telling the agent to call a tool it does not have would be a trap."""
    goals = " ".join(goal.description for goal in greeter_goals())
    assert "call_agent" not in goals
    assert TRANSLATOR_AGENT_NAME not in goals
    assert "extract_visitor_details" in goals
    assert "sign_guestbook" in goals


# ------------------------------------------------------- one flag at a time

#: flag -> (tool count, capability classes, environment class)
FLAG_EXPECTATIONS = {
    "time_aware": (5, [TimeAwareCapability], PythonEnvironment),
    "plan_first": (5, [PlanFirstCapability], PythonEnvironment),
    "progress_tracking": (5, [ProgressTrackingCapability], PythonEnvironment),
    "verbose_logging": (5, [LoggingCapability], PythonEnvironment),
    "delegation": (6, [], PythonEnvironment),
    "staged_review": (6, [], StagedActionEnvironment),
    "dynamic_experts": (9, [], PythonEnvironment),
}


@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_each_flag_changes_the_assembled_agent(flag, registered_tools):
    """Every flag is visible in the agent it produces -- tools, capabilities, or E."""
    tools, capabilities, environment = FLAG_EXPECTATIONS[flag]
    agent = build_greeter_agent(Features(**{flag: True}), FakeLLM([]))

    assert len(agent.actions.get_actions()) == tools
    assert [type(c) for c in agent.capabilities] == capabilities
    assert type(agent.environment) is environment


def test_delegation_adds_call_agent_only(registered_tools):
    """Ruling R15: the greeter gets message passing, not all four patterns."""
    agent = build_greeter_agent(Features(delegation=True), FakeLLM([]))
    names = {action.name for action in agent.actions.get_actions()}

    assert "call_agent" in names
    assert names.isdisjoint(
        {"call_agent_with_reflection", "hand_off_to_agent", "call_agent_with_selected_context"}
    )


def test_staged_review_adds_undo_and_stages_only_the_write(registered_tools):
    """Ruling R12: one flag, both halves of environmental safety."""
    agent = build_greeter_agent(Features(staged_review=True), FakeLLM([]))
    names = {action.name for action in agent.actions.get_actions()}

    assert "undo_last_guestbook_entry" in names
    # Undo has no registered reverser, so staging it would create an approved
    # action nobody can roll back.
    assert agent.environment.staged_tools == {"sign_guestbook"}


def test_dynamic_experts_adds_the_self_prompting_tools(registered_tools):
    agent = build_greeter_agent(Features(dynamic_experts=True), FakeLLM([]))
    names = {action.name for action in agent.actions.get_actions()}

    assert "create_and_consult_expert" in names
    assert {"prompt_llm", "prompt_llm_for_json", "prompt_expert"} <= names


def test_delegation_adds_the_translator_goal_and_registry(registered_tools):
    """The greeter delegates because a goal says to, not because code branches."""
    features = Features(delegation=True)
    goals = " ".join(goal.description for goal in greeter_goals(features))
    assert TRANSLATOR_AGENT_NAME in goals

    props = greeter_context_props(features, FakeLLM([]))
    assert props["agent_registry"].list_agents() == [TRANSLATOR_AGENT_NAME]
    assert "agent_registry" not in greeter_context_props(Features(), FakeLLM([]))


@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_each_flag_still_completes_a_scripted_run(flag, guestbook_path, registered_tools):
    """Turning a flag on must not break the run it was added to."""
    features = Features(**{flag: True})
    memory, llm = run_visit(features)

    assert final_message(memory) == FINAL
    assert [tool for tool, _ in tool_trace(memory)] == [
        "extract_visitor_details",
        "compose_greeting",
        "sign_guestbook",
        "terminate",
    ]
    assert llm.responses == [], "the script should be consumed exactly, with nothing left over"
    assert json.loads(guestbook_path.read_text())["entries"][0]["name"] == VISITOR["name"]


def test_all_flags_on_completes_a_scripted_run(guestbook_path, registered_tools):
    """Seven flags at once: four capabilities, nine tools, staged writes."""
    features = Features.all_on()
    agent = build_greeter_agent(features, FakeLLM([]))
    assert len(agent.capabilities) == 4
    assert [c.name for c in agent.capabilities][-1] == "Logging", "logging composes last"
    # 5 default + undo_last_guestbook_entry + 4 `prompts` tools + call_agent
    assert len(agent.actions.get_actions()) == 11

    memory, llm = run_visit(features, translate=True)

    assert final_message(memory) == FINAL
    assert llm.responses == []
    entry = json.loads(guestbook_path.read_text())["entries"][0]
    assert entry["name"] == VISITOR["name"]


def test_delegation_run_reaches_the_translator(guestbook_path, registered_tools):
    """One FakeLLM covers both agents, and the sub-agent's answer comes back (R16)."""
    memory, llm = run_visit(Features(delegation=True), translate=True)

    trace = dict(tool_trace(memory))
    assert TRANSLATED in trace["call_agent"], "call_agent returns the unwrapped result"
    assert llm.responses == []


def test_capabilities_are_fresh_per_agent(registered_tools):
    """All four count iterations on themselves; a shared instance would leak state."""
    features = Features(progress_tracking=True, verbose_logging=True)
    first = build_greeter_agent(features, FakeLLM([]))
    second = build_greeter_agent(features, FakeLLM([]))

    assert first.capabilities[0] is not second.capabilities[0]
    assert first.capabilities[1] is not second.capabilities[1]


# ------------------------------------------------------------------ the CLI


def test_help_documents_every_flag_and_env_var():
    help_text = build_parser().format_help()

    for flag in ALL_FLAGS:
        assert f"--{flag.replace('_', '-')}" in help_text
        assert Features.env_var(flag) in help_text
    for variable in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LLM_PROVIDER",
        "AGENT_MODEL_POWERFUL",
        "AGENT_MODEL_FAST",
        "AGENT_MAX_TOKENS",
        "AGENT_MAX_ITERATIONS",
        "AGENT_MAX_DURATION_SECONDS",
        "AGENT_GUESTBOOK_PATH",
        "AGENT_TIMEZONE",
        "AGENT_REVIEWER",
    ):
        assert variable in help_text
    # The staged-review/stdin trap is documented, not left to be discovered.
    assert "--reviewer llm" in help_text


def test_cli_without_a_key_exits_2_and_shows_no_traceback(capsys):
    """A missing key is a configuration mistake, not a crash."""
    assert main(["Ada Lovelace is visiting."]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "ANTHROPIC_API_KEY" in captured.err and "OPENAI_API_KEY" in captured.err


def test_cli_reads_stdin_when_given_no_argument(monkeypatch, guestbook_path, registered_tools):
    monkeypatch.setattr("sys.stdin", io.StringIO(VISITOR_TEXT))
    llm = FakeLLM(script_for(Features()))

    assert main(["--quiet"], generate_response=llm) == 0


def test_cli_with_nothing_to_greet_exits_2(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("   "))

    assert main([], generate_response=FakeLLM([])) == 2
    assert "no visitor to greet" in capsys.readouterr().err


def test_cli_refuses_to_take_stdin_under_staged_review(monkeypatch, capsys):
    """The human reviewer answers on stdin, so the visitor's prose cannot arrive there."""
    monkeypatch.setattr("sys.stdin", io.StringIO(VISITOR_TEXT))

    assert main(["--staged-review"], generate_response=FakeLLM([])) == 2
    assert "--staged-review" in capsys.readouterr().err


def test_cli_staged_review_without_a_terminal_writes_nothing(
    guestbook_path, registered_tools, capsys
):
    """Fail closed: no terminal, no human, no write -- and the agent is told why.

    This is the behaviour ``--help`` warns about. It looks like a bug the first
    time you meet it in CI, so it is pinned here as intended behaviour.
    """
    features = Features(staged_review=True)
    llm = FakeLLM(script_for(features))

    assert main(["--staged-review", VISITOR_TEXT], generate_response=llm) == 0
    assert not guestbook_path.exists(), "a refused write must leave no file behind"
    assert "rejected it" in capsys.readouterr().out


def test_cli_flags_turn_features_on_over_the_environment(monkeypatch, guestbook_path):
    """``AGENT_ENABLE_*`` first, command line on top."""
    monkeypatch.setenv("AGENT_ENABLE_TIME_AWARE", "1")
    args = build_parser().parse_args(["--delegation", "x"])

    from hello_agent.__main__ import features_from

    assert features_from(args).enabled() == ["time_aware", "delegation"]


# ------------------------------------------------------- the whole path


def test_end_to_end_happy_path(capsys, guestbook_path, registered_tools):
    """extract -> compose -> sign -> terminate, through the CLI, onto stdout."""
    llm = FakeLLM(script_for(Features()))

    assert main([VISITOR_TEXT], generate_response=llm) == 0

    # 1. every tool ran, in order, and the script was consumed exactly
    assert llm.responses == []
    assert [prompt for prompt in llm.calls]  # the fake saw real prompts

    # 2. the guestbook on disk holds one complete, well-formed entry
    document = json.loads(guestbook_path.read_text(encoding="utf-8"))
    assert document["next_entry_id"] == 2
    assert len(document["entries"]) == 1
    entry = document["entries"][0]
    assert entry["entry_id"] == 1
    assert entry["name"] == VISITOR["name"]
    assert entry["greeting"] == GREETING
    assert entry["note"] == NOTE
    assert entry["signed_at"]

    # 3. the terminate message reached the user, unchanged (ruling R7)
    captured = capsys.readouterr()
    assert FINAL in captured.out
    assert captured.out.rstrip().endswith("Signed the guestbook as entry #1.")

    # 4. and the trace showed the work, as in BUILD_PLAN §2
    for tool in ("extract_visitor_details", "compose_greeting", "sign_guestbook", "terminate"):
        assert f"[greeter] {tool}" in captured.out


def test_end_to_end_uses_the_fast_tier_for_extraction(guestbook_path, registered_tools):
    """MATE model efficiency survives assembly: extraction is cheap, composition is not."""
    _, llm = run_visit(Features())

    # calls: [agent, extract(fast), agent, compose(powerful), agent, agent]
    assert llm.tiers == ["powerful", "fast", "powerful", "powerful", "powerful", "powerful"]


def test_quiet_prints_only_the_final_message(capsys, guestbook_path, registered_tools):
    llm = FakeLLM(script_for(Features()))

    assert main(["--quiet", VISITOR_TEXT], generate_response=llm) == 0
    assert capsys.readouterr().out.strip() == FINAL
