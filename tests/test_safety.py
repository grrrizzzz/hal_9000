"""Environmental safety: undo, all-or-nothing, and asking before writing.

The guestbook is what makes these tests mean something. Every assertion about a
rollback or a refusal ends at a real file on disk -- either it was written or it
was not, and "unchanged" is checked byte for byte.
"""

from __future__ import annotations

import json

import pytest

from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import Action, PythonActionRegistry
from hello_agent.core.agent import Agent
from hello_agent.core.agent_language import AgentFunctionCallingActionLanguage
from hello_agent.core.environment import PythonEnvironment
from hello_agent.core.goals import Goal
from hello_agent.safety import (
    ActionTransaction,
    HumanReviewer,
    LLMReviewer,
    ReversibleAction,
    ReviewDecision,
    ReviewRequest,
    StagedActionEnvironment,
    reverser_for,
    reviewer_for,
)
from hello_agent.safety.review import REVIEWER_ENV_VAR
from hello_agent.storage.guestbook import GuestbookStore
from hello_agent.tools.guestbook import sign_guestbook, undo_last_guestbook_entry
from tests.fakes import FakeLLM, tool_call

GOALS = [Goal(priority=1, name="greet", description="Greet the visitor, then stop.")]


@pytest.fixture
def store(tmp_path) -> GuestbookStore:
    return GuestbookStore("guestbook.json", root=tmp_path)


def approve(request: ReviewRequest) -> ReviewDecision:
    """A reviewer that always says yes."""
    return ReviewDecision(True, "test", "Approved by the test.")


def reject(request: ReviewRequest) -> ReviewDecision:
    """A reviewer that always says no, with a reason worth reading."""
    return ReviewDecision(False, "test", "The visitor asked not to be recorded.")


# --- ReversibleAction -------------------------------------------------------


def test_run_records_the_args_result_and_timestamp():
    action = ReversibleAction(lambda entry: {"id": entry.upper()}, lambda **record: None, "add")

    result = action.run(entry="ada")

    assert result == {"id": "ADA"}
    record = action.execution_record
    assert record["args"] == {"entry": "ada"}
    assert record["result"] == {"id": "ADA"}
    assert record["timestamp"].startswith("20")  # an ISO-8601 timestamp


def test_undo_hands_the_whole_record_to_the_reverse_function():
    seen = {}
    action = ReversibleAction(lambda n: n * 2, lambda **record: seen.update(record), "double")

    action.run(n=21)
    action.undo()

    assert seen["args"] == {"n": 21}
    assert seen["result"] == 42
    assert "timestamp" in seen


def test_undo_reverses_the_effect():
    book: list[str] = []
    add = ReversibleAction(
        execute_func=lambda name: book.append(name) or {"index": len(book) - 1},
        reverse_func=lambda **record: book.pop(record["result"]["index"]),
        name="add",
    )

    add.run(name="Ada")
    assert book == ["Ada"]

    add.undo()
    assert book == []


def test_undo_without_a_prior_run_raises():
    action = ReversibleAction(lambda: None, lambda **record: None, "never_run")

    with pytest.raises(ValueError) as excinfo:
        action.undo()

    assert "No action to reverse" in str(excinfo.value)
    assert "never_run" in str(excinfo.value)


def test_undoing_twice_raises_rather_than_reversing_twice():
    """The record is cleared on undo, so a double rollback cannot double-delete."""
    calls: list[str] = []
    action = ReversibleAction(lambda: "done", lambda **record: calls.append("undo"), "once")
    action.run()
    action.undo()

    with pytest.raises(ValueError):
        action.undo()
    assert calls == ["undo"]


def test_the_name_defaults_to_the_executed_function():
    def sign_the_book():
        return None

    assert ReversibleAction(sign_the_book, lambda **record: None).name == "sign_the_book"


# --- ActionTransaction ------------------------------------------------------


def logged_action(log: list[str], number: int, explode: bool = False) -> ReversibleAction:
    """An action that appends to ``log`` when it runs, and again when it is undone."""

    def execute():
        log.append(f"run-{number}")
        if explode:
            raise RuntimeError(f"action {number} failed")
        return number

    return ReversibleAction(execute, lambda **record: log.append(f"undo-{number}"), f"step{number}")


def test_a_transaction_runs_every_queued_action_in_order():
    log: list[str] = []
    transaction = ActionTransaction()
    for number in (1, 2, 3):
        transaction.add(logged_action(log, number))

    assert transaction.execute() == [1, 2, 3]
    assert log == ["run-1", "run-2", "run-3"]
    assert len(transaction.executed) == 3


def test_a_mid_sequence_failure_reverses_everything_in_reverse_order():
    """Three actions, the third fails: 1 and 2 are undone newest-first, and the error escapes."""
    log: list[str] = []
    transaction = ActionTransaction()
    transaction.add(logged_action(log, 1))
    transaction.add(logged_action(log, 2))
    transaction.add(logged_action(log, 3, explode=True))

    with pytest.raises(RuntimeError) as excinfo:
        transaction.execute()

    assert "action 3 failed" in str(excinfo.value)
    assert log == ["run-1", "run-2", "run-3", "undo-2", "undo-1"]
    assert log[3:] == ["undo-2", "undo-1"]  # reverse order, stated outright
    assert transaction.executed == []
    assert transaction.results == []


def test_a_failure_on_the_first_action_reverses_nothing():
    log: list[str] = []
    transaction = ActionTransaction()
    transaction.add(logged_action(log, 1, explode=True))
    transaction.add(logged_action(log, 2))

    with pytest.raises(RuntimeError):
        transaction.execute()

    assert log == ["run-1"]


def test_arguments_are_passed_through_to_each_action():
    transaction = ActionTransaction()
    transaction.add(ReversibleAction(lambda a, b: a + b, lambda **r: None), a=2, b=3)

    assert transaction.execute() == [5]


def test_a_committed_transaction_refuses_further_actions():
    transaction = ActionTransaction()
    transaction.commit()

    with pytest.raises(ValueError) as excinfo:
        transaction.add(ReversibleAction(lambda: None, lambda **r: None))
    assert "already committed" in str(excinfo.value)


def test_describe_lists_the_queued_actions_as_plain_data():
    transaction = ActionTransaction(transaction_id="task-1")
    transaction.add(ReversibleAction(lambda name: name, lambda **r: None, "sign"), name="Ada")

    assert transaction.describe() == [{"tool": "sign", "args": {"name": "Ada"}}]
    assert json.dumps(transaction.describe())  # review prompts need it serialisable


# --- reversers --------------------------------------------------------------


def test_the_guestbook_reverser_removes_the_entry_it_created(store):
    """Reversal keys off entry_id, so a later signing does not confuse it."""
    context = ActionContext({"guestbook": store})
    signing = ReversibleAction(
        execute_func=lambda **args: sign_guestbook(_guestbook=store, **args),
        reverse_func=reverser_for("sign_guestbook", context),
        name="sign_guestbook",
    )

    signing.run(name="Ada", greeting="Welcome.")
    sign_guestbook("Grace", "Welcome, Grace.", _guestbook=store)  # somebody else signs in between
    signing.undo()

    assert [entry["name"] for entry in store.entries()] == ["Grace"]


def test_an_unreversible_tool_refuses_loudly_rather_than_silently():
    reverse = reverser_for("send_email", ActionContext({}))

    with pytest.raises(ValueError) as excinfo:
        reverse(args={}, result={}, timestamp="now")
    assert "No reversal is registered for 'send_email'" in str(excinfo.value)


# --- reviewers --------------------------------------------------------------


@pytest.mark.parametrize(("answer", "approved"), [("y", True), ("Y", True), ("yes", True)])
def test_the_human_reviewer_approves_on_yes(answer, approved):
    decision = HumanReviewer(input_func=lambda _: answer, output=lambda _: None)(
        ReviewRequest("task-1", [{"tool": "sign_guestbook", "args": {"name": "Ada"}}])
    )
    assert decision.approved is approved
    assert decision.reviewer == "human"


@pytest.mark.parametrize("answer", ["n", "no", "", "   ", "maybe"])
def test_the_human_reviewer_defaults_to_no(answer):
    """It is y/N: silence, nonsense and refusal all mean no."""
    decision = HumanReviewer(input_func=lambda _: answer, output=lambda _: None)(
        ReviewRequest("task-1", [])
    )
    assert decision.approved is False


def test_the_human_reviewer_is_shown_the_staged_actions():
    shown: list[str] = []
    HumanReviewer(input_func=lambda _: "n", output=shown.append)(
        ReviewRequest("task-7", [{"tool": "sign_guestbook", "args": {"name": "Ada"}}])
    )
    assert "sign_guestbook" in shown[0]
    assert "Ada" in shown[0]
    assert "task-7" in shown[0]


def test_a_cancelled_prompt_is_a_refusal():
    def cancel(_):
        raise EOFError

    decision = HumanReviewer(input_func=cancel, output=lambda _: None)(ReviewRequest("t", []))
    assert decision.approved is False
    assert "cancelled" in decision.reason


def test_with_no_terminal_the_human_reviewer_refuses_instead_of_blocking(monkeypatch):
    """Nothing here can ever wait on real stdin -- it fails closed."""

    def explode(_prompt):  # pragma: no cover - must never be reached
        raise AssertionError("the reviewer tried to read stdin")

    monkeypatch.setattr("builtins.input", explode)
    decision = HumanReviewer()(ReviewRequest("task-1", []))

    assert decision.approved is False
    assert "No interactive terminal" in decision.reason


@pytest.mark.parametrize(
    ("response", "approved"),
    [
        ("APPROVED - all three actions are necessary.", True),
        ("Yes, this looks safe.", True),
        ("REJECTED - this would overwrite an entry.", False),
        ("This is not approved.", False),
        ("The order is unsafe.", False),
        ("", False),
    ],
)
def test_the_llm_reviewer_reads_its_own_verdict(response, approved):
    llm = FakeLLM([response])
    decision = LLMReviewer(generate_response=llm)(ReviewRequest("task-1", []))

    assert decision.approved is approved
    assert decision.reviewer == "llm"
    assert decision.reason == response.strip()


def test_the_llm_reviewer_uses_the_powerful_tier():
    """Reviewing is judgement work: the one place MATE says not to economise."""
    llm = FakeLLM(["APPROVED"])
    LLMReviewer(generate_response=llm)(
        ReviewRequest("task-1", [{"tool": "sign_guestbook", "args": {"name": "Ada"}}])
    )

    assert llm.tiers == ["powerful"]
    sent = llm.calls[0].messages[-1]["content"]
    assert "sign_guestbook" in sent and "Ada" in sent
    assert "unintended consequences" in sent  # the note's review checklist


def test_the_llm_reviewer_takes_the_llm_from_the_action_context():
    llm = FakeLLM(["APPROVED"])
    request = ReviewRequest("task-1", [], action_context=ActionContext({"llm": llm}))

    assert LLMReviewer()(request).approved is True


def test_with_no_llm_at_all_the_review_is_a_refusal():
    decision = LLMReviewer()(ReviewRequest("task-1", [], action_context=ActionContext({})))

    assert decision.approved is False
    assert "No LLM was available" in decision.reason


def test_reviewer_for_resolves_names_callables_and_the_environment(monkeypatch):
    assert isinstance(reviewer_for("human"), HumanReviewer)
    assert isinstance(reviewer_for("llm"), LLMReviewer)
    assert reviewer_for(approve) is approve
    assert isinstance(reviewer_for(None), HumanReviewer)  # the safe default

    monkeypatch.setenv(REVIEWER_ENV_VAR, "llm")
    assert isinstance(reviewer_for(None), LLMReviewer)


def test_an_unknown_reviewer_name_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        reviewer_for("vibes")
    assert "Unknown reviewer 'vibes'" in str(excinfo.value)
    assert "human" in str(excinfo.value) and "llm" in str(excinfo.value)


# --- StagedActionEnvironment ------------------------------------------------


def guestbook_registry() -> PythonActionRegistry:
    return PythonActionRegistry(tool_names=["sign_guestbook", "read_guestbook", "terminate"])


def run_tool(environment, registry, store, tool: str, **args) -> dict:
    return environment.execute_action(
        agent=None,
        action_context=ActionContext({"guestbook": store}),
        action=registry.get_action(tool),
        args=args,
    )


def signing(**args) -> str:
    """The response a function-calling model gives to sign the guestbook.

    ``tool_call``'s first parameter is itself called ``name``, which a guestbook
    signing also uses -- so this spells the call out rather than fighting it.
    """
    return json.dumps({"tool": "sign_guestbook", "args": args})


def test_an_unstaged_tool_executes_normally(store, registered_tools):
    environment = StagedActionEnvironment(reviewer=reject)
    sign_guestbook("Ada", "Welcome.", _guestbook=store)

    result = run_tool(environment, guestbook_registry(), store, "read_guestbook")

    assert result["tool_executed"] is True
    assert result["result"]["entries"][0]["name"] == "Ada"
    assert environment.decisions == []  # the reviewer was never troubled


def test_an_approved_staged_tool_executes_with_its_dependencies_injected(store, registered_tools):
    environment = StagedActionEnvironment(reviewer=approve)

    result = run_tool(
        environment, guestbook_registry(), store, "sign_guestbook", name="Ada", greeting="Welcome."
    )

    assert result["tool_executed"] is True
    assert result["result"]["entry_id"] == 1
    assert result["review"] == {
        "approved": True,
        "reviewer": "test",
        "reason": "Approved by the test.",
    }
    assert store.entries()[0]["name"] == "Ada"  # the injected store, not a configured one


def test_a_rejected_staged_tool_returns_a_structured_refusal(store, registered_tools):
    """Not an exception, not a bare False: a result the agent can read and react to."""
    environment = StagedActionEnvironment(reviewer=reject)

    result = run_tool(
        environment, guestbook_registry(), store, "sign_guestbook", name="Ada", greeting="Welcome."
    )

    assert result["tool_executed"] is False
    assert result["staged"] is True
    assert result["status"] == "rejected"
    assert result["tool"] == "sign_guestbook"
    assert result["args"] == {"name": "Ada", "greeting": "Welcome."}
    assert result["review"] == {
        "approved": False,
        "reviewer": "test",
        "reason": "The visitor asked not to be recorded.",
    }
    assert "The visitor asked not to be recorded." in result["error"]
    assert "Nothing was written" in result["error"]
    assert json.dumps(result)  # it goes into memory, so it must serialise
    assert not store.exists()


def test_a_rejection_leaves_the_store_untouched(store, registered_tools):
    environment = StagedActionEnvironment(reviewer=reject)
    run_tool(environment, guestbook_registry(), store, "sign_guestbook", name="A", greeting="G")

    assert store.count() == 0
    assert environment.decisions[0].approved is False


def test_the_staged_tool_list_is_configurable(store, registered_tools):
    """Stage nothing and the environment is an ordinary PythonEnvironment."""
    environment = StagedActionEnvironment(staged_tools=[], reviewer=reject)

    result = run_tool(
        environment, guestbook_registry(), store, "sign_guestbook", name="Ada", greeting="Welcome."
    )

    assert result["tool_executed"] is True
    assert store.count() == 1


def test_the_staged_environment_is_an_environment_that_still_injects(store, registered_tools):
    assert isinstance(StagedActionEnvironment(reviewer=approve), PythonEnvironment)


def test_a_tool_that_raises_after_approval_still_reaches_the_agent_as_feedback(
    store, registered_tools
):
    environment = StagedActionEnvironment(reviewer=approve)

    result = run_tool(
        environment, guestbook_registry(), store, "sign_guestbook", name="", greeting="Welcome."
    )

    assert result["tool_executed"] is False
    assert "'name'" in result["error"]
    assert not store.exists()


def test_reviewing_an_unknown_task_raises():
    with pytest.raises(ValueError) as excinfo:
        StagedActionEnvironment(reviewer=approve).review_transaction("no-such-task")
    assert "No staged transaction found" in str(excinfo.value)


def test_every_staged_call_is_kept_as_an_audit_trail(store, registered_tools):
    environment = StagedActionEnvironment(reviewer=approve)
    registry = guestbook_registry()
    run_tool(environment, registry, store, "sign_guestbook", name="Ada", greeting="Welcome.")
    run_tool(environment, registry, store, "sign_guestbook", name="Grace", greeting="Welcome.")

    assert len(environment.staged_transactions) == 2
    assert [decision.approved for decision in environment.decisions] == [True, True]
    assert all(t.committed for t in environment.staged_transactions.values())


# --- end to end: the file on disk -------------------------------------------


def build_agent(environment, llm) -> Agent:
    return Agent(
        goals=GOALS,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=guestbook_registry(),
        generate_response=llm,
        environment=environment,
        max_iterations=3,
    )


def test_end_to_end_a_rejected_signing_never_creates_the_guestbook(store, registered_tools):
    """The headline guarantee: reviewer says no, the file is not there at all."""
    llm = FakeLLM(
        [
            signing(name="Ada Lovelace", greeting="Welcome, Ada."),
            tool_call("terminate", message="The signing was refused."),
        ]
    )
    agent = build_agent(StagedActionEnvironment(reviewer=reject), llm)

    memory = agent.run("Ada Lovelace is here.", action_context_props={"guestbook": store})

    assert not store.path.exists()
    assert store.count() == 0
    feedback = json.loads(memory.get_memories()[2]["content"])
    assert feedback["status"] == "rejected"
    assert "Nothing was written" in feedback["error"]


def test_end_to_end_a_rejection_leaves_an_existing_guestbook_byte_identical(
    store, registered_tools
):
    sign_guestbook("Grace Hopper", "Welcome, Grace.", _guestbook=store)
    before = store.path.read_bytes()

    llm = FakeLLM(
        [
            signing(name="Ada", greeting="Welcome, Ada."),
            tool_call("terminate", message="Refused."),
        ]
    )
    agent = build_agent(StagedActionEnvironment(reviewer=reject), llm)
    agent.run("Ada is here.", action_context_props={"guestbook": store})

    assert store.path.read_bytes() == before


def test_end_to_end_an_approved_signing_writes_exactly_one_entry(store, registered_tools):
    llm = FakeLLM(
        [
            signing(name="Ada Lovelace", greeting="Welcome, Ada."),
            tool_call("terminate", message="Signed as entry 1."),
        ]
    )
    agent = build_agent(StagedActionEnvironment(reviewer=approve), llm)

    agent.run("Ada Lovelace is here.", action_context_props={"guestbook": store})

    assert store.count() == 1
    assert store.entries()[0]["name"] == "Ada Lovelace"


def test_end_to_end_an_llm_reviewer_can_veto_the_write(store, registered_tools):
    """The reviewing LLM is separate from the acting one, and it is asked first."""
    reviewing_llm = FakeLLM(["REJECTED - the visitor did not consent to being recorded."])
    acting_llm = FakeLLM(
        [
            signing(name="Ada", greeting="Welcome, Ada."),
            tool_call("terminate", message="Refused."),
        ]
    )
    environment = StagedActionEnvironment(reviewer=LLMReviewer(generate_response=reviewing_llm))

    build_agent(environment, acting_llm).run("Ada", action_context_props={"guestbook": store})

    assert not store.path.exists()
    assert environment.decisions[0].reviewer == "llm"
    assert reviewing_llm.tiers == ["powerful"]


# --- undo_last_guestbook_entry ----------------------------------------------


def test_undoing_removes_the_most_recent_entry(store):
    sign_guestbook("Ada", "Welcome, Ada.", _guestbook=store)
    sign_guestbook("Grace", "Welcome, Grace.", _guestbook=store)

    result = undo_last_guestbook_entry(_guestbook=store)

    assert result["status"] == "undone"
    assert result["entry_id"] == 2
    assert result["name"] == "Grace"
    assert [entry["name"] for entry in store.entries()] == ["Ada"]


def test_entry_ids_are_not_recycled_after_an_undo(store):
    sign_guestbook("Ada", "Welcome.", _guestbook=store)
    undo_last_guestbook_entry(_guestbook=store)

    assert sign_guestbook("Grace", "Welcome.", _guestbook=store)["entry_id"] == 2


def test_undoing_an_empty_guestbook_instructs_the_agent_instead_of_failing(store):
    result = undo_last_guestbook_entry(_guestbook=store)

    assert result["status"] == "nothing_to_undo"
    assert "nothing to undo" in result["error"]
    assert "sign_guestbook" in result["error"]
    assert "Do not call undo_last_guestbook_entry again" in result["error"]


def test_undo_falls_back_to_the_configured_path(guestbook_path):
    sign_guestbook("Ada", "Welcome.", _guestbook=None)

    assert undo_last_guestbook_entry(_guestbook=None)["entry_id"] == 1


def test_undo_is_tagged_safety_and_is_not_in_the_lean_default(registered_tools):
    """The ``safety`` tag is what keeps a destructive tool out of the default five."""
    assert registered_tools.tools_by_tag["safety"] == ["undo_last_guestbook_entry"]
    assert "undo_last_guestbook_entry" not in registered_tools.tools_by_tag["guestbook"]

    default = PythonActionRegistry(tags=["greeting", "guestbook", "system"])
    assert default.get_action("undo_last_guestbook_entry") is None
    assert PythonActionRegistry(tags=["safety"]).get_action("undo_last_guestbook_entry") is not None


def test_the_model_sees_no_parameters_on_undo(registered_tools):
    schema = registered_tools.tools["undo_last_guestbook_entry"]["parameters"]

    assert schema == {"type": "object", "properties": {}, "required": []}


def test_undo_has_a_description_good_enough_to_stand_alone(registered_tools):
    description = registered_tools.tools["undo_last_guestbook_entry"]["description"]

    assert "Returns:" in description
    assert "correct a mistake" in description


def test_the_undo_tool_runs_through_the_environment(store, registered_tools):
    sign_guestbook("Ada", "Welcome.", _guestbook=store)
    registry = PythonActionRegistry(tool_names=["undo_last_guestbook_entry"])

    result = PythonEnvironment().execute_action(
        agent=None,
        action_context=ActionContext({"guestbook": store}),
        action=registry.get_action("undo_last_guestbook_entry"),
        args={},
    )

    assert result["tool_executed"] is True
    assert result["result"]["status"] == "undone"
    assert store.count() == 0


def test_an_action_object_is_all_the_staged_environment_needs(store, registered_tools):
    """Staging works on any Action, not just registered tools -- it keys off the name."""
    action = Action("risky_write", lambda text: text.upper(), "Write.", {}, terminal=False)
    environment = StagedActionEnvironment(staged_tools=["risky_write"], reviewer=reject)

    result = environment.execute_action(None, ActionContext({}), action, {"text": "hi"})

    assert result["status"] == "rejected"
    assert result["tool"] == "risky_write"
