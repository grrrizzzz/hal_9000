"""Staged execution with review: the write waits until somebody says yes.

Rollback is a cure; this is the prevention. A :class:`StagedActionEnvironment`
is an ordinary :class:`~hello_agent.core.environment.Environment` in every
respect except one: for the tools you name, it does not execute the call. It
*stages* the call into an
:class:`~hello_agent.safety.reversible.ActionTransaction`, shows that
transaction to a reviewer, and executes only on approval.

Because the swap happens at the environment -- the E in GAME -- goals, tools and
loop are untouched. The model still calls ``sign_guestbook`` and still reads a
result dict. It simply may get back a *refusal* instead of an entry id, and a
refusal is deliberately shaped like every other failed-tool result:
``tool_executed: False`` plus an ``error`` string written to be read by a model.
Rejection is feedback the agent can react to -- not an exception that kills the
run, and not a bare ``False`` it cannot interpret.

Naming which tools to stage is a policy decision, and the single-safe-tool
pattern is what makes it a short list: this template has exactly one tool that
writes, so :data:`DEFAULT_STAGED_TOOLS` has exactly one name in it.

None of this is on by default. ``Features.staged_review`` switches it on, and
the lean default never stops to ask a question -- which is what keeps the
quickstart two minutes long.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import Action
from hello_agent.core.environment import PythonEnvironment
from hello_agent.safety.reversible import ActionTransaction, ReversibleAction, reverser_for
from hello_agent.safety.review import ReviewDecision, ReviewRequest, reviewer_for

__all__ = ["DEFAULT_STAGED_TOOLS", "StagedActionEnvironment"]

#: Every tool that writes. One name, because there is one tool that writes.
DEFAULT_STAGED_TOOLS: tuple[str, ...] = ("sign_guestbook",)


class StagedActionEnvironment(PythonEnvironment):
    """An environment that holds the risky tools back until a reviewer approves.

    Subclasses :class:`~hello_agent.core.environment.PythonEnvironment` rather
    than ``Environment`` so dependency injection is inherited verbatim: a staged
    tool is eventually called with exactly the arguments it would have received
    unstaged.

    Args:
        staged_tools: Tool names to hold for review. Defaults to
            :data:`DEFAULT_STAGED_TOOLS`; everything else executes normally.
        reviewer: A name (``"human"``, ``"llm"``) or any callable taking a
            :class:`~hello_agent.safety.review.ReviewRequest`. ``None`` resolves
            via :func:`~hello_agent.safety.review.reviewer_for`.
        reversers: Overrides
            :data:`~hello_agent.safety.reversible.DEFAULT_REVERSERS`, so an
            approved transaction can still be rolled back if it fails part-way.

    Example:
        >>> env = StagedActionEnvironment(reviewer=lambda request: None)  # doctest: +SKIP
    """

    def __init__(
        self,
        staged_tools: list[str] | tuple[str, ...] | None = None,
        reviewer: str | Callable | None = None,
        reversers: dict[str, Callable] | None = None,
    ):
        self.staged_tools = set(DEFAULT_STAGED_TOOLS if staged_tools is None else staged_tools)
        self.reviewer = reviewer_for(reviewer)
        self.reversers = reversers
        #: task_id -> transaction, kept for inspection and as an audit trail
        self.staged_transactions: dict[str, ActionTransaction] = {}
        #: every decision this environment has taken, oldest first
        self.decisions: list[ReviewDecision] = []

    # --- the Environment contract ----------------------------------------

    def execute_action(
        self, agent, action_context: ActionContext, action: Action, args: dict
    ) -> dict:
        """Execute ``action`` normally, or stage it for review if it is risky."""
        if action.name not in self.staged_tools:
            return super().execute_action(agent, action_context, action, args)

        task_id = str(uuid.uuid4())
        transaction = self.stage_actions(task_id)
        transaction.add(self._reversible(agent, action_context, action), **args)

        decision = self.review_transaction(task_id, action_context)
        self.decisions.append(decision)
        if not decision.approved:
            return self.format_refusal(action, args, decision, task_id)

        try:
            results = transaction.execute()
            transaction.commit()
        except Exception as error:
            return self.format_error(error)

        formatted = self.format_result(results[-1])
        formatted["review"] = decision.as_dict()
        return formatted

    # --- staging and review ----------------------------------------------

    def stage_actions(self, task_id: str) -> ActionTransaction:
        """Create and remember a transaction for ``task_id``."""
        transaction = ActionTransaction(transaction_id=task_id)
        self.staged_transactions[task_id] = transaction
        return transaction

    def review_transaction(
        self, task_id: str, action_context: ActionContext | None = None
    ) -> ReviewDecision:
        """Show the staged transaction to the reviewer and return its decision.

        Raises:
            ValueError: if ``task_id`` names no staged transaction.
        """
        transaction = self.staged_transactions.get(task_id)
        if transaction is None:
            raise ValueError(f"No staged transaction found for task {task_id!r}.")
        request = ReviewRequest(
            task_id=task_id, actions=transaction.describe(), action_context=action_context
        )
        return self.reviewer(request)

    def format_refusal(
        self, action: Action, args: dict, decision: ReviewDecision, task_id: str
    ) -> dict:
        """Turn a rejection into a result the agent can read and act on.

        ``tool_executed`` and ``error`` match the shape of every other failed
        call, so no capability, memory writer or agent language needs to know
        that review exists. The extra keys let code that *does* know react
        precisely.
        """
        return {
            "tool_executed": False,
            "staged": True,
            "status": "rejected",
            "task_id": task_id,
            "tool": action.name,
            "args": dict(args),
            "review": decision.as_dict(),
            "error": (
                f"{action.name} was staged for review and the {decision.reviewer} reviewer "
                f"rejected it: {decision.reason} Nothing was written and nothing was changed. "
                f"Do not call {action.name} again with the same arguments -- either revise "
                "them in light of that reason, or tell the user the action was refused and stop."
            ),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # --- plumbing ---------------------------------------------------------

    def _reversible(self, agent, action_context: ActionContext, action: Action) -> ReversibleAction:
        """Wrap ``action`` so the transaction can run it later -- and undo it.

        Injection is deferred into the closure, so the transaction holds only the
        model's own arguments. That keeps the review prompt readable and keeps
        storage handles and LLM callables out of it.
        """

        def execute(**model_args) -> Any:
            injected = self.inject_dependencies(agent, action_context, action, model_args)
            return action.execute(**injected)

        return ReversibleAction(
            execute_func=execute,
            reverse_func=reverser_for(action.name, action_context, self.reversers),
            name=action.name,
        )

    def __repr__(self) -> str:
        return (
            f"StagedActionEnvironment(staged={sorted(self.staged_tools)}, "
            f"reviewer={getattr(self.reviewer, 'name', type(self.reviewer).__name__)!r})"
        )
