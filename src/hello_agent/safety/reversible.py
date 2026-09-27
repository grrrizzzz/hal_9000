"""Undo, and undo-everything: the two smallest safety nets an acting agent can have.

An agent that writes to the world will eventually write the wrong thing. Not
usually because it reasoned badly -- more often because the world moved after it
decided. So the interesting question is not "how do we stop it" but "how do we
take it back".

**A reversible action** answers that for one call: a pair of functions -- do it,
and undo it -- plus the *record* of what happened, because you cannot reverse
what you did not write down. :meth:`ReversibleAction.run` captures the arguments,
the result and the time; :meth:`ReversibleAction.undo` hands that whole record to
the reverse function, which picks out whatever it needs. For a guestbook signing
that is the ``entry_id`` the tool returned::

    reverse_func=lambda **record: store.remove_entry(record["result"]["entry_id"])

Note it keys off the *id*, not "the last entry". Ids are never reused by
:class:`~hello_agent.storage.guestbook.GuestbookStore`, so this stays correct
even when another visitor signed in between.

**A transaction** answers it for a sequence. Actions are queued, then executed
together; if any one of them fails, everything already done is undone **in
reverse order** and the original error is re-raised. That is what stops a
half-finished task from leaving the world in a state no one designed.

Not every action has an honest reverse -- an email that has been read cannot be
unsent. :func:`reverser_for` is deliberate about that: a tool with no registered
reversal gets a reverse function that *raises*, so the gap shows up loudly at
rollback time instead of being silently skipped.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from hello_agent.storage.guestbook import GuestbookStore

__all__ = [
    "DEFAULT_REVERSERS",
    "ActionTransaction",
    "ReversibleAction",
    "reverser_for",
]


class ReversibleAction:
    """One action plus the knowledge of how to take it back.

    Args:
        execute_func: The callable to run. Receives the arguments given to
            :meth:`run`.
        reverse_func: Called as ``reverse_func(**record)`` with the execution
            record -- ``args``, ``result`` and ``timestamp`` -- so it can pull
            out whichever handle undoing needs.
        name: What to call this in logs and review prompts. Defaults to the
            executed function's ``__name__``.

    """

    def __init__(
        self,
        execute_func: Callable[..., Any],
        reverse_func: Callable[..., Any],
        name: str | None = None,
    ):
        self.execute = execute_func
        self.reverse = reverse_func
        self.name = name or getattr(execute_func, "__name__", "action")
        #: What the last :meth:`run` did. ``None`` until it has run.
        self.execution_record: dict | None = None

    def run(self, **args) -> Any:
        """Execute the action and record everything needed to reverse it."""
        result = self.execute(**args)
        self.execution_record = {
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        return result

    def undo(self) -> Any:
        """Reverse the recorded execution.

        The record is cleared first, so a second ``undo`` raises rather than
        reversing the same action twice -- a real risk when one action object is
        rolled back by two different code paths.

        Raises:
            ValueError: if the action never ran, or has already been undone.
        """
        record = self.execution_record
        if not record:
            raise ValueError(
                f"No action to reverse: {self.name!r} has not been run, or has already "
                "been undone. Call run() before undo()."
            )
        self.execution_record = None
        return self.reverse(**record)

    def __repr__(self) -> str:
        return f"ReversibleAction(name={self.name!r}, ran={self.execution_record is not None})"


class ActionTransaction:
    """A queue of reversible actions that all happen, or none of them stick.

    Queue with :meth:`add`, run with :meth:`execute`. If any action raises,
    :meth:`rollback` undoes everything already executed in reverse order and the
    original exception propagates -- the caller learns the task failed *and* the
    world is back where it started.

    Three actions where the third raises leave the log
    ``["do1", "do2", "undo2", "undo1"]`` -- forward in order, back in reverse.
    """

    def __init__(self, transaction_id: str | None = None):
        #: queued ``(action, args)`` pairs, in the order they will run
        self.actions: list[tuple[ReversibleAction, dict]] = []
        #: actions that completed, in the order they completed
        self.executed: list[ReversibleAction] = []
        #: what each completed action returned
        self.results: list[Any] = []
        self.committed = False
        self.transaction_id = transaction_id or str(uuid.uuid4())

    def add(self, action: ReversibleAction, **args) -> None:
        """Queue an action for execution.

        Raises:
            ValueError: if the transaction has already been committed.
        """
        if self.committed:
            raise ValueError(
                f"Transaction {self.transaction_id} is already committed; "
                "start a new transaction rather than adding to a closed one."
            )
        self.actions.append((action, args))

    def execute(self) -> list[Any]:
        """Run every queued action, rolling back completely if one fails.

        Returns:
            Each action's result, in order.

        Raises:
            Exception: whatever the failing action raised, re-raised after the
                rollback so the caller sees the real cause.
        """
        try:
            for action, args in self.actions:
                self.results.append(action.run(**args))
                self.executed.append(action)
        except Exception:
            self.rollback()
            raise
        return list(self.results)

    def rollback(self) -> None:
        """Undo every executed action, most recent first."""
        for action in reversed(self.executed):
            action.undo()
        self.executed = []
        self.results = []

    def commit(self) -> None:
        """Mark the transaction closed. Nothing further may be added."""
        self.committed = True

    def describe(self) -> list[dict]:
        """The queued actions as plain data, for logging or a review prompt."""
        return [{"tool": action.name, "args": dict(args)} for action, args in self.actions]

    def __len__(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        return (
            f"ActionTransaction(id={self.transaction_id[:8]}, queued={len(self.actions)}, "
            f"executed={len(self.executed)}, committed={self.committed})"
        )


def _reverse_guestbook_signing(action_context) -> Callable[..., Any]:
    """Build the reversal for ``sign_guestbook`` against this run's store."""
    store = (action_context.get("guestbook") if action_context is not None else None) or (
        GuestbookStore()
    )
    return lambda **record: store.remove_entry(record["result"]["entry_id"])


#: tool name -> ``(action_context) -> reverse_func``. The template's one write
#: is the only entry; add yours here when you add a side-effecting tool.
DEFAULT_REVERSERS: dict[str, Callable[[Any], Callable[..., Any]]] = {
    "sign_guestbook": _reverse_guestbook_signing,
}


def reverser_for(
    tool_name: str,
    action_context,
    reversers: dict[str, Callable] | None = None,
) -> Callable[..., Any]:
    """The reverse function for ``tool_name``, bound to this run's resources.

    A tool with no registered reversal gets one that raises. Refusing loudly is
    the point: a rollback that silently skips an irreversible step is worse than
    no rollback, because it reports success.
    """
    table = DEFAULT_REVERSERS if reversers is None else reversers
    factory = table.get(tool_name)
    if factory is not None:
        return factory(action_context)

    def refuse(**record) -> Any:
        raise ValueError(
            f"No reversal is registered for {tool_name!r}, so it cannot be rolled back. "
            "Register one in hello_agent.safety.reversible.DEFAULT_REVERSERS, or keep the "
            "tool out of transactions that may need to undo it."
        )

    return refuse
