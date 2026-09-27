"""MATE's E: the patterns that make an acting agent safe to leave running.

Four ideas from ``agent_design_principles/environment_safety.md`` -- three of
them here, and one of them everywhere else in the template:

===========================  ==========================================================
Reversible actions           :class:`ReversibleAction` -- record enough to undo
Transactions                 :class:`ActionTransaction` -- all of it, or none of it
Staged execution + review    :class:`StagedActionEnvironment` -- ask before writing
Single safe tool             ``sign_guestbook``: see :mod:`hello_agent.tools.guestbook`
===========================  ==========================================================

The fourth is why the other three are cheap here. Because there is exactly one
tool that writes -- no ``write_file``, no ``delete_entry``, no ``set_path`` --
there is exactly one reversal to register and one tool to stage.

Everything in this package is off by default. ``Features.staged_review`` swaps
the environment; the ``safety`` tag adds ``undo_last_guestbook_entry`` to the
toolkit. The lean default never stops to ask a question, which is what keeps the
quickstart two minutes long.
"""

from __future__ import annotations

from hello_agent.safety.reversible import (
    DEFAULT_REVERSERS,
    ActionTransaction,
    ReversibleAction,
    reverser_for,
)
from hello_agent.safety.review import (
    DEFAULT_REVIEWER,
    REVIEWERS,
    HumanReviewer,
    LLMReviewer,
    ReviewDecision,
    ReviewRequest,
    reviewer_for,
)
from hello_agent.safety.staged import DEFAULT_STAGED_TOOLS, StagedActionEnvironment

__all__ = [
    "DEFAULT_REVERSERS",
    "DEFAULT_REVIEWER",
    "DEFAULT_STAGED_TOOLS",
    "REVIEWERS",
    "ActionTransaction",
    "HumanReviewer",
    "LLMReviewer",
    "ReviewDecision",
    "ReviewRequest",
    "ReversibleAction",
    "StagedActionEnvironment",
    "reverser_for",
    "reviewer_for",
]
