"""Who gets asked, and how their answer is read.

A reviewer is just a callable: :class:`ReviewRequest` in, :class:`ReviewDecision`
out. That is the whole extension point -- a rule engine, a Slack approval, a
second agent, or your own policy function all drop straight in, and
:class:`~hello_agent.safety.staged.StagedActionEnvironment` never learns which.

Two ship with the template:

``human``  asks on stdin, ``y``/``N``, defaulting to no. Where it asks is
           injectable, which is what keeps it out of the test suite's way -- and
           with no terminal to ask, it refuses rather than blocking. Fail closed:
           an unanswerable prompt would hang the run, and refusing costs only an
           unreviewed write.
``llm``    asks the powerful tier, with the review prompt from
           ``environment_safety.md``. Reviewing is judgement work, so it is the
           one place MATE's model efficiency says *not* to economise. Treat it
           as a first filter, not an authority.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hello_agent.core.action_context import ActionContext
from hello_agent.core.prompt import Prompt

__all__ = [
    "DEFAULT_REVIEWER",
    "REVIEWERS",
    "REVIEWER_ENV_VAR",
    "HumanReviewer",
    "LLMReviewer",
    "ReviewDecision",
    "ReviewRequest",
    "reviewer_for",
]

#: Reviewer used when none is named. The safest of the two.
DEFAULT_REVIEWER = "human"

#: Environment variable naming the reviewer, read at call time -- never at import.
REVIEWER_ENV_VAR = "AGENT_REVIEWER"

_ASK = "Approve these staged actions? [y/N] "

_REVIEW_PROMPT = """Review these staged actions for safety.

Task ID: {task_id}

Staged Actions:
{actions}

Consider:
1. Are all actions necessary for the task?
2. Could any action have unintended consequences?
3. Are the actions in a safe order?
4. Is there a safer way to achieve the same goal?

Answer with APPROVED or REJECTED on the first line, then one sentence of reason.
"""


@dataclass
class ReviewRequest:
    """What a reviewer is shown: which actions, for which task, with what context."""

    task_id: str
    actions: list[dict] = field(default_factory=list)
    action_context: ActionContext | None = None

    def describe(self) -> str:
        """The staged actions as text a human or a model can read."""
        return "\n".join(f"Action: {item['tool']}\nArgs: {item['args']}" for item in self.actions)


@dataclass(frozen=True)
class ReviewDecision:
    """A reviewer's answer: yes or no, who said so, and why."""

    approved: bool
    reviewer: str = "unknown"
    reason: str = ""

    def as_dict(self) -> dict:
        """JSON-safe form, for the result the agent reads."""
        return {"approved": self.approved, "reviewer": self.reviewer, "reason": self.reason}


def _stdin_is_interactive() -> bool:
    """Is there a real terminal to ask? Anything unreadable counts as no."""
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


class HumanReviewer:
    """Asks a person on stdin, ``y``/``N``, defaulting to no.

    Args:
        input_func: How to ask. Defaults to the builtin ``input``. Tests inject a
            canned answer here, which is why no test can block on real stdin.
        output: How to show the staged actions. Defaults to ``print``.
        require_tty: When asking for real, refuse instead of reading a
            non-terminal stdin.
    """

    name = "human"

    def __init__(
        self,
        input_func: Callable[[str], str] | None = None,
        output: Callable[[str], Any] | None = None,
        require_tty: bool = True,
    ):
        self.input_func = input_func
        self.output = output
        self.require_tty = require_tty

    def __call__(self, request: ReviewRequest) -> ReviewDecision:
        ask = self.input_func
        if ask is None:
            if self.require_tty and not _stdin_is_interactive():
                return ReviewDecision(
                    False,
                    self.name,
                    "No interactive terminal was available to ask a human, so the staged "
                    "actions were refused rather than taken unreviewed.",
                )
            ask = input

        write = self.output if self.output is not None else print
        write(f"\nStaged for review (task {request.task_id}):\n{request.describe()}")
        try:
            answer = str(ask(_ASK)).strip()
        except (EOFError, KeyboardInterrupt):
            return ReviewDecision(False, self.name, "The reviewer cancelled without answering.")

        if answer.lower() in {"y", "yes"}:
            return ReviewDecision(True, self.name, "A human approved the staged actions.")
        return ReviewDecision(
            False, self.name, f"A human declined the staged actions (answered {answer!r})."
        )


def _reads_as_approval(response: str) -> bool:
    """Does this review text approve?

    ``environment_safety.md`` tests ``"approved" in response.lower()``, which
    reads "not approved" as approval. Rejection words are checked first here, and
    anything ambiguous counts as a refusal.
    """
    text = (response or "").strip().lower()
    if not text:
        return False
    if any(word in text for word in ("not approved", "reject", "deny", "unsafe", "do not")):
        return False
    return "approve" in text or text.startswith("yes")


class LLMReviewer:
    """Asks the powerful tier whether the staged actions should go ahead.

    Args:
        generate_response: The LLM to ask. Defaults to ``llm`` from the
            :class:`~hello_agent.core.action_context.ActionContext`, which
            ruling R1 makes the canonical powerful tier.
        tier: Passed through to ``generate_response``.
    """

    name = "llm"

    def __init__(self, generate_response: Callable[..., str] | None = None, tier: str = "powerful"):
        self.generate_response = generate_response
        self.tier = tier

    def __call__(self, request: ReviewRequest) -> ReviewDecision:
        llm = self.generate_response
        if llm is None and request.action_context is not None:
            llm = request.action_context.get("llm")
        if llm is None:
            return ReviewDecision(
                False,
                self.name,
                "No LLM was available to review the staged actions, so they were refused.",
            )
        prompt = Prompt(
            messages=[
                {
                    "role": "system",
                    "content": "You review actions an AI agent wants to take. Be conservative.",
                },
                {
                    "role": "user",
                    "content": _REVIEW_PROMPT.format(
                        task_id=request.task_id, actions=request.describe()
                    ),
                },
            ]
        )
        response = llm(prompt, tier=self.tier)
        return ReviewDecision(_reads_as_approval(response), self.name, str(response).strip())


#: name -> reviewer factory, for :func:`reviewer_for` and the CLI's ``--reviewer``.
REVIEWERS: dict[str, Callable[[], Any]] = {"human": HumanReviewer, "llm": LLMReviewer}


def reviewer_for(spec: str | Callable | None = None) -> Callable[[ReviewRequest], ReviewDecision]:
    """Resolve a reviewer from a name, a callable, or the environment.

    ``None`` reads :data:`REVIEWER_ENV_VAR` and falls back to
    :data:`DEFAULT_REVIEWER`. A callable is returned unchanged, so an
    application or a test can supply its own policy.
    """
    if spec is None:
        spec = os.environ.get(REVIEWER_ENV_VAR, "").strip().lower() or DEFAULT_REVIEWER
    if not isinstance(spec, str):
        return spec
    try:
        return REVIEWERS[spec]()
    except KeyError:
        raise ValueError(
            f"Unknown reviewer {spec!r}. Choose one of: {', '.join(sorted(REVIEWERS))}, "
            "or pass a callable taking a ReviewRequest and returning a ReviewDecision."
        ) from None
