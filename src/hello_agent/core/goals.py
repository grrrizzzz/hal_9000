"""G is for Goals: what the agent is for, and how it should go about it.

A ``Goal`` is deliberately a tiny frozen record rather than a wall of prompt
text. Splitting the instructions into named, prioritised objects means the agent
language decides how to *render* them, tests can assert on one goal without
string-matching a paragraph, and you can reuse a house-rules goal across several
agents.

"Goal" covers both the *what* ("greet the visitor") and the *how* ("always sign
the guestbook before terminating"). The notes group instructions under G for
exactly this reason: an agent that knows its objective but not its method is not
usefully specified.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Goal"]


@dataclass(frozen=True)
class Goal:
    """One named objective or instruction for an agent.

    Attributes:
        priority: Lower sorts first when goals are rendered into a prompt.
        name: Short identifier, e.g. ``"greet_visitor"``.
        description: The instruction itself, in plain English, addressed to the
            model.

    Example:
        >>> Goal(priority=1, name="greet", description="Greet the visitor warmly.")
        Goal(priority=1, name='greet', description='Greet the visitor warmly.')
    """

    priority: int
    name: str
    description: str
