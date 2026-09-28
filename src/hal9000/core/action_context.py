"""The ActionContext: how a tool gets what it needs without importing the agent.

A tool that wants the conversation history, an LLM, or a storage handle has a
coupling problem. Importing the agent to reach its memory welds the tool to one
agent and makes it untestable. Taking every dependency as a normal parameter
exposes them to the model, which then has to invent an auth token.

The ActionContext is the third answer: a per-run bag of resources that the
:class:`~hal9000.core.environment.PythonEnvironment` injects into exactly
the tools that ask for them, by parameter name, invisibly to the model.

Keys this template puts in it (see :meth:`hal9000.core.agent.Agent.run`):

=================  ===========================================================
``memory``         the running :class:`~hal9000.core.memory.Memory`
``llm``            ``generate_response``, powerful tier. **Canonical.**
``powerful_llm``   alias for ``llm`` -- the same callable, same object
``fast_llm``       ``generate_response`` pinned to the fast tier (MATE)
``action_registry``  the agent's :class:`~hal9000.core.actions.ActionRegistry`
``agent_registry``   an :class:`~hal9000.core.registry.AgentRegistry`, when
                   delegation is enabled -- deliberately absent by default
=================  ===========================================================

Anything else you pass to ``Agent.run(action_context_props=...)`` lands here too,
which is how per-request values such as an auth token reach a tool.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from hal9000.core.actions import ActionRegistry
    from hal9000.core.memory import Memory
    from hal9000.core.registry import AgentRegistry

__all__ = ["ActionContext", "default_properties"]


class ActionContext:
    """A container for the resources available during one agent run.

    Example:
        >>> ctx = ActionContext({"auth_token": "abc"})
        >>> ctx.get("auth_token")
        'abc'
        >>> ctx.get("missing", "fallback")
        'fallback'
    """

    def __init__(self, properties: dict | None = None):
        #: unique per context, so logs from concurrent runs can be told apart
        self.context_id: str = str(uuid.uuid4())
        self.properties: dict[str, Any] = dict(properties or {})

    def get(self, key, default=None):
        """Return the property ``key``, or ``default`` if it was not provided."""
        return self.properties.get(key, default)

    def get_memory(self) -> Memory:
        """The run's :class:`~hal9000.core.memory.Memory`, or ``None``."""
        return self.properties.get("memory", None)

    def get_action_registry(self) -> ActionRegistry:
        """The tools available on this run, or ``None``."""
        return self.properties.get("action_registry", None)

    def get_agent_registry(self) -> AgentRegistry:
        """Other agents this run may call, or ``None`` when delegation is off.

        ``None`` is the default and is *load-bearing*: a sub-agent that cannot
        see the registry cannot delegate, which is what stops two agents calling
        each other forever.
        """
        return self.properties.get("agent_registry", None)

    def __repr__(self) -> str:
        return f"ActionContext(id={self.context_id[:8]}, keys={sorted(self.properties)})"


def default_properties(generate_response, memory, action_registry: ActionRegistry) -> dict:
    """The resources every run of every agent makes available to its tools.

    ``llm`` and ``powerful_llm`` are deliberately the same object: ``llm`` is
    canonical and every tool in this template reads it, while the alias lets the
    MATE note's examples run verbatim. ``fast_llm`` is the same callable pinned
    to the cheap tier -- that pinning *is* MATE's model efficiency.
    """

    def fast_llm(prompt, tier: str = "fast") -> str:
        """``generate_response`` on the fast tier, whatever tier is asked for."""
        return generate_response(prompt, tier="fast")

    return {
        "memory": memory,
        "llm": generate_response,
        "powerful_llm": generate_response,
        "fast_llm": fast_llm,
        "action_registry": action_registry,
    }
