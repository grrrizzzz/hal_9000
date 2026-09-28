"""Two global registries: one for tools, one for agents.

**Tools.** ``@register_tool`` (see :mod:`hal9000.tools.decorators`) writes
into the module-level :data:`tools` and :data:`tools_by_tag` dicts here as a
side effect of importing a tool module. An agent then asks for a *slice* of that
catalogue by tag -- ``PythonActionRegistry(tags=["greeting", "guestbook"])`` --
rather than assembling ``Action`` objects by hand. Tags are how you give one
agent write access and another read-only access to the same codebase of tools.

The registries are global and mutable, which is a real trade-off: it is what
makes a decorator on a function in another file enough to publish a tool, and it
is also why tests must reset them. Use :func:`clear_tools` (or the
``clean_tool_registry`` fixture) rather than reassigning the names, so importers
that did ``from ... import tools`` keep seeing the same object.

**Agents.** :class:`AgentRegistry` is the equivalent directory for whole agents.
It maps a name to an agent's ``run`` function, which is all ``call_agent`` needs
in order to delegate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["AgentRegistry", "clear_tools", "tools", "tools_by_tag"]

#: name -> tool metadata. One entry per registered tool. The metadata dict is::
#:
#:     {
#:         "description": str,   # model-facing; the docstring unless overridden
#:         "parameters": dict,   # JSON Schema, dependencies already excluded
#:         "function": Callable, # the undecorated function
#:         "terminal": bool,     # True ends the agent loop after it runs
#:         "tags": list[str],    # used to slice the catalogue into registries
#:     }
tools: dict[str, dict[str, Any]] = {}

#: tag -> list of tool names carrying that tag, e.g.
#: ``{"guestbook": ["sign_guestbook", "read_guestbook"]}``.
tools_by_tag: dict[str, list[str]] = {}


def clear_tools() -> None:
    """Empty both registries **in place**.

    In place matters: modules that imported the dicts by name hold references to
    these exact objects. Rebinding the names would leave them looking at stale
    copies.
    """
    tools.clear()
    tools_by_tag.clear()


class AgentRegistry:
    """A directory of agents, keyed by name, valued by their ``run`` function.

    Storing the bound ``run`` method rather than the ``Agent`` keeps the contract
    to one call -- ``run(user_input, memory, action_context_props)`` -- so a
    callee can be an ``Agent``, a stub, or any callable with that shape.

    Example:
        >>> registry = AgentRegistry()
        >>> registry.register_agent("echo", lambda user_input, **kw: user_input)
        >>> registry.get_agent("echo")("hi")
        'hi'
        >>> registry.get_agent("nope") is None
        True
    """

    def __init__(self) -> None:
        self.agents: dict[str, Callable] = {}

    def register_agent(self, name: str, run_function: Callable) -> None:
        """Register an agent's ``run`` function under ``name``."""
        self.agents[name] = run_function

    def get_agent(self, name: str) -> Callable | None:
        """Look up an agent's ``run`` function, or ``None`` if there is no such agent."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """Names of every registered agent, so an agent can be told who it may call."""
        return list(self.agents.keys())
