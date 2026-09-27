"""A is for Actions: what the agent *may* do, and how it looks those up.

An :class:`Action` is an interface description -- a name, a model-facing
description, a JSON Schema for its arguments, and the Python function behind it.
The :class:`~hello_agent.core.environment.Environment` is the implementation
side: it decides *how* the call actually happens. Splitting the two is what lets
the same agent run against a real filesystem, a sandbox, or a staged-review
queue without touching its reasoning.

An :class:`ActionRegistry` is the lookup table. The model answers with a tool
*name*; the registry turns that string into something executable.
:class:`PythonActionRegistry` builds one from the global catalogue that
``@register_tool`` populates, so assembling an agent's toolkit is a list of tags
rather than a page of ``register(Action(...))`` calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hello_agent.core import registry

__all__ = ["Action", "ActionRegistry", "PythonActionRegistry"]


class Action:
    """One tool the agent can invoke: its contract plus its implementation.

    Args:
        name: What the model calls it. Lower snake_case, verb-first.
        function: The Python callable to run.
        description: Model-facing explanation -- usually the docstring. This is
            prompt text, so it is worth writing well.
        parameters: JSON Schema for the arguments the *model* supplies.
            Injected dependencies are deliberately absent from it.
        terminal: ``True`` if running this action ends the agent loop.
    """

    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: dict,
        terminal: bool = False,
    ):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Call the underlying function.

        No error handling here on purpose: the environment owns that, because
        *how* a failure is reported to the agent is an environment concern.
        """
        return self.function(**args)

    def __repr__(self) -> str:
        return f"Action(name={self.name!r}, terminal={self.terminal})"


class ActionRegistry:
    """Name-to-:class:`Action` lookup for one agent.

    Example:
        >>> reg = ActionRegistry()
        >>> reg.register(Action("noop", lambda: None, "Do nothing.", {}))
        >>> reg.get_action("noop")
        Action(name='noop', terminal=False)
        >>> reg.get_action("absent") is None
        True
    """

    def __init__(self) -> None:
        self.actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        """Add an action, replacing any earlier action of the same name."""
        self.actions[action.name] = action

    def get_action(self, name: str) -> Action | None:
        """Look up an action by name, or ``None``.

        ``None`` rather than an exception: the model invents tool names, and an
        invented name is feedback to write into memory, not a crash.
        """
        return self.actions.get(name, None)

    def get_actions(self) -> list[Action]:
        """Every registered action, in registration order."""
        return list(self.actions.values())

    def __len__(self) -> int:
        return len(self.actions)


class PythonActionRegistry(ActionRegistry):
    """An :class:`ActionRegistry` built from the global ``@register_tool`` catalogue.

    Args:
        tags: Keep only tools carrying at least one of these tags. ``None``
            means "no tag filter". An unknown tag simply matches nothing -- an
            empty registry, not an error, because a feature flag that is off
            should not explode.
        tool_names: Keep only tools with these exact names. ``None`` means "no
            name filter".

    The two filters are combined with **and**, so
    ``PythonActionRegistry(tags=["guestbook"], tool_names=["sign_guestbook"])``
    yields a tool only if it is both.

    The catalogue is read at construction time, so import your tool modules
    first -- ``import hello_agent.tools`` does that for all of them.

    Example:
        >>> from hello_agent.tools import guestbook  # doctest: +SKIP
        >>> PythonActionRegistry(tags=["guestbook"]).get_action("sign_guestbook")  # doctest: +SKIP
        Action(name='sign_guestbook', terminal=False)
    """

    def __init__(self, tags: list[str] = None, tool_names: list[str] = None):
        super().__init__()
        self.tags = tags
        self.tool_names = tool_names
        for name, metadata in registry.tools.items():
            if self._selects(name, metadata):
                self.register(self._build_action(name, metadata))

    def _selects(self, name: str, metadata: dict) -> bool:
        """Does this catalogue entry pass both filters?"""
        if self.tool_names is not None and name not in self.tool_names:
            return False
        if self.tags is not None:
            if not any(tag in metadata.get("tags", []) for tag in self.tags):
                return False
        return True

    @staticmethod
    def _build_action(name: str, metadata: dict) -> Action:
        """Turn one catalogue entry into an :class:`Action`."""
        return Action(
            name=name,
            function=metadata["function"],
            description=metadata["description"],
            parameters=metadata["parameters"],
            terminal=metadata.get("terminal", False),
        )
