"""E is for Environment: where actions actually happen, and what comes back.

If :class:`~hal9000.core.actions.Action` is the interface, the Environment
is the implementation. It owns three things the agent loop should never know
about:

1. **Execution.** Calling the function.
2. **Dependency injection.** Reading the tool's signature and supplying the
   resources it asked for from the :class:`~hal9000.core.action_context.ActionContext`.
3. **Error containment.** A tool that raises must produce a *result*, not a
   crash. An agent that can read "that failed, because X" can retry; an agent
   that dies cannot.

That last point is the whole reason the loop has no ``try`` in it.
"""

from __future__ import annotations

import inspect
import time
import traceback
from collections.abc import Callable
from typing import Any

from hal9000.core.action_context import ActionContext
from hal9000.core.actions import Action

__all__ = ["Environment", "PythonEnvironment", "has_named_parameter"]


def has_named_parameter(function: Callable, name: str) -> bool:
    """Does ``function`` declare a parameter called ``name``?

    Used to decide what to inject. A tool opts into a dependency by naming it;
    a tool that names nothing receives nothing.
    """
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):  # builtins and other unintrospectable callables
        return False


class Environment:
    """Executes actions and formats their results. Never raises at the agent.

    Subclass this to change *how* execution happens -- inject dependencies
    (:class:`PythonEnvironment`), require review before a write, or dispatch to
    a remote worker -- without touching goals, memory, or the loop.
    """

    def execute_action(
        self,
        agent,
        action_context: ActionContext,
        action: Action,
        args: dict,
    ) -> dict:
        """Run ``action`` with ``args`` and return a result dict.

        Returns:
            ``{"tool_executed": True, "result": ..., "timestamp": ...}`` on
            success, or ``{"tool_executed": False, "error": ..., "traceback": ...}``
            if the tool raised. Both shapes are JSON-serialisable, because they
            go straight back into the prompt as the agent's feedback.
        """
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return self.format_error(e)

    def format_result(self, result: Any) -> dict:
        """Wrap a successful result with the metadata the agent sees."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def format_error(self, error: Exception) -> dict:
        """Turn an exception into feedback the model can act on.

        The traceback is included deliberately. It is the single most useful
        thing you can hand a model that just called a tool wrongly.
        """
        return {
            "tool_executed": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


class PythonEnvironment(Environment):
    """An :class:`Environment` that injects dependencies by parameter name.

    Two rules, both driven by the tool's own signature:

    - a parameter named ``action_context`` receives the
      :class:`~hal9000.core.action_context.ActionContext` itself;
    - a parameter named ``_foo`` receives ``action_context.get("foo")``.

    (``action_agent`` receives the running agent, for the rare tool that needs
    to introspect its caller.)

    ``@register_tool`` strips all three shapes from the JSON Schema, so the model
    never sees them, never has to supply them, and cannot forge them. A tool that
    declares none of them is called with exactly the arguments the model chose.

    Example -- the model supplies only ``name``::

        @register_tool(tags=["guestbook"])
        def sign_guestbook(action_context: ActionContext, name: str, _guestbook) -> dict:
            ...
    """

    def execute_action(
        self,
        agent,
        action_context: ActionContext,
        action: Action,
        args: dict,
    ) -> dict:
        """Inject the declared dependencies, then execute."""
        try:
            args_copy = self.inject_dependencies(agent, action_context, action, args)
            result = action.execute(**args_copy)
            return self.format_result(result)
        except Exception as e:
            return self.format_error(e)

    def inject_dependencies(
        self,
        agent,
        action_context: ActionContext,
        action: Action,
        args: dict,
    ) -> dict:
        """Return a copy of ``args`` with the tool's declared dependencies added."""
        args_copy = dict(args)
        function = action.function

        if has_named_parameter(function, "action_context"):
            args_copy["action_context"] = action_context
        if has_named_parameter(function, "action_agent"):
            args_copy["action_agent"] = agent

        for key, value in action_context.properties.items():
            param_name = "_" + key
            if has_named_parameter(function, param_name):
                args_copy[param_name] = value

        return args_copy
