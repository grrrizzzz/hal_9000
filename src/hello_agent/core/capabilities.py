"""The Capability: ten hooks into the agent loop, so the loop never grows.

Tools extend what an agent can *do*. Capabilities extend how it *runs* -- seed
memory before the first turn, rewrite every prompt, veto an action, log a
result, stop early. The alternative is a loop that accumulates an ``if`` per
feature until nobody can read it.

Each hook has a default that does nothing (or passes its value through), so a
capability overrides only the moments it cares about. The loop calls all ten on
every run, whether or not any capabilities are registered -- which is what makes
``capabilities=[TimeAwareCapability()]`` a genuine one-line change.

Where each hook fires (see :meth:`hello_agent.core.agent.Agent.run`)::

    init                              once, before the first iteration
    ┌ start_agent_loop                 top of each iteration; False stops the run
    │ process_prompt                   prompt built, not yet sent
    │ process_response                 raw text back from the LLM
    │ process_action                   response parsed into {"tool", "args"}
    │ process_result                   action executed
    │ process_new_memories             memories drafted, not yet stored
    │ end_agent_loop                   bottom of each iteration
    └ should_terminate                 True stops the run
    terminate                         once, after the loop

The ``process_*`` hooks compose left to right via :func:`functools.reduce`: with
``capabilities=[A, B]``, the loop uses ``B(A(value))``. Order in the list is
therefore meaningful -- the first capability sees the raw value, the last one
decides what the loop actually gets.
"""

from __future__ import annotations

from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import Action
from hello_agent.core.memory import Memory
from hello_agent.core.prompt import Prompt

__all__ = ["Capability"]


class Capability:
    """Base class for agent-loop extensions. Override only the hooks you need.

    Example:
        >>> class ShoutCapability(Capability):
        ...     def __init__(self):
        ...         super().__init__("Shout", "Uppercases every LLM response.")
        ...     def process_response(self, agent, action_context, response):
        ...         return response.upper()
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def init(self, agent, action_context: ActionContext) -> dict:
        """Called once before the first iteration. Set up state, seed memory."""
        pass

    def start_agent_loop(self, agent, action_context: ActionContext) -> bool:
        """Called at the top of each iteration. Return ``False`` to stop the run."""
        return True

    def process_prompt(self, agent, action_context: ActionContext, prompt: Prompt) -> Prompt:
        """Called with the constructed prompt, before it goes to the LLM."""
        return prompt

    def process_response(self, agent, action_context: ActionContext, response: str) -> str:
        """Called with the LLM's raw response, before it is parsed."""
        return response

    def process_action(self, agent, action_context: ActionContext, action: dict) -> dict:
        """Called with the parsed ``{"tool": ..., "args": ...}``, before execution."""
        return action

    def process_result(
        self,
        agent,
        action_context: ActionContext,
        response: str,
        action_def: Action,
        action: dict,
        result: any,
    ) -> any:
        """Called with the action's result, before it is written to memory."""
        return result

    def process_new_memories(
        self,
        agent,
        action_context: ActionContext,
        memory: Memory,
        response,
        result,
        memories: list[dict],
    ) -> list[dict]:
        """Called with the memory items about to be stored. Add, drop, or rewrite them."""
        return memories

    def end_agent_loop(self, agent, action_context: ActionContext):
        """Called at the bottom of each iteration. Log, flush, measure."""
        pass

    def should_terminate(self, agent, action_context: ActionContext, response: str) -> bool:
        """Called at the end of each iteration. Return ``True`` to stop the run."""
        return False

    def terminate(self, agent, action_context: ActionContext) -> dict:
        """Called once after the loop ends, however it ended. Clean up."""
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
