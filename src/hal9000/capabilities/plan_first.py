"""Plan first: make the agent commit to a strategy before it touches a tool.

Without this, an agent picks its next action from memory alone, one step at a
time. That is fine when the task is three tool calls long. When it is fifteen,
the agent starts repeating steps it already finished, or wandering off the task
somewhere around step nine, because nothing in the prompt says where it was
going.

:class:`PlanFirstCapability` spends one LLM call in ``init`` writing a numbered
plan, then puts it in memory as a ``system`` item. From then on the plan is part
of every prompt the loop builds -- the agent's own stated intentions, visible to
it on every turn.

**Cost.** One extra LLM call per run, plus the plan's tokens in every subsequent
prompt. That is the whole price, and it is paid once regardless of how long the
run is -- which makes this by far the cheaper of the two planning capabilities.

**Once, not once per hook.** ``init`` fires once per ``Agent.run``, but the same
capability instance may be reused across runs, and a nested agent may re-init.
A ``first_call`` latch makes re-planning impossible: the plan is written exactly
once in the life of the instance.
"""

from __future__ import annotations

from hal9000.core.action_context import ActionContext
from hal9000.core.capabilities import Capability
from hal9000.tools.planning import create_plan

__all__ = ["PlanFirstCapability"]

#: Prefixed to the generated plan so the model reads it as instruction, not notes.
PLAN_PREAMBLE = "You must follow these instructions carefully to complete the task:\n"


class PlanFirstCapability(Capability):
    """Create a plan before the first iteration and keep it in memory.

    Args:
        plan_memory_type: The memory ``type`` the plan is stored under. The
            default ``"system"`` marks it as framing rather than conversation,
            which means :meth:`~hal9000.core.memory.Memory.copy_without_system_memories`
            strips it before a hand-off -- a sub-agent should get the task, not
            this agent's plan for it. Pass ``"user"`` if you want the plan to
            travel with the conversation.
        track_progress: Kept from ``plan_first_capability.md`` so its published
            example runs verbatim. It is recorded on the instance and nothing
            else: progress tracking is a separate, composable capability, and
            folding it in here would mean two ways to enable the same behaviour.
            Use ``ProgressTrackingCapability`` instead.

    Example:
        >>> Agent(..., capabilities=[PlanFirstCapability()])  # doctest: +SKIP
    """

    def __init__(self, plan_memory_type: str = "system", track_progress: bool = False):
        super().__init__(
            name="Plan First Capability",
            description="The Agent will always create a plan and add it to memory",
        )
        self.plan_memory_type = plan_memory_type
        self.track_progress = track_progress
        #: Latched ``False`` after the first plan; see the module docstring.
        self.first_call = True

    def init(self, agent, action_context: ActionContext) -> None:
        """Write the plan into memory, the first time and only the first time."""
        if not self.first_call:
            return
        self.first_call = False

        memory = action_context.get_memory()
        plan = create_plan(
            action_context=action_context,
            _memory=memory,
            _action_registry=action_context.get_action_registry(),
        )
        memory.add_memory({"type": self.plan_memory_type, "content": PLAN_PREAMBLE + plan})
