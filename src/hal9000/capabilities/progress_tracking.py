"""Progress tracking: make the agent stop and look at what it just did.

A plan written before the first action is a guess. This capability adds the
other half of the human loop -- plan, act, *evaluate*, adjust -- by asking the
model, at the end of an iteration, what actually happened and what should come
next. The report lands in memory, so the next prompt carries a fresh assessment
rather than only the original guess.

End of the iteration, not the start, and deliberately: by then memory holds the
result of the action just taken, so the assessment is about real output instead
of intentions.

**Cost, stated plainly.** ``progress_tracking_capability.md`` is explicit that
this "adds overhead and slows down our agent, since it will add one prompt per
agent loop". At the default ``track_frequency=1`` a run that made 6 LLM calls
now makes 12, each report accumulates in memory, and every later prompt carries
all of them. This is the most expensive capability in the template by a wide
margin. ``track_frequency=3`` buys most of the benefit at a third of the price;
for the default greeter -- three tool calls, no room to drift -- the honest
answer is not to enable it at all.
"""

from __future__ import annotations

from hal9000.core.action_context import ActionContext
from hal9000.core.capabilities import Capability
from hal9000.tools.planning import track_progress

__all__ = ["ProgressTrackingCapability"]


class ProgressTrackingCapability(Capability):
    """Generate a progress report every *n*-th iteration and store it in memory.

    Args:
        memory_type: The memory ``type`` reports are stored under. ``"system"``
            keeps them out of a hand-off to another agent, which is right: they
            are this agent's self-assessment, not shared context.
        track_frequency: Report every *n*-th iteration. ``1`` means every
            iteration (one extra LLM call per loop); ``2`` means every second,
            so a four-iteration run produces two reports. Must be at least 1.

    Raises:
        ValueError: If ``track_frequency`` is less than 1.

    Example:
        >>> ProgressTrackingCapability(track_frequency=2)  # doctest: +SKIP
    """

    def __init__(self, memory_type: str = "system", track_frequency: int = 1):
        super().__init__(
            name="Progress Tracking",
            description="Tracks progress and enables reflection after actions",
        )
        if track_frequency < 1:
            raise ValueError(
                f"track_frequency must be at least 1, got {track_frequency}. "
                "Use 1 to report after every iteration, or a larger number to "
                "report less often; there is no value that disables reporting "
                "-- leave the capability out instead."
            )
        self.memory_type = memory_type
        self.track_frequency = track_frequency
        #: Iterations completed by this instance, counted across runs.
        self.iteration_count = 0

    def end_agent_loop(self, agent, action_context: ActionContext) -> None:
        """Count the iteration and, when it is due, spend a call assessing it."""
        self.iteration_count += 1
        if self.iteration_count % self.track_frequency != 0:
            return

        memory = action_context.get_memory()
        report = track_progress(
            action_context=action_context,
            _memory=memory,
            _action_registry=action_context.get_action_registry(),
        )
        memory.add_memory(
            {
                "type": self.memory_type,
                "content": f"Progress Report (Iteration {self.iteration_count}):\n{report}",
            }
        )
