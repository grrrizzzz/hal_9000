"""Time awareness: the smallest capability that is genuinely useful.

An LLM has no clock. Ask one to "schedule this for today" and it will answer
from whatever date its training data suggested, confidently. The fix is not a
tool the agent must remember to call -- it is putting the time in front of the
model on *every* turn, whether or not the agent thought to ask.

That is the shape of a capability in one paragraph: a cross-cutting concern that
no single tool owns, injected at a hook, with the loop none the wiser.

Two hooks do the work:

``init``
    seeds memory once, so the agent's own history says what time the run began.
``process_prompt``
    re-reads the clock on every iteration. A long run drifts; a prompt built at
    16:58 and answered at 17:03 should not still claim it is 16:58.

Costs nothing -- no LLM calls, a couple of dozen tokens per prompt.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from hello_agent import config
from hello_agent.core.action_context import ActionContext
from hello_agent.core.capabilities import Capability
from hello_agent.core.prompt import Prompt

__all__ = ["TimeAwareCapability"]

#: ``17:05 Tuesday, September 21, 2026`` -- what a person reads.
HUMAN_FORMAT = "%H:%M %A, %B %d, %Y"
#: ``2026-09-21T17:05:00-0500`` -- what a model can parse and do arithmetic on.
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
#: Used when the configured zone does not exist. A wrong zone beats no clock.
FALLBACK_TIME_ZONE = "UTC"

_MEMORY_TEMPLATE = """\
Right now, it is {human} (ISO: {iso}).
You are in the {zone} timezone.
Please consider the day/time, if relevant, when responding."""


def resolve_time_zone(action_context: ActionContext) -> tuple[str, ZoneInfo]:
    """Work out which timezone this run is in, and never raise doing it.

    Precedence: a ``time_zone`` property on the
    :class:`~hello_agent.core.action_context.ActionContext` (per-run, set by the
    caller), then :func:`hello_agent.config.timezone` (``AGENT_TIMEZONE``,
    defaulting to ``America/Chicago``). An unrecognised name falls back to UTC
    rather than killing the run -- a typo in an environment variable should cost
    you accuracy, not the agent.

    Returns:
        ``(zone_name, ZoneInfo)``.
    """
    name = action_context.get("time_zone") or config.timezone()
    try:
        return name, ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return FALLBACK_TIME_ZONE, ZoneInfo(FALLBACK_TIME_ZONE)


class TimeAwareCapability(Capability):
    """Tell the agent what time it is, once in memory and again in every prompt.

    Takes no arguments: the timezone is a per-run property, not a property of the
    capability, so the same instance behaves correctly for a caller in Osaka and
    a caller in Chicago.

    Example:
        >>> Agent(..., capabilities=[TimeAwareCapability()])  # doctest: +SKIP
    """

    def __init__(self):
        super().__init__(
            name="Time Awareness",
            description="Allows the agent to be aware of time",
        )

    def init(self, agent, action_context: ActionContext) -> None:
        """Seed memory with the time the run started, in both formats."""
        zone_name, zone = resolve_time_zone(action_context)
        now = datetime.now(zone)
        memory = action_context.get_memory()
        if memory is None:  # pragma: no cover - the loop always provides one
            return
        memory.add_memory(
            {
                "type": "system",
                "content": _MEMORY_TEMPLATE.format(
                    human=now.strftime(HUMAN_FORMAT),
                    iso=now.strftime(ISO_FORMAT),
                    zone=zone_name,
                ),
            }
        )

    def process_prompt(self, agent, action_context: ActionContext, prompt: Prompt) -> Prompt:
        """Prefix the system message with the time *now*, every single iteration.

        A fresh :class:`~hello_agent.core.prompt.Prompt` is returned rather than
        the argument mutated: the next capability in the chain, and any test
        holding the original, should see exactly what they were given.
        """
        zone_name, zone = resolve_time_zone(action_context)
        header = f"Current time: {datetime.now(zone).strftime(HUMAN_FORMAT)} ({zone_name})\n\n"

        messages = list(prompt.messages)
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": header + messages[0]["content"]}
        else:
            messages.insert(0, {"role": "system", "content": header})

        return Prompt(messages=messages, tools=prompt.tools, metadata=prompt.metadata)
