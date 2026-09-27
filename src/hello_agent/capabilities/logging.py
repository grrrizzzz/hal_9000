"""Structured logging: watch the loop without putting ``print`` in the loop.

When an agent misbehaves the question is almost always "which step went wrong?"
-- did the prompt not contain what you assumed, did the model answer something
else, did the parse pick a different tool, did the tool return an error the
model then ignored? Every one of those values already passes through a hook, so
answering it needs no change to :mod:`hello_agent.core.agent` at all, and
turning the answer off again costs a list entry rather than a pass through the
loop deleting ``print`` calls.

**This capability mutates nothing.** Every ``process_*`` hook returns the object
it was handed, unchanged and identical, and the test suite asserts that with
``is``. A logger that quietly reformatted a prompt would be the worst possible
bug to own: you would be debugging the observer.

**Cost.** No LLM calls. One line per hook per iteration, truncated to
:data:`MAX_LOGGED_CHARS`. Output goes to the standard library logger
``hello_agent.capabilities.logging``, so it is silent until the application
configures logging -- pass ``sink=print`` (or any ``(str) -> None``) to see it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import Action
from hello_agent.core.capabilities import Capability
from hello_agent.core.memory import Memory
from hello_agent.core.prompt import Prompt

__all__ = ["LoggingCapability", "MAX_LOGGED_CHARS"]

#: Prompts and responses are truncated to this many characters in the log.
MAX_LOGGED_CHARS = 300

_LOGGER = logging.getLogger(__name__)


def _abbreviate(value: object) -> str:
    """Render ``value`` on one line, short enough to read in a terminal."""
    text = " ".join(str(value).split())
    if len(text) <= MAX_LOGGED_CHARS:
        return text
    return text[:MAX_LOGGED_CHARS] + f"...(+{len(text) - MAX_LOGGED_CHARS} chars)"


class LoggingCapability(Capability):
    """Emit one structured line per loop event. Changes nothing else.

    Lines are ``key=value`` pairs led by the event name, e.g.::

        process_response iteration=2 chars=48 response="{"tool": "terminate", ...}"

    ``process_prompt`` logs the *first* message, the system message every
    capability injects into, rather than the last -- which is the previous
    iteration's tool result and was already logged by ``process_result``.

    Args:
        log_level: Level name used when writing through the standard library
            logger -- ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, and so on.
            Ignored when ``sink`` is given.
        sink: Where to write. ``None`` (the default) uses the
            ``hello_agent.capabilities.logging`` logger, which stays silent
            until the application configures handlers. Pass ``print`` to see the
            lines immediately, or ``a_list.append`` to capture them.

    Example:
        >>> lines = []
        >>> Agent(..., capabilities=[LoggingCapability(sink=lines.append)])  # doctest: +SKIP

    Put this **last** in the capability list. The ``process_*`` hooks compose
    left to right, so a logger at the end records the values the loop actually
    used, after every other capability has had its say.
    """

    def __init__(self, log_level: str = "INFO", sink: Callable[[str], None] | None = None):
        super().__init__(
            name="Logging",
            description="Writes a structured log line for each step of the agent loop",
        )
        self.log_level = log_level
        #: ``log_level`` as an integer; an unknown name degrades to INFO.
        self.level = getattr(logging, log_level.upper(), logging.INFO)
        self.sink = sink
        #: Iterations observed by this instance, counted across runs.
        self.iteration_count = 0
        #: Every line this capability has emitted, for tests and post-mortems.
        self.lines: list[str] = []

    # ------------------------------------------------------------------ output
    def log(self, event: str, **fields: object) -> str:
        """Format and emit one event line. Returns the line, for convenience."""
        rendered = " ".join(f"{key}={value!r}" for key, value in fields.items())
        line = f"{event} {rendered}".strip()
        self.lines.append(line)
        if self.sink is not None:
            self.sink(line)
        else:
            _LOGGER.log(self.level, line)
        return line

    # ------------------------------------------------------------------- hooks
    def init(self, agent, action_context: ActionContext) -> None:
        """Record the start of a run, with the shape of the agent running it."""
        self.iteration_count = 0
        self.log(
            "init",
            context=action_context.context_id,
            goals=[goal.name for goal in agent.goals],
            tools=len(agent.actions.get_actions()),
            capabilities=[capability.name for capability in agent.capabilities],
        )

    def start_agent_loop(self, agent, action_context: ActionContext) -> bool:
        """Count the iteration and announce it. Never vetoes."""
        self.iteration_count += 1
        self.log("start_agent_loop", iteration=self.iteration_count)
        return True

    def process_prompt(self, agent, action_context: ActionContext, prompt: Prompt) -> Prompt:
        """Log the prompt about to be sent. Returns the identical Prompt."""
        self.log(
            "process_prompt",
            iteration=self.iteration_count,
            messages=len(prompt.messages),
            tools=len(prompt.tools or []),
            first_message=_abbreviate(prompt.messages[0]["content"] if prompt.messages else ""),
        )
        return prompt

    def process_response(self, agent, action_context: ActionContext, response: str) -> str:
        """Log the raw model output. Returns the identical string."""
        self.log(
            "process_response",
            iteration=self.iteration_count,
            chars=len(response),
            response=_abbreviate(response),
        )
        return response

    def process_action(self, agent, action_context: ActionContext, action: dict) -> dict:
        """Log the parsed tool call. Returns the identical dict, unmodified."""
        self.log(
            "process_action",
            iteration=self.iteration_count,
            tool=action.get("tool"),
            args=_abbreviate(action.get("args", {})),
        )
        return action

    def process_result(
        self,
        agent,
        action_context: ActionContext,
        response: str,
        action_def: Action | None,
        action: dict,
        result: object,
    ) -> object:
        """Log the tool's outcome. Returns the identical result object."""
        executed = result.get("tool_executed") if isinstance(result, dict) else None
        self.log(
            "process_result",
            iteration=self.iteration_count,
            tool=action_def.name if action_def is not None else action.get("tool"),
            terminal=bool(action_def is not None and action_def.terminal),
            tool_executed=executed,
            result=_abbreviate(result),
        )
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
        """Log what is about to be remembered. Returns the identical list."""
        self.log(
            "process_new_memories",
            iteration=self.iteration_count,
            new=len(memories),
            total=len(memory) + len(memories),
        )
        return memories

    def end_agent_loop(self, agent, action_context: ActionContext) -> None:
        """Close out the iteration."""
        self.log("end_agent_loop", iteration=self.iteration_count)

    def terminate(self, agent, action_context: ActionContext) -> None:
        """Record the end of the run and how many iterations it took."""
        memory = action_context.get_memory()
        self.log(
            "terminate",
            context=action_context.context_id,
            iterations=self.iteration_count,
            memories=len(memory) if memory is not None else 0,
        )
