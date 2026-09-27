"""The agent loop: prompt, decide, execute, remember, repeat.

Everything that varies between agents lives in the GAME components handed to the
constructor. What is left here -- the order of operations, the limits, the
capability hooks -- is identical for every agent, which is why this is the file
you should almost never need to edit. One iteration, in full::

    construct_prompt   goals + tools + memory  ->  Prompt
    generate_response  Prompt                  ->  str
    parse_response     str                     ->  {"tool": ..., "args": ...}
    execute_action     Action + args           ->  result dict
    update_memory      response + result       ->  two new memory items

The loop stops on a terminal action, on ``max_iterations``, on
``max_duration_seconds``, or when a capability says so -- never because a tool
raised. The environment turns that into a result the agent reads and reacts to.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from functools import reduce

from hello_agent.core.action_context import ActionContext, default_properties
from hello_agent.core.actions import Action, ActionRegistry
from hello_agent.core.agent_language import AgentLanguage
from hello_agent.core.capabilities import Capability
from hello_agent.core.environment import Environment
from hello_agent.core.goals import Goal
from hello_agent.core.memory import Memory
from hello_agent.core.prompt import Prompt

__all__ = ["Agent"]


class Agent:
    """A GAME agent: goals, actions, memory and environment, plus a loop."""

    def __init__(
        self,
        goals: list[Goal],
        agent_language: AgentLanguage,
        action_registry: ActionRegistry,
        generate_response: Callable[[Prompt], str],
        environment: Environment,
        capabilities: list[Capability] = None,
        max_iterations: int = 10,
        max_duration_seconds: int = 180,
    ):
        """Assemble an agent from its GAME components.

        ``generate_response`` is ``(Prompt, tier="powerful") -> str``, injected so
        tests can pass a scripted fake. ``capabilities`` is empty by default.
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment
        self.capabilities = capabilities or []
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds

    # ---------------------------------------------------------------- seams
    def construct_prompt(self, action_context: ActionContext, memory: Memory) -> Prompt:
        """Render goals, tools and memory into a prompt."""
        return self.agent_language.construct_prompt(
            actions=self.actions.get_actions(),
            environment=self.environment,
            goals=self.goals,
            memory=memory,
        )

    def prompt_llm_for_action(self, action_context: ActionContext, prompt: Prompt) -> str:
        """Ask the model what to do next. Always the powerful tier."""
        return self.generate_response(prompt)

    def set_current_task(self, memory: Memory, task: str) -> None:
        """Record the user's request as the first memory of the run."""
        memory.add_memory({"type": "user", "content": task})

    def execute_action(
        self, action_context: ActionContext, action_def: Action | None, invocation: dict
    ) -> dict:
        """Run the chosen action, or explain why it could not be run.

        A hallucinated tool name is feedback, not a crash.
        """
        if action_def is None:
            available = ", ".join(a.name for a in self.actions.get_actions())
            return {
                "tool_executed": False,
                "error": (
                    f"Unknown tool: {invocation.get('tool')!r}. "
                    f"Choose one of the available tools: {available}."
                ),
                "traceback": None,
            }
        return self.environment.execute_action(
            self, action_context, action_def, invocation.get("args", {}) or {}
        )

    def update_memory(
        self, action_context: ActionContext, memory: Memory, response: str, result: dict
    ) -> None:
        """Store the decision and its outcome, after capabilities have had a say."""
        memories = [
            {"type": "assistant", "content": response},
            {"type": "user", "content": json.dumps(result, default=str)},
        ]
        memories = self._compose(
            "process_new_memories", action_context, memories, memory, response, result
        )
        for item in memories:
            memory.add_memory(item)

    def should_terminate(
        self, action_context: ActionContext, response: str, action_def: Action | None
    ) -> bool:
        """Stop if any capability says so, or if the action just run was terminal.

        Votes are collected into a list first, so no capability is short-circuited.
        """
        votes = [c.should_terminate(self, action_context, response) for c in self.capabilities]
        return any(votes) or bool(action_def is not None and action_def.terminal)

    # -------------------------------------------------------------- the loop
    def run(self, user_input: str, memory=None, action_context_props: dict = None) -> Memory:
        """Run the agent to completion and return its memory.

        ``action_context_props`` adds or overrides resources tools can ask for.
        """
        memory = memory or Memory()
        action_context = self._build_action_context(memory, action_context_props)
        self.set_current_task(memory, user_input)

        for capability in self.capabilities:
            capability.init(self, action_context)

        deadline = time.time() + self.max_duration_seconds
        for _ in range(self.max_iterations):
            if time.time() >= deadline:
                break
            if not all([c.start_agent_loop(self, action_context) for c in self.capabilities]):
                break
            if self._run_iteration(action_context, memory):
                break

        for capability in self.capabilities:
            capability.terminate(self, action_context)
        return memory

    def _run_iteration(self, action_context: ActionContext, memory: Memory) -> bool:
        """One pass of the loop. Returns ``True`` when the agent should stop."""
        prompt = self._compose(
            "process_prompt", action_context, self.construct_prompt(action_context, memory)
        )
        response = self._compose(
            "process_response", action_context, self.prompt_llm_for_action(action_context, prompt)
        )
        # Parsing and lookup are separate steps because process_action sits
        # between them: a capability may redirect the call to a different tool.
        invocation = self._compose(
            "process_action", action_context, self.agent_language.parse_response(response)
        )
        action_def = self.actions.get_action(invocation.get("tool"))
        result = self._compose(
            "process_result",
            action_context,
            self.execute_action(action_context, action_def, invocation),
            response,
            action_def,
            invocation,
        )
        self.update_memory(action_context, memory, response, result)

        for capability in self.capabilities:
            capability.end_agent_loop(self, action_context)
        return self.should_terminate(action_context, response, action_def)

    # ------------------------------------------------------------- plumbing
    def _compose(self, hook: str, action_context: ActionContext, value, *extra):
        """Thread ``value`` through every capability's ``hook``, left to right.

        Every ``process_*`` hook takes the value it transforms last, so one
        helper covers all five.
        """
        return reduce(
            lambda carried, capability: getattr(capability, hook)(
                self, action_context, *extra, carried
            ),
            self.capabilities,
            value,
        )

    def _build_action_context(self, memory: Memory, props: dict | None) -> ActionContext:
        """Assemble the resources tools may ask for. Caller-supplied props win."""
        properties = default_properties(self.generate_response, memory, self.actions)
        properties.update(props or {})
        return ActionContext(properties)
