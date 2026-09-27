"""The AgentLanguage: translator between GAME components and the model.

Goals, actions and memory are Python objects; an LLM takes messages and returns
text. The AgentLanguage owns both halves of that translation and nothing else:
``construct_prompt`` renders goals, tools and memory into a
:class:`~hello_agent.core.prompt.Prompt`; ``parse_response`` reads the reply back
as ``{"tool": name, "args": {...}}``. Because both halves live here, changing how
an agent talks to a model is a one-line swap at assembly time.

Two implementations ship:

- :class:`AgentFunctionCallingActionLanguage` -- the default. Tool schemas go in
  the API's ``tools`` field and the provider guarantees well-formed arguments.
- :class:`AgentJsonActionLanguage` -- tools described in a system message; the
  model thinks out loud and emits a fenced ``action`` block. You see the
  reasoning, at the cost of parsing free text. Useful for models without tool
  support, and for watching an agent reason.
"""

from __future__ import annotations

import json

from hello_agent.core.actions import Action
from hello_agent.core.goals import Goal
from hello_agent.core.memory import Memory
from hello_agent.core.prompt import Prompt

__all__ = [
    "AgentFunctionCallingActionLanguage",
    "AgentJsonActionLanguage",
    "AgentLanguage",
]

#: memory item ``type`` -> ChatML role. Anything unrecognised becomes ``user``,
#: because tool results are fed back to the model as user turns.
_ROLE_BY_TYPE = {"system": "system", "assistant": "assistant", "user": "user"}


class AgentLanguage:
    """Base class. Subclasses must implement both halves of the translation.

    :meth:`format_goals` and :meth:`format_memory` are concrete because every
    language needs them; override either to change how goals or history render.
    """

    def construct_prompt(
        self, actions: list[Action], environment, goals: list[Goal], memory: Memory
    ) -> Prompt:
        """Render the GAME components into one :class:`Prompt`."""
        raise NotImplementedError("Subclasses must implement this method")

    def parse_response(self, response: str) -> dict:
        """Parse the model's reply into ``{"tool": name, "args": {...}}``."""
        raise NotImplementedError("Subclasses must implement this method")

    def format_goals(self, goals: list[Goal]) -> list[dict]:
        """Render goals as one system message, highest priority first."""
        separator = "\n-------------------\n"
        ordered = sorted(goals, key=lambda goal: goal.priority)
        content = "\n\n".join(
            f"{goal.name}:{separator}{goal.description}{separator}" for goal in ordered
        )
        return [{"role": "system", "content": content}]

    def format_memory(self, memory: Memory) -> list[dict]:
        """Render memory items as ChatML messages.

        Non-string content is JSON-encoded rather than ``str()``-ed, so a dict
        result reaches the model as valid JSON it can parse back.
        """
        messages = []
        for item in memory.get_memories():
            content = item.get("content", None)
            if not isinstance(content, str):
                content = json.dumps(content)
            messages.append(
                {"role": _ROLE_BY_TYPE.get(item.get("type"), "user"), "content": content}
            )
        return messages


class AgentFunctionCallingActionLanguage(AgentLanguage):
    """Tools go in the API's ``tools`` field; the reply is already structured.

    The default, because the provider validates arguments against the schema and
    the loop never has to parse prose.
    """

    def format_actions(self, actions: list[Action]) -> list[dict]:
        """Render actions as OpenAI-style function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    # Providers cap tool descriptions; truncate rather than 400.
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            }
            for action in actions
        ]

    def construct_prompt(
        self, actions: list[Action], environment, goals: list[Goal], memory: Memory
    ) -> Prompt:
        """Goals as a system message, memory as conversation, tools as schemas."""
        messages = self.format_goals(goals) + self.format_memory(memory)
        return Prompt(messages=messages, tools=self.format_actions(actions))

    def parse_response(self, response: str) -> dict:
        """Parse the tool-call JSON that :mod:`hello_agent.llm` produced.

        When the model answers with prose instead of calling a tool -- which it
        does when it thinks it is finished -- that prose becomes a ``terminate``
        message. Nothing is lost and the loop ends cleanly instead of erroring.
        """
        try:
            return json.loads(response)
        except Exception:
            return {"tool": "terminate", "args": {"message": response}}


class AgentJsonActionLanguage(AgentLanguage):
    """Tools described in prose; the model replies with a fenced ``action`` block.

    The model thinks out loud before the block, which puts its reasoning in
    memory -- valuable when debugging why an agent chose what it chose.
    """

    action_format = """
<Stop and think step by step. Insert your thoughts here.>

```action
{
    "tool": "tool_name",
    "args": {...fill in arguments...}
}
```
"""

    def format_actions(self, actions: list[Action]) -> list[dict]:
        """Render the tool catalogue and the required reply format as a system message."""
        action_descriptions = [
            {
                "name": action.name,
                "description": action.description,
                "args": action.parameters,
            }
            for action in actions
        ]
        return [
            {
                "role": "system",
                "content": (
                    f"Available Tools: {json.dumps(action_descriptions, indent=4)}\n"
                    f"\n{self.action_format}\n"
                ),
            }
        ]

    def construct_prompt(
        self, actions: list[Action], environment, goals: list[Goal], memory: Memory
    ) -> Prompt:
        """Goals, then the tool catalogue, then the conversation. No ``tools`` field."""
        messages = (
            self.format_goals(goals) + self.format_actions(actions) + self.format_memory(memory)
        )
        return Prompt(messages=messages, tools=None)

    def parse_response(self, response: str) -> dict:
        """Extract and parse the ```` ```action ```` block.

        Raises:
            ValueError: if no well-formed block is present. There is no sensible
                fallback here -- a missing block means the model ignored the
                required format, and silently terminating would hide that.
        """
        start_marker = "```action"
        end_marker = "```"

        stripped = response.strip()
        start_index = stripped.find(start_marker)
        if start_index == -1:
            raise ValueError("No ```action block found in the response.")

        body_start = start_index + len(start_marker)
        end_index = stripped.rfind(end_marker)
        if end_index <= body_start:
            raise ValueError("Unterminated ```action block in the response.")

        try:
            return json.loads(stripped[body_start:end_index].strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in the ```action block: {e}") from e
