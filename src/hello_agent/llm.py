"""The one place this template talks to an LLM.

Everything hangs off :func:`generate_response`: it takes a
:class:`~hello_agent.core.prompt.Prompt` and returns a **string**.

That return type is the keystone of the design. When the prompt carries tools and
the model chooses one, the string is a JSON object
``{"tool": name, "args": {...}}``; otherwise it is the model's plain text. Either
way the agent loop receives a string, which means a test can replace this whole
module with a list of canned strings (``tests/fakes.py::FakeLLM``) and the suite
runs green with no API key and no network.
"""

from __future__ import annotations

import json
from typing import Any

from hello_agent import config
from hello_agent.core.prompt import Prompt

__all__ = ["generate_response"]


def _tool_call_to_json(call: Any) -> str:
    """Normalise one provider tool call into ``{"tool": ..., "args": {...}}``."""
    arguments = call.function.arguments
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    return json.dumps({"tool": call.function.name, "args": arguments or {}})


def generate_response(prompt: Prompt, tier: str = "powerful") -> str:
    """Call the LLM via LiteLLM. Returns a tool-call JSON string, or plain text.

    Args:
        prompt: messages, and optionally the tools the model may call.
        tier: ``"powerful"`` for judgement, ``"fast"`` for cheap mechanical work.
            This is MATE's model efficiency: see :func:`hello_agent.config.model_for`.

    Raises:
        hello_agent.config.ConfigError: if no provider can be resolved.
    """
    # Imported here rather than at module scope: litellm is slow to import, and
    # nothing in the test suite should pay for it.
    from litellm import completion

    response = completion(
        model=config.model_for(tier),
        messages=prompt.messages,
        tools=prompt.tools or None,
        max_tokens=config.max_tokens(),
    )
    message = response.choices[0].message

    # Shape handling lives *only* here, so provider drift is a one-file fix.
    tool_calls = getattr(message, "tool_calls", None)
    if prompt.tools and tool_calls:
        return _tool_call_to_json(tool_calls[0])

    return message.content or ""
