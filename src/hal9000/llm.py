"""The one place HAL talks to an LLM.

Everything hangs off :func:`generate_response`: it takes a
:class:`~hal9000.core.prompt.Prompt` and returns a **string**. When the prompt
carries tools and the model chooses one, the string is
``{"tool": name, "args": {...}}``; otherwise it is the model's plain text. A test
can therefore replace this whole module with a list of canned strings.

Two additions over the template:

- **Usage reporting.** ``on_usage`` receives token counts after every call
  (prompt, cached, completion, reasoning, plus the model). The cost ledger is
  built on it, and the return type stays a string.
- **One tool call per turn.** ``parallel_tool_calls=False`` is sent whenever
  tools are offered, because the loop only ever executes the first call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from hal9000 import config
from hal9000.core.prompt import Prompt

__all__ = ["Usage", "generate_response"]


class Usage(dict):
    """Token counts for one call.

    Keys: ``model``, ``prompt``, ``cached``, ``completion``, ``reasoning``.
    """


def _tool_call_to_json(call: Any) -> str:
    arguments = call.function.arguments
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    return json.dumps({"tool": call.function.name, "args": arguments or {}})


def _get(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_usage(response: Any, model: str) -> Usage:
    """Normalise a provider response's usage block into a :class:`Usage`."""
    usage = _get(response, "usage", None)
    prompt_details = _get(usage, "prompt_tokens_details", None)
    completion_details = _get(usage, "completion_tokens_details", None)
    return Usage(
        model=model,
        prompt=int(_get(usage, "prompt_tokens", 0) or 0),
        cached=int(_get(prompt_details, "cached_tokens", 0) or 0),
        completion=int(_get(usage, "completion_tokens", 0) or 0),
        reasoning=int(_get(completion_details, "reasoning_tokens", 0) or 0),
    )


def generate_response(
    prompt: Prompt,
    tier: str = "powerful",
    *,
    on_usage: Callable[[Usage], None] | None = None,
) -> str:
    """Call the LLM via LiteLLM. Returns a tool-call JSON string, or plain text."""
    import litellm
    from litellm import completion

    litellm.drop_params = True  # tolerate params a given model does not accept

    model = config.model_for(tier)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": prompt.messages,
        "tools": prompt.tools or None,
        "max_tokens": config.max_tokens(),
    }
    if prompt.tools:
        kwargs["parallel_tool_calls"] = False
    if model.startswith("openai/"):
        kwargs["reasoning_effort"] = config.reasoning_effort(tier)
        cache_key = prompt.metadata.get("cache_key")
        if cache_key:
            kwargs["extra_body"] = {"prompt_cache_key": cache_key}

    response = completion(**kwargs)
    if on_usage is not None:
        on_usage(extract_usage(response, model))
    message = response.choices[0].message

    tool_calls = getattr(message, "tool_calls", None)
    if prompt.tools and tool_calls:
        return _tool_call_to_json(tool_calls[0])
    return message.content or ""
