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

The knowledge base adds two calls with the same usage sink (plan §13):

- :func:`stream_response` (KF1) yields text deltas, then one final
  :class:`Usage`, for streamed answers.
- :func:`embed` (KF2) embeds many texts in token-aware batches.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from hal9000 import config
from hal9000.core.prompt import Prompt

__all__ = ["Embeddings", "Usage", "embed", "generate_response", "stream_response"]


class Usage(dict):
    """Token counts for one call.

    Keys: ``model``, ``prompt``, ``cached``, ``completion``, ``reasoning``; and
    ``estimated=True`` when the provider reported none and the counts come from
    the local tokenizer instead.
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


def _completion_kwargs(prompt: Prompt, tier: str) -> tuple[str, dict[str, Any]]:
    model = config.model_for(tier)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": prompt.messages,
        "max_tokens": config.max_tokens(tier),
    }
    if model.startswith("openai/"):
        kwargs["reasoning_effort"] = config.reasoning_effort(tier)
        cache_key = prompt.metadata.get("cache_key")
        if cache_key:
            kwargs["extra_body"] = {"prompt_cache_key": cache_key}
    return model, kwargs


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
        "max_tokens": config.max_tokens(tier),
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


# ----------------------------------------------------------------------------
# KF1: streaming
# ----------------------------------------------------------------------------
def _messages_text(messages: Sequence[dict]) -> str:
    """All text in chat messages (string content or ``{"type": "text"}`` parts)."""
    texts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "\n".join(texts)


def stream_response(
    prompt: Prompt,
    tier: str = "fast",
    *,
    on_usage: Callable[[Usage], None] | None = None,
    count_tokens: Callable[[str], int] | None = None,
    **kw: Any,
) -> Iterator[str | Usage]:
    """Stream a plain-text completion: yields ``str`` deltas in order, then one :class:`Usage`.

    Usage is requested with ``stream_options={"include_usage": True}``; the
    provider sends it on the final chunk. ``on_usage`` receives it exactly once,
    before it is yielded. When the provider sends none, prompt and completion
    tokens are counted locally (``count_tokens``, default the model's tiktoken
    encoding) and the Usage carries ``estimated=True``, so cost caps still bite.
    Tools are not offered. ``kw`` overrides the request (``max_tokens``...).
    """
    import litellm
    from litellm import completion

    litellm.drop_params = True
    model, kwargs = _completion_kwargs(prompt, tier)
    kwargs.update(kw)
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}

    usage: Usage | None = None
    parts: list[str] = []
    for chunk in completion(**kwargs):
        if _get(chunk, "usage", None):
            usage = extract_usage(chunk, model)
        choices = _get(chunk, "choices", None) or []
        if not choices:
            continue
        delta = _get(choices[0], "delta", None)
        text = _get(delta, "content", None)
        if text:
            parts.append(text)
            yield text
    if usage is None:
        counter = count_tokens or _default_token_counter(model)
        usage = Usage(
            model=model,
            prompt=counter(_messages_text(prompt.messages)),
            cached=0,
            completion=counter("".join(parts)),
            reasoning=0,
            estimated=True,
        )
    if on_usage is not None:
        on_usage(usage)
    yield usage


# ----------------------------------------------------------------------------
# KF2: embeddings
# ----------------------------------------------------------------------------
#: OpenAI's embedding endpoint takes at most 2,048 inputs and 300,000 tokens per
#: request; 250,000 leaves headroom for tokenizer differences.
EMBED_MAX_INPUTS = 2048
EMBED_MAX_TOKENS = 250_000


@dataclass
class Embeddings:
    """Vectors in input order, and the summed :class:`Usage` of every request."""

    vectors: list[list[float]]
    usage: Usage
    requests: int = 0


def _default_token_counter(model: str) -> Callable[[str], int]:
    from hal9000.knowledge.tokens import count_tokens, encoding_for_model

    name = encoding_for_model(model)
    return lambda text: count_tokens(text, name)


def embedding_batches(
    token_counts: Sequence[int],
    max_inputs: int = EMBED_MAX_INPUTS,
    max_tokens: int = EMBED_MAX_TOKENS,
) -> list[range]:
    """Split inputs into consecutive index ranges within both per-request limits.

    An input larger than ``max_tokens`` on its own gets a request of its own
    (the provider will reject it, which is the honest outcome).
    """
    batches: list[range] = []
    start = 0
    tokens = 0
    for i, count in enumerate(token_counts):
        full = (i - start) >= max_inputs or (i > start and tokens + count > max_tokens)
        if full:
            batches.append(range(start, i))
            start, tokens = i, 0
        tokens += count
    if start < len(token_counts):
        batches.append(range(start, len(token_counts)))
    return batches


def embed(
    texts: Sequence[str],
    model: str,
    dimensions: int | None = None,
    *,
    on_usage: Callable[[Usage], None] | None = None,
    count_tokens: Callable[[str], int] | None = None,
    before_request: Callable[[], None] | None = None,
) -> Embeddings:
    """Embed ``texts`` with ``litellm.embedding`` in token-aware batches.

    ``on_usage`` is called once per request, so every paid request reaches the
    ledger even if a later one fails; the returned usage is the sum. When a
    response carries no usage, the locally counted tokens stand in (marked
    ``estimated``), so cost caps still bite. ``before_request`` runs before
    every request; the meter uses it to re-check the cap between batches.
    """
    import litellm
    from litellm import embedding

    litellm.drop_params = True
    counter = count_tokens or _default_token_counter(model)
    counts = [counter(t) for t in texts]
    vectors: list[list[float]] = []
    total = Usage(model=model, prompt=0, cached=0, completion=0, reasoning=0)
    requests = 0
    for batch in embedding_batches(counts):
        if before_request is not None:
            before_request()
        kwargs: dict[str, Any] = {"model": model, "input": [texts[i] for i in batch]}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = embedding(**kwargs)
        requests += 1
        data = sorted(_get(response, "data", []) or [], key=lambda d: _get(d, "index", 0))
        if len(data) != len(batch):
            raise RuntimeError(
                f"embedding returned {len(data)} vectors for {len(batch)} inputs ({model})"
            )
        vectors.extend(list(_get(d, "embedding", [])) for d in data)
        prompt_tokens = int(_get(_get(response, "usage", None), "prompt_tokens", 0) or 0)
        usage = Usage(
            model=model,
            prompt=prompt_tokens or sum(counts[i] for i in batch),
            cached=0,
            completion=0,
            reasoning=0,
        )
        if not prompt_tokens:
            usage["estimated"] = total["estimated"] = True
        total["prompt"] += usage["prompt"]
        if on_usage is not None:
            on_usage(usage)
    return Embeddings(vectors=vectors, usage=total, requests=requests)
