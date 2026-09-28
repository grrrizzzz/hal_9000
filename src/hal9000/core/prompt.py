"""The Prompt: everything the LLM is told on one turn.

A prompt is ChatML messages (``system`` / ``user`` / ``assistant``) plus, when
the agent language is function calling, the JSON-Schema tool definitions the
model may choose from. Keeping it a plain dataclass rather than a pile of
keyword arguments is what lets tests record and assert on prompts, and what lets
``AgentLanguage`` own prompt *construction* while ``llm.py`` owns transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prompt:
    """One turn's worth of input for the model.

    Attributes:
        messages: ChatML messages, e.g. ``{"role": "system", "content": "..."}``.
        tools: JSON-Schema tool definitions, or ``None`` for a plain text call.
        metadata: Free-form notes for capabilities and logging. Never sent to the
            model.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict] | None = None
    metadata: dict = field(default_factory=dict)
