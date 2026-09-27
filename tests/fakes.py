"""Deterministic stand-ins for the LLM, so the suite never needs a key.

Because :func:`hello_agent.llm.generate_response` is just
``(Prompt, tier) -> str``, faking it needs no mocking library and no HTTP
interception: a list of strings is a complete LLM.
"""

from __future__ import annotations

import json

from hello_agent.core.prompt import Prompt

__all__ = ["FakeLLM", "tool_call"]


def tool_call(name: str, /, **args: object) -> str:
    """Build the string a function-calling LLM would return for one tool call.

    ``name`` is **positional-only** (ruling R18). Without the ``/`` this helper
    could not script a call to any tool that itself takes a ``name`` argument --
    and ``sign_guestbook(name, greeting, note="")`` is the template's central
    write, so that is not a corner case:

        >>> tool_call("sign_guestbook", name="Ada", greeting="Welcome!")
        '{"tool": "sign_guestbook", "args": {"name": "Ada", "greeting": "Welcome!"}}'

    Every call site already passes the tool name positionally, so the marker is
    purely additive: it widens what ``**args`` can carry and breaks nothing.
    """
    return json.dumps({"tool": name, "args": args})


class FakeLLM:
    """Deterministic scripted ``generate_response``. Records every Prompt it saw.

    Example:
        >>> llm = FakeLLM([tool_call("terminate", message="done")])
        >>> llm(Prompt(messages=[{"role": "user", "content": "hi"}]))
        '{"tool": "terminate", "args": {"message": "done"}}'
        >>> len(llm.calls)
        1
    """

    def __init__(self, responses: list[str]):
        self.responses: list[str] = list(responses)
        #: every Prompt this fake was called with, in order
        self.calls: list[Prompt] = []
        #: the tier requested on each call, positionally aligned with ``calls``
        self.tiers: list[str] = []

    def __call__(self, prompt: Prompt, tier: str = "powerful") -> str:
        self.calls.append(prompt)
        self.tiers.append(tier)
        if not self.responses:
            raise AssertionError(
                f"FakeLLM ran out of scripted responses on call {len(self.calls)}. "
                "Add another response, or check why the agent kept going."
            )
        return self.responses.pop(0)

    @property
    def last_prompt(self) -> Prompt:
        """The most recent Prompt. Raises IndexError if the fake was never called."""
        return self.calls[-1]
