"""The translator: a sub-agent, and the smallest complete agent in the template.

The greeter has an honest reason to delegate. ``extract_visitor_details`` records
a ``language`` field, and a visitor who would rather be welcomed in Japanese
should be. Translating well is a different job from greeting well -- different
instructions, different failure modes, a different voice -- so it gets a
different agent rather than a sixth goal bolted onto the greeter's.

This file is also the worked example for ``docs/EXTENDING.md``: an agent is four
components and a builder. Note how little there is. One goal block carrying a
persona (``persona_pattern.md`` -- the persona lives in G, not in a tool, because
here it describes the whole agent), one tool, the default language, the default
environment. It has no delegation tools of its own and no agent registry, so it
is a leaf: it answers and stops.

Nothing registers this agent by default. ``Features.delegation`` is what builds
it and puts it in an :class:`~hello_agent.core.registry.AgentRegistry` -- see
:mod:`hello_agent.agents` -- and until then a run costs exactly what the lean
default costs.
"""

from __future__ import annotations

from collections.abc import Callable

import hello_agent.tools  # noqa: F401  -- importing registers the tool catalogue
from hello_agent.core.actions import PythonActionRegistry
from hello_agent.core.agent import Agent
from hello_agent.core.agent_language import AgentFunctionCallingActionLanguage
from hello_agent.core.environment import PythonEnvironment
from hello_agent.core.goals import Goal

__all__ = ["TRANSLATOR_AGENT_NAME", "build_translator_agent", "translator_goals"]

#: The name the translator is registered under, and the name the greeter is told
#: to call. Defined here so the caller and the callee cannot drift apart.
TRANSLATOR_AGENT_NAME = "translator_agent"

#: One request in, one answer out. A translator that needs ten turns has
#: misunderstood the task, and a low ceiling makes that fail fast and cheap.
TRANSLATOR_MAX_ITERATIONS = 3

_PERSONA = """\
You are a conference interpreter with twenty-five years at the lectern and in \
the booth. You have rendered condolences, jokes, and technical specifications \
between a dozen languages, and you know that the literal words are the least of \
it: what must survive translation is what the speaker was *doing* -- welcoming, \
apologising, teasing, reassuring.

You translate register, warmth and formality as carefully as you translate \
vocabulary. You use the forms of address and the politeness level a native \
speaker would actually use in that setting. You never add flourishes the \
original did not have, and you never flatten ones it did."""

_METHOD = """\
Translate the text you are given into the requested language, then call \
terminate with the translation as your message.

Rules:
1. Reply with the translation itself, and nothing else. No transliteration, no \
back-translation, no notes on your choices, no quotation marks around it.
2. Preserve names, titles and honorifics as the target language would render \
them for that person.
3. If no target language is named, or you do not know it well enough to be \
trusted, call terminate and say so plainly rather than guessing.
4. Never invent content. If the source text says nothing about a topic, neither \
does your translation."""


def translator_goals() -> list[Goal]:
    """The translator's goals: who it is, then how it works.

    Two goals rather than one paragraph, because they change for different
    reasons -- you would rewrite the persona to change the *voice* and the
    method to change the *output format*.
    """
    return [
        Goal(priority=1, name="persona", description=_PERSONA),
        Goal(priority=2, name="translate", description=_METHOD),
    ]


def build_translator_agent(generate_response: Callable = None) -> Agent:
    """Assemble the translator sub-agent.

    Args:
        generate_response: ``(Prompt, tier="powerful") -> str``. Defaults to the
            real LiteLLM call. Tests pass a scripted fake, which is why this
            argument exists at all -- no network, no key, no recorded cassettes.

    Returns:
        An :class:`~hello_agent.core.agent.Agent` with one tool, ``terminate``.
        Register its ``run`` method in an
        :class:`~hello_agent.core.registry.AgentRegistry` to make it callable by
        another agent.
    """
    if generate_response is None:
        # Imported lazily: llm.py pulls in litellm, and a test that never calls
        # the real model should never pay for that import.
        from hello_agent.llm import generate_response as generate_response_default

        generate_response = generate_response_default

    return Agent(
        goals=translator_goals(),
        agent_language=AgentFunctionCallingActionLanguage(),
        # By name, not by tag: the translator gets `terminate` and nothing else.
        # Giving a sub-agent the smallest toolkit that can do its job is the
        # cheapest reliability measure available.
        action_registry=PythonActionRegistry(tool_names=["terminate"]),
        generate_response=generate_response,
        environment=PythonEnvironment(),
        max_iterations=TRANSLATOR_MAX_ITERATIONS,
    )
