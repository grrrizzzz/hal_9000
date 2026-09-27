"""The greeter: the agent this template is for, and the only place flags matter.

Everything else in this repository is a component. This file is the assembly --
the one place that turns a :class:`~hello_agent.config.Features` record into a
concrete GAME agent, and therefore the one file you edit to change what the
default run costs.

**The lean default is the headline.** ``build_greeter_agent()`` with no arguments
gives you five tools, zero capabilities, a plain
:class:`~hello_agent.core.environment.PythonEnvironment` and the function-calling
language. That agent extracts, composes, signs and stops. Every flag below is
off, and every one of them makes the agent bigger, slower or more expensive --
which is the point of making them flags rather than defaults.

**What each flag does here**, and nothing is hidden elsewhere:

======================  ==================================================
``time_aware``          appends :class:`TimeAwareCapability`
``plan_first``          appends :class:`PlanFirstCapability`
``progress_tracking``   appends :class:`ProgressTrackingCapability`
``verbose_logging``     appends :class:`LoggingCapability`, always last
``delegation``          adds ``call_agent`` + the translator goal + registry
``staged_review``       adds the ``safety`` tag *and* swaps the environment
``dynamic_experts``     adds the ``prompts`` tag (four self-prompting tools)
======================  ==================================================

``staged_review`` governing both the tag and the environment is ruling R12:
staging a write and being able to reverse one are a single concern, and
``Features`` is a frozen contract with no eighth field to spend on splitting
them. ``delegation`` handing over ``call_agent`` *alone*, rather than all four
delegation tools, is ruling R15: the greeter's one honest sub-task is "translate
this", which is message passing. The other three remain registered and tagged
for anyone who wants them -- ``PythonActionRegistry(tags=["delegation"])``.

Capability order is load-bearing. ``process_*`` hooks compose left to right, so
:class:`LoggingCapability` goes last and therefore records the prompt the model
actually receives, after ``TimeAwareCapability`` has injected the clock into it.
All four are stateful across runs, so they are constructed fresh here rather
than shared as module-level singletons.
"""

from __future__ import annotations

from collections.abc import Callable

import hello_agent.tools  # noqa: F401  -- importing registers the tool catalogue
from hello_agent import config
from hello_agent.agents.translator import TRANSLATOR_AGENT_NAME, build_translator_agent
from hello_agent.capabilities import (
    LoggingCapability,
    PlanFirstCapability,
    ProgressTrackingCapability,
    TimeAwareCapability,
)
from hello_agent.config import Features
from hello_agent.core.actions import ActionRegistry, PythonActionRegistry
from hello_agent.core.agent import Agent
from hello_agent.core.agent_language import AgentFunctionCallingActionLanguage
from hello_agent.core.capabilities import Capability
from hello_agent.core.environment import Environment, PythonEnvironment
from hello_agent.core.goals import Goal
from hello_agent.core.registry import AgentRegistry
from hello_agent.safety import StagedActionEnvironment
from hello_agent.storage.guestbook import GuestbookStore

__all__ = [
    "DEFAULT_TAGS",
    "DELEGATION_TOOL",
    "build_greeter_action_registry",
    "build_greeter_agent",
    "build_greeter_capabilities",
    "greeter_context_props",
    "greeter_goals",
]

#: The lean default toolkit: five tools, and not one of them optional.
DEFAULT_TAGS: tuple[str, ...] = ("greeting", "guestbook", "system")

#: The single delegation tool the greeter is given -- see ruling R15.
DELEGATION_TOOL = "call_agent"

_MISSION = """\
You are the greeter at the front desk of this building. Someone has just arrived, \
and a colleague has told you about them in their own words. Welcome that visitor \
properly, and record their visit in the guestbook so that whoever reads the book \
next knows who came and what was said to them.

A visit is handled when both of those are true: the visitor has a greeting written \
for them, and the guestbook holds an entry recording it. Neither one alone is \
enough, and a greeting nobody wrote down is a visit nobody can remember."""

_METHOD = """\
Use your tools in this order, one call per turn, feeding the result of each call \
into the next:

1. extract_visitor_details -- pass it what you were told, verbatim. Do not \
summarise or rewrite it first; the extractor reads the original words better \
than it reads a paraphrase of them.
2. compose_greeting -- pass it the details object the extractor returned, whole \
and unchanged. What comes back is the greeting, word for word. Do not edit it.
3. sign_guestbook -- record the visit. `name` is the visitor's name from the \
extracted details, `greeting` is the composed greeting in full, and `note` is one \
short line of context taken from the details: their role, where they travelled \
from, or what brought them here.
4. terminate -- deliver the result, exactly as your final goal describes.

Rules you do not break:
- Never skip a step and never reorder them. compose_greeting needs the extracted \
details; sign_guestbook needs the composed greeting.
- Call each tool once. If a tool returns an error, read it -- the message tells \
you what to do next. Fix what it names and try again; if it tells you to stop, stop.
- Invent nothing. Everything you record must come from what you were told or from \
what a tool returned.
- You do not need to read the guestbook before signing it."""

_TRANSLATE = """\
The extracted details may carry a `language` field naming the language the visitor \
would rather be greeted in. If that field is present and names a language other \
than English, then after compose_greeting and before sign_guestbook, call \
call_agent with agent_name="{translator}" and a `task` that gives the translator \
both the target language and the complete English greeting to translate.

The translation is the greeting from that point on: it is what you pass to \
sign_guestbook and what you put in your terminate message. If the call comes back \
with "success": false, keep the English greeting, carry on, and say plainly at the \
end that it could not be translated.

If no language is named, or the language named is English, do not call call_agent \
at all."""

_DELIVERY = """\
Once the guestbook is signed, call terminate exactly once and put the whole of your \
answer in its `message`. Nothing you say outside terminate reaches anyone.

The message has two parts, separated by a blank line:

1. The greeting itself, word for word as it was composed for this visitor.
2. One line recording the visit, in this form:

Signed the guestbook as entry #N.

where N is the `entry_id` that sign_guestbook returned.

No preamble, no list of the tools you used, no offer of further help. The visitor \
reads this; the plumbing is not their business."""


def greeter_goals(features: Features | None = None) -> list[Goal]:
    """The greeter's goals: what it is for, how it works, and how it signs off.

    Three goals by default, four with ``delegation`` on. They are separate
    because they change for different reasons -- you would rewrite the method to
    add a tool and the delivery goal to change the output format -- and because
    the delegation instruction must not be in the prompt when ``call_agent`` is
    not in the toolkit. An agent told to call a tool it does not have will try.
    """
    features = features if features is not None else Features()
    goals = [
        Goal(priority=1, name="welcome_and_record", description=_MISSION),
        Goal(priority=2, name="method", description=_METHOD),
    ]
    if features.delegation:
        goals.append(
            Goal(
                priority=3,
                name="greet_in_their_language",
                description=_TRANSLATE.format(translator=TRANSLATOR_AGENT_NAME),
            )
        )
    goals.append(Goal(priority=4, name="deliver", description=_DELIVERY))
    return goals


def build_greeter_action_registry(features: Features) -> ActionRegistry:
    """The greeter's toolkit: five tools, plus whatever the flags add.

    ``dynamic_experts`` and ``staged_review`` add a whole tag. ``delegation``
    adds one tool *by name* -- ruling R15 -- which is why it is registered
    separately rather than by appending ``"delegation"`` to the tag list.
    """
    tags = list(DEFAULT_TAGS)
    if features.staged_review:
        tags.append("safety")  # undo_last_guestbook_entry
    if features.dynamic_experts:
        tags.append("prompts")  # create_and_consult_expert, and its three siblings

    registry = PythonActionRegistry(tags=tags)
    if features.delegation:
        one_tool = PythonActionRegistry(tags=["delegation"], tool_names=[DELEGATION_TOOL])
        for action in one_tool.get_actions():
            registry.register(action)
    return registry


def build_greeter_capabilities(features: Features) -> list[Capability]:
    """Fresh capability instances for one agent, in composition order.

    Order is the contract: ``process_*`` hooks compose left to right, so logging
    goes last and records what the model is actually sent. New instances every
    time, because all four count iterations on themselves.
    """
    capabilities: list[Capability] = []
    if features.time_aware:
        capabilities.append(TimeAwareCapability())
    if features.plan_first:
        capabilities.append(PlanFirstCapability())
    if features.progress_tracking:
        capabilities.append(ProgressTrackingCapability())
    if features.verbose_logging:
        capabilities.append(LoggingCapability())
    return capabilities


def build_greeter_environment(features: Features) -> Environment:
    """A plain environment, or one that holds writes back for review (R12)."""
    if features.staged_review:
        # Reviewer and staged-tool list come from StagedActionEnvironment's own
        # defaults: AGENT_REVIEWER (default "human") and ("sign_guestbook",).
        return StagedActionEnvironment()
    return PythonEnvironment()


def greeter_context_props(
    features: Features | None = None,
    generate_response: Callable | None = None,
    guestbook: GuestbookStore | None = None,
) -> dict:
    """The per-run resources to hand ``Agent.run(action_context_props=...)``.

    The guestbook store is injected rather than left to the tools' own fallback
    so that one object -- one lock, one file -- serves the whole run, and so a
    test can point the run at a temporary file.

    The agent registry is built **only** when ``delegation`` is on. Its absence
    is what keeps delegation off: ``agent_registry`` is deliberately missing from
    :func:`~hello_agent.core.action_context.default_properties`, and the
    delegation tools check for it rather than for a flag.
    """
    features = features if features is not None else Features()
    props: dict = {"guestbook": guestbook if guestbook is not None else GuestbookStore()}
    if features.delegation:
        registry = AgentRegistry()
        # The bound `run` method, not the Agent: that is the whole callee
        # contract. Sharing generate_response means one fake covers both agents.
        registry.register_agent(
            TRANSLATOR_AGENT_NAME, build_translator_agent(generate_response).run
        )
        props["agent_registry"] = registry
    return props


def build_greeter_agent(
    features: Features = None,
    generate_response: Callable = None,
) -> Agent:
    """Assemble the greeter.

    Args:
        features: Which optional patterns to switch on. ``None`` means the lean
            default -- every flag off. Pass ``Features.from_env()`` to take them
            from ``AGENT_ENABLE_*``.
        generate_response: ``(Prompt, tier="powerful") -> str``. ``None`` means
            the real LiteLLM call, imported lazily so a test never pays for it.

    Returns:
        An :class:`~hello_agent.core.agent.Agent`. Run it with the properties
        :func:`greeter_context_props` builds for the same ``features``.

    Example:
        >>> agent = build_greeter_agent()  # doctest: +SKIP
        >>> len(agent.actions.get_actions()), agent.capabilities  # doctest: +SKIP
        (5, [])
    """
    features = features if features is not None else Features()
    if generate_response is None:
        from hello_agent.llm import generate_response as generate_response_default

        generate_response = generate_response_default

    return Agent(
        goals=greeter_goals(features),
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=build_greeter_action_registry(features),
        generate_response=generate_response,
        environment=build_greeter_environment(features),
        capabilities=build_greeter_capabilities(features),
        # Read from the environment here rather than defaulted in Agent, so a
        # deployment can cap a run without editing code (ruling R2).
        max_iterations=config.max_iterations(),
        max_duration_seconds=config.max_duration_seconds(),
    )
