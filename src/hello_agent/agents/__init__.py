"""The agents this template assembles, and the registry that lets them call each other.

An *agent* here is a builder function returning a configured
:class:`~hello_agent.core.agent.Agent`. Builders rather than module-level
instances, so nothing is constructed -- and no LLM is resolved -- until someone
asks for it, and so tests can inject a scripted ``generate_response``.

:func:`build_translator_agent` is the sub-agent. It exists only when
``Features.delegation`` is on: the greeter that assembles it is also what puts it
in an :class:`~hello_agent.core.registry.AgentRegistry` and hands that registry
to ``Agent.run(action_context_props={"agent_registry": ...})``. Absent that one
property, the delegation tools have nobody to call -- which is exactly how a
sub-agent stays out of the default run's cost.

To add your own sub-agent: write a builder beside ``translator.py``, register
its ``run`` under a name, and tell the calling agent that name in a goal. There
is no other wiring.
"""

from __future__ import annotations

from hello_agent.agents.translator import (
    TRANSLATOR_AGENT_NAME,
    build_translator_agent,
    translator_goals,
)

# Imported after the translator, which it builds and registers when delegation
# is on. Submodule import, so there is no package-level cycle between the two.
from hello_agent.agents.greeter import (  # isort: skip
    build_greeter_agent,
    greeter_context_props,
    greeter_goals,
)

__all__ = [
    "TRANSLATOR_AGENT_NAME",
    "build_greeter_agent",
    "build_translator_agent",
    "greeter_context_props",
    "greeter_goals",
    "translator_goals",
]
