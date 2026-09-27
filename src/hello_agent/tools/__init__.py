"""Importing this package registers every tool in the template.

``@register_tool`` publishes a tool as a *side effect of the module being
imported*, so a tool module nobody imports is a tool that does not exist. This
``__init__`` is the one place that guarantees the whole catalogue is loaded, and
:class:`~hello_agent.core.actions.PythonActionRegistry` reads that catalogue at
construction time -- so import this before you build a registry.

Registering everything is not the same as *offering* everything. Selection
happens later, by tag: the default greeter asks for ``greeting``, ``guestbook``
and ``system`` and gets five tools, while ``prompts``, ``planning``,
``delegation`` and ``safety`` sit in the catalogue until a feature flag adds
their tag. Registration is cheap; a crowded tool list is not.
"""

from __future__ import annotations

from hello_agent.tools import delegation, greeting, guestbook, planning, self_prompting, system
from hello_agent.tools.decorators import get_json_type, get_tool_metadata, register_tool

__all__ = [
    "delegation",
    "get_json_type",
    "get_tool_metadata",
    "greeting",
    "guestbook",
    "planning",
    "register_tool",
    "self_prompting",
    "system",
]
