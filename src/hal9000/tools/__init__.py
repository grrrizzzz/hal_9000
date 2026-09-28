"""Importing this package registers the framework's general-purpose tools.

The solar and triage tools live in :mod:`hal9000.solar.tools`. Registration is a
side effect of import; selection happens later, by tag.
"""

from __future__ import annotations

from hal9000.tools import delegation, planning, self_prompting, system
from hal9000.tools.decorators import get_json_type, get_tool_metadata, register_tool

__all__ = [
    "delegation",
    "get_json_type",
    "get_tool_metadata",
    "planning",
    "register_tool",
    "self_prompting",
    "system",
]
