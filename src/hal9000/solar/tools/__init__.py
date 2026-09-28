"""Importing this package registers the solar and triage tools in the global catalogue."""

from __future__ import annotations

import importlib

from hal9000.core import registry
from hal9000.solar.tools import solar_tools, triage_tools

__all__ = ["ensure_registered", "solar_tools", "triage_tools"]


def ensure_registered() -> None:
    """Re-register the tools if the global catalogue was cleared (tests, reloads)."""
    if "get_home_energy" not in registry.tools or "submit_verdict" not in registry.tools:
        importlib.reload(solar_tools)
        importlib.reload(triage_tools)
