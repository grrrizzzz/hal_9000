"""The solar and triage tool registries as JSON (handoff KWP10, rule K5).

Run as a script against any checkout to print the snapshot::

    PYTHONPATH=<checkout>/src python tests/knowledge/ops/registry_snapshot.py

``registries_008d35d.json`` was produced this way from ``solar-agent`` at
commit ``008d35d`` (before the knowledge base existed), in a temporary
worktree, with this very file. ``test_integration.py`` rebuilds both
registries on the current tree, after registering the knowledge tool, and
compares the bytes.

It imports only the framework and the solar agent, never ``hal9000.knowledge``,
so it runs on the old commit unchanged.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["describe", "dumps", "snapshot"]


def describe(registry: Any) -> dict[str, Any]:
    """Everything an agent's registry decides: its tools, their schemas, what the model sees."""
    from hal9000.core.agent_language import AgentFunctionCallingActionLanguage

    actions = registry.get_actions()
    return {
        "tools": [
            {
                "name": action.name,
                "function": f"{action.function.__module__}.{action.function.__qualname__}",
                "terminal": action.terminal,
                "description": action.description,
                "parameters": action.parameters,
            }
            for action in actions
        ],
        "model_tools": AgentFunctionCallingActionLanguage().format_actions(actions),
    }


def snapshot() -> dict[str, Any]:
    """Build the solar and triage registries from the global catalogue, as it stands."""
    import hal9000.solar.tools  # noqa: F401 - registration is a side effect of import
    import hal9000.tools  # noqa: F401
    from hal9000.solar.agents.solar import SOLAR_TAGS, solar_registry
    from hal9000.solar.agents.triage import TRIAGE_TAGS, triage_registry

    return {
        "solar": {"tags": list(SOLAR_TAGS), **describe(solar_registry())},
        "triage": {"tags": list(TRIAGE_TAGS), **describe(triage_registry())},
    }


def dumps(snap: dict[str, Any]) -> str:
    return json.dumps(snap, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":  # pragma: no cover - run against the old commit by hand
    import sys

    sys.stdout.write(dumps(snapshot()))
