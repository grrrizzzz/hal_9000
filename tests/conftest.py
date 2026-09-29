"""Shared fixtures.

The important one is :func:`clean_env`: it is autouse, so every test starts from
an environment with no provider keys and no ``AGENT_*`` settings. That keeps the
suite hermetic -- it behaves the same on a laptop with real keys exported as it
does in CI, where there are deliberately no secrets.
"""

from __future__ import annotations

import os

# LiteLLM fetches its model-price map over the network at import time (with
# retries and backoff). Pin it to the bundled copy so the suite makes no
# outbound connection on any machine, sandboxed CI included. Must be set
# before litellm is first imported, which is why it lives at module scope.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from collections.abc import Callable

import pytest

from tests.fakes import FakeLLM

PROVIDER_VARS = ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


#: Markers that run only when named in ``-m`` (registered in pyproject.toml).
OPT_IN_MARKERS = ("live", "models", "house")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip opt-in tests unless selected: ``pytest -m house`` runs the house tests.

    A plain ``pytest`` on a machine that has keys, models and the house bundle must
    still spend nothing and read no house data (knowledge rules K2, K3, K9).
    """
    selected = config.getoption("markexpr") or ""
    for item in items:
        for marker in OPT_IN_MARKERS:
            if item.get_closest_marker(marker) and marker not in selected:
                item.add_marker(pytest.mark.skip(reason=f"opt-in: select it with -m {marker}"))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip provider keys and every ``AGENT_*`` variable for the duration of a test."""
    for name in PROVIDER_VARS:
        monkeypatch.delenv(name, raising=False)
    prefixes = ("AGENT_", "HA_", "HAL_", "SOLAR_", "TELEGRAM_", "DEADMAN_")
    for name in [key for key in os.environ if key.startswith(prefixes)]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_llm() -> Callable[..., FakeLLM]:
    """Factory for a scripted LLM: ``fake_llm(tool_call("terminate", message="hi"))``."""

    def _make(*responses: str) -> FakeLLM:
        return FakeLLM(list(responses))

    return _make


@pytest.fixture
def anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend an Anthropic key is configured."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend an OpenAI key is configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


@pytest.fixture
def clean_tool_registry():
    """Hand a test an **empty** global tool catalogue, then put the real one back.

    The catalogue is module-level mutable state -- that is what lets a decorator
    in one file publish a tool readable from another -- so a test that registers
    throwaway tools would otherwise leak them into every later test. Contents are
    saved and restored rather than rebuilt, because other modules hold references
    to these exact dict objects (see :func:`hal9000.core.registry.clear_tools`).
    """
    from hal9000.core import registry

    saved_tools = dict(registry.tools)
    saved_by_tag = {tag: list(names) for tag, names in registry.tools_by_tag.items()}
    registry.clear_tools()
    try:
        yield registry
    finally:
        registry.clear_tools()
        registry.tools.update(saved_tools)
        registry.tools_by_tag.update(saved_by_tag)


@pytest.fixture
def registered_tools():
    """The global tool catalogue, guaranteed populated.

    Registration is a side effect of importing a tool module, and Python imports
    a module once per process -- so a test that runs *after* one which called
    :func:`hal9000.core.registry.clear_tools` would otherwise find the
    catalogue empty, and would pass or fail depending on file ordering. Reloading
    the tool modules re-runs their decorators and refills it.

    Any test that reads the real catalogue (rather than registering throwaway
    tools into an empty one, which is what ``clean_tool_registry`` is for) should
    depend on this fixture.
    """
    import importlib
    import sys

    from hal9000.core import registry

    importlib.import_module("hal9000.tools")
    importlib.import_module("hal9000.solar.tools")
    if "call_agent" not in registry.tools:
        for name, module in list(sys.modules.items()):
            if name.startswith("hal9000.tools.") and not name.endswith(".decorators"):
                importlib.reload(module)
    from hal9000.solar.tools import ensure_registered

    ensure_registered()
    return registry


# --- HAL solar fixtures -------------------------------------------------------


@pytest.fixture
def hal_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A hermetic HAL environment: temp data dir, HA 'configured', the real solar correction."""
    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HA_BASE_URL", "http://ha.test:8123")
    monkeypatch.setenv("HA_TOKEN", "test-token")
    monkeypatch.setenv("SOLAR_SCALE", "2.0")
    monkeypatch.setenv("SOLAR_FIXED_AT", "2026-09-27T18:35:00-07:00")
    return tmp_path


@pytest.fixture
def write_fixture(tmp_path):
    """Write a replay fixture dict to a file and return its path."""
    import json

    def _write(fx: dict, name: str = "fixture.json") -> str:
        path = tmp_path / name
        path.write_text(json.dumps(fx))
        return str(path)

    return _write
