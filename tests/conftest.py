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


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip provider keys and every ``AGENT_*`` variable for the duration of a test."""
    for name in PROVIDER_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in [key for key in os.environ if key.startswith("AGENT_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_llm() -> Callable[..., FakeLLM]:
    """Factory for a scripted LLM: ``fake_llm(tool_call("terminate", message="hi"))``."""

    def _make(*responses: str) -> FakeLLM:
        return FakeLLM(list(responses))

    return _make


@pytest.fixture
def guestbook_path(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway guestbook file path, also exported as ``AGENT_GUESTBOOK_PATH``."""
    path = tmp_path / "guestbook.json"
    monkeypatch.setenv("AGENT_GUESTBOOK_PATH", str(path))
    return path


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
    to these exact dict objects (see :func:`hello_agent.core.registry.clear_tools`).
    """
    from hello_agent.core import registry

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
    :func:`hello_agent.core.registry.clear_tools` would otherwise find the
    catalogue empty, and would pass or fail depending on file ordering. Reloading
    the tool modules re-runs their decorators and refills it.

    Any test that reads the real catalogue (rather than registering throwaway
    tools into an empty one, which is what ``clean_tool_registry`` is for) should
    depend on this fixture.
    """
    import importlib
    import sys

    from hello_agent.core import registry

    importlib.import_module("hello_agent.tools")
    if not registry.tools:
        for name, module in list(sys.modules.items()):
            if name.startswith("hello_agent.tools.") and not name.endswith(".decorators"):
                importlib.reload(module)
    return registry
