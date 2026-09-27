"""The two ways a new user's very first run goes wrong.

Both were found after the build was otherwise complete, and both are failures
of the quickstart rather than of the agent: the docs told the reader to create
a ``.env`` that nothing read, and a mistyped key answered with a stack trace.
"""

from __future__ import annotations

import os

import pytest
from dotenv import find_dotenv, load_dotenv

from hello_agent import __main__ as cli


def test_a_dotenv_in_the_working_directory_is_found(tmp_path, monkeypatch):
    """`cp .env.example .env` has to work from wherever the user is standing.

    ``load_dotenv()`` defaults to searching from the *calling module's*
    directory, which in a fresh clone is inside the installed package and never
    where the user put their .env. ``usecwd=True`` is what makes the documented
    workflow real; this test fails if someone drops it.
    """
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    load_dotenv(find_dotenv(usecwd=True), override=False)

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-dotenv"


def test_a_real_exported_key_beats_a_stale_dotenv(tmp_path, monkeypatch):
    """`override=False`: the shell wins. A forgotten .env must not shadow it."""
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-stale\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-exported")

    load_dotenv(find_dotenv(usecwd=True), override=False)

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-exported"


def test_main_loads_the_dotenv_before_resolving_a_provider(tmp_path, monkeypatch):
    """Ordering matters: resolution fails fast, so the load must precede it."""
    calls: list[str] = []
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: calls.append("load"))
    monkeypatch.setattr(cli.config, "resolve_provider", lambda: calls.append("resolve"))
    monkeypatch.setenv("AGENT_GUESTBOOK_PATH", str(tmp_path / "gb.json"))

    from tests.fakes import FakeLLM, tool_call

    cli.main(
        ["Ada is here", "--quiet"],
        generate_response=FakeLLM([tool_call("terminate", message="done")]),
    )

    assert calls[0] == "load", f"expected the .env load first, got {calls}"


def test_a_rejected_key_is_an_auth_error(monkeypatch):
    """A bad key gets one line, not a stack trace."""
    from litellm.exceptions import AuthenticationError

    error = AuthenticationError(message="bad key", llm_provider="anthropic", model="claude-opus-5")
    assert cli._is_auth_error(error) is True


@pytest.mark.parametrize(
    "error", [ValueError("a real bug"), KeyError("missing"), RuntimeError("boom")]
)
def test_every_other_failure_still_propagates(error):
    """A traceback is the right answer to a genuine bug -- don't swallow it."""
    assert cli._is_auth_error(error) is False
