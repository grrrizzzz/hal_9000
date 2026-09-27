"""The one test in this repository that talks to a real model.

Everything else in ``tests/`` runs on :class:`tests.fakes.FakeLLM` and proves
that *our* code is wired correctly. That is necessary and not sufficient: a fake
cannot notice that LiteLLM changed the shape of a tool-call response, that a
provider stopped honouring a field, or that a model id was retired. One real
round trip per provider is what catches that -- it is the mitigation named in
BUILD_PLAN §10 against "LiteLLM's tool-call response shape drifts between
providers".

**These tests skip themselves when their key is absent**, which is why the suite
stays green in CI (no secrets, by design) and on a fresh clone. They are not
deselected, weakened or mocked: given a key they run for real, unmodified.

    uv run pytest -m live                  # just these, needs a key
    uv run pytest -m "not live"            # everything else, if you have keys
                                           #   exported and don't want to spend

Running one costs a handful of LLM calls: two on the fast tier and two on the
powerful tier for the lean path (extract, compose, sign, terminate), plus the
model's own tool-selection turns.

**Why the lean default path.** No flags, five tools -- that is the command a new
user runs first, and the one the README's quickstart prints. A smoke test of a
configuration nobody runs proves nothing about the configuration everybody runs.
Notably ``--staged-review`` is *not* exercised here: its default reviewer reads
stdin, so a live run of it would block forever on a CI machine. If you add one,
set ``AGENT_REVIEWER=llm``.

**Why the assertions are loose.** A live model writes the greeting, so its exact
wording is not ours to predict and asserting on it would make this test flaky
rather than strict. What *is* predictable is the shape of a completed run, and
that is what is asserted: exit code 0, a non-empty final message on stdout, and
exactly one guestbook entry carrying the visitor's name and a real greeting.
Every one of those would break under a genuine regression -- a response shape
LiteLLM no longer parses gives a fallback ``terminate`` and zero entries; a
broken argument path gives an entry whose ``name`` is not the visitor's; a
retired model id gives a non-zero exit.
"""

from __future__ import annotations

import json
import os

import pytest

from hello_agent.__main__ import main

pytestmark = pytest.mark.live

#: provider -> the environment variable that pays for it.
PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Captured at import time, which is collection time -- *before* conftest's
# autouse ``clean_env`` fixture strips provider keys from os.environ for the
# duration of each test. That fixture is what makes the rest of the suite
# hermetic; these tests are the deliberate exception, so they keep their own
# copy and hand it back with monkeypatch.setenv below.
_KEYS_AT_IMPORT = {name: os.environ.get(var) for name, var in PROVIDER_KEYS.items()}

VISITOR_TEXT = "Ada Lovelace, mathematician, visiting from London."

#: Bound so a wedged run fails the suite instead of hanging it. Four tool calls
#: on the lean path finish well inside this on either provider.
MAX_DURATION_SECONDS = "150"


def _require(provider: str) -> str:
    """The live key for ``provider``, or skip this test with a usable reason."""
    key = _KEYS_AT_IMPORT.get(provider)
    if not key:
        pytest.skip(
            f"no {PROVIDER_KEYS[provider]} in the environment -- "
            f"export one and re-run `uv run pytest -m live` to verify {provider} live."
        )
    return key


@pytest.mark.parametrize("provider", sorted(PROVIDER_KEYS))
def test_live_run_greets_and_signs_the_guestbook(provider, tmp_path, monkeypatch, capsys):
    """A real run on ``provider``: the agent finishes, greets, and signs once.

    This is BUILD_PLAN DoD #4 (Anthropic) and #5 (OpenAI) executed as a test.
    """
    key = _require(provider)
    monkeypatch.setenv(PROVIDER_KEYS[provider], key)
    # Pin the provider rather than letting resolution choose: on a machine with
    # both keys exported, precedence would otherwise run the Anthropic models
    # twice and the OpenAI case never.
    monkeypatch.setenv("LLM_PROVIDER", provider)
    # Never write into the repository. A live run is still a real run: without
    # this it would append to ./guestbook.json.
    guestbook_path = tmp_path / "guestbook.json"
    monkeypatch.setenv("AGENT_GUESTBOOK_PATH", str(guestbook_path))
    monkeypatch.setenv("AGENT_MAX_DURATION_SECONDS", MAX_DURATION_SECONDS)

    # No flags: the lean default, exactly as the quickstart runs it. --quiet so
    # stdout is the agent's final message and nothing else.
    exit_code = main([VISITOR_TEXT, "--quiet"])
    captured = capsys.readouterr()

    assert exit_code == 0, f"run failed (exit {exit_code}); stderr: {captured.err}"

    final_message = captured.out.strip()
    assert final_message, "the agent terminated without printing anything"

    assert guestbook_path.exists(), "the run finished without ever writing the guestbook"
    document = json.loads(guestbook_path.read_text())
    entries = document["entries"]
    assert len(entries) == 1, f"expected exactly one entry, got {len(entries)}: {entries}"

    entry = entries[0]
    # The visitor's surname is a fact in the input, not a matter of style, so
    # requiring it is loose about wording while still proving that what the
    # extractor found actually reached the write.
    assert "lovelace" in str(entry.get("name", "")).lower(), (
        f"the signed name does not look like the visitor's: {entry.get('name')!r}"
    )
    greeting = str(entry.get("greeting", "")).strip()
    assert len(greeting) > 20, f"the recorded greeting is not a greeting: {greeting!r}"
