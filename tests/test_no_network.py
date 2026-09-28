"""The suite must not touch the network -- enforced, not assumed (ruling R8).

BUILD_PLAN DoD #1 asks for a green suite "with no API keys in the environment,
no network access". The keys half is easy to see. The network half was, for most
of this build, true only by accident: ``import litellm`` fetches its model-price
map from raw.githubusercontent.com at import time. Ruling R8 measured four
outbound attempts with retries and backoff; the installed version makes one.
Either way it degrades to a bundled copy, so nothing failed -- it just quietly
needed the internet, and would have cost a sandboxed runner the connect timeout
before every single test session.

``LITELLM_LOCAL_MODEL_COST_MAP=True`` turns that off. It is set in exactly two
places, and this module is the alarm on both:

* ``tests/conftest.py``, at module scope, before the first litellm import;
* ``.github/workflows/ci.yml``, on the Test step.

It is deliberately *not* set in ``src/hal9000/llm.py``: in production a
current price map is worth one fetch. This is a test and CI guarantee, not a
library default -- which is precisely why it needs a test to defend it.

The real work happens in :func:`test_importing_the_package_makes_no_connection`,
which imports every module in the package in a subprocess with
``socket.socket.connect`` hard-blocked. If someone later adds a dependency, or a
module-scope client, that dials out on import, that test is what says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

COST_MAP_VAR = "LITELLM_LOCAL_MODEL_COST_MAP"

# Runs in a fresh interpreter. Blocks every socket connection, proves the block
# works against a reserved TEST-NET-1 address, then imports the whole package
# and reports what tried to dial out.
_PROBE = """
import json, pkgutil, socket

attempts = []


def _refuse(address, *args, **kwargs):
    attempts.append(repr(address))
    raise OSError("blocked by the no-network regression test")


socket.socket.connect = lambda self, address: _refuse(address)
socket.socket.connect_ex = lambda self, address: _refuse(address)
socket.create_connection = _refuse

# Self-check: a blocker that silently fails to install would make this probe
# pass no matter what the imports did. 192.0.2.1 is TEST-NET-1 (RFC 5737).
try:
    socket.socket().connect(("192.0.2.1", 80))
except OSError:
    pass
blocker_works = len(attempts)
attempts.clear()

import hal9000
import hal9000.tools  # importing this registers every tool

for module in pkgutil.walk_packages(hal9000.__path__, hal9000.__name__ + "."):
    __import__(module.name)

# litellm last, and explicitly: hal9000.llm imports it *inside*
# generate_response rather than at module scope, so walking the package would
# never reach it -- and it is the one dependency known to dial out on import.
import litellm

print(json.dumps({"blocker_works": blocker_works, "attempts": attempts}))
"""


def test_conftest_pins_the_litellm_cost_map():
    """The setting conftest makes must still be in force by the time a test runs."""
    assert os.environ.get(COST_MAP_VAR), (
        f"{COST_MAP_VAR} is not set. tests/conftest.py sets it at module scope so that "
        "importing litellm reads its bundled price map instead of fetching one over "
        "the network. Putting that line back is the fix; see ruling R8."
    )


def test_ci_workflow_pins_the_litellm_cost_map():
    """CI gets no conftest guarantee for free -- its own step must set it too."""
    workflow = CI_WORKFLOW.read_text()

    assert COST_MAP_VAR in workflow, (
        f"{COST_MAP_VAR} has gone missing from {CI_WORKFLOW.name}. Without it every CI "
        "job reaches raw.githubusercontent.com before it can import litellm, and pays "
        "the connect timeout when it cannot; see ruling R8."
    )


def test_importing_the_package_makes_no_connection():
    """Import every module in ``hal9000`` with sockets blocked. Nothing may dial out."""
    environment = dict(os.environ)
    environment[COST_MAP_VAR] = "True"

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, (
        f"importing the package failed under a socket block:\n{completed.stderr}"
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["blocker_works"] == 1, (
        "the socket block did not install, so this test could not have failed -- "
        "fix the probe rather than trusting its result."
    )
    assert report["attempts"] == [], (
        "importing hal9000 attempted these outbound connections: "
        f"{report['attempts']}. An import must never need the network; see ruling R8."
    )
