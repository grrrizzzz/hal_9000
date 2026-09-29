"""HAL integration (handoff KWP10, rule K5): the knowledge tool stays out of the solar agents.

- The tool's schema exposes only ``query`` and ``doc_type``.
- **The solar and triage registries are unchanged**: built on this tree, after
  the knowledge tool is registered, they equal byte for byte the snapshot in
  ``tests/knowledge/ops/registries_008d35d.json``, produced from ``solar-agent``
  at commit 008d35d (before the knowledge base) by
  ``tests/knowledge/ops/registry_snapshot.py``.
- **Import boundary** (AST, extending ``test_kb_import.py``'s checks):
  ``hal9000.knowledge`` imports nothing from ``hal9000.solar`` except
  ``solar.domain.numerals`` -- relative imports and ``importlib`` strings
  included -- and ``hal9000.solar`` / ``hal9000.coordinator`` import nothing
  from ``hal9000.knowledge``. At run time, importing every knowledge module
  loads no solar module but numerals and its packages.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.knowledge.ops.registry_snapshot import dumps, snapshot

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hal9000"
SNAPSHOT = REPO / "tests" / "knowledge" / "ops" / "registries_008d35d.json"
NUMERALS = "hal9000.solar.domain.numerals"
#: the packages Python must import on the way to numerals (both are docstring-only)
NUMERALS_PARENTS = {"hal9000.solar", "hal9000.solar.domain"}


# ------------------------------------------------------------------ the tool's schema
def test_the_schema_exposes_only_query_and_doc_type():
    from hal9000.core import registry
    from hal9000.knowledge import tools

    tools.ensure_registered()
    entry = registry.tools["search_house_docs"]
    assert entry["parameters"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}, "doc_type": {"type": "string"}},
        "required": ["query"],
    }
    assert entry["tags"] == ["knowledge_read"] and entry["terminal"] is False
    assert "_house_docs" not in json.dumps(entry["parameters"])
    assert "house_docs" not in entry["description"]
    assert len(entry["description"]) <= 1024  # providers cap tool descriptions
    for doc_type in tools.DOC_TYPES:
        assert doc_type in entry["description"]


def test_the_tool_is_only_in_its_own_tag():
    from hal9000.core import registry
    from hal9000.knowledge import tools

    tools.ensure_registered()
    holders = sorted(
        t for t, names in registry.tools_by_tag.items() if "search_house_docs" in names
    )
    assert holders == ["knowledge_read"]


# ------------------------------------------------------------------ registries
def test_solar_and_triage_registries_equal_the_pre_knowledge_snapshot():
    from hal9000.knowledge import tools  # registers search_house_docs first

    tools.ensure_registered()
    current = dumps(snapshot())
    assert current.encode() == SNAPSHOT.read_bytes()


def test_the_snapshot_is_the_real_one():
    """Guard against a vacuous snapshot: it holds both agents' known toolkits."""
    snap = json.loads(SNAPSHOT.read_text())
    solar = [tool["name"] for tool in snap["solar"]["tools"]]
    triage = [tool["name"] for tool in snap["triage"]["tools"]]
    assert "deliver_report" in solar and "run_fault_triage" in solar and len(solar) >= 10
    assert "submit_verdict" in triage and "read_handover" in triage and len(triage) >= 6
    assert all("search_house_docs" not in names for names in (solar, triage))
    assert "knowledge_read" not in snap["solar"]["tags"] + snap["triage"]["tags"]


def test_the_comparison_would_catch_a_leak(clean_tool_registry):
    """If the knowledge tool carried a solar tag, the snapshot comparison would fail."""
    from hal9000.solar.tools import ensure_registered
    from hal9000.tools.decorators import register_tool

    ensure_registered()

    @register_tool(tags=["knowledge_read", "memory_read"])
    def search_house_docs(query: str) -> dict:  # pragma: no cover - never called
        """Leaky."""
        return {}

    assert dumps(snapshot()).encode() != SNAPSHOT.read_bytes()


def test_the_snapshot_script_runs_standalone():
    """The helper imports nothing from the knowledge base, so it runs on the old commit."""
    env = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True", PYTHONPATH=str(REPO / "src"))
    done = subprocess.run(
        [sys.executable, str(REPO / "tests/knowledge/ops/registry_snapshot.py")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.encode() == SNAPSHOT.read_bytes()
    source = (REPO / "tests/knowledge/ops/registry_snapshot.py").read_text()
    assert "hal9000.knowledge" not in {name for name in _imports_of(ast.parse(source), "")}


# ------------------------------------------------------------------ import boundary (AST)
def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(module: str | None, level: int, current: str, is_package: bool) -> str:
    if level == 0:
        return module or ""
    base = current.split(".") if is_package else current.split(".")[:-1]
    base = base[: len(base) - (level - 1)] if level > 1 else base
    return ".".join([*base, module] if module else base)


def _imports_of(tree: ast.AST, current: str, is_package: bool = False) -> set[str]:
    """Every module a file names: imports (relative ones resolved) and importlib strings."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve(node.module, node.level, current, is_package)
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("import_module", "__import__") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.add(first.value)
    return found


def _file_imports(path: Path) -> set[str]:
    return _imports_of(ast.parse(path.read_text()), _module_name(path), path.name == "__init__.py")


def _solar_violation(name: str) -> bool:
    if not (name == "hal9000.solar" or name.startswith("hal9000.solar.")):
        return False
    if name == NUMERALS or name.startswith(NUMERALS + "."):
        return False
    # "from hal9000.solar.domain import numerals" names the package and the module
    return name not in NUMERALS_PARENTS


def test_knowledge_imports_nothing_from_solar_but_numerals():
    offenders = {}
    for path in sorted((SRC / "knowledge").rglob("*.py")):
        names = _file_imports(path)
        bad = sorted(n for n in names if _solar_violation(n))
        # a bare parent-package import is only allowed alongside the numerals import itself
        parents = names & NUMERALS_PARENTS
        if parents and not any(n.startswith(NUMERALS) for n in names):
            bad += sorted(parents)
        if bad:
            offenders[str(path.relative_to(REPO))] = bad
    assert offenders == {}


def test_solar_and_coordinator_import_nothing_from_knowledge():
    offenders = {}
    for folder in ("solar", "coordinator"):
        for path in sorted((SRC / folder).rglob("*.py")):
            bad = sorted(
                n
                for n in _file_imports(path)
                if n == "hal9000.knowledge" or n.startswith("hal9000.knowledge.")
            )
            if bad:
                offenders[str(path.relative_to(REPO))] = bad
    assert offenders == {}


@pytest.mark.parametrize(
    ("source", "where", "expected_bad"),
    [
        ("from hal9000.solar.domain import numerals\n", "hal9000.knowledge.x", False),
        (
            "from hal9000.solar.domain.numerals import extract_numerals\n",
            "hal9000.knowledge.x",
            False,
        ),
        ("from hal9000.solar import pipeline\n", "hal9000.knowledge.x", True),
        ("import hal9000.solar.weather\n", "hal9000.knowledge.x", True),
        ("import hal9000.solar\n", "hal9000.knowledge.x", True),
        ("from ...solar import energy\n", "hal9000.knowledge.index.x", True),
        ("from ..solar.domain import numerals\n", "hal9000.knowledge.x", False),
        ("importlib.import_module('hal9000.solar.pipeline')\n", "hal9000.knowledge.x", True),
        ("__import__('hal9000.solar.energy')\n", "hal9000.knowledge.x", True),
    ],
)
def test_the_boundary_scanner_catches_every_shape(source, where, expected_bad):
    names = _imports_of(ast.parse(source), where)
    bad = any(_solar_violation(n) for n in names) or bool(
        names & NUMERALS_PARENTS and not any(n.startswith(NUMERALS) for n in names)
    )
    assert bad is expected_bad, names


def test_the_scanner_resolves_relative_imports_into_knowledge():
    names = _imports_of(ast.parse("from ..knowledge import tools\n"), "hal9000.solar.x")
    assert "hal9000.knowledge.tools" in names


def test_importing_all_of_knowledge_loads_no_solar_module_but_numerals():
    code = """
import json, pkgutil, sys
import hal9000.knowledge
for m in pkgutil.walk_packages(hal9000.knowledge.__path__, "hal9000.knowledge."):
    __import__(m.name)
print(json.dumps(sorted(n for n in sys.modules if n.startswith("hal9000."))))
"""
    env = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True")
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=180
    )
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert "hal9000.knowledge.tools" in loaded and "hal9000.knowledge.cli.ops" in loaded
    solar = {n for n in loaded if n == "hal9000.solar" or n.startswith("hal9000.solar.")}
    assert solar <= {NUMERALS, *NUMERALS_PARENTS}
    assert not any(n.startswith("hal9000.coordinator") for n in loaded)
