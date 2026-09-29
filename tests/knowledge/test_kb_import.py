"""Import hygiene: lean imports, offline tiktoken, and the knowledge ↔ solar boundary.

Acceptance (handoff KWP0):
- ``import hal9000.knowledge`` does not import docling, fastembed or torch;
- with sockets blocked, ``tiktoken.get_encoding('cl100k_base')`` works.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hal9000"
HEAVY = ("docling", "fastembed", "torch", "onnxruntime", "transformers")

_SOCKET_BLOCK = """
import socket
attempts = []
def _refuse(address, *args, **kwargs):
    attempts.append(repr(address))
    raise OSError("blocked by the knowledge offline test")
socket.socket.connect = lambda self, address: _refuse(address)
socket.socket.connect_ex = lambda self, address: _refuse(address)
socket.create_connection = _refuse
try:
    socket.socket().connect(("192.0.2.1", 80))
except OSError:
    pass
blocker_works = len(attempts) == 1
attempts.clear()
"""


def _run(code: str, env: dict[str, str] | None = None) -> dict:
    environment = dict(os.environ)
    environment["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    environment.update(env or {})
    for key in [k for k, v in environment.items() if v is None]:
        environment.pop(key)
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def heavy_modules(names) -> list[str]:
    return sorted(n for n in names if n.split(".")[0] in HEAVY)


def test_heavy_module_detector_catches_submodules():
    """Guard against a detector that can't fail."""
    assert heavy_modules(["torch.nn", "docling", "json", "fastembed.rerank"]) == [
        "docling",
        "fastembed.rerank",
        "torch.nn",
    ]


def test_import_knowledge_and_every_submodule_stays_lean():
    report = _run(
        """
import json, pkgutil, sys
import hal9000.knowledge
top = sorted(sys.modules)
for m in pkgutil.walk_packages(hal9000.knowledge.__path__, "hal9000.knowledge."):
    __import__(m.name)
print(json.dumps({"top": top, "all": sorted(sys.modules)}))
"""
    )
    assert "hal9000.knowledge.contracts" in report["all"], "the walk imported nothing"
    assert "hal9000.knowledge.api.app" in report["all"]
    assert heavy_modules(report["top"]) == []
    assert heavy_modules(report["all"]) == []


@pytest.mark.skipif(
    importlib.util.find_spec("docling") is None, reason="knowledge-docling group not installed"
)
def test_heavy_groups_are_installed_so_the_lean_check_means_something():
    # With --all-groups the heavy packages *are* importable, so the lean check
    # above proves nobody imports them, not merely that they are absent.
    assert importlib.util.find_spec("fastembed") is not None


def test_tiktoken_encodings_load_with_sockets_blocked():
    report = _run(
        _SOCKET_BLOCK
        + """
import json
import hal9000.knowledge  # points TIKTOKEN_CACHE_DIR at vendor/tiktoken
import os, tiktoken
out = {"blocker_works": blocker_works, "cache": os.environ.get("TIKTOKEN_CACHE_DIR")}
for name in ("cl100k_base", "o200k_base"):
    enc = tiktoken.get_encoding(name)
    out[name] = enc.decode(enc.encode("The furnace was installed in 2009."))
out["attempts"] = attempts
print(json.dumps(out))
""",
        env={"TIKTOKEN_CACHE_DIR": ""},
    )
    assert report["blocker_works"]
    assert report["cache"] == str(REPO / "vendor" / "tiktoken")
    assert report["cl100k_base"] == "The furnace was installed in 2009."
    assert report["o200k_base"] == "The furnace was installed in 2009."
    assert report["attempts"] == []


def test_without_the_vendored_files_tiktoken_would_dial_out(tmp_path):
    """The same probe fails with an empty cache dir: the block is real, and so is the risk."""
    report = _run(
        _SOCKET_BLOCK
        + """
import json, tiktoken
try:
    tiktoken.get_encoding("cl100k_base")
    ok = True
except Exception:
    ok = False
print(json.dumps({"blocker_works": blocker_works, "ok": ok, "attempts": len(attempts)}))
""",
        env={"TIKTOKEN_CACHE_DIR": str(tmp_path)},
    )
    assert report["blocker_works"]
    assert not report["ok"] and report["attempts"] >= 1


def test_existing_tiktoken_cache_dir_wins(monkeypatch, tmp_path):
    from hal9000 import knowledge

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    assert knowledge.configure_tiktoken() == str(tmp_path)
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    assert knowledge.configure_tiktoken() == str(knowledge.TIKTOKEN_VENDOR_DIR)
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(knowledge.TIKTOKEN_VENDOR_DIR)
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    monkeypatch.setattr(knowledge, "TIKTOKEN_VENDOR_DIR", tmp_path / "missing")
    assert knowledge.configure_tiktoken() is None


def test_vendored_files_are_the_official_encodings():
    import hashlib

    expected = {
        # sha1(blob URL) -> sha256 tiktoken expects (tiktoken_ext/openai_public.py)
        "9b5ad71b2ce5302211f9c61530b329a4922fc6a4": (
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
        ),
        "fb374d419588a4632f3f557e76b4b70aebbca790": (
            "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
        ),
    }
    vendor = REPO / "vendor" / "tiktoken"
    for name, sha256 in expected.items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == sha256


def test_token_counting_uses_the_right_encoding():
    from hal9000.knowledge.tokens import count_tokens, encoding_for_model

    assert encoding_for_model("openai/text-embedding-3-small") == "cl100k_base"
    assert encoding_for_model("openai/gpt-4-turbo") == "cl100k_base"
    assert encoding_for_model("openai/gpt-4o-mini") == "o200k_base"
    assert encoding_for_model("openai/gpt-5-mini") == "o200k_base"
    assert count_tokens("hello world") == 2
    assert count_tokens("<|endoftext|>") > 1  # special tokens counted as text, never raising


# --------------------------------------------------------------- boundaries
def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_knowledge_imports_nothing_from_solar_but_numerals():
    offenders = {}
    for path in (SRC / "knowledge").rglob("*.py"):
        bad = {
            name
            for name in _imports(path)
            if name.startswith("hal9000.solar")
            and not name.startswith("hal9000.solar.domain.numerals")
            and name not in {"hal9000.solar.domain", "hal9000.solar"}
        }
        if bad:
            offenders[str(path.relative_to(REPO))] = sorted(bad)
    assert offenders == {}


def test_solar_and_coordinator_never_import_knowledge():
    offenders = [
        str(path.relative_to(REPO))
        for folder in ("solar", "coordinator")
        for path in (SRC / folder).rglob("*.py")
        if any(name.startswith("hal9000.knowledge") for name in _imports(path))
    ]
    assert offenders == []


def test_boundary_scanner_sees_imports(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("from hal9000.solar.domain import numerals\nimport hal9000.knowledge.x\n")
    assert "hal9000.solar.domain.numerals" in _imports(probe)
    assert "hal9000.knowledge.x" in _imports(probe)
