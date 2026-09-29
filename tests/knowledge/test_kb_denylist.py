"""The house-data denylist check (plan §14, rule K2). Synthetic strings only."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "kb_denylist_check.py"

# Invented strings standing in for house data, split so this file never
# contains them itself (the real-repository test below scans it).
SECRET = "Zebulon" + " Quartermain"
REFERENCE = "QX-4471" + "-Z"


def _load():
    spec = importlib.util.spec_from_file_location("kb_denylist_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "README.md").write_text("A clean readme.\n")
    (root / "src.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    return root


def _track(root: Path, rel: str, content: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    subprocess.run(["git", "add", rel], cwd=root, check=True)


def _run(root: Path, env: dict[str, str], capsys) -> tuple[int, str]:
    code = _load().main(["--root", str(root)], env=env)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_a_listed_string_fails_and_is_not_printed(repo, capsys):
    _track(repo, "docs/notes.md", f"line one\nThe owner is {SECRET.swapcase()}.\n")
    code, output = _run(repo, {"KB_DENYLIST": f"{SECRET}\n{REFERENCE}"}, capsys)
    assert code == 1
    assert "docs/notes.md:2: contains denylist entry #1" in output  # case-insensitive
    assert SECRET.casefold() not in output.casefold()
    assert REFERENCE.casefold() not in output.casefold()


def test_clean_files_pass(repo, capsys):
    code, output = _run(repo, {"KB_DENYLIST": f"{SECRET}\n{REFERENCE}"}, capsys)
    assert code == 0 and "clean" in output


def test_no_list_warns_and_passes(repo, capsys):
    _track(repo, "leak.txt", SECRET)
    code, output = _run(repo, {}, capsys)
    assert code == 0 and "WARNING" in output


def test_environment_value_pipe_separated(repo, capsys):
    """A GitHub secret written as "a|b" must match both entries (review F1)."""
    _track(repo, "invoice.txt", f"ref {REFERENCE}\n")
    code, output = _run(repo, {"KB_DENYLIST": f"{SECRET}|{REFERENCE}"}, capsys)
    assert code == 1 and "invoice.txt:1: contains denylist entry #2" in output
    _track(repo, "owner.txt", f"{SECRET}\n")
    code, output = _run(repo, {"KB_DENYLIST": f"{SECRET} | {REFERENCE}\nthird entry"}, capsys)
    assert "owner.txt:1: contains denylist entry #1" in output
    assert "invoice.txt:1: contains denylist entry #2" in output


def test_tracked_file_paths_are_checked(repo, capsys):
    name = f"docs/{REFERENCE.lower()}-invoice.txt"
    _track(repo, name, "nothing inside\n")
    code, output = _run(repo, {"KB_DENYLIST": REFERENCE}, capsys)
    assert code == 1
    assert "docs/***-invoice.txt: file path contains denylist entry #1" in output
    assert REFERENCE.casefold() not in output.casefold()  # the path is redacted too


def test_redact_masks_every_occurrence():
    module = _load()
    assert module.redact("a/QX-1/b/qx-1.txt", ["qx-1", "zz"]) == "a/***/b/***.txt"


def test_no_list_under_github_actions_emits_an_annotation(repo, capsys):
    code, output = _run(repo, {"GITHUB_ACTIONS": "true"}, capsys)
    assert code == 0
    assert "::warning title=House-data denylist::no house-data denylist configured" in output
    code, output = _run(repo, {}, capsys)
    assert "::warning" not in output


def test_env_file_pipe_separated(repo, capsys):
    _track(repo, "invoice.txt", f"ref {REFERENCE}\n")
    (repo / ".env").write_text(f"OTHER=1\nKB_DENYLIST={SECRET}|{REFERENCE}\n")
    code, output = _run(repo, {}, capsys)
    assert code == 1 and "invoice.txt:1: contains denylist entry #2" in output


def test_env_file_parses_without_python_dotenv(repo, monkeypatch):
    module = _load()
    (repo / ".env").write_text(f'export KB_DENYLIST="{SECRET}|{REFERENCE}"\n')
    monkeypatch.setitem(sys.modules, "dotenv", None)  # import fails
    entries, source = module.load_denylist({}, repo / ".env")
    assert entries == [SECRET, REFERENCE] and ".env" in source
    (repo / ".env").write_text("NOTHING=here\n")
    assert module.load_denylist({}, repo / ".env") == ([], "")


def test_the_environment_wins_over_env_file(repo, capsys):
    _track(repo, "a.txt", f"{SECRET}\n")
    (repo / ".env").write_text(f"KB_DENYLIST={SECRET}\n")
    code, _ = _run(repo, {"KB_DENYLIST": "something else entirely"}, capsys)
    assert code == 0


def test_untracked_files_are_not_scanned(repo, capsys):
    (repo / "scratch.txt").write_text(SECRET)
    code, _ = _run(repo, {"KB_DENYLIST": SECRET}, capsys)
    assert code == 0


def test_binary_files_are_scanned_too(repo, capsys):
    _track(repo, "scan.pdf", b"%PDF-1.4\0\0" + REFERENCE.encode() + b"\0")
    code, output = _run(repo, {"KB_DENYLIST": REFERENCE}, capsys)
    assert code == 1 and "scan.pdf (binary): contains denylist entry #1" in output


def test_vendored_encodings_are_excluded(repo, capsys):
    _track(repo, "vendor/tiktoken/abc", f"{REFERENCE}\n")
    code, _ = _run(repo, {"KB_DENYLIST": REFERENCE}, capsys)
    assert code == 0


def test_too_short_entries_are_skipped(repo, capsys):
    _track(repo, "a.txt", "ab\n")
    code, output = _run(repo, {"KB_DENYLIST": f"ab\n{SECRET}"}, capsys)
    assert code == 0 and "entry #1 is under 3 characters" in output


def test_outside_a_git_repository(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert _load().main([], env={"KB_DENYLIST": SECRET}) == 2


def test_the_real_repository_passes_with_a_synthetic_list():
    done = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO,
        env={"KB_DENYLIST": f"{SECRET}\n{REFERENCE}", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr


def test_the_hook_and_ci_run_the_check():
    hooks = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())
    entries = [h["entry"] for r in hooks["repos"] for h in r["hooks"]]
    assert any("kb_denylist_check.py" in e for e in entries)

    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "knowledge.yml").read_text())
    steps = workflow["jobs"]["denylist"]["steps"]
    check = next(s for s in steps if "kb_denylist_check.py" in s.get("run", ""))
    assert check["env"]["KB_DENYLIST"] == "${{ secrets.KB_DENYLIST }}"
    python_steps = " ".join(s.get("run", "") for s in workflow["jobs"]["python"]["steps"])
    for gate in (
        "ruff check",
        "ruff format --check",
        "pytest",
        "kb_sqlite_vec_check.py",
        "kb_export_openapi.py --check",
    ):
        assert gate in python_steps, gate
    web = workflow["jobs"]["web"]["steps"]
    assert all("hashFiles('web/package.json')" in s["if"] for s in web if "if" in s)
    assert {"npm ci", "npm run lint", "npm run typecheck", "npm test"} <= {
        s.get("run") for s in web
    }
