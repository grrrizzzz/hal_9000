"""``hal9000 kb``: every plan §17 command is registered and exits 2 until its KWP lands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hal9000.__main__ import build_parser, main

REPO = Path(__file__).resolve().parents[2]


def test_no_command_is_still_a_stub():
    """Every plan §17 command is implemented: no command module falls back to the stub."""
    cli_dir = REPO / "src" / "hal9000" / "knowledge" / "cli"
    stubs = [
        path.name
        for path in cli_dir.glob("*.py")
        if path.name != "__init__.py" and "not_implemented(" in path.read_text(encoding="utf-8")
    ]
    assert stubs == []


def test_flags_parse_as_documented():
    parser = build_parser()
    args = parser.parse_args(["kb", "eval", "sweep", "--stages", "S5", "--force-judge", "--yes"])
    assert args.kb_command == "eval sweep" and args.force_judge and args.yes
    assert args.max_cost == 10.0 and args.resume is None
    args = parser.parse_args(["kb", "gold", "generate"])
    assert args.max_cost == 3.0 and not args.yes
    assert args.resume is None and args.only is None and args.count is None
    args = parser.parse_args(["kb", "gold", "generate", "--resume"])
    assert args.resume == "latest"
    args = parser.parse_args(["kb", "gold", "generate", "--resume", "gold-x", "--max-cost", "1"])
    assert args.resume == "gold-x" and args.max_cost == 1.0
    args = parser.parse_args(["kb", "gold", "generate", "--only", "unanswerable", "--count", "9"])
    assert args.only == "unanswerable" and args.count == 9
    args = parser.parse_args(["kb", "serve"])
    assert args.port is None and args.host == "0.0.0.0" and not args.fake
    args = parser.parse_args(["kb", "ingest", "x", "y"])
    assert args.paths == ["x", "y"] and args.collection is None and not args.no_llm
    args = parser.parse_args(["kb", "eval", "latency", "--profiles", "A"])
    # None: the S6 grid's replay and end_to_end (knowledge_eval.yaml) apply
    assert (args.n, args.end_to_end, args.ingest_test, args.sweep) == (None, None, None, None)


@pytest.mark.parametrize(
    "argv",
    [
        ["kb", "ingest"],
        ["kb", "eval", "sweep"],
        ["kb", "eval", "test", "RUN"],
        ["kb", "gold", "export"],
    ],
)
def test_missing_required_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["kb"], ["kb", "gold"], ["kb", "eval"], ["kb", "profile"]])
def test_a_group_without_a_command_prints_help_and_exits_2(argv, capsys):
    assert main(argv) == 2
    assert "usage: hal9000 kb" in capsys.readouterr().err


def test_kb_appears_in_the_top_level_help_beside_the_solar_commands():
    help_text = build_parser().format_help()
    for command in ("run", "check", "calibrate", "cost", "serve", "kb"):
        assert command in help_text


def test_config_errors_from_kb_commands_exit_2(monkeypatch, capsys):
    from hal9000.knowledge import config as kc
    from hal9000.knowledge.cli import check

    def boom(args):
        raise kc.ConfigMissing("KB_PASSCODE is not set.")

    monkeypatch.setattr(check, "run", boom)
    assert main(["kb", "check"]) == 2
    assert "KB_PASSCODE" in capsys.readouterr().err


def test_building_the_cli_imports_nothing_heavy():
    """`hal9000 run` builds the whole parser: the kb group must cost it nothing."""
    code = (
        "import json, sys\n"
        "from hal9000.__main__ import build_parser\n"
        "build_parser()\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith(('hal9000.knowledge', "
        "'tiktoken', 'prometheus_client', 'sqlite_vec', 'fastapi')))))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["hal9000.knowledge", "hal9000.knowledge.cli"]
