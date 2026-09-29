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

#: Plan §17 plus handoff KWP0's additions, with the owning work package.
COMMANDS = [
    (["check"], "KWP3"),
    (["ingest", "a.pdf", "Folder/", "--collection", "Appliances", "--no-llm", "--yes"], "KWP3"),
    (["ask", "How old is the furnace?", "--trace"], "KWP6"),
    (["serve", "--host", "127.0.0.1", "--port", "8911", "--fake"], "KWP6"),
    (["gold", "generate", "--max-cost", "2.00", "--yes"], "KWP5"),
    (["gold", "export", "data/knowledge/eval/gold.jsonl"], "KWP5"),
    (["gold", "import", "data/knowledge/eval/gold.jsonl"], "KWP5"),
    (["eval", "sweep", "--stages", "S1a,S1b,S2,S3,S4", "--max-cost", "10", "--yes"], "KWP5"),
    (["eval", "sweep", "--stages", "S5", "--resume", "RUN1", "--yes", "--force-judge"], "KWP5"),
    (["eval", "judge-prep", "RUN1", "--yes"], "KWP5"),
    (
        [
            "eval",
            "latency",
            "--profiles",
            "A,B,C",
            "--n",
            "200",
            "--end-to-end",
            "30",
            "--ingest-test",
            "ab" * 32,
        ],
        "KWP5",
    ),
    (["eval", "export", "RUN1"], "KWP5"),
    (["eval", "report", "RUN1"], "KWP5"),
    (["eval", "regression"], "KWP5"),
    (["profile", "list"], "KWP10"),
    (["profile", "activate", "pdfium.fixed-512.hdr.te3s-1536"], "KWP10"),
    (["backup", "--to", "/tmp/backups"], "KWP10"),
    (["stats"], "KWP10"),
]


@pytest.mark.parametrize(
    ("argv", "kwp"), COMMANDS, ids=lambda v: " ".join(v) if isinstance(v, list) else v
)
def test_every_command_is_a_stub_that_exits_2(argv, kwp, capsys):
    assert main(["kb", *argv]) == 2
    err = capsys.readouterr().err
    assert "not implemented yet" in err and f"({kwp})" in err


def test_flags_parse_as_documented():
    parser = build_parser()
    args = parser.parse_args(["kb", "eval", "sweep", "--stages", "S5", "--force-judge", "--yes"])
    assert args.kb_command == "eval sweep" and args.force_judge and args.yes
    assert args.max_cost == 10.0 and args.resume is None
    args = parser.parse_args(["kb", "gold", "generate"])
    assert args.max_cost == 2.0 and not args.yes
    args = parser.parse_args(["kb", "serve"])
    assert args.port is None and args.host == "0.0.0.0" and not args.fake
    args = parser.parse_args(["kb", "ingest", "x", "y"])
    assert args.paths == ["x", "y"] and args.collection is None and not args.no_llm
    args = parser.parse_args(["kb", "eval", "latency", "--profiles", "A"])
    assert (args.n, args.end_to_end, args.ingest_test) == (200, 30, None)


@pytest.mark.parametrize(
    "argv",
    [
        ["kb", "ingest"],
        ["kb", "eval", "sweep"],
        ["kb", "eval", "latency"],
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
