"""``hal9000 kb eval sweep|latency|choose|report`` end to end through the CLI, offline."""
# ruff: noqa: F811 - pytest fixtures imported from test_sweep_fixtures are parameters here

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from hal9000.__main__ import main
from hal9000.knowledge.eval.runs import list_runs
from hal9000.knowledge.eval.sweep import read_manifest
from hal9000.knowledge.storage import open_db
from tests.knowledge.eval.test_sweep_fixtures import (  # noqa: F401 - fixtures
    ACTIVE,
    TINY_GRID,
    FakeProvider,
    _copy,
    corpus_root,
    fake_chat,
    fake_mode,
    fresh_query_cache,
    label_calibration,
    make_settings,
)

#: offline everywhere: no reranker model download, no Docling
CLI_GRID = TINY_GRID.replace("rerankers: [none, minilm]", "rerankers: [none]").replace(
    "parser: [pdfium, docling]", "parser: [pdfium]"
)


@pytest.fixture
def cli(corpus_root: Path, tmp_path: Path, monkeypatch):
    _copy(corpus_root, tmp_path)
    settings = make_settings(tmp_path, CLI_GRID)
    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HAL_CONFIG_DIR", str(settings.config_dir))
    log = io.StringIO()
    provider = FakeProvider(log)
    monkeypatch.setattr("hal9000.llm.generate_response", provider)
    monkeypatch.setattr("hal9000.llm.stream_response", fake_chat())
    return settings


def runs(settings, stage=None):
    conn = open_db(settings.db_path)
    try:
        return [r for r in list_runs(conn) if stage is None or r.stage == stage]
    finally:
        conn.close()


def active(settings) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        return conn.execute("SELECT name FROM profiles WHERE active = 1").fetchone()[0]
    finally:
        conn.close()


def sweep_id_from(out: str) -> str:
    return out.split("Sweep ", 1)[1].split(":", 1)[0]


def test_sweep_s5_report_and_choose(cli, capsys):
    settings = cli
    assert main(["kb", "eval", "sweep", "--stages", "S1a,S1b,S2,S3,S4", "--yes"]) == 0
    out = capsys.readouterr().out
    sweep = sweep_id_from(out)
    assert "Estimate for S1a" in out and "Next (gate G-K1b): hal9000 kb eval judge-prep" in out
    assert active(settings) == ACTIVE
    assert main(["kb", "eval", "sweep", "--stages", "S5", "--resume", sweep, "--yes"]) == 3
    assert "S5 refused" in capsys.readouterr().err

    conn = open_db(settings.db_path)
    try:
        leader = next(r for r in list_runs(conn) if r.stage == "S4" and r.config.get("k") is None)
        label_calibration(conn, leader.id)
    finally:
        conn.close()
    assert main(["kb", "eval", "sweep", "--stages", "S5", "--resume", sweep, "--yes"]) == 0
    assert runs(settings, "S5")

    assert main(["kb", "eval", "report", sweep]) == 0
    report = (settings.config_dir.parent / "docs" / "KNOWLEDGE_EVAL_REPORT.md").read_text()
    assert "### Recommendation (plan §8.6)" in report and "chat tier: fast" in report
    assert "flagged" in report  # no Pi numbers yet

    s4 = [r for r in runs(settings, "S4") if r.config.get("k") is None]
    chosen = s4[-1].config["name"]
    capsys.readouterr()
    assert main(["kb", "eval", "choose", sweep, "--config", chosen, "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"Chosen: {chosen}" in out
    assert active(settings) == s4[-1].config["profile"]
    yaml_text = settings.yaml_path.read_text()
    assert f"active_profile: {s4[-1].config['profile']}" in yaml_text
    assert f"method: {s4[-1].config['retrieval']}" in yaml_text and "# G-K2: chosen by" in yaml_text
    choice = read_manifest(settings.eval_dir, sweep)["choice"]
    assert choice["config"] == chosen and choice["chat_tier"] == "fast"
    assert main(["kb", "eval", "report", sweep]) == 0
    report = (settings.config_dir.parent / "docs" / "KNOWLEDGE_EVAL_REPORT.md").read_text()
    assert f"**Chosen at G-K2:** {chosen}" in report

    k_variant = next(r for r in runs(settings, "S4") if r.config.get("k") is not None)
    assert main(["kb", "eval", "choose", sweep, "--config", k_variant.id, "--yes"]) == 1
    assert main(["kb", "eval", "choose", sweep, "--config", "nope", "--yes"]) == 1
    assert main(["kb", "eval", "choose", "sweep-missing", "--config", chosen, "--yes"]) == 1


def test_the_sweep_needs_a_confirmation(cli, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["kb", "eval", "sweep", "--stages", "S1a"]) == 3
    captured = capsys.readouterr()
    assert "pass --yes" in captured.out and "not confirmed" in captured.err
    assert not runs(cli)


def test_the_sweep_stops_at_max_cost_and_resumes(cli, capsys):
    argv = ["kb", "eval", "sweep", "--stages", "S1a,S1b,S2,S3", "--yes", "--max-cost", "0.01"]
    assert main(argv) == 4
    out = capsys.readouterr().out
    sweep = sweep_id_from(out)
    assert "STOPPED at the cost cap" in out
    assert main([*argv[:-1], "1.0", "--resume", sweep]) == 0
    assert any(r.config.get("query_transform") == "multi" for r in runs(cli, "S3"))


def test_bad_arguments_are_errors(cli, capsys):
    assert main(["kb", "eval", "sweep", "--stages", "S5", "--yes"]) == 1
    assert "--resume" in capsys.readouterr().err
    (cli.config_dir / "knowledge_eval.yaml").write_text("sweep: {S1a: {sizes: [1]}}\n")
    assert main(["kb", "eval", "sweep", "--stages", "S1a", "--yes"]) == 1


def test_latency_command(cli, capsys):
    argv = ["kb", "eval", "latency", "--profiles", ACTIVE, "--n", "5", "--end-to-end", "1"]
    assert main([*argv, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "Stored s6-" in out and "local p50/p95/p99" in out
    (run,) = runs(cli, "S6")
    assert run.config["measured"]["end_to_end"] == 1
    assert main(["kb", "eval", "latency", "--profiles", "missing.x", "--yes"]) == 3
    assert "not built" in capsys.readouterr().err
