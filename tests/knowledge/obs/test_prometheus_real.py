"""Opt-in: a real Prometheus scrapes a real ``kb serve --fake`` and runs every panel query.

Set ``KB_PROMETHEUS_DIR`` to an unpacked Prometheus release (the directory
holding ``prometheus`` and ``promtool``) to run it; without it the module is
skipped, so the offline suite never needs the binary. It checks, with the
real tools:

- ``promtool check config`` accepts ``deploy/prometheus/prometheus.yml``;
- Prometheus scrapes ``/metrics`` of ``hal9000 kb serve --fake`` (a real
  uvicorn process on a local port, driven over HTTP with httpx) using that
  file's scrape job, pointed at the local port;
- every PromQL expression on the dashboards and in the alert rules evaluates
  without an error, and those whose data the session produced return series.

Everything stays on 127.0.0.1.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from tests.knowledge.obs.test_dashboards import PROM_QUERIES, REPO

PROM_DIR = os.environ.get("KB_PROMETHEUS_DIR", "")
pytestmark = pytest.mark.skipif(
    not (PROM_DIR and (Path(PROM_DIR) / "prometheus").is_file()),
    reason="set KB_PROMETHEUS_DIR to an unpacked Prometheus release to run",
)

PASSCODE = "prometheus-check"
SAME_ORIGIN = {"sec-fetch-site": "same-origin"}
QUESTIONS = (
    "How old is the furnace?",
    "Is there a swimming pool heater?",
    "retrieval only: how old is the furnace?",
    "unverified: when was the furnace installed?",
    "embedding down: how old is the furnace?",
)
#: Queries that may rightly return nothing after this session (no such event).
MAY_BE_EMPTY = {
    # no ingestion happens in this session, so no ingestion stage has failed
    "knowledge-ingestion.json:Ingestion errors by stage and kind",
    # nothing was uploaded: increase() over counters that never moved is still a
    # series, but these divide by a zero page rate
    "knowledge-ingestion.json:Parse and embed time per page",
    # alert expressions that are empty when nothing is wrong or nothing happened
    "alert:Knowledge: answer p95 over 15 s",
    # kb serve --fake writes no regression_runs row, and its seeded eval_runs name no
    # profile (test_live.py writes both, as the CLI does, and checks them)
    "knowledge-quality.json:Post-ingest regression recall @ 2k",
    "knowledge-quality.json:Latest regression recall @ 2k",
    "knowledge-quality.json:Latest sweep metrics by profile",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(check, what: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (httpx.HTTPError, OSError) as error:
            last = error
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {what} ({last})")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    root = tmp_path_factory.mktemp("prom")
    kb_port, prom_port = _free_port(), _free_port()
    env = {
        **os.environ,
        "KB_PASSCODE": PASSCODE,
        "TMPDIR": str(root),
        "OTEL_EXPORTER_OTLP_ENDPOINT": "",
    }
    kb = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hal9000",
            "kb",
            "serve",
            "--fake",
            "--host",
            "127.0.0.1",
            "--port",
            str(kb_port),
        ],  # fmt: skip
        env=env,
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    config = yaml.safe_load((REPO / "deploy" / "prometheus" / "prometheus.yml").read_text())
    config["global"]["scrape_interval"] = "1s"
    config["global"]["scrape_timeout"] = "1s"
    job = next(j for j in config["scrape_configs"] if j["job_name"] == "hal-kb")
    job["static_configs"] = [{"targets": [f"127.0.0.1:{kb_port}"]}]
    (root / "prometheus.yml").write_text(yaml.safe_dump(config))
    prom = subprocess.Popen(
        [str(Path(PROM_DIR) / "prometheus"), f"--config.file={root / 'prometheus.yml'}",
         f"--storage.tsdb.path={root / 'tsdb'}", f"--web.listen-address=127.0.0.1:{prom_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # fmt: skip
    kb_url, prom_url = f"http://127.0.0.1:{kb_port}", f"http://127.0.0.1:{prom_port}"
    try:
        _wait(lambda: httpx.get(f"{kb_url}/health").status_code == 200, "kb serve --fake")
        with httpx.Client(base_url=kb_url, headers=SAME_ORIGIN, timeout=30) as client:
            assert client.post("/api/session", json={"passcode": PASSCODE}).is_success
            for question in QUESTIONS:
                assert client.post("/api/chat", json={"message": question}).is_success

        def scraped_after_the_questions():
            body = httpx.get(
                f"{prom_url}/api/v1/query",
                params={"query": 'kb_questions_total{outcome="abstained"}'},
            ).json()
            return any(float(r["value"][1]) >= 1 for r in body["data"]["result"])

        _wait(scraped_after_the_questions, "Prometheus to scrape the session")
        time.sleep(3)  # a few more scrapes, so rate() has points on both sides
        yield prom_url
    finally:
        for process in (prom, kb):
            process.terminate()
        for process in (prom, kb):
            try:
                process.wait(20)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()


def test_the_may_be_empty_list_names_real_queries():
    assert MAY_BE_EMPTY <= {where for where, _ in PROM_QUERIES}


def test_promtool_accepts_the_prometheus_config():
    result = subprocess.run(
        [str(Path(PROM_DIR) / "promtool"), "check", "config",
         str(REPO / "deploy" / "prometheus" / "prometheus.yml")],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SUCCESS" in result.stdout + result.stderr


def test_prometheus_scrapes_hal_kb(stack):
    body = httpx.get(f"{stack}/api/v1/targets").json()
    targets = body["data"]["activeTargets"]
    assert [(t["labels"]["job"], t["health"], t["lastError"]) for t in targets] == [
        ("hal-kb", "up", "")
    ]


@pytest.mark.parametrize("where,expr", PROM_QUERIES, ids=[w for w, _ in PROM_QUERIES])
def test_every_panel_and_alert_query_evaluates(stack, where, expr):
    query = expr.replace("$__rate_interval", "30s")
    assert "$" not in query, f"unhandled Grafana variable in {expr}"
    response = httpx.get(f"{stack}/api/v1/query", params={"query": query})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "success", body
    result = body["data"]["result"]
    if where not in MAY_BE_EMPTY:
        assert result, f"{where}: no series ({query})"
    for series in result:
        value = series["value"][1]
        assert re.fullmatch(r"-?[\d.e+-]+|NaN|\+?Inf|-Inf", value), (where, value)
