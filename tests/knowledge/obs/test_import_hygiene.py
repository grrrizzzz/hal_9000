"""Rule K11 for observability: without an OTLP endpoint, nothing heavy is imported."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

CODE = """
import json, sys, tempfile, pathlib
from hal9000.knowledge.storage import open_db
from hal9000.knowledge.obs.timing import stage
from hal9000.knowledge.obs.traces import observe
from hal9000.knowledge.obs import metrics

conn = open_db(pathlib.Path(tempfile.mkdtemp()) / "knowledge.db")
with observe(conn, question="synthetic") as trace:
    with stage("select"):
        with stage("vector"):
            pass
    metrics.record_usage({"model": "fake", "prompt": 1, "completion": 1}, "answer", cost_usd=0)
    trace.outcome = "answered"
metrics.render_metrics()
written = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("opentelemetry", "grpc", "google"))
print(json.dumps({"heavy": heavy, "fastapi": "fastapi" in sys.modules, "written": written}))
"""


def test_observing_without_an_endpoint_loads_no_otel_grpc_or_fastapi():
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    done = subprocess.run(
        [sys.executable, "-c", CODE], cwd=REPO, env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {"heavy": [], "fastapi": False, "written": 1}
