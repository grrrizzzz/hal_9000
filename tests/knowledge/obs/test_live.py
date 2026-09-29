"""KWP8 phase 2: ``/metrics`` and every Grafana query against a live ``kb serve --fake``.

One fake-mode session runs through the real service, started by the real
``hal9000 kb serve --fake`` code path (``cli/serve.run``; only ``uvicorn.run``
is replaced by a driver that talks to the app it is handed), so the service
builds, seeds and mounts everything itself. The driver:

- asks questions that end answered (and a follow-up that is condensed),
  abstained, retrieval-only, with a failed grounding check, with the query
  embedding down, and cancelled by the client going away mid-answer;
- rates two answers; uploads a new file, the same bytes again and a broken
  PDF, and waits for the ingest jobs (the post-ingest regression runs too);
- runs a manual regression and writes an ``eval_runs`` row on a connection of
  its own, as ``kb eval regression`` and a sweep do;
- scrapes ``/metrics`` before and after, and runs every Grafana SQLite query
  against the live ``knowledge.db`` opened the way Grafana opens it
  (``mode=ro``), recording which table columns each query reads.

Counters are process-wide: the session first drops the labelled series other
tests made (a new process has none), and each "this happened" check is a
delta between the two scrapes. The one test-only change to the service is the reranker: the
ONNX cross-encoder can't be downloaded here, so ``knowledge.yaml`` names one
and the retriever is handed a word-overlap stand-in, so ``rerank`` is timed
like every other question stage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from hal9000.knowledge.auth import COOKIE
from hal9000.knowledge.cli import serve as serve_cli
from hal9000.knowledge.config import DEFAULT_PORT
from hal9000.knowledge.contracts import EvalMetrics, MetricCI
from hal9000.knowledge.eval.regression import run_regression
from hal9000.knowledge.eval.runs import record_run
from hal9000.knowledge.eval.wiring import retriever_for
from hal9000.knowledge.index.fake import enable_fake_mode
from hal9000.knowledge.obs import metrics
from hal9000.knowledge.obs.timing import INGEST_STAGES, QUESTION_STAGES
from hal9000.knowledge.retrieve.retriever import Retriever
from tests.knowledge.obs.test_dashboards import GRAFANA, PROM_QUERIES, REPO, SQL_QUERIES

SAME_ORIGIN = {"sec-fetch-site": "same-origin"}
PROFILE = serve_cli.FAKE_PROFILE

QUESTIONS = {
    "answered": "How old is the furnace?",
    "abstained": "Is there a swimming pool heater?",
    "retrieval_only": "retrieval only: how old is the furnace?",
    "unverified": "unverified: when was the furnace installed?",
    "embedding_down": "embedding down: how old is the furnace?",
    "cancelled": "Tell me everything about the furnace.",
}
FOLLOW_UP = "And when was it installed?"
NOTE = b"Synthetic gutter note.\n\nThe gutters were cleaned in spring 2024 by Example Gutters.\n"
BROKEN_PDF = b"%PDF-1.7\n" + b"\x00 not a real pdf body \x00" * 40

#: What Go's time.RFC3339 parses (fractional seconds allowed): frser-sqlite-datasource's
#: time columns are these strings or unix seconds.
RFC3339 = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:\d\d)")

#: Columns Grafana must never read (plan §9.3, §14): question and document text,
#: the trace body, names that can identify the house.
FORBIDDEN_COLUMNS = {
    "body", "question", "standalone_query", "content", "text", "header", "answer",
    "reference_answer", "evidence", "comment", "title", "original_name", "relative_path",
    "detail", "citations", "issues", "human_labels", "judge_labels",
}  # fmt: skip


class OverlapReranker:
    """A cross-encoder stand-in: the share of the query's words found in each passage."""

    name = "fake-overlap"

    def score(self, query: str, passages) -> list[float]:
        words = set(re.findall(r"\w+", query.lower()))
        return [
            len(words & set(re.findall(r"\w+", p.lower()))) / max(1, len(words)) for p in passages
        ]


@dataclass
class SqlResult:
    columns: list[str]
    rows: list[tuple]
    reads: set[tuple[str, str]]


@dataclass
class Live:
    before: dict[tuple[str, frozenset], float]
    after: dict[tuple[str, frozenset], float]
    families: dict
    exposition: str
    answers: dict[str, dict] = field(default_factory=dict)
    uploads: dict[str, str] = field(default_factory=dict)
    sql: dict[str, SqlResult] = field(default_factory=dict)
    trace_status: dict[str, int] = field(default_factory=dict)
    month_cap: float = 0.0
    regression: tuple | None = None
    ingest_regressions: int = 0
    regression_jobs: int = 0
    after_ingest: dict = field(default_factory=dict)

    def delta(self, name: str, **labels) -> float:
        key = (name, frozenset(labels.items()))
        return self.after.get(key, 0.0) - self.before.get(key, 0.0)

    def value(self, name: str, **labels) -> float | None:
        return self.after.get((name, frozenset(labels.items())))

    def samples(self, name: str) -> list[tuple[dict, float]]:
        return [(dict(labels), v) for (n, labels), v in self.after.items() if n == name]


# ---------------------------------------------------------------- the driver
def _parse(text: str) -> tuple[dict, dict]:
    samples, families = {}, {}
    for family in text_string_to_metric_families(text):
        families[family.name] = family
        for s in family.samples:
            samples[(s.name, frozenset(s.labels.items()))] = s.value
    return samples, families


def _sse(text: str) -> list[tuple[str, dict]]:
    out, event, data = [], None, []
    for line in [*text.splitlines(), ""]:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line.strip():
            if event and data:
                out.append((event, json.loads("\n".join(data))))
            event, data = None, []
    return out


def _chat(client: TestClient, message: str, **body) -> dict:
    response = client.post("/api/chat", json={"message": message, **body})
    assert response.status_code == 200, response.text
    return next(data for name, data in _sse(response.text) if name == "done")["answer"]


async def _chat_then_leave(app, token: str, message: str) -> bytes:
    """POST /api/chat at the ASGI level and disconnect after two tokens."""
    body = json.dumps({"message": message}).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/api/chat", "raw_path": b"/api/chat", "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 50000), "server": ("test", 80),
        "headers": [
            (b"host", b"test"), (b"cookie", f"{COOKIE}={token}".encode()),
            (b"sec-fetch-site", b"same-origin"), (b"content-type", b"application/json"),
        ],
    }  # fmt: skip
    received = bytearray()
    gone = asyncio.Event()
    sent = {"request": False}

    async def receive():
        if not sent["request"]:
            sent["request"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            received.extend(message.get("body", b""))
            if received.count(b"event: token") >= 2:
                gone.set()

    await asyncio.wait_for(app(scope, receive, send), 30)
    return bytes(received)


def _wait_for(check, what: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _grafana_connection(db_path: Path) -> sqlite3.Connection:
    """knowledge.db as the frser-sqlite-datasource opens it: ``file:...?mode=ro``."""
    conn = sqlite3.connect(f"file:{quote(str(db_path))}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = 1")
    return conn


def run_panel_sql(conn: sqlite3.Connection, query: str, now: datetime) -> SqlResult:
    """Run one panel query with Grafana's time macros; record every column it reads."""
    rendered = query.replace(
        "$__from", str(int((now - timedelta(hours=6)).timestamp() * 1000))
    ).replace("$__to", str(int((now + timedelta(minutes=5)).timestamp() * 1000)))
    assert "$" not in rendered, f"unhandled Grafana macro in {query}"
    reads: set[tuple[str, str]] = set()

    def authorize(action, arg1, arg2, db, source):
        if action == sqlite3.SQLITE_READ and arg1 and arg2:
            reads.add((arg1, arg2))
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    try:
        cursor = conn.execute(rendered)
        rows = cursor.fetchall()
    finally:
        conn.set_authorizer(None)
    return SqlResult([c[0] for c in cursor.description], rows, reads)


def _session(app, root: Path, live: dict) -> None:
    services = app.state.services
    client = TestClient(app, headers=SAME_ORIGIN)
    assert client.post("/api/session", json={"passcode": services.settings.passcode}).is_success
    token = client.cookies.get(COOKIE)
    live["month_cap"] = services.settings.monthly_cap_usd
    # the seeded library queued a regression check; the worker runs it at start
    assert services.worker.wait_idle(30), "the worker did not finish the seeded jobs"
    with sqlite3.connect(services.settings.db_path) as conn:
        seeded_regressions = conn.execute(
            "SELECT COUNT(*) FROM regression_runs WHERE trigger = 'ingest'"
        ).fetchone()[0]

    # the reranker: configured in knowledge.yaml (read per question), scored by the stand-in
    yaml_path = services.settings.yaml_path
    config = yaml.safe_load(yaml_path.read_text()) or {}
    yaml_path.write_text(yaml.safe_dump({**config, "reranker": "minilm"}))
    deps = services.answerer.deps
    deps.retriever_cls = partial(Retriever, reranker=OverlapReranker())

    live["before"] = client.get("/metrics").text

    answers = live["answers"]
    first = _chat(client, QUESTIONS["answered"])
    answers["answered"] = first
    answers["follow_up"] = _chat(client, FOLLOW_UP, conversation_id=first["conversation_id"])
    for key in ("abstained", "retrieval_only", "unverified", "embedding_down"):
        answers[key] = _chat(client, QUESTIONS[key])

    deps.stream_fn.delay_s = 0.05  # slow enough to leave mid-answer
    sent = asyncio.run(_chat_then_leave(app, token, QUESTIONS["cancelled"]))
    deps.stream_fn.delay_s = 0.0
    assert b"event: token" in sent and b"event: done" not in sent
    db_path = services.settings.db_path

    def cancelled_trace():
        with sqlite3.connect(db_path) as conn:
            return conn.execute("SELECT id FROM traces WHERE outcome = 'cancelled'").fetchone()

    _wait_for(cancelled_trace, "the cancelled question's trace")
    # the regressions below build their own retriever from knowledge.yaml, which
    # would load the real ONNX model: back to no reranker
    yaml_path.write_text(yaml.safe_dump(config))

    for key, rating in (("answered", 1), ("unverified", -1)):
        response = client.post(
            "/api/feedback", json={"message_id": answers[key]["message_id"], "rating": rating}
        )
        assert response.is_success, response.text

    files = {
        "Notes/gutters.txt": NOTE,
        "Notes/Copies/gutters-again.txt": NOTE,
        "Broken/broken.pdf": BROKEN_PDF,
    }
    parts = [("files", (Path(p).name, b, "application/octet-stream")) for p, b in files.items()]
    response = client.post("/api/uploads", files=parts, data={"relative_path": list(files)})
    assert response.is_success, response.text
    live["uploads"] = {r["relative_path"]: r["status"] for r in response.json()["results"]}
    job_ids = {r["job_id"] for r in response.json()["results"] if r["job_id"]}

    def jobs_finished():
        jobs = {j["id"]: j for j in client.get("/api/jobs").json()}
        return all(jobs.get(i, {}).get("status") in ("done", "failed") for i in job_ids)

    _wait_for(jobs_finished, "the ingest jobs")
    with sqlite3.connect(db_path) as conn:
        live["ingest_regressions"] = (
            conn.execute(
                "SELECT COUNT(*) FROM regression_runs WHERE trigger = 'ingest'"
            ).fetchone()[0]
            - seeded_regressions
        )
        live["regression_jobs"] = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = 'regression'"
        ).fetchone()[0]
    live["after_ingest"] = client.get("/metrics").text

    # what `kb eval regression` and a sweep write from their own processes
    conn = services.connect()
    try:
        binding = retriever_for(conn, services.settings, meter=None)
        run_regression(
            conn,
            binding.retrieve,
            profile_id=binding.profile_id,
            chunk_size=binding.chunk_size,
            trigger="manual",
        )
        record_run(
            conn,
            eval_dir=services.settings.eval_dir,
            stage="S9",  # later than any seeded stage, so it is the one exposed
            config={"profile": PROFILE, "chunker": "structure", "size": 64},
            metrics=EvalMetrics(
                n_questions=12,
                budgeted_recall={2000: MetricCI(value=0.75, low=0.6, high=0.9)},
                mrr=MetricCI(value=0.5, low=0.4, high=0.6),
            ),
            records=[],
        )
        row = conn.execute(
            "SELECT trigger, recall_2k FROM regression_runs ORDER BY at DESC, id DESC LIMIT 1"
        ).fetchone()
        live["regression"] = tuple(row) if row else None
    finally:
        conn.close()

    live["after"] = client.get("/metrics").text

    now = datetime.now(UTC)
    grafana = _grafana_connection(db_path)
    try:
        for where, query, _target in SQL_QUERIES:
            live["sql"][where] = run_panel_sql(grafana, query, now)
    finally:
        grafana.close()
    traces = live["sql"]["knowledge-live.json:Latest 50 traces"]
    for row in traces.rows:
        trace_id = row[traces.columns.index("id")]
        live["trace_status"][trace_id] = client.get(f"/api/traces/{trace_id}").status_code
    client.close()


def _as_a_fresh_process() -> None:
    """Drop the labelled series other tests created, as a new ``kb serve`` process has none.

    Without this a query could select a series some earlier test made (an
    ingestion error, another model's tokens), not one this session did.
    Unlabelled metrics are always present; deltas still measure the session.
    """
    for metric in (
        metrics.STAGE_SECONDS, metrics.QUESTIONS_TOTAL, metrics.ERRORS_TOTAL,
        metrics.TOKENS_TOTAL, metrics.COST_USD_TOTAL, metrics.GROUNDING_FAILURES_TOTAL,
        metrics.FEEDBACK_TOTAL, metrics.INGEST_FILES_TOTAL, metrics.INGEST_PAGES_TOTAL,
    ):  # fmt: skip
        metric.clear()
    metrics._start_closed_sets_at_zero()


@pytest.fixture(scope="module")
def live(tmp_path_factory) -> Live:
    root = tmp_path_factory.mktemp("live")
    _as_a_fresh_process()
    metrics.bind_month_budget(lambda: 99.0, 1.0)  # stale: kb serve --fake must replace it
    captured: dict = {"answers": {}, "sql": {}, "trace_status": {}}

    def drive(app, host, port, log_level, proxy_headers):
        _session(app, root / "fake", captured)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("uvicorn.run", drive)
        mp.setattr(serve_cli, "_temp_root", lambda: root / "fake")
        mp.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        try:
            assert serve_cli.run(argparse.Namespace(host="127.0.0.1", port=0, fake=True)) == 0
        finally:
            enable_fake_mode(False)
            metrics.install_db_collector(None)
    assert "after" in captured, "the session did not finish"
    before, _ = _parse(captured["before"])
    after_ingest, _ = _parse(captured["after_ingest"])
    after, families = _parse(captured["after"])
    return Live(
        before=before,
        after=after,
        families=families,
        exposition=captured["after"],
        answers=captured["answers"],
        uploads=captured["uploads"],
        sql=captured["sql"],
        trace_status=captured["trace_status"],
        month_cap=captured["month_cap"],
        regression=captured["regression"],
        ingest_regressions=captured["ingest_regressions"],
        regression_jobs=captured["regression_jobs"],
        after_ingest=after_ingest,
    )


# ------------------------------------------------- task 1: every §9.2 metric
def test_the_session_produced_every_outcome(live: Live):
    outcomes = {key: answer["outcome"] for key, answer in live.answers.items()}
    assert outcomes == {
        "answered": "answered",
        "follow_up": "answered",
        "abstained": "abstained",
        "retrieval_only": "retrieval_only",
        "unverified": "answered",
        "embedding_down": "answered",
    }
    assert live.answers["unverified"]["grounding"] == "failed"
    assert live.uploads == {
        "Notes/gutters.txt": "added",
        "Notes/Copies/gutters-again.txt": "duplicate",
        "Broken/broken.pdf": "added",  # accepted, then fails to parse in its job
    }


def test_every_plan_metric_is_exposed_with_exactly_its_labels(live: Live):
    for name, (kind, labels) in metrics.METRICS.items():
        family_name = name.removesuffix("_total") if kind == "counter" else name
        family = live.families.get(family_name)
        assert family is not None, f"{name} is not on /metrics"
        assert family.type == kind, (name, family.type)
        assert family.samples, f"{name} has no samples"
        for sample in family.samples:
            assert set(sample.labels) - {"le"} == set(labels), (name, sample.labels)


def test_questions_are_counted_by_outcome(live: Live):
    for outcome, n in (("answered", 4), ("abstained", 1), ("retrieval_only", 1), ("cancelled", 1)):
        assert live.delta("kb_questions_total", outcome=outcome) == n, outcome
    assert live.delta("kb_questions_total", outcome="error") == 0


def test_every_question_and_ingest_stage_is_timed(live: Live):
    for stage in (*QUESTION_STAGES, *INGEST_STAGES):
        assert live.delta("kb_stage_seconds_count", stage=stage.value) >= 1, stage.value
        assert live.delta("kb_stage_seconds_sum", stage=stage.value) > 0, stage.value
        assert live.delta("kb_stage_seconds_bucket", stage=stage.value, le="+Inf") >= 1


def test_per_question_histograms(live: Live):
    questions = 7  # six asked to the end, one cancelled
    assert live.delta("kb_answer_seconds_count") == questions
    assert live.delta("kb_answer_seconds_sum") > 0
    assert live.delta("kb_ttft_seconds_count") >= 5  # every answer that streamed a token
    assert live.delta("kb_retrieval_seconds_count") == questions
    # the abstain score needs a dense or rerank score: not the BM25-only retrieval-only one
    assert live.delta("kb_top_score_count") >= 5
    assert live.delta("kb_context_tokens_count") >= 4  # the answered questions
    assert live.delta("kb_context_tokens_sum") > 0


def test_tokens_and_cost_by_purpose(live: Live):
    tokens = defaultdict(float)
    for labels, _ in live.samples("kb_tokens_total"):
        tokens[labels["purpose"]] += live.delta("kb_tokens_total", **labels)
    for purpose in ("answer", "condense", "embed"):
        assert tokens[purpose] > 0, purpose
    costs = {(d["model"], d["purpose"]): v for d, v in live.samples("kb_cost_usd_total")}
    for labels, _ in live.samples("kb_tokens_total"):
        assert (labels["model"], labels["purpose"]) in costs  # fake models are free: 0
        assert labels["purpose"] in metrics.PURPOSES


def test_errors_are_counted_once_by_stage_and_kind(live: Live):
    assert live.delta("kb_errors_total", stage="embed_query", kind="RuntimeError") == 1
    total = sum(live.delta("kb_errors_total", **d) for d, _ in live.samples("kb_errors_total"))
    assert total == 2  # the embedding and the broken PDF, once each


def test_a_file_that_fails_to_parse_is_counted_under_parse(live: Live):
    assert live.delta("kb_errors_total", stage="parse", kind="ParseError") == 1
    assert live.delta("kb_errors_total", stage="request", kind="ParseError") == 0


def test_grounding_feedback_and_ingest_counters(live: Live):
    assert live.delta("kb_grounding_failures_total", kind="numeral") >= 1
    assert live.delta("kb_feedback_total", rating="up") == 1
    assert live.delta("kb_feedback_total", rating="down") == 1
    assert live.delta("kb_ingest_files_total", status="added") == 1
    assert live.delta("kb_ingest_files_total", status="duplicate") == 1
    assert live.delta("kb_ingest_files_total", status="failed") == 1
    assert live.delta("kb_ingest_pages_total", text_layer="yes") >= 1


def test_corpus_and_spend_gauges(live: Live):
    assert live.delta("kb_documents", profile=PROFILE) >= 1  # the new note
    assert live.delta("kb_chunks", profile=PROFILE) >= 1
    assert live.value("kb_db_bytes", profile="all") > 0
    assert live.value("kb_db_bytes", profile=PROFILE) > 0
    assert live.value("kb_month_spend_usd") == 0  # nothing is paid in fake mode
    assert live.value("kb_month_cap_usd") == live.month_cap > 0


def test_regression_recall_and_eval_metric_come_from_the_tables(live: Live):
    assert live.regression is not None and live.regression[0] == "manual"
    assert live.value("kb_regression_recall") == pytest.approx(live.regression[1])
    evals = {d["metric"]: v for d, v in live.samples("kb_eval_metric") if d["profile"] == PROFILE}
    assert evals["budgeted_recall@2000"] == 0.75 and evals["mrr"] == 0.5
    assert not any("run" in d for d, _ in live.samples("kb_eval_metric"))
    # kb serve --fake seeds 28 sweep runs over 25 named profiles; plus this session's
    profiles = {d["profile"] for d, _ in live.samples("kb_eval_metric")}
    assert len(profiles) == 26 and "pdfium.structure-512.hdr.fake-64" in profiles


def test_an_ingestion_runs_the_post_ingest_regression(live: Live):
    # the seeded question set has an approved answerable question: the upload (a file
    # and a broken PDF queued together) is checked once, after both jobs
    assert live.ingest_regressions == 1
    assert live.regression_jobs == 2  # the seeded library's check, and the upload's
    # and it retrieved without an error (test_errors_... counts every error of the session)
    assert live.delta("kb_stage_seconds_count", stage="regression") >= 2  # hook + manual run


def test_the_post_ingest_retrieval_stays_out_of_the_question_latencies(live: Live):
    """The upload's regression retrieved (vector, bm25, ...) inside its ingest trace."""
    assert live.ingest_regressions == 1

    def moved(stage: str) -> float:
        key = ("kb_stage_seconds_count", frozenset({("stage", stage)}))
        return live.after_ingest.get(key, 0.0) - live.before.get(key, 0.0)

    # five questions searched by vector (not the retrieval-only one or the one whose
    # embedding failed); the regression's search is not among them
    assert moved("vector") == 5
    assert moved("select") == 7  # every question selected, the regression did not
    assert moved("regression") >= 1  # the check itself is an ingestion stage


def test_no_question_or_document_text_on_metrics(live: Live):
    lowered = live.exposition.lower()
    for word in ("furnace", "swimming", "gutter", "installed"):
        assert word not in lowered, word


# ----------------------------------- task 2: every Grafana query, live
_SELECTOR = re.compile(r"([a-zA-Z_:][\w:]*)\s*\{([^}]*)\}")
_MATCHER = re.compile(r'(\w+)\s*(=~|!~|!=|=)\s*"([^"]*)"')


def _matches(labels: dict, label: str, op: str, value: str) -> bool:
    actual = labels.get(label, "")
    if op == "=":
        return actual == value
    if op == "!=":
        return actual != value
    hit = re.fullmatch(value, actual) is not None
    return hit if op == "=~" else not hit


def check_promql_live(expr: str, live: Live) -> None:
    """Every selector in ``expr`` selects live samples; every grouping label exists on them."""
    names = set()
    for name, body in _SELECTOR.findall(expr):
        names.add(name)
        samples = [labels for labels, _ in live.samples(name)]
        assert samples, f"{name} is not on the live /metrics ({expr})"
        matchers = _MATCHER.findall(body)
        selected = [s for s in samples if all(_matches(s, *m) for m in matchers)]
        assert selected, f"{name}{{{body}}} selects no live series ({expr})"
        for label, op, value in matchers:
            assert any(label in s for s in samples), f"{name} has no label {label}"
            if name.startswith("kb_stage_seconds") and op == "=~":
                for alternative in value.split("|"):  # every stage on a panel is live
                    assert any(s.get(label) == alternative for s in samples), alternative
    bare = re.sub(r'"[^"]*"|\{[^}]*\}|\[[^\]]*\]', "", expr)
    names |= {w for w in re.findall(r"\b(kb_\w+)\b", bare)}
    for name in names:
        assert live.samples(name), f"{name} is not on the live /metrics ({expr})"
    available = {label for name in names for labels, _ in live.samples(name) for label in labels}
    for group in re.findall(r"\b(?:by|without)\s*\(([^)]*)\)", bare):
        for label in filter(None, (g.strip() for g in group.split(","))):
            assert label in available, f"grouping by {label!r}: not on {names} ({expr})"


@pytest.mark.parametrize("where,expr", PROM_QUERIES, ids=[w for w, _ in PROM_QUERIES])
def test_every_prometheus_query_selects_live_series(live: Live, where, expr):
    check_promql_live(expr, live)


def test_the_live_promql_check_can_fail(live: Live):
    for bad in (
        'kb_questions_total{outcome="failed"}',
        'kb_stage_seconds_count{stage=~"vector|bogus"}',
        "sum by (profile) (rate(kb_questions_total[5m]))",
        "kb_not_a_metric",
    ):
        with pytest.raises(AssertionError):
            check_promql_live(bad, live)


@pytest.mark.parametrize("where", [w for w, _, _ in SQL_QUERIES])
def test_every_sqlite_panel_returns_rows_from_the_live_database(live: Live, where):
    result = live.sql[where]
    assert result.rows, f"{where} returned no rows after a full session"
    assert result.reads, where
    for table, column in result.reads:
        assert column not in FORBIDDEN_COLUMNS, f"{where} reads {table}.{column}"
    if "time" in result.columns:  # the plugin's time columns: epoch seconds or RFC 3339
        i = result.columns.index("time")
        for row in result.rows:
            value = row[i]
            assert isinstance(value, int) or RFC3339.fullmatch(value), (where, value)


def test_the_traces_table_lists_the_session_without_text_and_links_to_the_inspector(live: Live):
    result = live.sql["knowledge-live.json:Latest 50 traces"]
    assert {c for _, c in result.reads} <= {
        "id", "created_at", "kind", "outcome", "profile", "total_ms", "ttft_ms", "cost_usd",
    }  # fmt: skip
    outcomes = {row[result.columns.index("outcome")] for row in result.rows}
    assert {"answered", "abstained", "retrieval_only", "cancelled"} <= outcomes
    newest = result.rows[0][result.columns.index("time")]
    assert datetime.fromisoformat(newest) > datetime.now(UTC) - timedelta(minutes=10)
    assert live.trace_status and set(live.trace_status.values()) == {200}  # Query Inspector data


def test_the_cost_per_question_series_has_epoch_second_times(live: Live):
    result = live.sql["knowledge-live.json:Cost per question (hourly average)"]
    assert result.columns[0] == "time"
    now = datetime.now(UTC).timestamp()
    for when, cost in result.rows:
        assert isinstance(when, int) and now - 7 * 3600 < when <= now
        assert cost == 0  # fake models are free


def test_the_failed_jobs_panel_shows_the_broken_upload_by_error_kind(live: Live):
    result = live.sql["knowledge-ingestion.json:Failed jobs"]
    assert result.columns == ["time", "kind", "file", "error_kind"]
    assert [row[1:] for row in result.rows] == [("ingest", result.rows[0][2], "ParseError")]


def test_no_uploaded_file_name_reaches_a_sqlite_panel(live: Live):
    """The job error names the file ("Could not parse broken.pdf: ..."); Grafana must not."""
    names = {Path(p).name.lower() for p in live.uploads} | {Path(p).stem for p in live.uploads}
    assert {"broken.pdf", "gutters.txt", "gutters"} <= names
    for where, result in live.sql.items():
        for row in result.rows:
            for value in row:
                text = str(value).lower()
                assert not any(name in text for name in names), (where, value)


def test_the_regression_alert_query_reads_the_live_runs(live: Live):
    result = live.sql["alert:Knowledge: regression recall down after an ingestion"]
    assert result.columns == ["drop_points"] and len(result.rows) == 1
    assert isinstance(result.rows[0][0], int | float)


def test_the_panel_sql_check_can_fail(tmp_path):
    from hal9000.knowledge.storage import open_db

    open_db(tmp_path / "k.db").close()
    conn = _grafana_connection(tmp_path / "k.db")
    try:
        result = run_panel_sql(conn, "SELECT id, body FROM traces", datetime.now(UTC))
        assert ("traces", "body") in result.reads and not result.rows
        with pytest.raises(sqlite3.OperationalError):
            run_panel_sql(conn, "SELECT nope FROM traces", datetime.now(UTC))
        with pytest.raises(sqlite3.OperationalError):  # mode=ro: Grafana can't write
            conn.execute("DELETE FROM traces")
    finally:
        conn.close()


# ------------------------------------------- the service's address and routes
def test_the_inspector_link_and_scrape_target_match_the_service():
    live_dash = json.loads((GRAFANA / "dashboards" / "knowledge-live.json").read_text())
    kb_url = next(v for v in live_dash["templating"]["list"] if v["name"] == "kb_url")
    assert kb_url["query"].endswith(f":{DEFAULT_PORT}")  # KB_PORT's default
    app_routes = (REPO / "web" / "src" / "App.tsx").read_text()
    assert 'path="traces/:id"' in app_routes  # the Query Inspector route the link opens
    prometheus = yaml.safe_load((REPO / "deploy" / "prometheus" / "prometheus.yml").read_text())
    job = next(j for j in prometheus["scrape_configs"] if j["job_name"] == "hal-kb")
    assert job["static_configs"][0]["targets"] == [f"hal-kb:{DEFAULT_PORT}"]


def test_the_real_service_reports_the_ledgers_month_spend_and_cap(tmp_path, monkeypatch):
    """Without --fake the meter is bound: knowledge spend from the ledger, eval spend excluded."""
    from hal9000.knowledge.api import create_app
    from hal9000.knowledge.api.services import _meter, build_services
    from tests.knowledge.answer.conftest import make_settings

    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path / "hal"))
    settings = make_settings(tmp_path, monthly_cap_usd=0.5)
    services = build_services(settings, meter=_meter(settings), start_worker=False)
    try:
        client = TestClient(create_app(services, web_dist=False))
        ledger = services.meter.db
        ledger.add_ledger("earlier", "knowledge", {"model": "openai/gpt-5-mini"}, 0.2)
        ledger.add_ledger("sweep", "knowledge-eval", {"model": "openai/gpt-5-mini"}, 3.0)
        samples, _ = _parse(client.get("/metrics").text)
        assert samples[("kb_month_spend_usd", frozenset())] == pytest.approx(0.2)
        assert samples[("kb_month_cap_usd", frozenset())] == 0.5
        ledger.add_ledger("later", "knowledge", {"model": "openai/gpt-5-mini"}, 0.3)
        samples, _ = _parse(client.get("/metrics").text)  # read at scrape time
        assert samples[("kb_month_spend_usd", frozenset())] == pytest.approx(0.5)
    finally:
        services.close()
        metrics.use_configured_month_budget()
        metrics.install_db_collector(None)
