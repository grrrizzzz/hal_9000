"""Static checks for deploy/grafana and deploy/prometheus (handoff KWP8).

Every dashboard is well-formed; every Prometheus query names only metrics and
labels the knowledge registry exposes (with values from the closed label sets);
every SQLite query runs against a freshly migrated knowledge.db, so each table
and column exists; and no query reads question text.
"""

from __future__ import annotations

import configparser
import json
import re
from pathlib import Path

import pytest
import yaml

from hal9000.knowledge.obs import metrics
from hal9000.knowledge.obs.timing import Stage
from tests.knowledge.obs.conftest import seed_corpus

REPO = Path(__file__).resolve().parents[3]
GRAFANA = REPO / "deploy" / "grafana"
DASHBOARDS = sorted((GRAFANA / "dashboards").glob("*.json"))
RULES = GRAFANA / "provisioning" / "alerting" / "knowledge-rules.yaml"
PROM_UID, SQL_UID = "kb-prometheus", "kb-sqlite"

PROMQL_WORDS = {
    "sum", "avg", "min", "max", "count", "rate", "increase", "irate", "delta",
    "histogram_quantile", "clamp_min", "clamp_max", "max_over_time", "min_over_time",
    "avg_over_time", "by", "without", "on", "ignoring", "offset", "and", "or", "unless",
    "bool", "abs", "topk", "bottomk", "time", "vector", "scalar", "e",
}  # fmt: skip

#: (metric, label) -> the closed set a query's label value must come from.
CLOSED = {
    ("kb_stage_seconds", "stage"): {s.value for s in Stage},
    ("kb_errors_total", "stage"): set(metrics.ERROR_STAGES),
    ("kb_questions_total", "outcome"): set(metrics.OUTCOMES),
    ("kb_tokens_total", "purpose"): set(metrics.PURPOSES),
    ("kb_cost_usd_total", "purpose"): set(metrics.PURPOSES),
    ("kb_grounding_failures_total", "kind"): set(metrics.GROUNDING_KINDS),
    ("kb_feedback_total", "rating"): set(metrics.FEEDBACK_RATINGS),
    ("kb_ingest_files_total", "status"): set(metrics.INGEST_STATUSES),
    ("kb_ingest_pages_total", "text_layer"): {"yes", "no"},
}


def _series() -> dict[str, tuple[str, set[str]]]:
    """Exposed sample name -> (metric name, its labels)."""
    out = {}
    for name, (kind, labels) in metrics.METRICS.items():
        if kind == "histogram":
            out[f"{name}_bucket"] = (name, {*labels, "le"})
            out[f"{name}_sum"] = (name, set(labels))
            out[f"{name}_count"] = (name, set(labels))
        else:
            out[name] = (name, set(labels))
    return out


SERIES = _series()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _panels(dashboard: dict) -> list[dict]:
    out = []
    for panel in dashboard["panels"]:
        out.append(panel)
        out.extend(panel.get("panels", []))
    return out


def _targets():
    """(where, uid, target) for every dashboard target and alert query."""
    for path in DASHBOARDS:
        for panel in _panels(_load(path)):
            for target in panel.get("targets", []):
                uid = (target.get("datasource") or panel["datasource"])["uid"]
                yield f"{path.name}:{panel['title']}", uid, target
    for group in yaml.safe_load(RULES.read_text())["groups"]:
        for rule in group["rules"]:
            for item in rule["data"]:
                if item["datasourceUid"] != "__expr__":
                    yield f"alert:{rule['title']}", item["datasourceUid"], item["model"]


PROM_QUERIES = [(where, t["expr"]) for where, uid, t in _targets() if uid == PROM_UID]
SQL_QUERIES = [(where, t["queryText"], t) for where, uid, t in _targets() if uid == SQL_UID]


def check_promql(expr: str) -> None:
    """Raise AssertionError unless ``expr`` uses only exposed metrics, labels and values."""
    assert expr.count("(") == expr.count(")") and expr.count("{") == expr.count("}"), expr
    used: list[str] = []
    selectors = re.findall(r"([a-zA-Z_:][\w:]*)\s*\{([^}]*)\}", expr)
    for name, body in selectors:
        assert name in SERIES, f"unknown metric {name!r} in {expr}"
        metric, labels = SERIES[name]
        for label, op, value in re.findall(r'(\w+)\s*(=~|!~|!=|=)\s*"([^"]*)"', body):
            assert label in labels, f"{name} has no label {label!r} ({expr})"
            closed = CLOSED.get((metric, label))
            if closed is not None:
                values = set(value.split("|")) if "~" in op else {value}
                assert values <= closed, f"{label} values {values - closed} not in {closed}"
    bare = re.sub(r'"[^"]*"', '""', expr)
    bare = re.sub(r"\{[^}]*\}", "", bare)
    bare = re.sub(r"\[[^\]]*\]", "", bare)
    groupings = re.findall(r"\b(?:by|without)\s*\(([^)]*)\)", bare)
    bare = re.sub(r"\b(by|without)\s*\([^)]*\)", " ", bare)
    for word in re.findall(r"\b[a-zA-Z_:][\w:]*\b", bare):
        if word in PROMQL_WORDS:
            continue
        assert word in SERIES, f"unknown metric or function {word!r} in {expr}"
        used.append(word)
    used += [name for name, _ in selectors]
    assert used, f"no kb metric in {expr}"
    available = set().union(*(SERIES[name][1] for name in used))
    for group in groupings:
        for label in filter(None, (g.strip() for g in group.split(","))):
            assert label in available, f"grouping by {label!r} which {used} lack ({expr})"


def test_there_are_three_dashboards_with_the_plan_titles():
    titles = {_load(p)["title"] for p in DASHBOARDS}
    assert titles == {"Knowledge: Live", "Knowledge: Ingestion", "Knowledge: Quality"}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboard_json_is_well_formed(path):
    dashboard = _load(path)
    for key in ("uid", "title", "panels", "schemaVersion", "time", "templating"):
        assert key in dashboard, key
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", dashboard["uid"])
    assert "id" not in dashboard or dashboard["id"] is None  # provisioned: no fixed DB id
    panels = _panels(dashboard)
    assert panels
    assert len({p["id"] for p in panels}) == len(panels)
    assert len({p["title"] for p in panels}) == len(panels)
    for panel in panels:
        grid = panel["gridPos"]
        assert grid["w"] > 0 and grid["h"] > 0 and grid["x"] >= 0 and grid["x"] + grid["w"] <= 24
        assert panel["type"] in {"timeseries", "stat", "gauge", "bargauge", "table", "heatmap"}
        assert panel["datasource"]["uid"] in {PROM_UID, SQL_UID}
        targets = panel["targets"]
        assert targets, panel["title"]
        assert len({t["refId"] for t in targets}) == len(targets)
        for target in targets:
            assert target["datasource"]["uid"] == panel["datasource"]["uid"]
            if target["datasource"]["uid"] == PROM_UID:
                assert target["expr"].strip()
            else:
                assert target["queryText"] == target["rawQueryText"]
                assert target["queryType"] in {"table", "time series"}
    # panels don't overlap
    cells = set()
    for panel in panels:
        g = panel["gridPos"]
        mine = {
            (x, y) for x in range(g["x"], g["x"] + g["w"]) for y in range(g["y"], g["y"] + g["h"])
        }
        assert not cells & mine, panel["title"]
        cells |= mine


def test_dashboard_uids_are_unique():
    uids = [_load(p)["uid"] for p in DASHBOARDS]
    assert len(set(uids)) == len(uids)


def test_the_plan_panels_are_present():
    titles = {
        _load(p)["title"]: " | ".join(x["title"] for x in _panels(_load(p))).lower()
        for p in DASHBOARDS
    }
    live = titles["Knowledge: Live"]
    for phrase in (
        "retrieval latency p50", "retrieval latency p95", "retrieval latency p99",
        "end-to-end latency", "time to first token", "questions per hour by outcome",
        "error rate", "cost per question", "spend vs cap", "top-score distribution",
        "abstain rate", "grounding failures", "feedback", "latest 50 traces",
    ):  # fmt: skip
        assert phrase in live, phrase
    for phrase in (
        "files by status", "without a text layer", "parse and embed time per page",
        "chunks by profile", "database size", "failed jobs",
    ):  # fmt: skip
        assert phrase in titles["Knowledge: Ingestion"], phrase
    for phrase in ("regression recall", "latest sweep metrics by profile"):
        assert phrase in titles["Knowledge: Quality"], phrase


def test_prometheus_queries_exist():
    assert len(PROM_QUERIES) >= 20


@pytest.mark.parametrize("where,expr", PROM_QUERIES, ids=[w for w, _ in PROM_QUERIES])
def test_every_prometheus_query_uses_exposed_metrics_and_labels(where, expr):
    check_promql(expr)


def test_every_plan_metric_is_on_a_dashboard_or_alert():
    text = " ".join(expr for _, expr in PROM_QUERIES)
    for name in metrics.METRICS:
        assert re.search(rf"\b{name}(_bucket|_sum|_count)?\b", text), name


@pytest.mark.parametrize(
    "bad",
    [
        "rate(kb_questons_total[5m])",
        'kb_questions_total{outcom="error"}',
        'kb_questions_total{outcome="failed"}',
        'kb_stage_seconds_count{stage=~"vector|retrieve"}',
        "sum by (profile) (kb_questions_total)",
        "sum(rate(kb_questions_total[5m])",
        "vector(1)",
    ],
)
def test_the_promql_check_can_fail(bad):
    with pytest.raises(AssertionError):
        check_promql(bad)


def _run_sql(conn, query: str) -> list:
    rendered = query.replace("$__from", "1700000000000").replace("$__to", "1900000000000")
    assert "$" not in rendered, f"unhandled Grafana macro in {query}"
    return conn.execute(rendered).fetchall()


def test_sqlite_queries_exist():
    assert len(SQL_QUERIES) >= 5


@pytest.mark.parametrize("where,query,target", SQL_QUERIES, ids=[w for w, _, _ in SQL_QUERIES])
def test_every_sqlite_query_runs_against_the_schema(kb_db, where, query, target):
    seed_corpus(kb_db)
    with kb_db:
        kb_db.execute(
            "INSERT INTO traces (id, created_at, outcome, total_ms, ttft_ms, cost_usd, body)"
            " VALUES ('t1', '2026-09-01T00:00:00.000+00:00', 'answered', 1200, 300, 0.001, '{}')"
        )
    _run_sql(kb_db, query)  # sqlite3 raises on an unknown table or column
    lowered = re.sub(r"'[^']*'", "''", query.lower())  # string literals aren't columns
    assert not re.search(r"\b(body|question|standalone_query|content|text)\b", lowered), (
        "no question or document text in Grafana"
    )
    if target.get("queryType") == "time series":
        assert re.search(r"\bas time\b", lowered)


def test_the_traces_table_panel_shows_the_plan_columns(kb_db):
    live = _load(GRAFANA / "dashboards" / "knowledge-live.json")
    panel = next(p for p in live["panels"] if p["title"] == "Latest 50 traces")
    with kb_db:
        kb_db.execute(
            "INSERT INTO traces (id, created_at, outcome, total_ms, cost_usd, body)"
            " VALUES ('abc', '2026-09-01T00:00:00.000+00:00', 'answered', 900, 0.002, '{\"question\": \"q\"}')"
        )
    cursor = kb_db.execute(panel["targets"][0]["queryText"])
    columns = [c[0] for c in cursor.description]
    assert {"time", "outcome", "latency_ms", "cost_usd", "id"} <= set(columns)
    links = [
        prop["value"][0]["url"]
        for override in panel["fieldConfig"]["overrides"]
        if override["matcher"]["options"] == "id"
        for prop in override["properties"]
        if prop["id"] == "links"
    ]
    assert links == ["${kb_url}/traces/${__value.raw}"]
    assert any(v["name"] == "kb_url" for v in live["templating"]["list"])


# ------------------------------------------------------------------ alert rules
def _rules() -> list[dict]:
    return [r for g in yaml.safe_load(RULES.read_text())["groups"] for r in g["rules"]]


def test_there_are_four_alert_rules_matching_the_plan():
    rules = {r["uid"]: r for r in _rules()}
    assert set(rules) == {"kb-error-rate", "kb-latency-p95", "kb-month-spend", "kb-regression-drop"}

    def threshold(rule):
        cond = next(d for d in rule["data"] if d["refId"] == rule["condition"])
        return cond["model"]["conditions"][0]["evaluator"]

    assert threshold(rules["kb-error-rate"]) == {"type": "gt", "params": [0.1]}
    assert rules["kb-error-rate"]["for"] == "15m"
    assert threshold(rules["kb-latency-p95"]) == {"type": "gt", "params": [15]}
    assert rules["kb-latency-p95"]["for"] == "30m"
    assert "kb_answer_seconds_bucket" in rules["kb-latency-p95"]["data"][0]["model"]["expr"]
    assert threshold(rules["kb-month-spend"]) == {"type": "gt", "params": [0.8]}
    assert threshold(rules["kb-regression-drop"]) == {"type": "gt", "params": [5]}
    for rule in rules.values():
        refs = {d["refId"] for d in rule["data"]}
        assert rule["condition"] in refs
        for item in rule["data"]:
            expression = item["model"].get("expression")
            assert expression is None or expression in refs


def _drop_points(conn) -> float:
    rule = next(r for r in _rules() if r["uid"] == "kb-regression-drop")
    return _run_sql(conn, rule["data"][0]["model"]["queryText"])[0][0]


def test_the_regression_alert_compares_the_latest_ingest_run_with_the_previous(kb_db):
    ids = seed_corpus(kb_db)
    pid = ids["fixed-512"]

    def run(at, recall, trigger="ingest", profile=pid):
        with kb_db:
            kb_db.execute(
                "INSERT INTO regression_runs (at, profile_id, trigger, recall_2k) VALUES (?, ?, ?, ?)",
                (at, profile, trigger, recall),
            )

    assert _drop_points(kb_db) == 0
    run("2026-09-01T00:00:00+00:00", 0.90)
    assert _drop_points(kb_db) == 0  # nothing to compare with
    run("2026-09-02T00:00:00+00:00", 0.87)
    assert _drop_points(kb_db) == pytest.approx(3)
    run("2026-09-03T00:00:00+00:00", 0.80)
    assert _drop_points(kb_db) == pytest.approx(7)  # fires (> 5)
    run("2026-09-04T00:00:00+00:00", 0.70, trigger="manual")
    assert _drop_points(kb_db) == 0  # only after an ingestion
    run("2026-09-05T00:00:00+00:00", 0.95, profile=ids["recursive-1024"])
    run("2026-09-06T00:00:00+00:00", 0.60, profile=ids["recursive-1024"])
    assert _drop_points(kb_db) == pytest.approx(35)  # profiles compared with themselves


# ------------------------------------------------------ datasources and servers
def test_datasources_open_knowledge_db_read_only_and_match_the_dashboards():
    doc = yaml.safe_load((GRAFANA / "provisioning" / "datasources" / "knowledge.yaml").read_text())
    by_uid = {d["uid"]: d for d in doc["datasources"]}
    assert set(by_uid) == {PROM_UID, SQL_UID}
    sqlite = by_uid[SQL_UID]
    assert sqlite["type"] == "frser-sqlite-datasource"
    assert sqlite["jsonData"]["path"] == "/data/knowledge.db"
    assert "mode=ro" in sqlite["jsonData"]["pathOptions"]
    assert by_uid[PROM_UID]["url"] == "http://prometheus:9090"
    used = {uid for _, uid, _ in _targets()}
    assert used <= set(by_uid)


def test_anonymous_access_is_off():
    ini = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    ini.read(GRAFANA / "grafana.ini")
    assert ini.getboolean("auth.anonymous", "enabled") is False
    assert ini.getboolean("users", "allow_sign_up") is False
    assert "frser-sqlite-datasource" in ini.get("plugins", "preinstall")
    assert ini.get("server", "http_port") == "3001"
    assert not ini.has_option("security", "admin_password")  # it comes from .env


def test_the_dashboard_provider_loads_the_dashboards_folder():
    doc = yaml.safe_load((GRAFANA / "provisioning" / "dashboards" / "knowledge.yaml").read_text())
    provider = doc["providers"][0]
    assert provider["type"] == "file" and provider["options"]["path"] == "/etc/grafana/dashboards"


def test_telegram_is_optional_and_holds_no_secret():
    text = (GRAFANA / "optional" / "telegram.yaml").read_text()
    doc = yaml.safe_load(text)
    settings = doc["contactPoints"][0]["receivers"][0]["settings"]
    assert settings["bottoken"] == "${TELEGRAM_BOT_TOKEN}"
    assert settings["chatid"] == "${TELEGRAM_CHAT_ID}"
    assert not re.search(r"\d{6,}:[A-Za-z0-9_-]{20,}", text)  # no literal bot token
    assert not list((GRAFANA / "provisioning").rglob("telegram*"))


def test_prometheus_scrapes_hal_kb():
    doc = yaml.safe_load((REPO / "deploy" / "prometheus" / "prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in doc["scrape_configs"]}
    job = jobs["hal-kb"]
    assert job["metrics_path"] == "/metrics"
    assert job["static_configs"][0]["targets"] == ["hal-kb:8910"]
