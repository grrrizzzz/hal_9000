"""Scrape-time gauges from knowledge.db, including rows written by another process."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from hal9000.knowledge.obs import collector as collector_mod
from hal9000.knowledge.obs import metrics
from hal9000.knowledge.obs.collector import (
    KnowledgeDbCollector,
    eval_profile,
    flatten_eval_metrics,
)
from tests.knowledge.obs.conftest import add_eval_run, seed_corpus

REPO = Path(__file__).resolve().parents[3]


def _samples(name: str) -> dict[tuple, float]:
    """``{sorted label items: value}`` for ``name`` from a full /metrics render."""
    text = metrics.render_metrics(process=False).decode()
    out = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name:
                out[tuple(sorted(sample.labels.items()))] = sample.value
    return out


def _write_regression_row_in_another_process(db: Path, at: str, recall: float) -> None:
    code = textwrap.dedent(
        f"""
        from hal9000.knowledge.storage import open_db
        conn = open_db({str(db)!r})
        with conn:
            conn.execute(
                "INSERT INTO regression_runs (at, trigger, recall_2k, recall_low, recall_high,"
                " n_questions) VALUES (?, 'ingest', ?, ?, ?, 40)",
                ({at!r}, {recall!r}, {recall - 0.05!r}, {recall + 0.05!r}),
            )
        conn.close()
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr


def test_a_regression_row_from_another_process_appears_on_the_next_scrape(db_path, kb_db):
    metrics.install_db_collector(db_path)
    assert _samples("kb_regression_recall") == {}

    _write_regression_row_in_another_process(db_path, "2026-09-01T10:00:00+00:00", 0.82)
    assert _samples("kb_regression_recall") == {(): 0.82}

    # the latest run wins, whatever order the rows arrive in
    _write_regression_row_in_another_process(db_path, "2026-09-02T10:00:00+00:00", 0.74)
    _write_regression_row_in_another_process(db_path, "2026-08-01T10:00:00+00:00", 0.99)
    assert _samples("kb_regression_recall") == {(): 0.74}


def test_eval_metric_is_the_latest_finished_run_per_profile_without_a_run_label(db_path, kb_db):
    ci = lambda v: {"value": v, "low": v - 0.1, "high": v + 0.1}  # noqa: E731
    add_eval_run(
        kb_db,
        "old",
        "fixed-512",
        {"n_questions": 40, "mrr": ci(0.3)},
        finished="2026-09-01T00:00:00+00:00",
    )
    add_eval_run(
        kb_db,
        "new",
        "fixed-512",
        {
            "n_questions": 40,
            "budgeted_recall": {"1000": ci(0.6), "2000": ci(0.7)},
            "mrr": ci(0.5),
            "abstain_auc": None,
            "latency_ms_p95": 120.0,
            "by_category": {"money": {"recall": 0.2}},
        },
        finished="2026-09-02T00:00:00+00:00",
    )
    add_eval_run(kb_db, "running", "fixed-512", {"n_questions": 1, "mrr": ci(0.9)}, finished=None)
    add_eval_run(
        kb_db,
        "other",
        {"name": "recursive-1024", "size": 1024},
        {"n_questions": 10, "mrr": ci(0.4)},
    )
    add_eval_run(kb_db, "no-profile", None, {"n_questions": 10, "mrr": ci(0.1)})
    with kb_db:
        kb_db.execute(
            "INSERT INTO eval_runs (id, stage, config, metrics, started_at, finished_at)"
            " VALUES ('broken', 'S1a', '{', 'nope', '2026-09-03', '2026-09-03')"
        )
    metrics.install_db_collector(db_path)
    got = _samples("kb_eval_metric")
    assert got == {
        (("metric", "budgeted_recall@1000"), ("profile", "fixed-512")): 0.6,
        (("metric", "budgeted_recall@2000"), ("profile", "fixed-512")): 0.7,
        (("metric", "latency_ms_p95"), ("profile", "fixed-512")): 120.0,
        (("metric", "mrr"), ("profile", "fixed-512")): 0.5,
        (("metric", "n_questions"), ("profile", "fixed-512")): 40.0,
        (("metric", "mrr"), ("profile", "recursive-1024")): 0.4,
        (("metric", "n_questions"), ("profile", "recursive-1024")): 10.0,
    }


def test_corpus_gauges_per_profile(db_path, kb_db):
    ids = seed_corpus(kb_db)
    kb_db.execute(f"CREATE VIRTUAL TABLE fts_{ids['fixed-512']} USING fts5(header, text)")
    kb_db.execute(
        f"INSERT INTO fts_{ids['fixed-512']} VALUES ('h', ?)", ("synthetic words " * 500,)
    )
    kb_db.commit()
    metrics.install_db_collector(db_path)
    assert _samples("kb_chunks") == {
        (("profile", "fixed-512"),): 3.0,
        (("profile", "recursive-1024"),): 6.0,
    }
    assert _samples("kb_documents") == {
        (("profile", "fixed-512"),): 1.0,
        (("profile", "recursive-1024"),): 2.0,
    }
    sizes = _samples("kb_db_bytes")
    assert sizes[(("profile", "all"),)] >= db_path.stat().st_size > 0
    assert sizes[(("profile", "fixed-512"),)] > 0  # its FTS5 shadow tables, via dbstat
    assert (("profile", "recursive-1024"),) not in sizes  # it has no search tables yet


def test_dbstat_is_cached_and_optional(db_path, kb_db):
    seed_corpus(kb_db)
    reader = KnowledgeDbCollector(db_path, dbstat_ttl_s=3600)
    conn = reader._connect()
    first = reader._profile_bytes(conn)
    calls = []

    class NoDbstat:
        def execute(self, sql, *args):
            calls.append(sql)
            raise sqlite3.OperationalError("no such table: dbstat")

    assert reader._profile_bytes(NoDbstat()) is first  # cached: not queried again
    assert calls == []
    uncached = KnowledgeDbCollector(db_path, dbstat_ttl_s=0)
    assert uncached._profile_bytes(NoDbstat()) == {}
    assert calls
    conn.close()


def test_a_missing_or_unmigrated_database_gives_empty_families(tmp_path):
    registry = CollectorRegistry()
    registry.register(KnowledgeDbCollector(tmp_path / "absent.db"))
    names = {m.name: m.samples for m in registry.collect()}
    assert set(names) == {
        "kb_regression_recall",
        "kb_eval_metric",
        "kb_documents",
        "kb_chunks",
        "kb_db_bytes",
    }
    assert all(not samples for samples in names.values())

    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    registry = CollectorRegistry()
    registry.register(KnowledgeDbCollector(empty))
    got = {m.name: m.samples for m in registry.collect()}
    assert [s.labels for s in got["kb_db_bytes"]] == [{"profile": "all"}]
    assert not got["kb_regression_recall"] and not got["kb_chunks"]


def test_an_unreadable_database_does_not_break_the_scrape(tmp_path, caplog):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"this is not a sqlite database at all" * 100)
    registry = CollectorRegistry()
    registry.register(KnowledgeDbCollector(bad))
    assert {m.name for m in registry.collect()} >= {"kb_regression_recall"}
    assert "knowledge.db metrics" in caplog.text


def test_open_failure_is_logged(tmp_path, caplog, monkeypatch):
    reader = KnowledgeDbCollector(tmp_path / "x.db")

    def refuse():
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(reader, "_connect", refuse)
    assert len(list(reader.collect())) == 5
    assert "cannot open" in caplog.text


def test_the_database_is_opened_read_only(db_path, kb_db):
    conn = KnowledgeDbCollector(db_path)._connect()
    try:
        try:
            conn.execute("DELETE FROM traces")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc) or "read-only" in str(exc) or "query_only" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("a write succeeded on the metrics connection")
    finally:
        conn.close()


def test_flatten_and_profile_helpers():
    assert flatten_eval_metrics(None) == {}
    assert flatten_eval_metrics({"mrr": {"value": True}, "x": "text", "k": {"1": "a"}}) == {}
    assert eval_profile("fixed") is None
    assert eval_profile({"profile": ""}) is None
    assert eval_profile({"profile": {"name": "p"}}) == "p"
    assert collector_mod.DBSTAT_TTL_S > 0


def test_eval_metric_compares_dev_runs_of_the_latest_stage(db_path, kb_db):
    ci = lambda v: {"value": v, "low": v, "high": v}  # noqa: E731
    # a later-finished S1b run doesn't beat an S2 run; S2 beats S1a; test split never shows
    add_eval_run(
        kb_db, "s2", "p", {"mrr": ci(0.6)}, stage="S2", finished="2026-09-02T00:00:00+00:00"
    )
    add_eval_run(
        kb_db, "s1b", "p", {"mrr": ci(0.9)}, stage="S1b", finished="2026-09-05T00:00:00+00:00"
    )
    add_eval_run(
        kb_db, "s2-rerun", "p", {"mrr": ci(0.65)}, stage="S2", finished="2026-09-03T00:00:00+00:00"
    )
    add_eval_run(
        kb_db,
        "s5-test",
        "p",
        {"mrr": ci(0.1)},
        stage="S5",
        split="test",
        finished="2026-09-09T00:00:00+00:00",
    )
    add_eval_run(kb_db, "only-test", "q", {"mrr": ci(0.2)}, split="test")
    metrics.install_db_collector(db_path)
    assert _samples("kb_eval_metric") == {(("metric", "mrr"), ("profile", "p")): 0.65}


def test_stage_order():
    from hal9000.knowledge.obs.collector import stage_order

    stages = ["S2", "S10", "S1b", "custom", "S1a", "s3"]
    assert sorted(stages, key=stage_order) == ["custom", "S1a", "S1b", "S2", "s3", "S10"]
    assert stage_order(None) < stage_order("S1a")


def test_the_scrape_connection_uses_a_mode_ro_uri(db_path, kb_db, monkeypatch):
    reader = KnowledgeDbCollector(db_path)
    assert reader.uri().startswith("file:") and reader.uri().endswith("?mode=ro")
    calls = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        calls.append((target, kwargs.get("uri")))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(collector_mod.sqlite3, "connect", spy)
    list(reader.collect())
    assert calls == [(reader.uri(), True)]
