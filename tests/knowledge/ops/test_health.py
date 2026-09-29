"""``data/knowledge_health.json`` (plan §16): fields, atomic writes, refresh on ingestion."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from hal9000.knowledge import health
from hal9000.knowledge.contracts import Job
from hal9000.knowledge.ingest.jobs import ProgressBus
from hal9000.knowledge.storage import open_db


def test_the_body_has_the_plan_fields_and_no_question_text(data_dir):
    conn = open_db(data_dir / "knowledge.db")
    try:
        with conn:
            conn.execute(
                "INSERT INTO traces (id, created_at, kind, outcome, body) VALUES "
                "('t1', '2026-09-01T10:00:00+00:00', 'question', 'answered', "
                '\'{"question": "How old is the furnace?"}\'), '
                "('t2', '2026-09-02T10:00:00+00:00', 'question', 'abstained', '{}')"
            )
        body = health.health_body(conn, month_spend=1.25, month_cap=10.0)
    finally:
        conn.close()
    assert set(body) == {
        "updated_at",
        "last_successful_question_at",
        "last_ingestion_at",
        "month_spend_usd",
        "month_cap_usd",
    }
    assert body["last_successful_question_at"] == "2026-09-01T10:00:00+00:00"
    assert body["last_ingestion_at"]  # the fixture library's ingestion jobs
    assert (body["month_spend_usd"], body["month_cap_usd"]) == (1.25, 10.0)
    assert "furnace" not in json.dumps(body)


def test_last_ingestion_ignores_failed_and_other_jobs(kb_db):
    now = datetime.now(UTC).isoformat()
    with kb_db:
        kb_db.execute(
            "INSERT INTO jobs (kind, status, created_at, updated_at) VALUES "
            "('ingest', 'done', ?, '2026-01-01T00:00:00'), "
            "('ingest', 'failed', ?, '2026-03-01T00:00:00'), "
            "('reindex', 'done', ?, '2026-04-01T00:00:00')",
            (now, now, now),
        )
    assert health.last_ingestion_at(kb_db) == "2026-01-01T00:00:00"
    assert health.last_question_at(kb_db) is None


def test_the_file_is_written_atomically(tmp_path, kb_db):
    body = health.write_health(kb_db, tmp_path / "data", month_spend=None, month_cap=10.0)
    written = json.loads((tmp_path / "data" / health.HEALTH_FILE).read_text())
    assert written == body and written["month_spend_usd"] is None
    assert not list((tmp_path / "data").glob("*.tmp"))


def test_an_unwritable_data_dir_is_logged_not_raised(tmp_path, kb_db, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    body = health.write_health(kb_db, blocker, month_spend=0.0, month_cap=10.0)
    assert body["month_cap_usd"] == 10.0 and "could not write" in caplog.text


def test_a_finished_ingestion_refreshes_the_file():
    bus = ProgressBus()
    calls = []
    unsubscribe = health.watch_ingestions(bus, lambda: calls.append(1))
    bus.publish(Job(kind="ingest", status="running"))
    bus.publish(Job(kind="reindex", status="done"))
    bus.publish(Job(kind="ingest", status="failed"))
    assert calls == []
    bus.publish(Job(kind="ingest", status="done"))
    assert calls == [1]
    unsubscribe()
    bus.publish(Job(kind="ingest", status="done"))
    assert calls == [1]


def test_a_failing_refresh_never_breaks_the_job(caplog):
    bus = ProgressBus()

    def broken():
        raise OSError("read-only")

    health.watch_ingestions(bus, broken)
    bus.publish(Job(kind="ingest", status="done"))
    assert "could not refresh" in caplog.text


def test_the_service_refreshes_the_file_when_an_ingestion_finishes(data_dir, monkeypatch):
    """KWP6's service subscribes its health file to the job bus (and unsubscribes on close)."""
    from hal9000.knowledge.api.services import build_services
    from hal9000.knowledge.config import settings
    from hal9000.knowledge.ingest.jobs import BUS

    monkeypatch.setenv("KB_PASSCODE", "synthetic-passcode")
    services = build_services(settings(), fake=True, start_worker=False)
    target = data_dir / health.HEALTH_FILE
    try:
        assert not target.exists()
        BUS.publish(Job(kind="ingest", status="done"))
        written = json.loads(target.read_text())
        assert written["last_ingestion_at"] and written["month_spend_usd"] == 0.0
    finally:
        services.close()
    target.unlink()
    BUS.publish(Job(kind="ingest", status="done"))
    assert not target.exists()
