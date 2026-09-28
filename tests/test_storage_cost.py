"""SQLite persistence, and the cost meter and caps."""

from __future__ import annotations

import threading
from datetime import UTC, date

import pytest
from pydantic import ValidationError

from hal9000.contracts import NightlyRecord
from hal9000.core.prompt import Prompt
from hal9000.cost.metering import BudgetExceeded, Meter, monthly_precheck, usage_cost
from hal9000.solar.run_context import RunFlags
from hal9000.storage.db import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "hal.db")


def record(day="2026-09-22", kind="provisional", **kw):
    base = dict(
        date=day, kind=kind, production_kwh=10.0, expected_kwh=11.0, ratio=0.9, band="normal"
    )
    base.update(kw)
    return NightlyRecord(**base)


def test_records_are_append_only_and_latest_wins(db):
    db.append_record(record(production_kwh=9.0))
    db.append_record(record(kind="final", production_kwh=9.4))
    assert db.latest_record("2026-09-22").kind == "final"
    assert [r.production_kwh for r in db.record_versions("2026-09-22")] == [9.0, 9.4]
    db.append_record(record(day="2026-09-21"))
    recent = db.recent_records(date(2026, 9, 23), 7)
    assert [r.date for r in recent] == ["2026-09-22", "2026-09-21"]
    assert len(db.all_latest_records()) == 2
    assert db.latest_record("2020-01-01") is None


def test_malformed_record_is_rejected():
    with pytest.raises(ValidationError):
        NightlyRecord(
            date="2026-09-22", kind="draft", production_kwh=1, expected_kwh=1, ratio=1, band="x"
        )
    with pytest.raises(ValidationError):
        NightlyRecord(
            date="2026-09-22",
            kind="final",
            production_kwh=1,
            expected_kwh=1,
            ratio=1,
            band="x",
            surprise=1,
        )


def test_hourly_and_weather_roundtrip(db):
    day = date(2026, 9, 22)
    db.upsert_hourly(
        day, [{"solar": 1.0, "import": 0.5, "export": 0.2, "flags": ["x"]}] * 24, complete=True
    )
    rows = db.hourly(day)
    assert len(rows) == 24 and rows[0]["flags"] == ["x"] and rows[0]["complete"]
    assert db.hourly_days(date(2026, 9, 1), date(2026, 9, 30)) == ["2026-09-22"]
    assert db.hourly(date(2026, 9, 21)) is None
    db.upsert_weather(day, [{"poa": 0.5, "ghi": 0.4, "cloud": 10, "temp": 20}] * 24, "archive")
    assert db.weather(day)[0]["source"] == "archive"
    assert db.weather(date(2026, 9, 21)) is None


def test_alert_dedup_escalation_and_close(db):
    assert db.alert("production_low") is None
    db.open_alert("production_low", "warn", "Low today.")
    first = db.alert("production_low")
    db.open_alert("production_low", "critical", "Zero today.")
    escalated = db.alert("production_low")
    assert escalated["severity"] == "critical" and escalated["opened_at"] == first["opened_at"]
    assert [a["alert_key"] for a in db.open_alerts()] == ["production_low"]
    db.close_alert("production_low")
    assert db.open_alerts() == []
    assert db.alert("production_low")["status"] == "closed"


def test_ledger_run_and_month_totals(db):
    db.add_ledger("run1", "solar", {"model": "openai/gpt-5-mini", "prompt": 10}, 0.01)
    db.add_ledger("run1", "triage", {"model": "openai/gpt-5"}, 0.02)
    db.add_ledger("run2", "solar", {"model": "openai/gpt-5-mini"}, 0.5)
    assert db.run_cost("run1") == pytest.approx(0.03)
    from datetime import datetime

    now = datetime.now(UTC)
    assert db.month_cost(now.year, now.month) == pytest.approx(0.53)
    assert db.month_cost(2000, 1) == 0
    assert {r["agent"] for r in db.cost_breakdown(now.year, now.month)} == {"solar", "triage"}


def test_runs_and_kv(db):
    db.start_run("r1", date(2026, 9, 22))
    db.finish_run("r1", "ok", {"x": 1})
    assert db.last_success() is not None
    assert db.last_runs(1)[0]["run_id"] == "r1"
    db.set_kv("k", "v")
    db.set_kv("k", "w")
    assert db.get_kv("k") == "w" and db.get_kv("nope") is None


def test_concurrent_writes(db):
    def writer(i):
        for j in range(20):
            db.add_ledger(f"run{i}", "solar", {"model": "m"}, 0.001)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(db.run_cost(f"run{i}") for i in range(4)) == pytest.approx(0.08)


# --- cost -----------------------------------------------------------------------


def test_usage_cost_hand_calculation():
    usage = {"model": "openai/gpt-5-mini", "prompt": 10_000, "cached": 4_000, "completion": 1_000}
    expected = (6_000 * 0.25 + 4_000 * 0.025 + 1_000 * 2.0) / 1e6
    assert usage_cost(usage) == pytest.approx(expected, abs=1e-12)


def fake_generate(prompt, tier="powerful", *, on_usage=None):
    if on_usage:
        on_usage({"model": "openai/gpt-5", "prompt": 100_000, "cached": 0, "completion": 10_000})
    return "ok"


def test_meter_soft_and_hard_caps(db):
    flags = RunFlags()
    meter = Meter(db, "run1", soft_cap=0.2, hard_cap=0.4, flags=flags)
    wrapped = meter.wrap(fake_generate, "solar")
    wrapped(Prompt())  # 0.225
    assert flags.soft_cap_hit and meter.spent == pytest.approx(0.225)
    wrapped(Prompt(), "fast")  # 0.45
    with pytest.raises(BudgetExceeded):
        wrapped(Prompt())
    assert flags.hard_cap_hit
    assert db.run_cost("run1") == pytest.approx(0.45)


def test_two_agents_share_one_ledger(db):
    meter = Meter(db, "run1", 10, 10)
    meter.wrap(fake_generate, "solar")(Prompt())
    meter.wrap(fake_generate, "triage")(Prompt())
    assert meter.calls == 2
    assert {r["agent"] for r in db.cost_breakdown(*_ym())} == {"solar", "triage"}


def _ym():
    from datetime import datetime

    now = datetime.now(UTC)
    return now.year, now.month


def test_meter_passes_through_generators_without_usage(db):
    meter = Meter(db, "run1", 1, 1)
    assert meter.wrap(lambda prompt, tier="powerful": tier, "solar")(Prompt(), "fast") == "fast"
    assert meter.calls == 0


def test_monthly_precheck(db):
    today = date.today()
    status = monthly_precheck(db, today, 10.0)
    assert not status.skip_agent and status.triage_enabled
    db.add_ledger("r", "solar", {"model": "m"}, 8.5)
    status = monthly_precheck(db, today, 10.0)
    assert not status.skip_agent and not status.triage_enabled
    db.add_ledger("r", "solar", {"model": "m"}, 2.0)
    assert monthly_precheck(db, today, 10.0).skip_agent


def test_ledger_rows_use_local_time_so_month_end_runs_count_in_their_month(tmp_path):
    """20:00 on Sep 30 in California is Oct 1 in UTC; the spend belongs to September."""
    from datetime import datetime as real_datetime
    from zoneinfo import ZoneInfo

    from hal9000.cost import metering
    from hal9000.cost.metering import Meter
    from hal9000.storage.db import Database

    la = ZoneInfo("America/Los_Angeles")

    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 9, 30, 20, 0, tzinfo=la).astimezone(tz)

    db = Database(tmp_path / "l.db")
    import pytest

    mp = pytest.MonkeyPatch()
    mp.setattr(metering, "datetime", FrozenDatetime)
    try:
        Meter(db, "r", 1.0, 2.0, tz=la).record(
            {"model": "openai/gpt-5-mini", "prompt": 1000, "completion": 100}, "solar"
        )
    finally:
        mp.undo()
    assert db.month_cost(2026, 9) > 0
    assert db.month_cost(2026, 10) == 0
