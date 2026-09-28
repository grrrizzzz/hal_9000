"""Data quality end to end: missing buckets, partial days, config validation, alerts.

Missing data is never zero production. These tests pin that down at each layer:
the fold (which hours are missing), the assessment (missing hours leave the
comparison; too many make the day ``partial_data``), the record and hourly
store (flags persist, flagged days never calibrate the model) and the runner
(announce status is honest, recovery notices only clear what the day proves).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from hal9000 import config
from hal9000.coordinator.runner import RunOptions, _alert_class, open_database, run_evening
from hal9000.solar.domain.sitecfg import load_site
from hal9000.solar.energy import UNFIT_FLAGS, fold_statistics
from tests.fakes import FakeLLM, tool_call
from tests.fixture_builder import (
    EXPORT,
    IMPORT,
    REAL_2026_09_22,
    SOLAR,
    TZ,
    five_minute_points,
    fixture,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"
DAY = date(2026, 9, 22)
D = DAY.isoformat()


@pytest.fixture
def site():
    return load_site(CONFIG)


def _five_minute(day, series, drop_hours=(), keep=12):
    out = {}
    for entity, values in series.items():
        pts = []
        for t, v in five_minute_points(day, values):
            local = datetime.fromisoformat(t).astimezone(TZ)
            if local.hour in drop_hours and local.minute >= 5 * keep:
                continue
            pts.append((datetime.fromisoformat(t).astimezone(UTC), v))
        out[entity] = pts
    return out


# --- the fold -----------------------------------------------------------------


def test_full_coverage_has_no_missing_hours(hal_env, site):
    points = _five_minute(DAY, REAL_2026_09_22)
    energy = fold_statistics(points, DAY, config.settings(), site, "5minute")
    assert energy.missing_solar_hours == []
    assert "missing_data" not in energy.day_flags()


def test_hours_with_too_few_buckets_are_missing(hal_env, site):
    # hour 11 keeps 9 of 12 buckets (missing); hour 12 keeps 10 (fine)
    points = _five_minute(DAY, REAL_2026_09_22, drop_hours=(11,), keep=9)
    more = _five_minute(DAY, REAL_2026_09_22, drop_hours=(12,), keep=10)
    points[SOLAR] = [p for p in points[SOLAR] if p[0].astimezone(TZ).hour != 12] + [
        p for p in more[SOLAR] if p[0].astimezone(TZ).hour == 12
    ]
    energy = fold_statistics(points, DAY, config.settings(), site, "5minute")
    assert energy.missing_solar_hours == [11]
    assert "missing_data" in energy.day_flags()


def test_hours_after_through_are_not_missing(hal_env, site):
    points = _five_minute(DAY, REAL_2026_09_22)
    until = datetime(2026, 9, 22, 20, 0, tzinfo=TZ)
    points = {e: [p for p in v if p[0] < until.astimezone(UTC)] for e, v in points.items()}
    energy = fold_statistics(points, DAY, config.settings(), site, "5minute", until=until)
    assert energy.missing_solar_hours == []


def test_ceiling_applies_after_scaling(hal_env, site):
    """A pre-fix bucket just under the ceiling raw is over it once doubled."""
    ceiling = site.dc_kw * 1.2 * 5 / 60
    start = datetime(2026, 9, 22, 12, 0, tzinfo=TZ)
    points = {
        SOLAR: [((start + timedelta(minutes=5 * m)).astimezone(UTC), 0.01) for m in range(12)]
        + [((start + timedelta(hours=1)).astimezone(UTC), ceiling * 0.9)],
        IMPORT: [],
        EXPORT: [],
    }
    energy = fold_statistics(points, DAY, config.settings(), site, "5minute")
    assert any("exceeds" in a for a in energy.artifacts)
    assert "data_artifact" in energy.day_flags()


def test_unfit_flags_cover_the_bad_days():
    assert {"sensor_fault", "transition_day", "missing_data"} == set(UNFIT_FLAGS)


# --- config ---------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["2026-09-27T18:35:00", "not a date", "2026-13-01T00:00:00-07:00"])
def test_solar_fixed_at_must_be_offset_aware_iso(monkeypatch, raw):
    monkeypatch.setenv("SOLAR_FIXED_AT", raw)
    with pytest.raises(config.ConfigError) as excinfo:
        config.settings()
    assert "SOLAR_FIXED_AT" in str(excinfo.value)


def test_solar_fixed_at_accepts_offset(monkeypatch):
    monkeypatch.setenv("SOLAR_FIXED_AT", "2026-09-27T18:35:00-07:00")
    assert config.settings().solar_fixed_at.utcoffset() == timedelta(hours=-7)


# --- whole runs -------------------------------------------------------------------


def _fx_with_5min(drop_hours, keep):
    fx = fixture(
        DAY, REAL_2026_09_22[SOLAR], REAL_2026_09_22[IMPORT], REAL_2026_09_22[EXPORT], 0.97
    )
    result = {}
    for entity, values in REAL_2026_09_22.items():
        pts = []
        for t, v in five_minute_points(DAY, values, until_hour=20):
            local = datetime.fromisoformat(t).astimezone(TZ)
            if entity == SOLAR and local.hour in drop_hours and local.minute >= 5 * keep:
                continue
            pts.append([t, v])
        result[entity] = pts
    fx["ha"].insert(0, {"entities": [SOLAR, IMPORT, EXPORT], "period": "5minute", "result": result})
    return fx


def test_a_missing_midday_hour_is_left_out_not_counted_as_zero(hal_env, write_fixture):
    result = run_evening(RunOptions(replay=write_fixture(_fx_with_5min((12,), 0)), no_llm=True))
    assert result.band == "normal"
    assert "were missing and were left out" in result.report_text
    db = open_database(config.settings())
    record = db.latest_record(DAY)
    assert record.missing_solar_hours == 1
    assert "missing_data" in db.hourly(DAY)[0]["flags"]


def test_most_of_the_day_missing_is_partial_data(hal_env, write_fixture):
    fx = _fx_with_5min(tuple(range(9, 16)), 0)
    result = run_evening(RunOptions(replay=write_fixture(fx), no_llm=True))
    assert result.band == "partial_data"
    assert "too much of today's solar data is missing" in result.report_text
    assert "below expectation" not in result.report_text


def test_ha_failure_in_the_agent_path_is_no_data(hal_env, write_fixture):
    fx = fixture(DAY, REAL_2026_09_22[SOLAR], REAL_2026_09_22[IMPORT], REAL_2026_09_22[EXPORT])
    fx["ha_unavailable"] = True
    llm = FakeLLM([tool_call("get_home_energy", date=D)])
    result = run_evening(RunOptions(replay=write_fixture(fx), generate_response=llm))
    assert result.band == "no_data"
    assert "production data is missing" in result.report_text
    record = open_database(config.settings()).latest_record(DAY)
    assert record.production_missing and record.production_kwh is None


def test_announce_failure_everywhere_degrades(hal_env, write_fixture, monkeypatch):
    from hal9000.coordinator import delivery

    monkeypatch.setattr(delivery.Delivery, "announce", lambda self, text: "failed")
    result = run_evening(
        RunOptions(
            replay=write_fixture(
                fixture(
                    DAY,
                    REAL_2026_09_22[SOLAR],
                    REAL_2026_09_22[IMPORT],
                    REAL_2026_09_22[EXPORT],
                    0.97,
                )
            ),
            no_llm=True,
        )
    )
    assert result.announced is False
    assert result.status == "degraded"
    assert any("not delivered" in r for r in result.degraded_reasons)


def test_announce_telegram_fallback_is_announced_but_degraded(hal_env, write_fixture, monkeypatch):
    from hal9000.coordinator import delivery

    monkeypatch.setattr(delivery.Delivery, "announce", lambda self, text: "telegram")
    result = run_evening(
        RunOptions(
            replay=write_fixture(
                fixture(
                    DAY,
                    REAL_2026_09_22[SOLAR],
                    REAL_2026_09_22[IMPORT],
                    REAL_2026_09_22[EXPORT],
                    0.97,
                )
            ),
            no_llm=True,
        )
    )
    assert result.announced is True and result.status == "degraded"


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("production_low", "production"),
        ("inverter_suspected", "production"),
        ("output_zero", "production"),
        ("solar_sensor_fault", "sensor"),
        ("sensor_fault", "sensor"),
        ("net_position_trending", None),
        ("net_export", None),
        ("something_else", None),
    ],
)
def test_alert_classes(key, kind):
    assert _alert_class(key) == kind


def test_recovery_clears_production_and_sensor_but_not_net(hal_env, write_fixture):
    db = open_database(config.settings())
    for key in ("production_low", "solar_sensor_fault", "net_position_trending", "custom"):
        db.open_alert(key, "warn", "x")
    fx = fixture(
        DAY, REAL_2026_09_22[SOLAR], REAL_2026_09_22[IMPORT], REAL_2026_09_22[EXPORT], 0.97
    )
    result = run_evening(RunOptions(replay=write_fixture(fx), no_llm=True))
    assert result.band == "normal"
    still_open = {a["alert_key"] for a in open_database(config.settings()).open_alerts()}
    assert still_open == {"net_position_trending", "custom"}
    transcript = json.loads(open(result.transcript_path).read())
    texts = [o["text"] for o in transcript["outbox"]]
    assert any("sensor is reading normally" in t for t in texts)


def test_recovery_does_not_clear_on_a_partial_day(hal_env, write_fixture):
    db = open_database(config.settings())
    db.open_alert("production_low", "warn", "x")
    result = run_evening(RunOptions(replay=write_fixture(_fx_with_5min((12,), 0)), no_llm=True))
    assert result.band == "normal"
    assert [a["alert_key"] for a in open_database(config.settings()).open_alerts()] == [
        "production_low"
    ]


# Real Home Assistant hourly statistics for 2026-09-27, the day the Emporia
# multiplier was fixed at 18:35 (the 18:00 bucket holds the +3.26 kWh rescale
# jump). 23 hours: the run happened before the 23:00 bucket was compiled.
REAL_2026_09_27_SOLAR = [
    0, 0, 0, 0, 0, 0, 0.00053, 0.04558, 0.15726, 0.24478, 0.50518, 0.44407, 0.61794,
    0.44701, 0.32171, 0.21422, 0.18699, 0.07265, 3.26477, 0.00016, 0, 0, 0,
]  # fmt: skip
REAL_2026_09_27_IMPORT = [
    7.48389, 9.55392, 9.59273, 3.1718, 1.20256, 0.61851, 0.63701, 0.46332, 0.33187,
    0.15653, 0.02549, 0.44171, 0.7508, 0.4891, 0.45119, 1.08221, 1.00082, 1.11944,
    1.59418, 2.54842, 3.55345, 2.50826, 0.78583,
]  # fmt: skip


def test_real_transition_day(hal_env, write_fixture):
    day = date(2026, 9, 27)
    fx = fixture(day, [0.0] * 24, [0.0] * 24, [0.0] * 24, 0.9)
    start = datetime(2026, 9, 27, tzinfo=TZ)
    result = fx["ha"][0]["result"]
    for entity, values in ((SOLAR, REAL_2026_09_27_SOLAR), (IMPORT, REAL_2026_09_27_IMPORT)):
        result[entity] = [
            [(start + timedelta(hours=h)).astimezone(UTC).isoformat(), v]
            for h, v in enumerate(values)
        ]
    result[EXPORT] = result[EXPORT][:23]
    fx["now"] = "2026-09-28T08:00:00-07:00"
    run = run_evening(RunOptions(replay=write_fixture(fx), no_llm=True))
    # pre-fix hours doubled, the rescale jump dropped, the gap not counted as zero
    assert run.report_text.startswith("Solar report for Sunday. The panels produced 6.5")
    assert run.band not in ("sensor_fault", "partial_data", "zero")
    db = open_database(config.settings())
    flags = db.hourly(day)[0]["flags"]
    assert {"transition_day", "data_artifact", "missing_data"} <= set(flags)
