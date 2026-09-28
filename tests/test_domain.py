"""Clear-sky model, expectation models, bands, anomaly profiles, net position, formatting."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from hal9000.solar.domain import fmt
from hal9000.solar.domain.anomaly import band_for, classify_profile
from hal9000.solar.domain.clearsky import clearsky_poa_hourly, daylight_hours
from hal9000.solar.domain.expectation import (
    ExpectationModel,
    bootstrap_model,
    fit_model,
)
from hal9000.solar.domain.net_position import day_net_position, window_net_position
from hal9000.solar.domain.sitecfg import load_site, load_thresholds
from hal9000.solar.domain.tariff import load_tariff

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def site():
    return load_site(CONFIG)


@pytest.fixture(scope="module")
def thresholds():
    return load_thresholds(CONFIG)


def test_site_config(site):
    assert site.dc_kw == 2.47 and site.tilt_deg == 33 and site.azimuth_deg == 180
    assert site.openmeteo_azimuth == 0  # Open-Meteo: 0 = south
    assert site.max_kwh_per_5min == pytest.approx(2.47 * 1.2 * 5 / 60)


def test_clearsky_solstice_sanity_band(site):
    poa = clearsky_poa_hourly(site, date(2026, 6, 21))
    energy = sum(bootstrap_model(site).predict_hourly(poa, [20.0] * 24))
    assert 11 <= energy <= 16
    assert poa[3] == 0 and poa[22] == 0
    assert daylight_hours(site, date(2026, 6, 21))[12] is True
    assert daylight_hours(site, date(2026, 6, 21))[2] is False


def test_production_after_20_00_is_negligible_all_year(site):
    """Validates the 20:00 run time: < 1% of the day's clear-sky energy comes later."""
    day = date(2026, 1, 1)
    while day.year == 2026:
        poa = clearsky_poa_hourly(site, day)
        assert sum(poa[20:]) < 0.01 * sum(poa), day
        day += timedelta(days=14)


def test_bootstrap_model(site):
    model = bootstrap_model(site)
    assert model.kind == "bootstrap"
    assert model.a == pytest.approx(2.47 * 0.85 * 0.945)
    hot = model.predict_hourly([1.0], [40.0])[0]
    cool = model.predict_hourly([1.0], [10.0])[0]
    assert hot < cool  # hotter cells produce less
    assert model.predict_hourly([-0.1], [20.0]) == [0.0]
    roundtrip = ExpectationModel.from_dict(model.to_dict())
    assert roundtrip.a == model.a and roundtrip.kind == "bootstrap"


def test_fit_recovers_the_arrays_true_coefficient(site):
    rng = np.random.default_rng(7)
    true_k = 1.9
    days = []
    start = date(2026, 1, 1)
    for i in range(120):
        day = start + timedelta(days=i)
        poa = [
            max(0.0, 0.9 * np.sin(np.pi * (h - 6) / 13)) if 6 <= h <= 19 else 0.0 for h in range(24)
        ]
        poa = [g * rng.uniform(0.6, 1.0) for g in poa]
        temp = [18.0] * 24
        actual = [
            true_k * g * (1 - 0.004 * (18 + 25 * g - 25)) + rng.normal(0, 0.01) if g > 0 else 0
            for g in poa
        ]
        days.append((day, poa, temp, [max(a, 0.0) for a in actual]))
    model = fit_model(days, site, ridge_lambda=1e-6, today=date(2026, 5, 1))
    assert model is not None and model.kind == "fitted"
    assert model.a == pytest.approx(true_k, rel=0.02)
    assert model.n_days == 120
    assert model.calibrated_on == "2026-05-01"


def test_fit_needs_90_usable_days_and_skips_outages(site):
    poa = [0.5 if 8 <= h <= 16 else 0.0 for h in range(24)]
    good = [
        (date(2026, 1, 1) + timedelta(days=i), poa, [20.0] * 24, [0.9 * g for g in poa])
        for i in range(50)
    ]
    outage = [
        (date(2026, 3, 1) + timedelta(days=i), poa, [20.0] * 24, [0.0] * 24) for i in range(60)
    ]
    assert fit_model(good + outage, site) is None


def test_band_for(thresholds):
    assert band_for(10, 10, thresholds) == ("normal", 1.0)
    assert band_for(7, 10, thresholds)[0] == "low"
    assert band_for(5, 10, thresholds)[0] == "severe"
    assert band_for(0.2, 10, thresholds)[0] == "zero"
    assert band_for(0.5, 0.5, thresholds)[0] == "unassessed"
    # exhaustive and non-overlapping over ratios
    seen = {band_for(r / 100, 1.0 * 10, thresholds)[0] for r in range(0, 2000, 10)}
    assert seen == {"zero", "severe", "low", "normal"}


def _bell(scale=1.0):
    return [
        max(0.0, scale * np.sin(np.pi * (h - 6) / 13)) if 6 <= h <= 19 else 0.0 for h in range(24)
    ]


def test_anomaly_profiles():
    expected = _bell(2.0)
    assert classify_profile(expected, expected).kind == "normal"
    assert classify_profile([0.0] * 24, expected).kind == "flat_zero"
    late = [0.0 if h < 11 else v for h, v in enumerate(expected)]
    assert classify_profile(late, expected).kind == "late_start"
    early = [0.0 if h >= 13 else v for h, v in enumerate(expected)]
    assert classify_profile(early, expected).kind == "early_cutoff"
    dropout = list(expected)
    dropout[12] = 0.1
    assert classify_profile(dropout, expected).kind == "midday_dropout"
    uniform = [v * 0.5 for v in expected]
    profile = classify_profile(uniform, expected)
    assert profile.kind == "uniform_shortfall" and "50%" in profile.description
    assert classify_profile([0.0] * 24, [0.0] * 24).kind == "normal"


def test_net_position_day_and_window():
    tariff = load_tariff(CONFIG)
    day = date(2026, 5, 10)  # a Sunday: super off-peak until 14:00
    imp = [0.4] * 9 + [0.0] * 8 + [0.3] * 7
    exp = [0.0] * 9 + [1.5] * 5 + [1.0] * 3 + [0.0] * 7
    daily = day_net_position(day, imp, exp, tariff.calendar)
    assert "super_off_peak" in daily.net_export_periods
    assert daily.net_kwh["on_peak"] > 0
    window = window_net_position(
        [(day, imp, exp)]
        + [(day - timedelta(days=i), [1.0] * 24, [0.0] * 24) for i in range(1, 30)],
        tariff.calendar,
    )
    assert window.net_export_periods == []


def test_formatting():
    assert fmt.kwh(8.24) == "8.2"
    assert fmt.dollars(1234.5) == "$1,234.50"
    assert fmt.percent(0.745) == "74%" or fmt.percent(0.745) == "75%"
    assert fmt.plain(3) == "3" and fmt.plain(2.0) == "2" and fmt.plain(2.25) == "2.2"
