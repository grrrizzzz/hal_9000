"""TOU calendar, tariff engine and savings: the highest-consequence numbers in HAL."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hal9000.solar.domain.savings import compute_savings
from hal9000.solar.domain.tariff import Tariff, TariffNotConfigured, load_tariff
from hal9000.solar.domain.tou_calendar import (
    OFF_PEAK,
    ON_PEAK,
    PERIODS,
    SUPER_OFF_PEAK,
    TouCalendar,
    sdge_holidays,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def tariff() -> Tariff:
    return load_tariff(CONFIG)


@pytest.fixture(scope="module")
def cal(tariff) -> TouCalendar:
    return tariff.calendar


# --- calendar ---------------------------------------------------------------

WEEKDAY = date(2026, 9, 28)  # Monday
SATURDAY = date(2026, 9, 26)
HOLIDAY = date(2026, 11, 26)  # Thanksgiving, a Thursday


@pytest.mark.parametrize(
    ("hour", "weekday", "weekend"),
    [
        (0, SUPER_OFF_PEAK, SUPER_OFF_PEAK),
        (5, SUPER_OFF_PEAK, SUPER_OFF_PEAK),
        (6, OFF_PEAK, SUPER_OFF_PEAK),
        (9, OFF_PEAK, SUPER_OFF_PEAK),
        (10, SUPER_OFF_PEAK, SUPER_OFF_PEAK),
        (13, SUPER_OFF_PEAK, SUPER_OFF_PEAK),
        (14, OFF_PEAK, OFF_PEAK),
        (15, OFF_PEAK, OFF_PEAK),
        (16, ON_PEAK, ON_PEAK),
        (20, ON_PEAK, ON_PEAK),
        (21, OFF_PEAK, OFF_PEAK),
        (23, OFF_PEAK, OFF_PEAK),
    ],
)
def test_boundaries_weekday_weekend_holiday(cal, hour, weekday, weekend):
    assert cal.period(WEEKDAY, hour) == weekday
    assert cal.period(SATURDAY, hour) == weekend
    assert cal.period(HOLIDAY, hour) == weekend


def test_midday_super_off_peak_rule_before_and_after_2026_05_01(cal):
    assert cal.period(date(2026, 1, 15), 11) == OFF_PEAK  # Thursday in January, before the change
    assert cal.period(date(2026, 3, 16), 11) == SUPER_OFF_PEAK  # March: always had it
    assert cal.period(date(2026, 4, 30), 11) == SUPER_OFF_PEAK  # April, day before the change
    assert cal.period(date(2026, 4, 30), 10) == SUPER_OFF_PEAK
    assert cal.period(date(2026, 5, 1), 11) == SUPER_OFF_PEAK  # the change
    assert cal.period(date(2025, 12, 2), 13) == OFF_PEAK
    assert cal.period(date(2026, 6, 15), 11) == SUPER_OFF_PEAK
    assert cal.period(date(2026, 4, 15), 9) == OFF_PEAK
    assert cal.period(date(2026, 4, 15), 14) == OFF_PEAK


def test_saturday_edges(cal):
    assert cal.period(SATURDAY, 13) == SUPER_OFF_PEAK
    assert cal.period(SATURDAY, 14) == OFF_PEAK
    assert cal.period(SATURDAY, 15) == OFF_PEAK


def test_seasons(cal):
    assert cal.season(date(2026, 5, 31)) == "winter"
    assert cal.season(date(2026, 6, 1)) == "summer"
    assert cal.season(date(2026, 10, 31)) == "summer"
    assert cal.season(date(2026, 11, 1)) == "winter"


def test_holidays_and_sunday_observance():
    days = sdge_holidays(2026)
    assert date(2026, 1, 1) in days
    assert date(2026, 2, 16) in days  # Presidents' Day
    assert date(2026, 5, 25) in days  # Memorial Day
    assert date(2026, 9, 7) in days  # Labor Day
    assert date(2026, 11, 26) in days  # Thanksgiving
    assert len(days) == 8
    # Independence Day 2027 is a Sunday: observed Monday the 5th
    assert date(2027, 7, 5) in sdge_holidays(2027)
    assert date(2027, 7, 4) not in sdge_holidays(2027)


def test_bad_hour_and_unknown_holiday():
    cal = TouCalendar({})
    with pytest.raises(ValueError):
        cal.period(WEEKDAY, 24)
    with pytest.raises(ValueError):
        sdge_holidays(2026, ("groundhog_day",))


def test_day_periods_has_24_and_period_at(cal):
    from datetime import datetime

    assert len(cal.day_periods(WEEKDAY)) == 24
    assert cal.period_at(datetime(2026, 9, 28, 17, 30)) == ON_PEAK


# --- tariff -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        # summer, current (August 2026 bill, 'Rate 2')
        (date(2026, 8, 15), {"on_peak": 0.86691, "off_peak": 0.50982, "super_off_peak": 0.12566}),
        # winter actual (February 2026 bill)
        (date(2026, 2, 15), {"on_peak": 0.60765, "off_peak": 0.52117, "super_off_peak": 0.12561}),
        # forward winter (provisional)
        (date(2026, 11, 15), {"on_peak": 0.59437, "off_peak": 0.50789, "super_off_peak": 0.12792}),
    ],
)
def test_marginal_rates_match_the_plan(tariff, day, expected):
    assert tariff.day_rates(day).rates == pytest.approx(expected, abs=1e-5)


def test_provisional_only_for_unconfirmed_winter(tariff):
    assert tariff.day_rates(date(2026, 11, 15)).provisional is True
    assert "provisional" in tariff.day_rates(date(2026, 11, 15)).schedule_version
    assert tariff.day_rates(date(2026, 9, 15)).provisional is False
    assert tariff.day_rates(date(2026, 2, 15)).provisional is False


def test_dates_before_configured_rates_raise(tariff):
    with pytest.raises(TariffNotConfigured) as excinfo:
        tariff.day_rates(date(2025, 12, 15))
    assert "2025-12-15" in str(excinfo.value)
    assert excinfo.value.recoverable is True


def test_a_missing_component_value_raises_never_zero():
    cfg = {
        "per_kwh_adders": {"wildfire_fund": 0.0, "ipca": 0.0, "state_regulatory_fee": 0.0},
        "components": {
            "udc": [
                {"from": "2026-01-01", "on_peak": 0.3, "off_peak": 0.3, "super_off_peak": None}
            ],
            "pcia": [{"from": "2026-01-01", "value": 0.02}],
            "cea_generation": [{"from": "2026-01-01", "summer": {}, "winter": {}}],
        },
    }
    with pytest.raises(TariffNotConfigured):
        Tariff(cfg).day_rates(date(2026, 6, 1))


def test_rate_for_an_hour(tariff):
    assert tariff.rate(date(2026, 8, 17), 17) == pytest.approx(0.86691)
    assert tariff.rate(date(2026, 8, 17), 11) == pytest.approx(0.12566)


# --- savings ----------------------------------------------------------------

hourly = st.lists(st.floats(min_value=0, max_value=3, allow_nan=False), min_size=24, max_size=24)
days = st.dates(min_value=date(2026, 1, 1), max_value=date(2027, 12, 31))


@settings(max_examples=500, deadline=None)
@given(values=hourly, day=days)
def test_savings_properties(tariff, values, day):
    result = compute_savings(day, values, tariff)
    assert result.total_usd >= 0
    assert sum(result.kwh_by_period.values()) == pytest.approx(sum(values), abs=1e-9)
    assert result.total_usd <= sum(values) * max(result.rates.values()) + 1e-9
    assert set(result.by_period_usd) == set(PERIODS)


@settings(max_examples=500, deadline=None)
@given(values=hourly, day=days, hour=st.integers(0, 23), extra=st.floats(0.01, 2))
def test_savings_monotonic_in_generation(tariff, values, day, hour, extra):
    more = list(values)
    more[hour] += extra
    assert (
        compute_savings(day, more, tariff).total_usd
        > compute_savings(day, values, tariff).total_usd
    )


def test_savings_rejects_bad_input(tariff):
    with pytest.raises(ValueError):
        compute_savings(date(2026, 8, 1), [1.0] * 23, tariff)
    with pytest.raises(ValueError):
        compute_savings(date(2026, 8, 1), [-1.0] + [0.0] * 23, tariff)


def test_savings_hand_computed(tariff):
    """One kWh in each period on a weekday in August: the sum of the three rates."""
    values = [0.0] * 24
    values[11] = 1.0  # super off-peak
    values[15] = 1.0  # off-peak
    values[17] = 1.0  # on-peak
    result = compute_savings(date(2026, 8, 17), values, tariff)
    assert result.total_usd == pytest.approx(0.86691 + 0.50982 + 0.12566, abs=1e-9)
