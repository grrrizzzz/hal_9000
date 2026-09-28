"""Numeral extraction and the numeric guard: nothing unbacked reaches the household."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from hal9000.solar.domain.numerals import extract_numerals, to_decimal
from hal9000.solar.reporting.guard import completeness_problems, find_unbacked_numerals
from hal9000.solar.run_context import RunFlags
from hal9000.storage.run_store import RunStore

DAY = date(2026, 9, 22)


def values(text):
    return [n.value for n in extract_numerals(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8.2 kilowatt-hours", [Decimal("8.2")]),
        ("worth $1.37 today", [Decimal("1.37")]),
        ("that is 37% of expected", [Decimal("37")]),
        ("1,024 kWh", [Decimal("1024")]),
        ("eight point two kilowatt hours", [Decimal("8.2")]),
        ("twelve hours", [Decimal("12")]),
        ("at 12:00 and 8 pm on September 22, 2026", []),
        ("one hundred and five", [Decimal("105")]),
        ("twenty-three", [Decimal("23")]),
        ("one of the best days", []),
        ("no one noticed", []),
        ("", []),
    ],
)
def test_extract_numerals(text, expected):
    assert values(text) == expected


def test_to_decimal_and_idempotent_normalisation():
    assert to_decimal("$1,024.50") == Decimal("1024.50")
    assert to_decimal("abc") is None
    once = values("8.20 and 8.2")
    assert once[0] == once[1]


@pytest.fixture
def store():
    s = RunStore()
    s.put("production_kwh", "8.2")
    s.put("expected_kwh", "11.0")
    s.put("performance_pct", "74%")
    s.put("savings_usd", "$1.37")
    s.put("peak_hour", "12:00")
    return s


@pytest.mark.parametrize(
    ("narrative", "unbacked"),
    [
        ("The panels made 8.2 kilowatt-hours.", []),
        ("That's 74% of the 11.0 expected.", []),
        ("Worth $1.37 today.", []),
        ("Eight point two kilowatt-hours today.", []),
        ("Peak output came at 12:00.", []),
        ("Production was normal.", []),
        ("The report runs at 8 pm.", []),
        ("On-peak runs 4 pm to 9 pm.", []),
        ("Today, September 22, 2026.", []),
        ("The eight pm report.", []),
        ("Peak at 12:00, 2.1 kilowatt-hours.", ["2.1"]),
        ("On 2026-09-22 production was normal.", []),
        ("It was one of the better days.", []),
        ("About 8 kilowatt-hours.", ["8"]),  # rounded: 8 != 8.2
        ("Roughly 8.0 kilowatt-hours.", ["8.0"]),
        ("Worth about $1.40.", ["$1.40"]),
        ("That is 75% of expected.", ["75%"]),
        ("Seventy-four percent.", []),  # 74 is backed
        ("Seventy-five percent.", ["seventy five"]),
        ("A shortfall of 2.8 kilowatt-hours.", ["2.8"]),  # derived, not stored
        ("Twelve kilowatt-hours were expected.", ["twelve"]),
        ("It produced 13.4 kilowatt-hours.", ["13.4"]),
        ("That saved $4.14.", ["$4.14"]),
        ("Output was 82% of normal.", ["82%"]),
        ("Two alerts are open.", ["two"]),
        ("8.2 and 11.0 and $1.37 and 74%.", []),
        ("Three hours of dropout.", ["three"]),
        ("1,100 watts peak.", ["1,100"]),
        ("Value was $0.99.", ["$0.99"]),
        # unit-aware: a value is backed only by a figure of the same unit
        ("That saved $8.2.", ["$8.2"]),  # 8.2 is kWh, not dollars
        ("It made 74 kilowatt-hours.", ["74"]),  # 74 is a percent
        ("That is 1.37 percent.", ["1.37"]),  # 1.37 is dollars
        ("Seventy-four kilowatt-hours.", ["seventy four"]),
        ("On-peak runs from 4 to 9 pm.", []),
        ("Above the 7-day average.", []),
        ("Over the last 30 days.", []),
        ("September twenty-second was sunny.", []),
    ],
)
def test_guard_cases(store, narrative, unbacked):
    assert find_unbacked_numerals(narrative, store, DAY) == unbacked


def test_completeness_problems():
    store = RunStore()
    flags = RunFlags()
    problems = completeness_problems(store, flags)
    assert len(problems) == 3
    store.put("production_kwh", "1.0")
    store.put_data("band", "normal")
    flags.savings_absent_reason = "rates not configured"
    assert completeness_problems(store, flags) == []


def test_run_store_basics():
    s = RunStore()
    s.put("a", "1.5")
    s.put("b", "$2.00")
    s.put_data("x", [1, 2])
    assert s.figures() == {"a": "1.5", "b": "$2.00"}
    assert s.figures("a") == {"a": "1.5"}
    assert s.numeric_values() == {("number", Decimal("1.5")), ("money", Decimal("2.00"))}
    assert s.data("x") == [1, 2] and s.data("missing", 7) == 7
    assert s.snapshot()["data_keys"] == ["x"]
