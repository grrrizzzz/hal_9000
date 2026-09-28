"""Savings: value each hour of generation at the marginal TOU rate in force.

Valid because the household is a net importer in every TOU period over every
billing cycle (verified against 12 cycles of bills, plan Appendix A): each kWh
the array produces displaces a kWh that would otherwise be bought at that
period's marginal rate. A blended daily rate would overstate value by about 2x.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from hal9000.solar.domain.tariff import Tariff
from hal9000.solar.domain.tou_calendar import PERIODS

__all__ = ["SavingsResult", "compute_savings"]


@dataclass(frozen=True)
class SavingsResult:
    total_usd: float
    by_period_usd: dict[str, float]
    kwh_by_period: dict[str, float]
    rates: dict[str, float]
    schedule_version: str
    provisional: bool


def compute_savings(day: date, hourly_kwh: list[float], tariff: Tariff) -> SavingsResult:
    """Savings for ``day`` from 24 hourly production values (index 0 = midnight)."""
    if len(hourly_kwh) != 24:
        raise ValueError(f"hourly_kwh must have 24 values, got {len(hourly_kwh)}")
    if any(v < 0 for v in hourly_kwh):
        raise ValueError("hourly_kwh values must be non-negative")
    day_rates = tariff.day_rates(day)
    periods = tariff.calendar.day_periods(day)
    kwh_by = {p: 0.0 for p in PERIODS}
    usd_by = {p: 0.0 for p in PERIODS}
    for hour, value in enumerate(hourly_kwh):
        period = periods[hour]
        kwh_by[period] += value
        usd_by[period] += value * day_rates.rates[period]
    return SavingsResult(
        total_usd=sum(usd_by.values()),
        by_period_usd=usd_by,
        kwh_by_period=kwh_by,
        rates=day_rates.rates,
        schedule_version=day_rates.schedule_version,
        provisional=day_rates.provisional,
    )
