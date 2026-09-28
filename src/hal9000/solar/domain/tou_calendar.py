"""EV-TOU-5 time-of-use calendar: which period any local hour falls in.

Rules (plan §2), all hours local time, periods half-open ``[start, end)``:

- On-peak 16:00-21:00 every day.
- Weekends and holidays: super off-peak 00:00-14:00, off-peak otherwise.
- Weekdays: super off-peak 00:00-06:00 and 10:00-14:00, off-peak otherwise. Before
  2026-05-01 the 10:00-14:00 window applied only in March and April.
- Summer is June 1 - October 31; winter is everything else.
- SDG&E holidays; one falling on a Sunday is observed on the Monday.
"""

from __future__ import annotations

import calendar as _cal
from datetime import date, datetime, timedelta
from functools import lru_cache

__all__ = [
    "ON_PEAK",
    "OFF_PEAK",
    "PERIODS",
    "SUPER_OFF_PEAK",
    "TouCalendar",
    "sdge_holidays",
]

ON_PEAK = "on_peak"
OFF_PEAK = "off_peak"
SUPER_OFF_PEAK = "super_off_peak"
PERIODS = (ON_PEAK, OFF_PEAK, SUPER_OFF_PEAK)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Mon=0) of a month; ``n=-1`` for the last."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last_day = _cal.monthrange(year, month)[1]
    last = date(year, month, last_day)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


_HOLIDAY_RULES = {
    "new_years_day": lambda y: date(y, 1, 1),
    "presidents_day": lambda y: _nth_weekday(y, 2, 0, 3),
    "memorial_day": lambda y: _nth_weekday(y, 5, 0, -1),
    "independence_day": lambda y: date(y, 7, 4),
    "labor_day": lambda y: _nth_weekday(y, 9, 0, 1),
    "veterans_day": lambda y: date(y, 11, 11),
    "thanksgiving": lambda y: _nth_weekday(y, 11, 3, 4),
    "christmas": lambda y: date(y, 12, 25),
}


@lru_cache(maxsize=64)
def sdge_holidays(year: int, names: tuple[str, ...] = tuple(_HOLIDAY_RULES)) -> frozenset[date]:
    """Observed SDG&E holidays for ``year`` (Sunday holidays move to Monday)."""
    days = set()
    for name in names:
        if name not in _HOLIDAY_RULES:
            raise ValueError(f"Unknown holiday {name!r}. Known: {sorted(_HOLIDAY_RULES)}")
        day = _HOLIDAY_RULES[name](year)
        if day.weekday() == 6:
            day = day + timedelta(days=1)
        days.add(day)
    return frozenset(days)


class TouCalendar:
    """Period and season lookup, configured from ``tariff.yaml``."""

    def __init__(self, tariff_cfg: dict | None = None):
        cfg = tariff_cfg or {}
        cal = cfg.get("calendar", {})
        self.on_peak = tuple(cal.get("on_peak", (16, 21)))
        self.weekend_sop = tuple(cal.get("weekend_super_off_peak", (0, 14)))
        self.weekday_sop = [tuple(w) for w in cal.get("weekday_super_off_peak", [(0, 6), (10, 14)])]
        self.midday_all_year_from = date.fromisoformat(
            cal.get("weekday_midday_sop_all_year_from", "2026-05-01")
        )
        seasons = cfg.get("seasons", {}).get("summer", {"start": "06-01", "end": "10-31"})
        self.summer_start = tuple(int(x) for x in seasons["start"].split("-"))
        self.summer_end = tuple(int(x) for x in seasons["end"].split("-"))
        self.holiday_names = tuple(cfg.get("holidays", list(_HOLIDAY_RULES)))

    def season(self, day: date) -> str:
        md = (day.month, day.day)
        return "summer" if self.summer_start <= md <= self.summer_end else "winter"

    def is_holiday(self, day: date) -> bool:
        return day in sdge_holidays(day.year, self.holiday_names)

    def is_weekend_or_holiday(self, day: date) -> bool:
        return day.weekday() >= 5 or self.is_holiday(day)

    def period(self, day: date, hour: int) -> str:
        """TOU period for the hour starting at ``hour`` (0-23) on local ``day``."""
        if not 0 <= hour <= 23:
            raise ValueError(f"hour must be 0-23, got {hour}")
        if self.on_peak[0] <= hour < self.on_peak[1]:
            return ON_PEAK
        if self.is_weekend_or_holiday(day):
            start, end = self.weekend_sop
            return SUPER_OFF_PEAK if start <= hour < end else OFF_PEAK
        for start, end in self.weekday_sop:
            if start <= hour < end:
                midday = start >= 6
                if midday and day < self.midday_all_year_from and day.month not in (3, 4):
                    continue
                return SUPER_OFF_PEAK
        return OFF_PEAK

    def period_at(self, moment: datetime) -> str:
        return self.period(moment.date(), moment.hour)

    def day_periods(self, day: date) -> list[str]:
        """The period for each of the 24 hours of ``day``."""
        return [self.period(day, h) for h in range(24)]
