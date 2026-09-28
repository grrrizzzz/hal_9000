"""Net position per TOU period: is the household importing or exporting?

NEM 1.0 nets each TOU period over the billing cycle. A single day being
net-export in a period is harmless; a period trending toward net export over a
whole cycle would break the savings model. This module computes both the daily
view and a rolling window view for trend warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from hal9000.solar.domain.tou_calendar import PERIODS, TouCalendar

__all__ = ["NetPosition", "day_net_position", "window_net_position"]


@dataclass(frozen=True)
class NetPosition:
    import_kwh: dict[str, float]
    export_kwh: dict[str, float]
    net_kwh: dict[str, float]  # import - export; positive = net import
    net_export_periods: list[str]
    trending_export: list[str]  # periods whose net import is below the warn fraction


def _net(import_by: dict, export_by: dict, warn_fraction: float) -> NetPosition:
    net = {p: import_by[p] - export_by[p] for p in PERIODS}
    exporting = [p for p in PERIODS if net[p] < 0]
    trending = [
        p
        for p in PERIODS
        if import_by[p] > 0 and net[p] >= 0 and net[p] < warn_fraction * import_by[p]
    ]
    return NetPosition(import_by, export_by, net, exporting, trending)


def day_net_position(
    day: date,
    hourly_import: list[float],
    hourly_export: list[float],
    calendar: TouCalendar,
    warn_fraction: float = 0.1,
) -> NetPosition:
    periods = calendar.day_periods(day)
    imp = {p: 0.0 for p in PERIODS}
    exp = {p: 0.0 for p in PERIODS}
    for hour in range(24):
        imp[periods[hour]] += hourly_import[hour]
        exp[periods[hour]] += hourly_export[hour]
    return _net(imp, exp, warn_fraction)


def window_net_position(
    days: list[tuple[date, list[float], list[float]]],
    calendar: TouCalendar,
    warn_fraction: float = 0.1,
) -> NetPosition:
    """Net position over several days, e.g. a rolling 30-day window."""
    imp = {p: 0.0 for p in PERIODS}
    exp = {p: 0.0 for p in PERIODS}
    for day, hourly_import, hourly_export in days:
        daily = day_net_position(day, hourly_import, hourly_export, calendar, warn_fraction)
        for p in PERIODS:
            imp[p] += daily.import_kwh[p]
            exp[p] += daily.export_kwh[p]
    return _net(imp, exp, warn_fraction)
