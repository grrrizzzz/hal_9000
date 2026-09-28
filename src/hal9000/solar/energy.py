"""Turn Home Assistant statistics into one day of clean hourly energy.

Three entities (solar, grid import, grid export) are read as statistics
``change`` buckets and folded into 24 local hours. Along the way:

- **Scale correction.** Solar buckets that started before ``SOLAR_FIXED_AT``
  are multiplied by ``SOLAR_SCALE`` (the Emporia channel read half before the
  multiplier fix, plan §2.1).
- **Artifact guard.** A solar bucket larger than the array can physically
  produce (for example Emporia's one-off rescale jump on 2026-09-27 at 18:35) is
  dropped and flagged ``data_artifact``.
- **Sanity check.** Solar must be at least grid export in every hour, because
  exported energy can only come from the array. Any hour that breaks this sets
  ``sensor_fault``.

Household load is derived as import + solar - export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from hal9000.config import Settings
from hal9000.solar.clients.homeassistant import HAProtocol
from hal9000.solar.domain.sitecfg import Site

__all__ = [
    "UNFIT_FLAGS",
    "DayEnergy",
    "fetch_day_energy",
    "fold_statistics",
    "FIVE_MINUTE_RETENTION_DAYS",
]

#: Home Assistant keeps 5-minute statistics for about 10 days by default.
FIVE_MINUTE_RETENTION_DAYS = 9
SANITY_TOLERANCE_KWH = 0.02
SANITY_TOLERANCE_FRACTION = 0.05
#: days with any of these flags are left out of calibration
UNFIT_FLAGS = frozenset({"sensor_fault", "transition_day", "missing_data"})


@dataclass
class DayEnergy:
    day: date
    solar: list[float]
    grid_import: list[float]
    grid_export: list[float]
    through: datetime  # data covers the day up to this local time
    complete: bool  # the whole day is covered
    resolution: str  # "5minute" | "hour"
    artifacts: list[str] = field(default_factory=list)  # descriptions of dropped buckets
    sensor_fault_hours: list[int] = field(default_factory=list)
    scaled_hours: list[int] = field(default_factory=list)
    transition_day: bool = False
    hours_with_data: int = 0
    #: hours (fully before ``through``) with too few solar buckets to trust:
    #: fewer than 10 of 12 five-minute buckets, or no hourly bucket
    missing_solar_hours: list[int] = field(default_factory=list)

    @property
    def household(self) -> list[float]:
        return [
            max(i + s - e, 0.0) for i, s, e in zip(self.grid_import, self.solar, self.grid_export)
        ]

    @property
    def sensor_fault(self) -> bool:
        return bool(self.sensor_fault_hours)

    def day_flags(self) -> list[str]:
        """Day-level data-quality flags, stored with every hourly row.

        A day carrying any of :data:`UNFIT_FLAGS` is never used to calibrate the
        expectation model.
        """
        flags = []
        if self.sensor_fault:
            flags.append("sensor_fault")
        if self.transition_day:
            flags.append("transition_day")
        if self.missing_solar_hours:
            flags.append("missing_data")
        if self.artifacts:
            flags.append("data_artifact")
        return flags

    def totals(self) -> dict[str, float]:
        return {
            "solar": sum(self.solar),
            "import": sum(self.grid_import),
            "export": sum(self.grid_export),
            "household": sum(self.household),
        }


def fold_statistics(
    points: dict[str, list[tuple[datetime, float]]],
    day: date,
    settings: Settings,
    site: Site,
    period: str,
    until: datetime | None = None,
) -> DayEnergy:
    """Fold raw ``(utc_start, change)`` buckets for the three entities into a DayEnergy."""
    tz = ZoneInfo(site.timezone)
    day_start = datetime(day.year, day.month, day.day, tzinfo=tz)
    day_end = day_start + timedelta(days=1)
    through = min(until, day_end) if until else day_end
    bucket_minutes = 5 if period == "5minute" else 60
    ceiling = site.dc_kw * 1.2 * bucket_minutes / 60

    solar, grid_import, grid_export = [0.0] * 24, [0.0] * 24, [0.0] * 24
    artifacts: list[str] = []
    scaled: set[int] = set()
    seen_hours: set[int] = set()
    solar_buckets = [0] * 24
    fixed_at = settings.solar_fixed_at

    def fold(entity: str, target: list[float], is_solar: bool) -> None:
        for start_utc, change in points.get(entity, []):
            local = start_utc.astimezone(tz)
            if not (day_start <= local < through):
                continue
            hour = local.hour
            seen_hours.add(hour)
            value = change
            if value < 0:
                if value < -0.05:
                    artifacts.append(f"{entity} negative change {value:.3f} at {local:%H:%M}")
                value = 0.0
            if is_solar:
                solar_buckets[hour] += 1
                if fixed_at is not None and local < fixed_at and settings.solar_scale != 1.0:
                    value *= settings.solar_scale
                    scaled.add(hour)
                if value > ceiling:
                    artifacts.append(
                        f"solar jump of {value:.2f} kWh at {local:%H:%M} exceeds the "
                        f"{ceiling:.2f} kWh physical limit; dropped"
                    )
                    continue
            target[hour] += value

    fold(settings.entity_solar, solar, True)
    fold(settings.entity_grid_import, grid_import, False)
    fold(settings.entity_grid_export, grid_export, False)

    needed = 10 if period == "5minute" else 1
    missing = [
        hour
        for hour in range(24)
        if day_start + timedelta(hours=hour + 1) <= through and solar_buckets[hour] < needed
    ]

    fault_hours = []
    for hour in range(24):
        if hour in missing:
            continue  # absent solar buckets are missing data, not a faulty reading
        tolerance = max(SANITY_TOLERANCE_KWH, SANITY_TOLERANCE_FRACTION * grid_export[hour])
        if grid_export[hour] - solar[hour] > tolerance:
            fault_hours.append(hour)

    transition = bool(fixed_at and fixed_at.astimezone(tz).date() == day)
    return DayEnergy(
        day=day,
        solar=solar,
        grid_import=grid_import,
        grid_export=grid_export,
        through=through,
        complete=through >= day_end,
        resolution=period,
        artifacts=artifacts,
        sensor_fault_hours=fault_hours,
        scaled_hours=sorted(scaled),
        transition_day=transition,
        hours_with_data=len(seen_hours),
        missing_solar_hours=missing,
    )


def fetch_day_energy(
    ha: HAProtocol,
    day: date,
    settings: Settings,
    site: Site,
    now: datetime,
) -> DayEnergy:
    """Read one local day from Home Assistant, up to ``now`` if the day is today."""
    tz = ZoneInfo(site.timezone)
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    end = min(start + timedelta(days=1), now.astimezone(tz))
    recent = (now.astimezone(tz).date() - day).days <= FIVE_MINUTE_RETENTION_DAYS
    period = "5minute" if recent else "hour"
    entities = [settings.entity_solar, settings.entity_grid_import, settings.entity_grid_export]
    points = ha.statistics(entities, start, start + timedelta(days=1), period)
    energy = fold_statistics(points, day, settings, site, period, until=end)
    if recent and energy.hours_with_data == 0:
        # 5-minute statistics may have been purged; fall back to hourly
        points = ha.statistics(entities, start, start + timedelta(days=1), "hour")
        energy = fold_statistics(points, day, settings, site, "hour", until=end)
    return energy
