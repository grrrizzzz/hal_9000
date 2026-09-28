"""Build replay fixtures from hourly arrays, and the real Home Assistant data used in tests.

The ``REAL_*`` arrays are hourly statistics ``change`` values copied from this
home's Home Assistant (Emporia Vue) on the dates named. Solar values before
2026-09-27 18:35 are raw (half-scale); the code multiplies them by SOLAR_SCALE.
Weather payloads are synthetic: pvlib clear-sky scaled per hour, shaped exactly
like Open-Meteo's hourly response (radiation = mean of the preceding hour).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
SOLAR = "sensor.electron_inspector_energy_today_13"
IMPORT = "sensor.mainstogrid_energy_today"
EXPORT = "sensor.mainsfromgrid_energy_today"

# fmt: off
REAL_2026_09_22 = {
    SOLAR: [0, 0, 0, 0, 0, 0, 0.00013, 0.04673, 0.16223, 0.32593, 0.73832, 0.84056, 1.04933,
            1.01961, 1.06875, 0.86222, 0.33551, 0.06022, 0.01450, 0, 0, 0, 0, 0],
    EXPORT: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02262, 0.60213, 1.15130, 1.55132, 1.53420, 1.29548,
             1.01497, 0.18072, 0, 0, 0, 0, 0, 0, 0],
    IMPORT: [9.10583, 9.98224, 8.23100, 9.96478, 6.86015, 0.49527, 0.54765, 0.45484, 0.33936,
             0, 0, 0, 0, 0, 0, 0, 0.01393, 0.59112, 1.18713, 1.14033, 0.78997, 0.58123, 0.76113,
             0.79836],
}
REAL_2026_05_10 = {
    SOLAR: [0, 0, 0, 0, 0, 0, 0.00657, 0.05425, 0.16643, 0.44557, 0.90408, 0.94000, 1.08363,
            1.28391, 1.09655, 1.00261, 0.74115, 0.10390, 0.06786, 0.02400, 0, 0, 0, 0],
    EXPORT: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.47463, 1.44733, 1.08391, 1.84715, 2.25710, 1.96780,
             1.76230, 1.23323, 0, 0, 0, 0, 0, 0, 0],
    IMPORT: [0.31374, 0.40899, 0.35930, 0.26580, 0.38861, 0.38185, 0.37334, 0.25587, 0.08632,
             0, 0, 0, 0, 0, 0, 0, 0, 0.05618, 0.22787, 0.35555, 0.44476, 0.50877, 0.57013,
             0.63859],
}
# fmt: on


def hourly_points(day: date, values: list[float]) -> list[list]:
    start = datetime(day.year, day.month, day.day, tzinfo=TZ)
    return [
        [(start + timedelta(hours=h)).astimezone(ZoneInfo("UTC")).isoformat(), v]
        for h, v in enumerate(values)
    ]


def five_minute_points(day: date, hourly: list[float], until_hour: int = 24) -> list[list]:
    """Spread each hour evenly over twelve 5-minute buckets."""
    start = datetime(day.year, day.month, day.day, tzinfo=TZ)
    out = []
    for h, v in enumerate(hourly[:until_hour]):
        for m in range(12):
            t = start + timedelta(hours=h, minutes=5 * m)
            out.append([t.astimezone(ZoneInfo("UTC")).isoformat(), v / 12.0])
    return out


def weather_payload(day: date, sky_factor: list[float] | float = 0.95, temp: float = 22.0) -> dict:
    """Open-Meteo-shaped hourly payload for ``day`` .. ``day+1`` from pvlib clear-sky."""
    from hal9000.solar.domain.clearsky import clearsky_poa_hourly
    from hal9000.solar.domain.sitecfg import Site

    site = Site(33.25, -117.30, 20, "America/Los_Angeles", 33, 180, 2.47, 0.945, 0.85)
    factors = sky_factor if isinstance(sky_factor, list) else [sky_factor] * 24
    times, gti, ghi, cloud, temps = [], [], [], [], []
    for offset in range(2):
        d = day + timedelta(days=offset)
        clear = clearsky_poa_hourly(site, d)
        for h in range(24):
            # Open-Meteo is asked for GMT: stamps are UTC wall-clock times
            t = datetime(d.year, d.month, d.day, h, tzinfo=TZ).astimezone(ZoneInfo("UTC"))
            times.append(t.strftime("%Y-%m-%dT%H:%M"))
            # value stamped h is the mean of the preceding hour (h-1 .. h)
            prev = h - 1
            energy = clear[prev] * factors[prev] if prev >= 0 else 0.0
            gti.append(round(energy * 1000.0, 1))
            ghi.append(round(energy * 900.0, 1))
            cloud.append(round((1 - factors[prev]) * 100) if prev >= 0 else 0)
            temps.append(temp)
    return {
        "latitude": 33.25,
        "longitude": -117.3,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly_units": {"global_tilted_irradiance": "W/m²"},
        "hourly": {
            "time": times,
            "global_tilted_irradiance": gti,
            "shortwave_radiation": ghi,
            "cloud_cover": cloud,
            "temperature_2m": temps,
        },
    }


def fixture(
    day: date,
    solar: list[float],
    grid_import: list[float],
    grid_export: list[float],
    sky_factor: list[float] | float = 0.95,
    now_hour: int = 20,
    extra_days: dict[date, dict] | None = None,
) -> dict:
    """A replay fixture with hourly statistics for ``day`` (and optional other days)."""
    now = datetime(day.year, day.month, day.day, now_hour, 0, tzinfo=TZ)
    days = {day: {SOLAR: solar, IMPORT: grid_import, EXPORT: grid_export}}
    days.update(extra_days or {})
    result = {SOLAR: [], IMPORT: [], EXPORT: []}
    for d, series in days.items():
        for entity, values in series.items():
            result[entity] += hourly_points(d, values)
    weather = [
        {
            "start": d.isoformat(),
            "end": (d + timedelta(days=1)).isoformat(),
            "payload": weather_payload(d, sky_factor if d == day else 0.95),
        }
        for d in days
    ]
    return {
        "date": day.isoformat(),
        "now": now.isoformat(),
        "ha": [{"entities": [SOLAR, IMPORT, EXPORT], "period": "hour", "result": result}],
        "weather": weather,
    }
