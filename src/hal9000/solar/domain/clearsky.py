"""Clear-sky plane-of-array irradiance for the site, hour by hour (pvlib)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

from hal9000.solar.domain.sitecfg import Site

__all__ = ["clearsky_poa_hourly", "daylight_hours"]


@lru_cache(maxsize=512)
def _clearsky(site: Site, day: date) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    import pandas as pd
    import pvlib

    tz = ZoneInfo(site.timezone)
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    end = start + timedelta(days=1)
    times = pd.date_range(start, end, freq="5min", inclusive="left")
    location = pvlib.location.Location(
        site.latitude, site.longitude, tz=site.timezone, altitude=site.altitude_m
    )
    solpos = location.get_solarposition(times)
    clear = location.get_clearsky(times, solar_position=solpos)
    poa = pvlib.irradiance.get_total_irradiance(
        site.tilt_deg,
        site.azimuth_deg,
        solpos["apparent_zenith"],
        solpos["azimuth"],
        clear["dni"],
        clear["ghi"],
        clear["dhi"],
    )["poa_global"].fillna(0.0)
    hourly = [0.0] * 24
    sun_up = [False] * 24
    for ts, value, elevation in zip(times, poa.values, solpos["apparent_elevation"].values):
        local_hour = ts.tz_convert(tz).hour
        hourly[local_hour] += max(float(value), 0.0) * (5 / 60) / 1000.0  # kWh/m2
        if elevation > 0:
            sun_up[local_hour] = True
    return tuple(hourly), tuple(sun_up)


def clearsky_poa_hourly(site: Site, day: date) -> list[float]:
    """Clear-sky plane-of-array insolation per local hour, kWh/m2."""
    return list(_clearsky(site, day)[0])


def daylight_hours(site: Site, day: date) -> list[bool]:
    """True for each local hour in which the sun is above the horizon."""
    return list(_clearsky(site, day)[1])
