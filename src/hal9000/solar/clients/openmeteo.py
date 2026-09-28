"""Open-Meteo client: hourly irradiance on the plane of the array, plus weather.

Radiation variables are **averages over the preceding hour**: the value stamped
13:00 describes 12:00-13:00. :func:`hourly_for_day` re-buckets them so index
``h`` always means the hour *starting* at ``h``, matching every other hourly
series in HAL. Open-Meteo's azimuth convention is 0 = south (compass 180).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Protocol

import httpx

__all__ = ["OpenMeteoClient", "WeatherProtocol", "WeatherUnavailable", "hourly_for_day"]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "global_tilted_irradiance,shortwave_radiation,cloud_cover,temperature_2m"
#: The forecast endpoint serves roughly the last three months; older dates use the archive.
FORECAST_MAX_PAST_DAYS = 80


class WeatherUnavailable(Exception):
    """Open-Meteo could not be reached. Recoverable: report production without a band."""

    error_type = "WeatherUnavailable"
    recoverable = True
    guidance = (
        "Weather data is unavailable, so expected production can't be computed. Report "
        "production without a performance band and say why."
    )


class WeatherProtocol(Protocol):
    def fetch(self, start: date, end: date) -> dict: ...


class OpenMeteoClient:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        tilt: float,
        openmeteo_azimuth: float,
        timezone: str,
        timeout: float = 30.0,
        today: date | None = None,
    ):
        self.params = {
            "latitude": latitude,
            "longitude": longitude,
            "tilt": tilt,
            "azimuth": openmeteo_azimuth,
            "timezone": timezone,
            "hourly": HOURLY_VARS,
        }
        self.timeout = timeout
        self._today = today

    def url_for(self, start: date) -> str:
        today = self._today or date.today()
        return FORECAST_URL if (today - start).days <= FORECAST_MAX_PAST_DAYS else ARCHIVE_URL

    def fetch(self, start: date, end: date) -> dict:
        """Raw Open-Meteo JSON for ``start`` .. ``end`` inclusive (local dates)."""
        params = dict(self.params, start_date=start.isoformat(), end_date=end.isoformat())
        try:
            response = httpx.get(self.url_for(start), params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise WeatherUnavailable(f"Open-Meteo request failed: {error}") from error
        if "hourly" not in payload:
            raise WeatherUnavailable(f"Open-Meteo returned no hourly data: {payload.get('reason')}")
        return payload


def hourly_for_day(payload: dict, day: date) -> list[dict]:
    """24 buckets for ``day``: ``poa``/``ghi`` in kWh/m2, ``cloud`` %, ``temp`` degC.

    Radiation stamped ``t`` is assigned to the hour starting ``t - 1h``. Temperature
    is instantaneous, so each hour takes the mean of its two endpoints.
    """
    hourly = payload["hourly"]
    times = [datetime.fromisoformat(t) for t in hourly["time"]]
    gti = hourly.get("global_tilted_irradiance") or [None] * len(times)
    ghi = hourly.get("shortwave_radiation") or [None] * len(times)
    cloud = hourly.get("cloud_cover") or [None] * len(times)
    temp = hourly.get("temperature_2m") or [None] * len(times)
    by_time = {t: i for i, t in enumerate(times)}

    buckets = []
    for hour in range(24):
        start = datetime(day.year, day.month, day.day, hour)
        end = start + timedelta(hours=1)
        i_end = by_time.get(end)
        i_start = by_time.get(start)
        if i_end is None or gti[i_end] is None:
            raise WeatherUnavailable(f"Open-Meteo data missing for {start.isoformat()}")
        temps = [temp[i] for i in (i_start, i_end) if i is not None and temp[i] is not None]
        buckets.append(
            {
                "poa": max(float(gti[i_end]), 0.0) / 1000.0,
                "ghi": max(float(ghi[i_end] or 0.0), 0.0) / 1000.0,
                "cloud": float(cloud[i_end]) if cloud[i_end] is not None else None,
                "temp": sum(temps) / len(temps) if temps else None,
            }
        )
    return buckets
