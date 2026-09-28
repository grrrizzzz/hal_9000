"""Record live client responses, and replay them offline.

Every live run saves what Home Assistant and Open-Meteo returned, so any evening
can be re-run exactly -- with a real model, a scripted fake, or no model at all
-- without touching the network. A fixture file looks like::

    {"date": "2026-09-22", "now": "2026-09-22T20:00:00-07:00",
     "ha": [{"entities": [...], "period": "5minute", "start": ..., "end": ...,
             "result": {"sensor.x": [["2026-09-22T07:00:00+00:00", 0.12], ...]}}],
     "weather": [{"start": "2026-09-22", "end": "2026-09-23", "payload": {...}}]}
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from hal9000.solar.clients.homeassistant import HAUnavailable
from hal9000.solar.clients.openmeteo import WeatherUnavailable

__all__ = ["FixtureHA", "FixtureWeather", "RecordingHA", "RecordingWeather", "load_fixture"]


def load_fixture(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


class RecordingHA:
    def __init__(self, inner):
        self.inner = inner
        self.log: list[dict] = []

    def statistics(self, entity_ids, start, end, period):
        result = self.inner.statistics(entity_ids, start, end, period)
        self.log.append(
            {
                "entities": entity_ids,
                "period": period,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "result": {k: [[t.isoformat(), v] for t, v in pts] for k, pts in result.items()},
            }
        )
        return result

    def speak(self, tts_entity, media_player, message):
        return self.inner.speak(tts_entity, media_player, message)

    def ping(self):
        return self.inner.ping()

    def state(self, entity_id):
        return self.inner.state(entity_id)


class RecordingWeather:
    def __init__(self, inner):
        self.inner = inner
        self.log: list[dict] = []

    def fetch(self, start: date, end: date) -> dict:
        payload = self.inner.fetch(start, end)
        self.log.append({"start": start.isoformat(), "end": end.isoformat(), "payload": payload})
        return payload


class FixtureHA:
    """Serves recorded statistics. Any bucket inside the requested window is returned."""

    def __init__(self, fixture: dict, spoken: list | None = None):
        self.series: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
        for call in fixture.get("ha", []):
            for entity, points in call["result"].items():
                bucket = self.series.setdefault((call["period"], entity), [])
                for t, v in points:
                    bucket.append((datetime.fromisoformat(t), float(v)))
        for key in self.series:
            self.series[key] = sorted(set(self.series[key]))
        self.spoken = spoken if spoken is not None else []
        self.fail = fixture.get("ha_unavailable", False)

    def statistics(self, entity_ids, start, end, period):
        if self.fail:
            raise HAUnavailable("Home Assistant unreachable (fixture)")
        out = {}
        for entity in entity_ids:
            points = self.series.get((period, entity), [])
            out[entity] = [(t, v) for t, v in points if start <= t < end]
        return out

    def speak(self, tts_entity, media_player, message):
        self.spoken.append({"tts": tts_entity, "player": media_player, "message": message})

    def ping(self):
        return "API running (fixture)"

    def state(self, entity_id):
        return {"entity_id": entity_id, "state": "fixture"}


class FixtureWeather:
    def __init__(self, fixture: dict):
        self.calls = fixture.get("weather", [])
        self.fail = fixture.get("weather_unavailable", False)

    def fetch(self, start: date, end: date) -> dict:
        if self.fail:
            raise WeatherUnavailable("Open-Meteo unreachable (fixture)")
        for call in self.calls:
            if date.fromisoformat(call["start"]) <= start and end <= date.fromisoformat(
                call["end"]
            ):
                return call["payload"]
        raise WeatherUnavailable(f"No recorded weather for {start}..{end}")
