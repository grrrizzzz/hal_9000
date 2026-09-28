"""Folding Home Assistant statistics into clean hourly energy, and the three clients."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from hal9000 import config
from hal9000.solar.clients.homeassistant import HAClient, HAUnavailable
from hal9000.solar.clients.openmeteo import (
    ARCHIVE_URL,
    FORECAST_URL,
    OpenMeteoClient,
    WeatherUnavailable,
    hourly_for_day,
)
from hal9000.solar.clients.replay import FixtureHA, FixtureWeather, RecordingHA, RecordingWeather
from hal9000.solar.clients.telegram import TelegramClient, TelegramError
from hal9000.solar.domain.sitecfg import load_site
from hal9000.solar.energy import fetch_day_energy, fold_statistics
from tests.fixture_builder import (
    EXPORT,
    IMPORT,
    REAL_2026_05_10,
    REAL_2026_09_22,
    SOLAR,
    TZ,
    five_minute_points,
    fixture,
    weather_payload,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"


def points_for(day, data):
    start = datetime(day.year, day.month, day.day, tzinfo=TZ)
    return {
        entity: [((start + timedelta(hours=h)).astimezone(UTC), v) for h, v in enumerate(values)]
        for entity, values in data.items()
    }


@pytest.fixture
def site():
    return load_site(CONFIG)


def test_may_10_unscaled_is_a_sensor_fault(monkeypatch, site):
    """Solar read 7.9 kWh while the house exported 12.1: impossible."""
    settings = config.settings()
    energy = fold_statistics(
        points_for(date(2026, 5, 10), REAL_2026_05_10), date(2026, 5, 10), settings, site, "hour"
    )
    assert energy.sensor_fault
    assert 12 in energy.sensor_fault_hours
    assert sum(energy.solar) == pytest.approx(7.92, abs=0.01)


def test_may_10_scaled_passes_the_sanity_check(hal_env, site):
    settings = config.settings()
    energy = fold_statistics(
        points_for(date(2026, 5, 10), REAL_2026_05_10), date(2026, 5, 10), settings, site, "hour"
    )
    assert not energy.sensor_fault
    assert sum(energy.solar) == pytest.approx(15.84, abs=0.02)
    assert energy.scaled_hours and energy.complete
    totals = energy.totals()
    assert totals["household"] == pytest.approx(
        totals["import"] + totals["solar"] - totals["export"], abs=0.2
    )
    assert not energy.transition_day


def test_transition_day_and_artifact_guard(hal_env, site):
    """2026-09-27: the multiplier fix at 18:35 caused a one-off +3.26 kWh jump."""
    settings = config.settings()
    day = date(2026, 9, 27)
    hourly = [0.0] * 24
    for h, v in {
        7: 0.05,
        8: 0.16,
        9: 0.24,
        10: 0.5,
        11: 0.44,
        12: 0.62,
        13: 0.45,
        14: 0.32,
        15: 0.21,
        16: 0.19,
        17: 0.07,
    }.items():
        hourly[h] = v
    solar = [[datetime.fromisoformat(t), v] for t, v in five_minute_points(day, hourly)]
    spike_at = datetime(2026, 9, 27, 18, 35, tzinfo=TZ).astimezone(UTC)
    solar.append([spike_at, 3.2613])
    after = datetime(2026, 9, 27, 18, 40, tzinfo=TZ).astimezone(UTC)
    solar.append([after, 0.001])
    points = {SOLAR: [(t, v) for t, v in solar], IMPORT: [], EXPORT: []}
    energy = fold_statistics(points, day, settings, site, "5minute")
    assert energy.transition_day
    assert any("exceeds" in a for a in energy.artifacts)
    assert sum(energy.solar) == pytest.approx(2 * sum(hourly) + 0.001, abs=0.01)


def test_negative_changes_are_clamped_and_flagged(site):
    settings = config.settings()
    day = date(2026, 9, 22)
    data = {k: list(v) for k, v in REAL_2026_09_22.items()}
    data[IMPORT][9] = -0.3
    energy = fold_statistics(points_for(day, data), day, settings, site, "hour")
    assert energy.grid_import[9] == 0.0
    assert any("negative" in a for a in energy.artifacts)


def test_partial_day_until_now(hal_env, site):
    settings = config.settings()
    day = date(2026, 9, 22)
    until = datetime(2026, 9, 22, 12, 0, tzinfo=TZ)
    energy = fold_statistics(
        points_for(day, REAL_2026_09_22), day, settings, site, "hour", until=until
    )
    assert not energy.complete
    assert energy.solar[12] == 0.0 and energy.solar[11] > 0
    assert energy.through == until


def test_fetch_day_energy_falls_back_to_hourly(hal_env, site):
    settings = config.settings()
    day = date(2026, 9, 22)
    ha = FixtureHA(fixture(day, *[REAL_2026_09_22[e] for e in (SOLAR, IMPORT, EXPORT)]))
    now = datetime(2026, 9, 22, 20, 0, tzinfo=TZ)
    energy = fetch_day_energy(ha, day, settings, site, now)
    assert energy.resolution == "hour"
    assert energy.hours_with_data > 0


# --- Open-Meteo ---------------------------------------------------------------


def test_hourly_for_day_realigns_preceding_hour_means():
    day = date(2026, 9, 22)
    payload = weather_payload(day, 1.0)
    rows = hourly_for_day(payload, day, TZ)
    stamped = payload["hourly"]["global_tilted_irradiance"]
    # the value stamped 13:00 describes 12:00-13:00, so it lands in hour 12
    assert rows[12]["poa"] == pytest.approx(stamped[13] / 1000)
    assert rows[23]["poa"] == pytest.approx(stamped[24] / 1000)
    assert len(rows) == 24 and all(r["temp"] == 22.0 for r in rows)


def test_hourly_for_day_missing_data_raises():
    payload = weather_payload(date(2026, 9, 22))
    payload["hourly"]["global_tilted_irradiance"] = payload["hourly"]["global_tilted_irradiance"][
        :10
    ]
    payload["hourly"]["time"] = payload["hourly"]["time"][:10]
    with pytest.raises(WeatherUnavailable):
        hourly_for_day(payload, date(2026, 9, 22), TZ)


def _utc_payload(start: datetime, hours: int) -> dict:
    times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
    return {
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly": {
            "time": times,
            "global_tilted_irradiance": [float(i) for i in range(hours)],
            "shortwave_radiation": [0.0] * hours,
            "cloud_cover": [0.0] * hours,
            "temperature_2m": [20.0] * hours,
        },
    }


@pytest.mark.parametrize(
    ("day", "hour", "utc_end"),
    [
        (date(2026, 9, 22), 12, datetime(2026, 9, 22, 20)),  # PDT: 12:00 local = 19:00Z
        (date(2026, 11, 1), 12, datetime(2026, 11, 1, 21)),  # fall back: PST after 2 am
        (date(2026, 3, 8), 12, datetime(2026, 3, 8, 20)),  # spring forward: PDT after 2 am
        (date(2026, 11, 1), 0, datetime(2026, 11, 1, 8)),  # still PDT at midnight
    ],
)
def test_hourly_for_day_uses_utc_stamps_across_dst(day, hour, utc_end):
    start = datetime(day.year, day.month, day.day) - timedelta(hours=2)
    payload = _utc_payload(start, 60)
    rows = hourly_for_day(payload, day, "America/Los_Angeles")
    assert len(rows) == 24
    index = int((utc_end - start).total_seconds() // 3600)
    assert rows[hour]["poa"] == pytest.approx(index / 1000)


def test_openmeteo_request_parameters_and_endpoint(monkeypatch, site):
    seen = {}

    def fake_get(url, params, timeout):
        seen["url"], seen["params"] = url, params
        return httpx.Response(
            200, json=weather_payload(date(2026, 9, 22)), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    client = OpenMeteoClient(
        site.latitude,
        site.longitude,
        site.tilt_deg,
        site.openmeteo_azimuth,
        site.timezone,
        today=date(2026, 9, 28),
    )
    client.fetch(date(2026, 9, 22), date(2026, 9, 23))
    assert seen["url"] == FORECAST_URL
    assert seen["params"]["timezone"] == "GMT"
    assert seen["params"]["tilt"] == 33 and seen["params"]["azimuth"] == 0  # not the compass 180
    assert "global_tilted_irradiance" in seen["params"]["hourly"]
    assert client.url_for(date(2026, 1, 1)) == ARCHIVE_URL


def test_openmeteo_errors_become_weather_unavailable(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(WeatherUnavailable):
        OpenMeteoClient(0, 0, 0, 0, "UTC").fetch(date(2026, 1, 1), date(2026, 1, 2))
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: httpx.Response(
            200, json={"reason": "bad"}, request=httpx.Request("GET", "x")
        ),
    )
    with pytest.raises(WeatherUnavailable):
        OpenMeteoClient(0, 0, 0, 0, "UTC").fetch(date(2026, 1, 1), date(2026, 1, 2))


# --- Home Assistant -------------------------------------------------------------


class FakeSocket:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def recv(self, timeout=None):
        return json.dumps(self.replies.pop(0))

    def send(self, message):
        self.sent.append(json.loads(message))


def _patch_ws(monkeypatch, replies):
    sock = FakeSocket(replies)
    import websockets.sync.client as ws_client

    monkeypatch.setattr(ws_client, "connect", lambda url, **kw: sock)
    return sock


def test_ha_statistics_over_websocket(monkeypatch):
    start_ms = datetime(2026, 9, 22, 19, 0, tzinfo=UTC).timestamp() * 1000
    sock = _patch_ws(
        monkeypatch,
        [
            {"type": "auth_required"},
            {"type": "auth_ok"},
            {
                "id": 1,
                "type": "result",
                "success": True,
                "result": {SOLAR: [{"start": start_ms, "end": start_ms + 3.6e6, "change": 1.05}]},
            },
        ],
    )
    client = HAClient("http://ha.test:8123", "tok")
    start = datetime(2026, 9, 22, tzinfo=TZ)
    result = client.statistics([SOLAR, EXPORT], start, start + timedelta(days=1), "hour")
    assert result[SOLAR][0][1] == 1.05
    assert result[SOLAR][0][0] == datetime(2026, 9, 22, 19, 0, tzinfo=UTC)
    assert result[EXPORT] == []
    auth, request = sock.sent
    assert auth == {"type": "auth", "access_token": "tok"}
    assert request["type"] == "recorder/statistics_during_period"
    assert request["types"] == ["change"] and request["period"] == "hour"
    assert client._ws_url() == "ws://ha.test:8123/api/websocket"
    assert HAClient("https://x.ui.nabu.casa", "t")._ws_url() == "wss://x.ui.nabu.casa/api/websocket"


def test_ha_bad_token_and_error_reply(monkeypatch):
    _patch_ws(monkeypatch, [{"type": "auth_required"}, {"type": "auth_invalid"}])
    with pytest.raises(HAUnavailable, match="token"):
        HAClient("http://ha", "bad").statistics(
            [SOLAR], datetime.now(UTC), datetime.now(UTC), "hour"
        )
    _patch_ws(
        monkeypatch,
        [
            {"type": "auth_required"},
            {"type": "auth_ok"},
            {"success": False, "error": {"code": "x"}},
        ],
    )
    with pytest.raises(HAUnavailable):
        HAClient("http://ha", "t").statistics([SOLAR], datetime.now(UTC), datetime.now(UTC), "hour")
    with pytest.raises(HAUnavailable):
        HAClient("", "")


def test_ha_rest_calls(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, json, headers["Authorization"]))
        return httpx.Response(200, json=[], request=httpx.Request("POST", url))

    def fake_get(url, headers, timeout):
        body = {"message": "API running."} if url.endswith("/api/") else {"state": "idle"}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    client = HAClient("http://ha.test:8123", "tok")
    client.speak("tts.home_assistant_cloud", "media_player.x", "hello")
    assert calls[0][0].endswith("/api/services/tts/speak")
    assert calls[0][1] == {
        "entity_id": "tts.home_assistant_cloud",
        "media_player_entity_id": "media_player.x",
        "message": "hello",
    }
    assert calls[0][2] == "Bearer tok"
    assert client.ping() == "API running."
    assert client.state("media_player.x")["state"] == "idle"

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: httpx.Response(500, request=httpx.Request("POST", "x"))
    )
    with pytest.raises(HAUnavailable):
        client.speak("tts.x", "media_player.x", "hi")


def test_telegram_client(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append((url, json))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    TelegramClient("123:abc").send("-100", "hello")
    assert sent[0][1] == {"chat_id": "-100", "text": "hello"}
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(400, json={"ok": False}, request=httpx.Request("POST", "x")),
    )
    with pytest.raises(TelegramError) as excinfo:
        TelegramClient("123:secret").send("-100", "hello")
    assert "secret" not in str(excinfo.value)


def test_recording_and_replay_roundtrip(monkeypatch):
    day = date(2026, 9, 22)
    fx = fixture(day, *[REAL_2026_09_22[e] for e in (SOLAR, IMPORT, EXPORT)])
    ha = RecordingHA(FixtureHA(fx))
    start = datetime(2026, 9, 22, tzinfo=TZ)
    out = ha.statistics([SOLAR], start, start + timedelta(days=1), "hour")
    assert len(out[SOLAR]) == 24 and ha.log[0]["period"] == "hour"
    weather = RecordingWeather(FixtureWeather(fx))
    weather.fetch(day, day + timedelta(days=1))
    assert weather.log and ha.ping().startswith("API")
    ha.speak("tts.x", "media_player.y", "hi")
    assert ha.state("x")["state"] == "fixture"
    with pytest.raises(WeatherUnavailable):
        FixtureWeather(fx).fetch(date(2020, 1, 1), date(2020, 1, 2))
    with pytest.raises(HAUnavailable):
        FixtureHA({"ha_unavailable": True}).statistics([SOLAR], start, start, "hour")
    with pytest.raises(WeatherUnavailable):
        FixtureWeather({"weather_unavailable": True}).fetch(day, day)
