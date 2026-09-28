"""CLI, API, calibration, delivery, the LLM boundary, and the remaining triage tools."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from hal9000 import config
from hal9000.coordinator.delivery import Delivery
from hal9000.core.prompt import Prompt
from hal9000.solar.clients.replay import FixtureHA, FixtureWeather
from tests.fixture_builder import (
    EXPORT,
    IMPORT,
    REAL_2026_05_10,
    REAL_2026_09_22,
    SOLAR,
    TZ,
    fixture,
    weather_payload,
)

DAY = date(2026, 9, 22)


def clear_fx():
    return fixture(
        DAY, REAL_2026_09_22[SOLAR], REAL_2026_09_22[IMPORT], REAL_2026_09_22[EXPORT], 0.97
    )


# --- CLI ------------------------------------------------------------------------


def test_cli_run_replay_no_llm(hal_env, write_fixture, capsys):
    from hal9000.__main__ import main

    code = main(["run", "--replay", write_fixture(clear_fx()), "--no-llm", "--no-announce"])
    out = capsys.readouterr().out
    assert code == 0
    assert "13.0 kilowatt-hours" in out
    assert "status=ok" in out and "renderer=fallback" in out
    assert "transcript:" in out


def test_cli_run_json(hal_env, write_fixture, capsys):
    from hal9000.__main__ import main

    assert main(["run", "--replay", write_fixture(clear_fx()), "--no-llm", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["report_delivered"] is True and body["band"] == "normal"


def test_cli_help_and_no_command(capsys):
    from hal9000.__main__ import build_parser, main

    assert main([]) == 2
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--date", "2026-09-20", "--shadow", "--speaker", "media_player.x"]
    )
    assert args.date == date(2026, 9, 20) and args.shadow and args.speaker == "media_player.x"


def test_cli_cost(hal_env, capsys):
    from hal9000.__main__ import main
    from hal9000.coordinator.runner import open_database

    open_database(config.settings()).add_ledger(
        "r", "solar", {"model": "openai/gpt-5-mini"}, 0.0123
    )
    assert main(["cost"]) == 0
    out = capsys.readouterr().out
    assert "$0.0123" in out and "solar" in out


def test_cli_check_reports_failures_without_network(hal_env, monkeypatch, capsys):
    import httpx

    from hal9000.__main__ import main

    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", boom)
    import websockets.sync.client as ws_client

    monkeypatch.setattr(ws_client, "connect", boom)
    code = main(["check"])
    out = capsys.readouterr().out
    assert code == 2
    assert "Home Assistant" in out and "Open-Meteo" in out and "LLM" in out
    assert "check(s) failed" in out


def test_cli_run_past_date_sets_evening_time(hal_env, monkeypatch, capsys):
    from hal9000 import __main__ as cli
    from hal9000.coordinator import runner

    seen = {}

    def fake_run(options):
        seen["options"] = options
        return SimpleNamespace(
            report_delivered=True, trace=[], report_text="ok", status="ok", renderer="fallback",
            band="normal", cost_usd=0.0, iterations=0, degraded_reasons=[], transcript_path="x",
            model_dump_json=lambda **k: "{}",
        )  # fmt: skip

    monkeypatch.setattr(runner, "run_evening", fake_run)
    assert cli.main(["run", "--date", "2026-09-01"]) == 0
    assert seen["options"].now.hour == 20 and seen["options"].day == date(2026, 9, 1)


# --- API ------------------------------------------------------------------------


@pytest.fixture
def client(hal_env, monkeypatch):
    from fastapi.testclient import TestClient

    from hal9000.coordinator.api import app

    monkeypatch.setenv("HAL_API_TOKEN", "secret")
    return TestClient(app)


def test_api_requires_token(client):
    assert client.post("/runs/evening").status_code == 401
    assert client.post("/runs/evening", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_api_without_server_token_refuses(hal_env):
    from fastapi.testclient import TestClient

    from hal9000.coordinator.api import app

    assert TestClient(app).post("/runs/evening").status_code == 503


def test_api_replay_health_cost(client, write_fixture):
    auth = {"Authorization": "Bearer secret"}
    response = client.post(
        "/runs/replay", json={"fixture": write_fixture(clear_fx())}, headers=auth
    )
    assert response.status_code == 200 and response.json()["report_delivered"]
    health = client.get("/health").json()
    assert health["last_status"] == "ok" and health["last_success"]
    cost = client.get("/cost", headers=auth).json()
    assert cost["cap_usd"] == 10.0


def test_api_evening_statuses(client, monkeypatch):
    from hal9000.contracts import RunResult
    from hal9000.coordinator import api

    auth = {"Authorization": "Bearer secret"}
    ok = RunResult(
        run_id="r",
        date="2026-09-22",
        status="degraded",
        report_delivered=True,
        record_written=True,
        announced=True,
        renderer="fallback",
        report_text="x",
    )
    monkeypatch.setattr(api, "run_evening", lambda options: ok)
    body = client.post("/runs/evening", json={"no_llm": True}, headers=auth).json()
    assert body["status"] == "degraded"

    def config_error(options):
        raise config.ConfigError("missing")

    monkeypatch.setattr(api, "run_evening", config_error)
    assert client.post("/runs/evening", headers=auth).status_code == 422
    failed = ok.model_copy(update={"report_delivered": False})
    monkeypatch.setattr(api, "run_evening", lambda options: failed)
    assert client.post("/runs/evening", headers=auth).status_code == 500


def test_api_health_before_any_run(client):
    assert client.get("/health").json()["last_success"] is None


# --- calibration -----------------------------------------------------------------


def test_calibrate_backfills_and_fits(hal_env, monkeypatch):
    from hal9000.coordinator.runner import open_database
    from hal9000.solar.calibration import calibrate, maybe_refit
    from hal9000.solar.domain import expectation
    from hal9000.solar.domain.sitecfg import load_site

    monkeypatch.setattr(expectation, "MIN_FIT_DAYS", 5)
    import hal9000.solar.calibration as cal

    monkeypatch.setattr(cal, "MIN_FIT_DAYS", 5)
    today = date(2026, 9, 27)
    days = {
        today - timedelta(days=i): (REAL_2026_09_22 if i % 2 else REAL_2026_05_10)
        for i in range(1, 11)
    }
    first = min(days)
    fx = fixture(
        first,
        *[days[first][e] for e in (SOLAR, IMPORT, EXPORT)],
        extra_days={d: v for d, v in days.items() if d != first},
    )
    fx["weather"] = [
        {
            "start": (today - timedelta(days=15)).isoformat(),
            "end": today.isoformat(),
            "payload": _range_payload(today - timedelta(days=12), today),
        }
    ]
    settings = config.settings()
    db = open_database(settings)
    site = load_site(settings.config_dir)
    result = calibrate(FixtureHA(fx), FixtureWeather(fx), db, settings, site, 10, today)
    assert result["energy_days"] == 10
    assert result["weather_days"] >= 9
    assert result["model"]["kind"] == "fitted"
    assert maybe_refit(db, site, today) is None  # fresh: no refit needed


def _range_payload(start: date, end: date) -> dict:
    payloads = [weather_payload(start + timedelta(days=i)) for i in range((end - start).days + 1)]
    merged = {"hourly": {k: [] for k in payloads[0]["hourly"]}}
    seen = set()
    for p in payloads:
        for i, t in enumerate(p["hourly"]["time"]):
            if t in seen:
                continue
            seen.add(t)
            for k in merged["hourly"]:
                merged["hourly"][k].append(p["hourly"][k][i])
    return merged


def test_cli_calibrate_uses_real_clients(hal_env, monkeypatch, capsys):
    from hal9000 import __main__ as cli
    from hal9000.solar import calibration

    monkeypatch.setattr(calibration, "calibrate", lambda *a, **k: {"energy_days": 0})
    assert cli.main(["calibrate", "--days", "5"]) == 0
    assert '"energy_days": 0' in capsys.readouterr().out


# --- delivery --------------------------------------------------------------------


class RecordingTelegram:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def send(self, chat_id, text):
        if self.fail:
            raise RuntimeError("down")
        self.sent.append((chat_id, text))


def _settings(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return config.settings()


def test_delivery_shadow_single_chat_prefix(monkeypatch):
    s = _settings(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="123:x",
        TELEGRAM_CHAT_ID="-100",
        TELEGRAM_TEST_CHAT_ID="-100",
    )
    tg = RecordingTelegram()
    d = Delivery(s, ha=None, telegram=tg, shadow=True)
    assert d.send_alert("Low output.", "warn")
    assert d.announce("Report text.") == "logged"
    assert all(text.startswith("[DRY RUN]") for _, text in tg.sent)
    assert tg.sent[0][0] == "-100"


def test_delivery_shadow_separate_test_chat(monkeypatch):
    s = _settings(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="123:x",
        TELEGRAM_CHAT_ID="-100",
        TELEGRAM_TEST_CHAT_ID="-200",
    )
    tg = RecordingTelegram()
    Delivery(s, ha=None, telegram=tg, shadow=True).send_alert("x", "info")
    assert tg.sent[0][0] == "-200" and not tg.sent[0][1].startswith("[DRY RUN]")


def test_delivery_cap_and_failures(monkeypatch):
    s = _settings(monkeypatch, TELEGRAM_BOT_TOKEN="123:x", TELEGRAM_CHAT_ID="-100")
    d = Delivery(s, ha=None, telegram=RecordingTelegram())
    for _ in range(3):
        d.send_alert("x")
    with pytest.raises(RuntimeError, match="limit"):
        d.send_alert("x")
    failing = Delivery(s, ha=None, telegram=RecordingTelegram(fail=True))
    assert failing.send_alert("x") is False


def test_delivery_announce_falls_back_to_telegram(monkeypatch):
    s = _settings(monkeypatch, TELEGRAM_BOT_TOKEN="123:x", TELEGRAM_CHAT_ID="-100")

    class BrokenHA:
        def speak(self, *a):
            raise RuntimeError("speaker offline")

    tg = RecordingTelegram()
    assert Delivery(s, ha=BrokenHA(), telegram=tg).announce("Report.") == "telegram"
    assert "speaker unavailable" in tg.sent[0][1]
    # speaker failed and Telegram not wired: printed only, which is not delivery
    assert Delivery(s, ha=None, telegram=None).announce("Report.") == "telegram_logged"

    class BrokenTelegram:
        def send(self, *a):
            raise RuntimeError("telegram down")

    assert Delivery(s, ha=BrokenHA(), telegram=BrokenTelegram()).announce("R.") == "failed"
    assert Delivery(s, ha=None, telegram=None, announce_enabled=False).announce("x") == "logged"


# --- the LLM boundary ---------------------------------------------------------------


def test_generate_response_reports_usage_and_openai_params(monkeypatch, openai_env):
    import litellm

    from hal9000 import llm

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="hello", tool_calls=None)
        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=60),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )  # fmt: skip
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    seen = []
    prompt = Prompt(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
        metadata={"cache_key": "hal-solar"},
    )
    assert llm.generate_response(prompt, "fast", on_usage=seen.append) == "hello"
    assert seen[0] == {
        "model": "openai/gpt-5-mini",
        "prompt": 100,
        "cached": 60,
        "completion": 20,
        "reasoning": 5,
    }
    assert captured["parallel_tool_calls"] is False
    assert captured["reasoning_effort"] == "low"
    assert captured["extra_body"] == {"prompt_cache_key": "hal-solar"}


def test_extract_usage_tolerates_missing_fields():
    from hal9000.llm import extract_usage

    assert extract_usage(SimpleNamespace(), "m") == {
        "model": "m",
        "prompt": 0,
        "cached": 0,
        "completion": 0,
        "reasoning": 0,
    }
    assert extract_usage({"usage": {"prompt_tokens": 3}}, "m")["prompt"] == 3


# --- remaining triage tools, called directly ------------------------------------------


@pytest.fixture
def run_ctx(hal_env):
    from hal9000.coordinator.runner import open_database
    from hal9000.solar import pipeline
    from hal9000.solar.domain.sitecfg import load_site, load_thresholds
    from hal9000.solar.domain.tariff import load_tariff
    from hal9000.solar.run_context import RunContext

    settings = config.settings()
    fx = clear_fx()
    ctx = RunContext(
        day=DAY,
        now=datetime(2026, 9, 22, 20, 0, tzinfo=TZ),
        run_id="t",
        settings=settings,
        site=load_site(settings.config_dir),
        thresholds=load_thresholds(settings.config_dir),
        tariff=load_tariff(settings.config_dir),
        db=open_database(settings),
        ha=FixtureHA(fx),
        weather=FixtureWeather(fx),
        delivery=Delivery(settings, None, None),
        features=config.HalFeatures(),
    )
    pipeline.ensure_core_steps(ctx)
    return ctx


def test_triage_tools_direct(run_ctx):
    from hal9000.solar.tools import triage_tools as t

    handover = t.read_handover(_run=run_ctx)
    assert handover["figures"]["production_kwh"] == "13.0"
    curve = t.get_intraday_curve(
        60, _run=run_ctx
    )  # fixture has hourly only -> empty 5-minute curve
    assert "max_kw" in curve
    with pytest.raises(ValueError):
        t.get_intraday_curve(7, _run=run_ctx)
    history = t.get_sensor_history("solar", 3, _run=run_ctx)
    assert history["sensor"] == "solar"
    with pytest.raises(ValueError):
        t.get_sensor_history("wind", _run=run_ctx)
    irradiance = t.get_irradiance_curve(_run=run_ctx)
    assert "12:00" in irradiance["hours"]
    clear = t.compare_to_clearsky(_run=run_ctx)
    assert clear["clear_sky_expected_kwh"]
    assert t.find_similar_days(3, _run=run_ctx)["similar_days"] == []


def test_intraday_curve_from_five_minute_data(run_ctx):
    from hal9000.solar.tools import triage_tools as t

    start = datetime(2026, 9, 22, 9, 0, tzinfo=TZ)
    points = [
        [(start + timedelta(minutes=5 * i)).astimezone(UTC).isoformat(), 0.05] for i in range(24)
    ]
    run_ctx.ha = FixtureHA({"ha": [{"period": "5minute", "result": {SOLAR: points}}]})
    curve = t.get_intraday_curve(15, _run=run_ctx)
    assert curve["curve"]["09:00"] == pytest.approx(0.05 * 3 * 4 * 2)  # scaled x2 before the fix
    assert run_ctx.store.figure("triage_max_kw") == "1.2"


def test_similar_days_uses_records(run_ctx):
    from hal9000.contracts import NightlyRecord
    from hal9000.solar.tools import triage_tools as t

    run_ctx.db.append_record(
        NightlyRecord(
            date="2026-09-20",
            kind="final",
            production_kwh=12.5,
            expected_kwh=12,
            ratio=1.04,
            band="normal",
            insolation_kwh_m2=6.4,
        )
    )
    out = t.find_similar_days(2, _run=run_ctx)
    assert out["similar_days"][0]["production_kwh"] == "12.5"


def test_history_and_briefing(run_ctx):
    from hal9000.contracts import NightlyRecord
    from hal9000.solar import pipeline
    from hal9000.solar.agents.briefing import build_briefing

    run_ctx.db.append_record(
        NightlyRecord(
            date="2026-09-21",
            kind="final",
            production_kwh=12.0,
            expected_kwh=12,
            ratio=1.0,
            band="normal",
        )
    )
    history = pipeline.recent_history(run_ctx, 7)
    assert history["days_found"] == 1 and history["average_production_kwh"] == "12.0"
    run_ctx.db.open_alert("production_low", "warn", "x")
    briefing = build_briefing(run_ctx)
    assert "production_low" in briefing and '"production_kwh": "12.0"' in briefing


def test_solar_tool_errors(run_ctx):
    from hal9000.solar.tools import solar_tools as s

    with pytest.raises(ValueError, match="severity"):
        s.record_alert("k", "urgent", "x", _run=run_ctx)
    with pytest.raises(ValueError, match="not from tool results"):
        s.record_alert("k", "warn", "Production was 99.9 kWh.", _run=run_ctx)
    with pytest.raises(ValueError, match="not from tool results"):
        s.send_telegram("It made 42.0 kWh", _run=run_ctx)
    assert s.check_alert_state("nothing", _run=run_ctx) == {"alert_key": "nothing", "open": False}
    run_ctx.db.open_alert("x", "warn", "t")
    assert s.check_alert_state("x", _run=run_ctx)["open"] is True
    run_ctx.features.triage_enabled = False
    assert s.run_fault_triage("why", _run=run_ctx)["ran"] is False


def test_weather_cache(hal_env, tmp_path):
    from hal9000.solar.weather import day_weather
    from hal9000.storage.db import Database

    db = Database(tmp_path / "w.db")

    class CountingWeather:
        calls = 0

        def fetch(self, start, end):
            CountingWeather.calls += 1
            return weather_payload(start)

    client = CountingWeather()
    day = date(2026, 9, 1)
    day_weather(client, db, day, date(2026, 9, 22), "America/Los_Angeles")
    day_weather(client, db, day, date(2026, 9, 22), "America/Los_Angeles")
    assert CountingWeather.calls == 1  # the second read came from the cache


def test_openmeteo_live_shape_is_documented():
    """The fixture builder mirrors Open-Meteo's documented response shape."""
    payload = weather_payload(DAY)
    assert set(payload["hourly"]) == {
        "time",
        "global_tilted_irradiance",
        "shortwave_radiation",
        "cloud_cover",
        "temperature_2m",
    }
    assert len(payload["hourly"]["time"]) == 48
    assert Path(__file__).exists()
    assert ZoneInfo("America/Los_Angeles")
