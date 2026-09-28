"""HAL 9000 command line.

    python -m hal9000 check                  test every connection (HA, Open-Meteo, Telegram, LLM)
    python -m hal9000 run                    today's evening run: agent report, announced on Sonos
    python -m hal9000 run --shadow           dry run: nothing played; report to Telegram/console
    python -m hal9000 run --no-llm           deterministic template report, zero API cost
    python -m hal9000 run --date 2026-09-22  re-run a past date
    python -m hal9000 run --speaker media_player.kitchen   announce somewhere else, once
    python -m hal9000 run --replay data/runs/<id>.fixture.json   offline re-run
    python -m hal9000 calibrate --days 300   backfill history and fit the expectation model
    python -m hal9000 cost                   month-to-date spend
    python -m hal9000 serve                  HTTP API for the scheduler and watchdog

Exit codes: 0 report delivered (status ok or degraded); 1 no report; 2 configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import find_dotenv, load_dotenv

from hal9000 import __version__, config

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hal9000", description="HAL 9000 solar monitoring.")
    parser.add_argument("--version", action="version", version=f"hal9000 {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the evening report")
    run.add_argument("--date", type=date.fromisoformat, help="report date (default: today)")
    run.add_argument("--shadow", action="store_true", help="dry run: no Sonos, Telegram prefixed")
    run.add_argument("--no-llm", action="store_true", help="deterministic report, no API cost")
    run.add_argument("--no-announce", action="store_true", help="don't play or send the report")
    run.add_argument("--speaker", help="media_player entity to announce on, this run only")
    run.add_argument("--replay", help="re-run from a recorded fixture file (offline)")
    run.add_argument("--json", action="store_true", help="print the result as JSON")

    check = sub.add_parser("check", help="test every connection")
    check.add_argument(
        "--llm", action="store_true", help="also make one tiny call to each model tier"
    )
    check.add_argument("--telegram", action="store_true", help="also send a test Telegram message")
    check.add_argument("--speak", action="store_true", help="also play a short test on the speaker")

    cal = sub.add_parser("calibrate", help="backfill history and fit the expectation model")
    cal.add_argument("--days", type=int, default=300)

    sub.add_parser("cost", help="month-to-date LLM spend")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8900)
    return parser


def _print_result(result, as_json: bool) -> None:
    if as_json:
        print(result.model_dump_json(indent=2))
        return
    print()
    for step, outcome in result.trace:
        print(f"  {step:<28} {outcome}")
    print()
    print(result.report_text or "(no report)")
    print()
    print(
        f"status={result.status}  renderer={result.renderer}  band={result.band}  "
        f"cost=${result.cost_usd:.4f}  steps={result.iterations}"
    )
    for reason in result.degraded_reasons:
        print(f"  degraded: {reason}")
    print(f"transcript: {result.transcript_path}")


def cmd_run(args) -> int:
    from hal9000.coordinator.runner import RunOptions, run_evening

    options = RunOptions(
        day=args.date,
        shadow=args.shadow,
        no_llm=args.no_llm,
        announce=not args.no_announce,
        speaker=args.speaker,
        replay=args.replay,
    )
    if args.date and not args.replay:
        tz = ZoneInfo(config.settings().timezone)
        today = datetime.now(tz).date()
        if args.date < today:
            options.now = datetime(args.date.year, args.date.month, args.date.day, 20, 0, tzinfo=tz)
    result = run_evening(options)
    _print_result(result, args.json)
    return 0 if result.report_delivered else 1


def _line(ok: bool | None, label: str, detail: str = "") -> None:
    mark = {True: "✓", False: "✗", None: "–"}[ok]
    print(f"  {mark} {label}" + (f": {detail}" if detail else ""))


def cmd_check(args) -> int:
    from hal9000.solar.clients.homeassistant import HAClient
    from hal9000.solar.clients.openmeteo import OpenMeteoClient, hourly_for_day
    from hal9000.solar.clients.telegram import TelegramClient
    from hal9000.solar.domain.sitecfg import load_site
    from hal9000.solar.domain.tariff import load_tariff
    from hal9000.solar.energy import fetch_day_energy

    try:
        settings = config.settings()
    except config.ConfigError as error:
        _line(False, "settings (.env)", str(error))
        return 1
    tz = ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    failures = 0
    print(f"HAL 9000 {__version__} — connection check ({now:%Y-%m-%d %H:%M})")

    try:
        site = load_site(settings.config_dir)
    except Exception as error:
        _line(False, "config/site.yaml", f"{type(error).__name__}: {error}")
        return 1
    try:
        tariff = load_tariff(settings.config_dir)
        rates = tariff.day_rates(now.date())
        _line(
            True,
            "config",
            f"site {site.dc_kw} kW at {site.tilt_deg}°/{site.azimuth_deg}°; "
            f"today's rates {rates.rates} ({rates.season}"
            f"{', provisional' if rates.provisional else ''})",
        )
    except Exception as error:
        _line(False, "config/tariff.yaml", f"{type(error).__name__}: {error}")
        failures += 1
    _line(True, "solar scale", f"x{settings.solar_scale} before {settings.solar_fixed_at}")

    if not settings.ha_configured:
        _line(False, "Home Assistant", "HA_BASE_URL / HA_TOKEN not set in .env")
        failures += 1
    else:
        try:
            ha = HAClient(settings.ha_base_url, settings.ha_token)
            _line(True, "Home Assistant API", ha.ping())
            for label, entity in (
                ("solar", settings.entity_solar),
                ("grid import", settings.entity_grid_import),
                ("grid export", settings.entity_grid_export),
                ("speaker", settings.entity_speaker),
                ("TTS engine", settings.tts_entity),
            ):
                try:
                    state = ha.state(entity)
                    _line(True, f"  {label}", f"{entity} = {state.get('state')}")
                except Exception as error:
                    _line(False, f"  {label}", f"{entity}: {error}")
                    failures += 1
            energy = fetch_day_energy(ha, now.date(), settings, site, now)
            totals = energy.totals()
            _line(
                energy.hours_with_data > 0,
                "  statistics (websocket)",
                f"today so far: solar {totals['solar']:.2f} kWh, import {totals['import']:.2f}, "
                f"export {totals['export']:.2f} ({energy.resolution}); "
                f"sensor_fault={energy.sensor_fault}",
            )
            if args.speak:
                ha.speak(settings.tts_entity, settings.entity_speaker, "This is HAL. Speaker test.")
                _line(True, "  speaker test", f"played on {settings.entity_speaker}")
        except Exception as error:
            _line(False, "Home Assistant", str(error))
            failures += 1

    try:
        client = OpenMeteoClient(
            site.latitude, site.longitude, site.tilt_deg, site.openmeteo_azimuth, site.timezone
        )
        payload = client.fetch(now.date(), now.date() + timedelta(days=1))
        rows = hourly_for_day(payload, now.date(), site.timezone)
        _line(
            True,
            "Open-Meteo",
            f"today's plane-of-array insolation {sum(r['poa'] for r in rows):.2f} kWh/m2",
        )
    except Exception as error:
        _line(False, "Open-Meteo", str(error))
        failures += 1

    if settings.telegram_configured:
        if args.telegram:
            try:
                TelegramClient(settings.telegram_bot_token).send(
                    settings.telegram_chat_id, "HAL 9000 connection test."
                )
                _line(True, "Telegram", f"test message sent to {settings.telegram_chat_id}")
            except Exception as error:
                _line(False, "Telegram", str(error))
                failures += 1
        else:
            _line(True, "Telegram", "configured (add --telegram to send a test message)")
    else:
        _line(None, "Telegram", "not configured; alerts will print to the console")

    try:
        config.require_llm_key()
        provider = config.resolve_provider()
        detail = (
            f"{provider}: fast={config.model_for('fast')}, powerful={config.model_for('powerful')}"
        )
        if args.llm:
            from hal9000.core.prompt import Prompt
            from hal9000.llm import generate_response

            for tier in ("fast", "powerful"):
                reply = generate_response(
                    Prompt(messages=[{"role": "user", "content": "Reply with the word ready."}]),
                    tier,
                )
                detail += f"; {tier} replied {reply.strip()[:20]!r}"
        _line(True, "LLM", detail)
    except Exception as error:
        _line(False, "LLM", f"{error} (runs will use the deterministic report)")
        failures += 1

    print()
    print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
    return 0 if failures == 0 else 2


def cmd_calibrate(args) -> int:
    from hal9000.coordinator.runner import open_database
    from hal9000.solar.calibration import calibrate
    from hal9000.solar.clients.homeassistant import HAClient
    from hal9000.solar.clients.openmeteo import OpenMeteoClient
    from hal9000.solar.domain.sitecfg import load_site

    settings = config.settings()
    site = load_site(settings.config_dir)
    ha = HAClient(settings.ha_base_url, settings.ha_token)
    weather = OpenMeteoClient(
        site.latitude, site.longitude, site.tilt_deg, site.openmeteo_azimuth, site.timezone
    )
    today = datetime.now(ZoneInfo(settings.timezone)).date()
    result = calibrate(ha, weather, open_database(settings), settings, site, args.days, today)
    print(json.dumps(result, indent=2))
    return 0


def cmd_cost(args) -> int:
    from hal9000.coordinator.runner import open_database

    settings = config.settings()
    today = datetime.now(ZoneInfo(settings.timezone)).date()
    db = open_database(settings)
    spent = db.month_cost(today.year, today.month)
    print(f"{today:%Y-%m}: ${spent:.4f} of ${settings.monthly_cap_usd:.2f} monthly cap")
    for row in db.cost_breakdown(today.year, today.month):
        print(f"  {row['agent']:<8} {row['model']:<28} calls={row['calls']:<4} ${row['usd']:.4f}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("hal9000.coordinator.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    handlers = {
        "run": cmd_run,
        "check": cmd_check,
        "calibrate": cmd_calibrate,
        "cost": cmd_cost,
        "serve": cmd_serve,
    }
    try:
        return handlers[args.command](args)
    except config.ConfigError as error:
        print(f"hal9000: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
