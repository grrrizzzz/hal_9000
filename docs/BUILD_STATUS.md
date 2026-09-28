# Build status

**Branch:** `solar-agent` · **State:** ready for the first live test
([`MORNING_TEST.md`](MORNING_TEST.md))

## What is built

The whole plan is implemented for manual CLI runs:

- the evening coordinator;
- the solar agent (fast tier) and the triage sub-agent (powerful tier);
- the thirteen solar tools and seven triage tools;
- Home Assistant (websocket statistics, TTS), Open-Meteo and Telegram clients;
- the TOU calendar and effective-dated tariff;
- savings and net position;
- clear-sky physics and the fitted expectation model;
- the anomaly profile;
- the numeric guard and the deterministic fallback report;
- the SQLite history, cost metering and caps;
- record and replay;
- the `check`, `run`, `calibrate`, `cost` and `serve` commands;
- the HTTP API for n8n later.

## Verification

| | |
|---|---|
| Offline suite | 427 tests pass; 95% coverage; ruff clean |
| Replay scenarios | clear, cloudy, zero output with inverter-first triage, midday dropout, sensor fault, weather down, HA down, tariff not configured, guard rejections, cost caps, monthly cap, shadow mode, alert dedup and recovery, finalize-yesterday, missing and partial data |
| Real data | Home Assistant statistics for 2026-09-22 (a clear day), 2026-05-10 (the day that exposed the half-reading solar channel) and 2026-09-27 (the multiplier fix and its rescale spike) are replayed in tests |
| Live paths | Exercised against local fakes of the Home Assistant websocket and REST API, an OpenAI-compatible server (the request LiteLLM actually sends was captured), and Open-Meteo-shaped payloads. Every CLI command ran end to end |
| Not yet verified | The real Home Assistant, OpenAI and Open-Meteo endpoints, because the build sandbox can't reach the LAN or those hosts. `hal9000 check` verifies all three in one command |

An independent review found issues, and all of them were fixed:

- **The numeric guard now checks units.** "$13" is no longer backed by 13.0 kWh.
- **Missing solar readings are no longer treated as zero.** They are left out of the comparison, the new `partial_data` band covers days with too much missing, and missing hours no longer trip the sensor-fault check.
- **Announce reports its real outcome.**
- **SOLAR_FIXED_AT is validated.**
- **Open-Meteo is requested in UTC**, so DST days bucket correctly.
- **Recovery notices clear only what the day proves.** Net-position alerts are never cleared by them.
- **Day quality flags are stored**, which keeps bad days out of calibration.
- **The guard, completeness and length checks each have their own correction counter.**
- **The cost ledger is stamped in local time**, so month-end runs land in the right month.
- **Triage tools scale readings before applying the physical limit, and fall back to hourly data.**
- **Home Assistant calls bypass any proxy.**

## Where the build differs from the plan

| Plan | Build | Why |
|---|---|---|
| Template's `safety/` package (staged review, reversible actions) | Removed | HAL's only side effects are the announcement, Telegram and its own records; budgets and shadow mode in `Delivery` cover them |
| Shadow mode as a capability | Implemented in `coordinator/delivery.py` | The coordinator owns every outbound channel, so one switch there catches all of them |
| Memory compaction capability | Not built | A run is at most 12 solar and 24 triage steps; prompts stay small and cached |
| Strict JSON-schema tool calling | Not sent | Tools use ordinary function schemas. The verdict is validated by pydantic, and a malformed verdict is returned for one correction |
| Output token ceiling 4096 | 8192 fast, 16384 powerful (`AGENT_MAX_TOKENS` overrides) | Reasoning tokens count toward the ceiling; 4096 could starve gpt-5 triage replies |
| — | New bands `partial_data` and `no_data` | Missing data must never read as a performance problem |
| Hourly series follow the clock | 24 wall-clock hours on DST days (the repeated or skipped hour is folded) | Keeps every series the same length; affects two nights a year |
| Triage tools take a date | Triage tools always use the run's date | The triage sub-agent investigates the day being reported, nothing else |
| Bootstrap expectation from clear-sky only | Bootstrap uses the day's actual plane-of-array irradiance × PR 0.85 | Compares like with like before 90 clean days exist for fitting |
| Live acceptance checks during the build | Deferred to `hal9000 check` and the morning test | The build sandbox can't reach the LAN or the external APIs |

## Known limits

- **Tariff.** Winter rates from 2026-11-01 are provisional until a winter bill
  confirms them; reports say "provisional" until then. Dates before 2026-01-01
  have no rates, so reports skip savings.
- **Expectation model.** It stays on the physics bootstrap until
  `hal9000 calibrate` has 90 clean days.
- **Model names.** The default models (`openai/gpt-5`, `openai/gpt-5-mini`)
  and their prices in `config.py` should be confirmed against your OpenAI
  account; `check --llm` confirms the models answer.
- **Telegram.** The bot token is a placeholder until a real bot is created.
- **Scheduling.** Nothing is scheduled yet. The Raspberry Pi and n8n are the
  next step.
