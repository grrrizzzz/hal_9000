# HAL 9000

HAL watches the rooftop solar array. Every evening at 20:00 it reads the day's
production from Home Assistant, compares it with what the weather allowed,
works out what the solar was worth on the EV-TOU-5 tariff, and says so on the
speaker, in four or five sentences. When something looks wrong it digs in
(a triage sub-agent looks at the intraday curve, the sensor history and the
irradiance) and sends a Telegram alert.

```
$ uv run hal9000 run
Solar report for Tuesday. The panels produced 13.0 kilowatt-hours, against
13.4 expected for the weather, which is 97%. Today's solar was worth $4.14.
```

It is built on the GAME framework (Goals, Actions, Memory, Environment) that
this repository started as. The framework lives in `src/hal9000/core/`; the
solar monitoring is in `src/hal9000/solar/`.

**First time?** Follow [`docs/MORNING_TEST.md`](docs/MORNING_TEST.md). It goes
from `uv sync` to a report on the speaker, one safe step at a time.

---

## How a run works

HAL itself is a deterministic coordinator, not an LLM. It:

1. checks the month's LLM spend against the cap;
2. finalizes yesterday's record now that the day is complete, and refits the
   expectation model about once a week;
3. hands the **solar agent** (fast model) a briefing. The agent calls tools, one
   per data source (Home Assistant, Open-Meteo), then assessment, savings,
   history, and, if production is unexplained, the **triage sub-agent**
   (powerful model);
4. makes sure there is a report and a record whatever happened: if the agent
   failed, overspent, or wrote a number no tool produced, the deterministic
   template report is used instead;
5. speaks the report on the configured speaker (Telegram if the speaker fails);
6. writes the transcript, `data/health.json` and the cost ledger.

The model never passes numbers between tools. Every figure is computed in code,
stored in the run store, and spoken exactly as stored: the **numeric guard**
rejects any number in the narrative that no tool produced, with its unit
("$13" is not backed by 13.0 kWh).

Missing data is never zero production. Hours with too few solar readings are
left out of the comparison, and a day missing too much becomes `partial_data`.

## Commands

```
hal9000 check [--llm] [--telegram] [--speak]   test every connection
hal9000 run [--date YYYY-MM-DD] [--shadow] [--no-llm] [--no-announce]
            [--speaker media_player.x] [--replay FILE] [--json]
hal9000 calibrate [--days 300]                 backfill history, fit the model
hal9000 cost                                   month-to-date LLM spend
hal9000 serve [--host] [--port 8900]           HTTP API (for n8n, later)
```

Run them with `uv run hal9000 ...` (or `uv run python -m hal9000 ...`).

| Flag | Effect |
|---|---|
| `--shadow` | Dry run: nothing plays on the speaker; the would-be announcement and any alerts go to the Telegram test chat, or the main chat prefixed `[DRY RUN]` |
| `--no-llm` | Deterministic template report, no API cost; the run still counts as `ok` |
| `--no-announce` | Do everything except speak or send the report |
| `--speaker` | Announce on a different media player for this run only |
| `--replay` | Re-run offline from a fixture recorded by an earlier run |

`run` exits 0 when a report was produced, with status `ok` or `degraded` (the
reasons are printed), and non-zero when it could not produce one.

## Configuration

| What | Where |
|---|---|
| Secrets, Home Assistant, entities, speaker, Telegram, caps | `.env` (see [`.env.example`](.env.example)) |
| Site: location, tilt, azimuth, array size, degradation | `config/site.yaml` |
| Performance bands | `config/thresholds.yaml` |
| Tariff components, effective-dated | `config/tariff.yaml` |
| How the report should sound | `config/report_policy.txt` |
| Models, reasoning effort, token limits | `.env` (`AGENT_MODEL_*`, `AGENT_REASONING_*`, `AGENT_MAX_TOKENS`) |

To move announcements to a different speaker, change `HA_ENTITY_SPEAKER` in
`.env`.

## Layout

```
src/hal9000/
├── __main__.py         CLI
├── config.py           settings from .env, model tiers and prices, feature switches
├── llm.py              generate_response(Prompt, tier, on_usage=) -> str (LiteLLM)
├── contracts.py        Verdict, NightlyRecord, RunResult
├── coordinator/        runner.py (the evening run), delivery.py (speaker/Telegram), api.py
├── cost/               per-run and monthly cost metering
├── storage/            db.py (SQLite history), run_store.py (this run's figures)
├── solar/
│   ├── agents/         solar agent, triage sub-agent, briefing
│   ├── tools/          the tools each agent may call
│   ├── clients/        Home Assistant, Open-Meteo, Telegram, record/replay
│   ├── domain/         TOU calendar, tariff, savings, clear-sky, expectation model,
│   │                   anomaly profile, numerals
│   ├── reporting/      numeric guard, deterministic template
│   ├── energy.py       HA statistics -> clean hourly energy (scale, artifacts, gaps)
│   ├── pipeline.py     the deterministic steps the tools call
│   └── calibration.py  backfill and model fitting
├── core/               the GAME framework: loop, memory, actions, registry, capabilities
├── capabilities/ tools/ agents/   framework extras from the template
```

## Development

```bash
uv sync
uv run pytest            # offline: replay fixtures and a scripted model; coverage >= 85%
uv run ruff check . && uv run ruff format --check .
```

The suite makes no network calls. Real Home Assistant data from 2026-09-22 and
2026-05-10 drives the replay scenarios.

## Documentation

| | |
|---|---|
| [`docs/MORNING_TEST.md`](docs/MORNING_TEST.md) | First live test, step by step |
| [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md) | What is built, what was verified, and where the build differs from the plan |
| [`docs/SOLAR_AGENT_PLAN.md`](docs/SOLAR_AGENT_PLAN.md) | The design |
| [`docs/SOLAR_BUILD_HANDOFF.md`](docs/SOLAR_BUILD_HANDOFF.md) | Build orchestration and acceptance criteria |
| [`docs/ADR/`](docs/ADR/) | Decisions: LiteLLM (0001), deterministic coordinator without LangGraph (0002) |
| [`docs/template/`](docs/template/) | The original GAME template's architecture notes |

## License

MIT — see [`LICENSE`](LICENSE).
