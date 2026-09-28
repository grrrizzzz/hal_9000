# First live test

Each step is safe to run on its own and builds on the one before. Stop at the
first step that doesn't look right; the troubleshooting table at the end covers
the likely causes. All commands run from the repository folder:

```bash
cd ~/AI/hal_9000
git branch --show-current        # should say: solar-agent
```

## 1. Install and add the OpenAI key (2 minutes)

```bash
uv sync                          # needs uv: brew install uv
```

Open `.env` and put your OpenAI key on the `OPENAI_API_KEY` line (remove the
leading `#`). Everything else in `.env` is already set: Home Assistant, the
three Emporia entities, the speaker (Michael's Room), and the solar scale
correction.

The Telegram bot token in `.env` is a placeholder, so Telegram messages will
fail harmlessly. To see alerts in the terminal instead, put a `#` in front of
`TELEGRAM_BOT_TOKEN` until you have a real bot.

## 2. Check every connection (no cost, nothing played)

```bash
uv run hal9000 check
```

You want a ✓ on the config line, on the Home Assistant API and on each entity
(solar, grid import, grid export, speaker, TTS engine), on statistics and on
Open-Meteo. The statistics line shows today's solar so far; right after
midnight it may be empty.

Then the two that make noise or cost a fraction of a cent:

```bash
uv run hal9000 check --speak     # plays "This is HAL. Speaker test." in Michael's Room
uv run hal9000 check --llm       # one tiny call to each model tier
```

## 3. Yesterday, without the model (no cost, nothing played)

Yesterday (2026-09-27) is the day the Emporia multiplier was fixed, so it
exercises the scale correction and the rescale-spike guard.

```bash
uv run hal9000 run --date 2026-09-27 --no-llm --no-announce
```

Expect about 6.5 kWh of production (the pre-fix readings doubled) with the
+3.26 kWh rescale jump at 18:35 dropped. The band depends on the real weather
that day.

## 4. Calibrate (optional, a few minutes, no cost)

```bash
uv run hal9000 calibrate --days 300
```

This backfills hourly energy and weather and fits the expectation model to
your array. Until it has 90 clean days it uses the physics bootstrap
(clear-sky × performance ratio 0.85), which is fine for testing. It prints how
many days it stored and which model it ended up with.

## 5. Yesterday, with the model, dry run (a cent or two, nothing played)

```bash
uv run hal9000 run --date 2026-09-27 --shadow
```

The solar agent writes the report, and triage runs if the day looks
unexplained. In shadow mode nothing plays; the report is printed. Look at:

- `renderer=llm`: the model's report passed the numeric guard. `fallback`
  means the template was used; the `degraded:` lines say why.
- `cost=`: typically a cent or two; up to about $0.25 if triage ran (the soft cap).
- the transcript path: every tool call and result for the run.

## 6. The real thing

```bash
uv run hal9000 run --date 2026-09-27      # plays on Michael's Room
uv run hal9000 run                        # tonight at 20:00: today's report
uv run hal9000 cost                       # month-to-date spend
```

Once you're happy with it, point `HA_ENTITY_SPEAKER` in `.env` at the
production speaker. You can also try another speaker for one run with
`--speaker media_player.other_room`.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `check`: Home Assistant ✗ with a connection error | The Mac can't reach your Home Assistant host (`HA_BASE_URL`) (Wi-Fi, VPN). HAL ignores proxy settings for Home Assistant on purpose. |
| `check`: 401 / auth_invalid | The token in `.env` was revoked or mistyped: make a new long-lived token in HA (Profile → Security). |
| `check`: an entity ✗ | The entity was renamed in HA; update the matching `HA_ENTITY_*` line. |
| `--speak` fails but the API is ✓ | The TTS engine isn't usable. Set `HA_TTS_ENTITY=tts.google_translate_en_com` in `.env`. |
| `hal9000: SOLAR_FIXED_AT=... has no UTC offset` | Keep the `-07:00` on the end of that line. |
| `degraded: no LLM available` | `OPENAI_API_KEY` isn't set, or the line still starts with `#`. |
| `check --llm` fails with "model not found" | Your key can't use that model. Set `AGENT_MODEL_FAST` / `AGENT_MODEL_POWERFUL` in `.env` (e.g. `openai/gpt-4.1-mini`, `openai/gpt-4.1`). |
| `renderer=fallback`, `degraded: numeric guard rejected the narrative twice` | The model kept rounding numbers. The report was still delivered from the template. The transcript shows which numbers it tried. |
| Band `sensor_fault` | Solar read below grid export in some hour: the Emporia solar channel is wrong. Check its multiplier in the Emporia app. |
| Band `partial_data` or "production data is missing" | Home Assistant has gaps in the solar statistics for that day. Missing data is never treated as zero. |

Everything a run writes is under `data/`: `hal9000.db` (history, alerts, cost
ledger), `runs/` (a transcript per run) and `health.json` (the last run's
outcome). Delete `data/` to start fresh.
