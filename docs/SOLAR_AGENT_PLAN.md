# HAL 9000 Solar Monitoring — Plan

**Repository:** `hal_9000`, which *is* the source code. Its GAME framework (Goals, Actions, Memory, Environment) is the foundation, and everything below is built into it in place.
**Site:** 2.47 kW DC array · Delta SOLIVIA 3.8 NA G4 TL inverter · Emporia Vue 3 · SDG&E delivery / Clean Energy Alliance generation · EV-TOU-5 · NEM 1.0
**Companion document:** [`SOLAR_BUILD_HANDOFF.md`](SOLAR_BUILD_HANDOFF.md) — work packages, agent roles and acceptance criteria for building this plan.

---

## 1. Purpose and success criteria

Every evening at 8 pm, HAL reports on the solar array aloud on Sonos: energy generated today, how that compares with what the weather warranted, what it was worth, trends, and anything that needs attention. Faults are caught early and diagnosed, and the inverter is named first when output collapses.

The build has succeeded when all of the following hold:
- A report is delivered every evening without a manual step.
- Every number the household hears comes from a deterministic Python function, never from the model's arithmetic.
- Savings figures reconcile with the utility bills.
- A real fault produces a specific, correctly routed alert, and an ordinary cloudy day produces none.
- Spend stays under the configured caps, and the design has a hard ceiling no bug can defeat.
- Nothing about the agents can cause a silent evening: every failure path ends in a deterministic report or an external alarm.

---

## 2. Site and utility facts

| Item | Value |
|---|---|
| Array | 2.47 kW DC, about 11 years old, with an estimated 5–6% cumulative degradation |
| Inverter | Delta SOLIVIA 3.8 NA G4 TL: transformerless, made around late 2014, out of warranty, inside the typical failure window for its generation |
| DC/AC ratio | ~0.65 (the inverter is heavily oversized, so clipping is not a factor) |
| Location | 33.25, −117.30 (Oceanside, CA) · coastal |
| Orientation | tilt ≈ 33°, azimuth 180° compass (due south) |
| Production meter | Emporia Vue 3 solar circuit, read through Home Assistant: `sensor.electron_inspector_energy_today_13` ("Solar Panels Energy Today"). The multiplier was set to 2.0 on 2026-09-27 at about 18:35. Earlier history reads half and is scaled ×2.0 (§2.1) |
| Grid import | `sensor.mainstogrid_energy_today` (Emporia `MainsToGrid`, friendly name "Cumulative Consumption"). Despite the name, this is **energy imported from the grid**, not total household use |
| Grid export | `sensor.mainsfromgrid_energy_today` (Emporia `MainsFromGrid`). Despite the name, this is **energy exported to the grid**. It matches the utility meter: 12.07 kWh vs Green Button 12.25 kWh on 2026-05-10 |
| Household load | Derived: import + solar − export (valid only once the solar channel is fixed) |
| Announcement speaker | `HA_ENTITY_SPEAKER` in `.env`, currently `media_player.unnamed_room_4` ("Michael's Room") for testing. The production speaker will differ: change that one line, or pass `--speaker <entity>` on the CLI for a single run. No code change either way |
| Telegram | One chat (`TELEGRAM_CHAT_ID`) for now. If `TELEGRAM_TEST_CHAT_ID` is unset or the same, dry-run messages go to the same chat prefixed `[DRY RUN]` |
| HA history | Long-term statistics (hourly, permanent) from **mid-November 2025**. March 2026 has a large gap |
| Utility meter | Meter 05203979 · Green Button 15-minute import/export channels (§12) |
| Utility | SDG&E (delivery) + Clean Energy Alliance (generation), EV-TOU-5, NEM 1.0 (grandfathered around 2015, valid to about 2035) |
| Net position | **Verified: a net consumer in every TOU period in all 12 billing cycles** (Sep 2025 – Aug 2026, NEM summary on the August bill; Appendix A). NEM nets each TOU period over the billing cycle, so single net-export days don't matter. Savings = generation × marginal rate for each period holds. The daily net-position check stays on as a trend watch |
| Marginal rate structure | Per TOU period: SDG&E delivery (UDC) + Wildfire Fund Charge $0.00591 + PCIA 2023 + IPCA $0.00006, all × 1.011 (Franchise Fee 1.10%), + State Regulatory Fee $0.001, + CEA generation + Clean Impact Plus $0.001. SDG&E's delivery charges are the **same in summer and winter**; only CEA's generation rate changes by season. CEA product: **Clean Impact Plus** |
| Summer marginal rates (current, from the August 2026 bill) | on-peak **$0.86691**/kWh · off-peak **$0.50982** · super off-peak **$0.12566**. Components: UDC $0.31218 / $0.31218 / $0.04114, PCIA $0.02613, CEA $0.51684 / $0.15975 / $0.04961. Delivery rates changed on 2026-08-01 (the bill's "Rate 1" was UDC $0.31711, PCIA $0.02828). Each rate is stored with an effective date, so historical dates use the rate in force then |
| PCIA | Vintage 2023: $0.02823 (Jan–Jul 2026), $0.02828 (late Jul 2026), **$0.02613** (from 2026-08-01). **Included** in the marginal rates above |
| Rate change 2026-05-01 | The weekday 10:00–14:00 super off-peak window now applies year-round. This cut the value of midday solar by about 75%, and a blended daily rate would overstate savings by about 2× |
| Winter marginal rates | **From 2026-11-01 (provisional):** current delivery (August 2026) + CEA winter generation ($0.24430 / $0.15782 / $0.05187) = on-peak **$0.59437** · off-peak **$0.50789** · super off-peak **$0.12792**. **Winter 2026 actual (February 2026 bill):** $0.60765 / $0.52117 / $0.12561, used for historical dates. The forward values are flagged `provisional: true` until the November bill confirms them |
| Seasons | Summer June 1 – October 31 · winter November 1 – May 31 |
| TOU periods (all year) | Confirmed by the August 2026 bill. **Weekdays:** super off-peak 00:00–06:00 and 10:00–14:00 · off-peak 06:00–10:00, 14:00–16:00, 21:00–24:00 · on-peak 16:00–21:00. **Weekends and holidays:** super off-peak 00:00–14:00 · off-peak 14:00–16:00, 21:00–24:00 · on-peak 16:00–21:00. Before 2026-05-01, the weekday 10:00–14:00 super off-peak applied only in March and April (the February bill shows weekday super off-peak as 00:00–06:00 only) |
| Holidays | SDG&E's list: New Year's Day, Presidents' Day, Memorial Day, Independence Day, Labor Day, Veterans Day, Thanksgiving, Christmas (confirm, input 2) |
| NEM | 12-month relevant period; true-up date September 28 each year. CEA net surplus compensation is $0.06/kWh, which is irrelevant while the household is a net importer |

### 2.1 Data issue — the Emporia solar channel read about half (fix applied 2026-09-27)

The solar sensor reports **less energy than the house exports**, which is physically impossible: exported energy can only come from the array.
- **On 2026-05-10:** solar read 7.92 kWh for the day, while export was 12.07 kWh (Emporia) and 12.25 kWh (utility meter). At noon, solar read 1.28 kWh for the hour against 2.26 kWh exported.
- **Across September 2026:** solar reads 5.3–6.8 kWh on clear days, often below that day's export (up to 8.5 kWh). A clear-sky model expects about 12 kWh.
- **The export and import channels agree with the utility meter.** Only the solar channel is wrong.
- **Most likely cause:** the 240 V inverter circuit is measured with one CT on one leg, or the circuit's multiplier or double-pole setting is wrong in the Emporia app. Either way, it reads about 50%.

**Fix applied.** Mike set the solar circuit's multiplier to 2.0 in the Emporia app, because it is a 240 V circuit.
- **When it took effect:** at about 18:35 on 2026-09-27. Emporia rescaled that day's running total, so HA statistics show a one-off jump of +3.26 kWh in the 18:35 five-minute bucket, and the month-to-date sensor doubled.
- **What stays unchanged:** HA's stored statistics for earlier days keep the old, half-scale values.
- **Back-check at ×2 on 2026-05-10:** solar ≥ export in every hour (e.g. 12:00: 2.57 vs 2.26 kWh). The implied daytime household load is 0.2–0.8 kWh/h, consistent with the overnight 0.3–0.6 kWh/h. So 2.0 is consistent with the history.
- **Still to confirm:** the first sunny, exporting day after the fix must show solar ≥ export in every hour and a daily total near the clear-sky expectation (`scripts/check_solar_sensor.py`).

**Consequences for the build:**
- **History correction.** `SOLAR_SCALE=2.0` and `SOLAR_FIXED_AT=2026-09-27T18:35-07:00` (in `.env`; mirrored in `site.yaml`). Statistics before that instant are multiplied by the scale.
- **2026-09-27 is a transition day.** It is excluded from baseline training. Its daily total is taken as the post-rescale value (6.5 kWh), and its hourly shape as the old values × 2.
- **Artifact guard.** Any 5-minute solar change larger than the array can physically produce (2.47 kW × 5 min ≈ 0.21 kWh) is flagged `data_artifact` and excluded; hourly values are recomputed. The same guard catches Emporia resets and HA restarts.
- **A sensor-sanity check runs every evening.** `get_home_energy` flags `sensor_fault` if solar < export in any hour, and the report says so instead of reporting a performance shortfall. This is the `emporia_ge_export` check, run daily from Home Assistant data, with no dependence on Green Button.

---

## 3. The central design decision

The LLM gets authority over **control flow** and none over **arithmetic**.

| Concern | Who decides |
|---|---|
| Fetching data | The agent calls **one tool per source** (Home Assistant, Open-Meteo). Each tool validates the response, stores it, and returns a summary |
| Savings, tariff lookup, expected production, performance ratio, net position | Deterministic Python tools that read the stored data. The model passes **no numbers** to them |
| Whether a shortfall warrants investigation, and how deep to go | The solar agent, within an iteration cap |
| Diagnosis | The triage agent, returning a schema-validated verdict |
| Narrative prose | The LLM, guarded: every numeral must appear among the figures stored for the run |
| Alert severity | Deterministic classification bands, which the narrative cannot override |

**Numbers never pass through the model.** Every tool writes its figures to a run-scoped store. Tools read their inputs from that store, keyed by date, rather than taking arrays from the model. The final report is checked against the store, not against anything the model supplies, because a hallucinated number could otherwise appear in both the narrative and a model-built payload and still pass the check.

Fault triage is the part of the job that really branches. A zero-output day, a mid-afternoon dropout, a slow degradation trend and a cloudy day that merely looks like a fault each need different follow-up. That branching lives in a **triage sub-agent** with detailed, read-only diagnostic tools. The solar agent's own routine is nearly fixed, so it runs on a small, cheap model.

---

## 4. Agents

| Agent | Kind | Model | Role |
|---|---|---|---|
| **HAL 9000 — coordinator** | Deterministic Python (the service) | none in this build | Receives the 8 pm trigger, checks budgets, finalizes yesterday, briefs and dispatches the solar agent, verifies completeness, runs the fallback, owns the delivery channels (Sonos, Telegram), ledgers and health. It is the single entry point for every current and future agent |
| **Solar agent** | GAME agent, ≤ 12 iterations | small tier | Gathers the day's data per source, runs the deterministic analysis tools, decides whether to triage, raises alerts, writes the record, and delivers the guarded report |
| **Triage agent** | GAME sub-agent, ≤ 24 iterations, read-only | mid tier | Called by the solar agent through `run_fault_triage`. Returns a structured verdict |

**Why HAL is not an LLM agent yet.** The coordinator's decisions tonight are fixed: one schedule, one domain agent. An LLM layer above the solar agent would add cost, latency and a failure point without making any real decision. HAL becomes an LLM coordinator when there is something to route: the Telegram query agent ("how did last week go?") and future home domains (§18). Its registry, delivery channels and ledgers are built now so that step adds an agent rather than a redesign.

---

## 5. The daily run

**Schedule:** the run happens at **20:00** local time, and HAL announces on Sonos **as soon as it finishes**, typically within a few minutes.

**How it's triggered:** **by hand first.** Mike runs HAL from the command line (or with one `curl` to the API) whenever he wants, for any date. Automatic scheduling (n8n at 20:00, the Mac's wake schedule, the HA watchdog) is added later as its own phase, before the 21-day dry run (§14).

**Why 20:00 works.** The array faces south at a 33° tilt. From about 18:55 in summer the sun is behind the plane of the panels; a clear-sky model gives under 0.01% of daily production after 20:00 on any day of the year. The day's generation, and therefore its savings, is complete at run time. Household consumption is not: 20:00–24:00 is still to come. So anything that needs a full day of consumption (net position) uses **yesterday**, which HAL finalizes at the start of each run.

### HAL, before the agent (plain Python, `POST /runs/evening`)
1. A trigger starts the run: a manual `python -m hal9000 run` or `POST /runs/evening` now, and n8n at 20:00 once scheduling is added. The target date defaults to today.
2. **Monthly budget check.** If month-to-date spend is over the monthly cap, the agent is skipped: HAL renders the deterministic report, announces it, and sends one low-severity notice.
3. **Finalize yesterday.** HAL re-reads yesterday's full 24 hours (the production tail and consumption through midnight) and appends a *final* version of yesterday's record. Net position for yesterday is computed from this complete data.
4. **Briefing.** Yesterday's final summary, open alerts and the 7-day production trend are seeded as the solar agent's first working-memory entry.

### Solar agent (small model, ≤ 12 iterations)
5. **Gather, one tool per source:**
   - `get_home_energy(date)` reads Home Assistant: Emporia solar production and household consumption, hourly through the run time.
   - `get_weather(date)` reads Open-Meteo: hourly plane-of-array irradiance for the site's tilt and azimuth, plus cloud cover and temperature.
6. **Analyze** with deterministic tools that read the stored data:
   - `assess_performance()` gives expected production (ridge model, or the clear-sky bootstrap until 90 days of history exist), actual ÷ expected, and the classification band.
   - `compute_savings()` values each hour at its TOU rate. If rates are unset for the date, it returns a recoverable "tariff not configured".
   - `check_net_position()` covers yesterday's finalized full day.
7. **Investigate if the band is below normal:**
   - `get_hourly_anomaly_profile()` finds when in the day the shortfall happened.
   - `get_recent_history(days)` shows whether it is one day or a trend.
   - If the shortfall is severe, or its cause is still unclear, `run_fault_triage` hands off to the **triage agent**, which returns a structured verdict. For zero or near-zero output, the inverter is named first.
8. **Alerts.** `check_alert_state` applies dedup: only new or escalated alerts are recorded and sent, plus recovery notices for cleared ones. Telegram is capped at 3 sends per run.
9. `write_nightly_record()` writes today's provisional record, schema-validated.
10. `deliver_report(narrative)` runs the numeric guard. If it rejects the narrative, the agent gets one more turn to correct it; a second rejection sends the Jinja2 report instead.

### HAL, after the agent (plain Python)
11. **Completeness check.** If a guarded report was not delivered, or a record was not written, the deterministic path supplies whichever is missing. This catches a model that answers in prose.
12. **Announce.** The report plays on Sonos through Home Assistant TTS immediately. If TTS fails, it goes to Telegram instead.
13. Spend is recorded in the run and monthly ledgers, and health is updated. Home Assistant polls `/health` and raises an alarm by itself if no successful run appears for 26 hours.

---

## 6. Codebase

hal_9000 already provides a complete, tested GAME loop:
- a `@register_tool` decorator whose schema comes from each function's signature and docstring
- tag-sliced tool registries
- dependency injection the model never sees (`_name` parameters)
- ten capability hooks
- sub-agent delegation, with the agent registry withheld from callees
- a key-free test harness (`FakeLLM`), a no-network test, an 85% coverage gate, CI, and LiteLLM as the only provider layer

### Conversion
- Rename the package `hello_agent` → **`hal9000`**. Update `pyproject.toml` (project name `hal9000`), CI, `README.md` and `.env.example`.
- **Delete the greeter domain:** `tools/greeting.py`, `tools/guestbook.py`, `storage/guestbook.py`, `agents/greeter.py`, `agents/translator.py`, `config/greeting_policy.txt`, and their tests.
- **Move the template's history docs** (`BUILD_PLAN.md`, `ORCHESTRATOR_RULINGS.md`, `COVERAGE.md`, `EXTENDING.md`, `ARCHITECTURE.md`) into `docs/template/`. Keep `docs/ADR/0001-litellm.md` and add `ADR 0002 — HAL 9000 coordinator and solar agents on the hal_9000 framework`.
- **Keep:** `core/`, `tools/decorators.py`, `tools/system.py`, `tools/delegation.py`, `capabilities/logging.py`, `tests/fakes.py`, `tests/test_no_network.py` and CI.
- **Keep but leave off:** the time_aware, plan_first and progress_tracking capabilities, `tools/planning.py`, `tools/self_prompting.py`, and `safety/`.

### Framework changes (small, each covered by tests)

| # | File | Change | Why |
|---|---|---|---|
| F1 | `core/agent.py` | `Agent(action_tier: str = "powerful")`, used by `prompt_llm_for_action` | The loop is pinned to the powerful tier today. The solar agent must run on the fast tier and triage on the powerful one |
| F2 | `llm.py` | (a) Report `usage` (prompt, cached, completion and reasoning tokens, plus the model) to an injectable usage sink, keeping the `str` return contract. (b) Send `parallel_tool_calls=False`. (c) Accept `prompt_cache_key` | Ledgers need token counts. The loop honours only the first tool call, so extras must not be generated. Caching needs a stable key |
| F3 | `config.py`, `.env` | OpenAI tiers and cost rates. Raise `AGENT_MAX_TOKENS` to 4096, because reasoning tokens count toward output. Add a `HalFeatures` dataclass (`shadow_mode`, `triage_enabled`, `verbose_logging`). Remove the greeter flags | Provider, budget and runtime switches |
| F4 | delegation usage | Triage receives a **filtered** props dict: no `delivery`, `record_store`, `alert_store` or `ledger_admin` | Callees inherit every resource except the registry by default. Tags already stop triage using these; filtering removes them entirely |

### Layout
```
hal_9000/
├── src/hal9000/
│   ├── core/                 GAME framework (+F1)
│   ├── llm.py                LiteLLM boundary (+F2)
│   ├── config.py             tiers, rates, HalFeatures (+F3)
│   ├── contracts.py          pydantic: HomeEnergy, Weather, Performance, Savings, NetPosition,
│   │                         Verdict, NightlyRecord, Alert, RunResult, LedgerEntry
│   ├── coordinator/          HAL: api.py (FastAPI) · __main__.py (CLI) · runner.py
│   │                         (precheck → finalize → brief → dispatch → complete → announce)
│   │                         delivery.py (Sonos TTS, Telegram, budgets) · health.py
│   ├── storage/              db.py records.py alerts.py ledger.py run_store.py  (SQLite)
│   ├── cost/                 metering.py budget.py
│   ├── solar/
│   │   ├── domain/           tou_calendar.py tariff.py savings.py net_position.py
│   │   │                     clearsky.py expectation.py anomaly.py numerals.py  (pure)
│   │   ├── clients/          homeassistant.py openmeteo.py  (injectable)
│   │   ├── tools/            sources.py analysis.py history.py alerts.py records.py
│   │   │                     report.py triage.py diagnostics.py
│   │   ├── agents/           solar.py triage.py briefing.py
│   │   ├── environments/     solar.py replay.py shadow.py
│   │   └── reporting/        fallback.py guard.py templates/report.j2
│   └── tools/                decorators.py system.py delegation.py (kept)
├── scripts/                  inspect_greenbutton.py import_greenbutton.py ha_history_depth.py
├── config/                   site.yaml tariff.yaml thresholds.yaml report_policy.txt
├── deploy/                   launchd/ docker/ n8n/ ha/
├── tests/                    domain, clients, storage, reporting, cost, agent, delegation,
│                             service, reconciliation, evals · fixtures/ · greenbutton/
└── docs/                     SOLAR_AGENT_PLAN.md SOLAR_BUILD_HANDOFF.md ADR/ template/
```

---

## 7. G — Goals

Goals are `Goal` objects: separate records with a priority, rendered into one system message. `config/report_policy.txt` holds the household-facing voice (length, tone, how figures are spoken). It is read fresh on each run, so style changes need no deploy.

### Solar agent
| P | Name | Instruction (summary) |
|---|---|---|
| 1 | `data_integrity` | Never report on data you have not fetched. Every figure comes from a tool result in this run. A tool error or partial data is stated, never estimated. Missing production data is the headline, and a missing sensor is never treated as zero. If `get_home_energy` reports `sensor_fault`, the sensor is the headline: do not call it a performance shortfall and do not triage |
| 2 | `method` | Call `get_home_energy`, then `get_weather`, then `assess_performance`, `compute_savings` and `check_net_position`. Investigate only if the band is below normal. Then handle alerts, write the record, and deliver. One tool per turn |
| 3 | `assess_performance` | Judge the day against expected production, never against the 2.47 kW nameplate. The array is about 11 years old, and a nameplate comparison reads normal ageing as a permanent fault |
| 4 | `quantify_savings` | Use only `compute_savings`. Never compute savings yourself or use a blended rate. If the tariff is not configured, deliver the report without savings, note the gap, and carry on |
| 5 | `triage_faults` | If the band is low, fetch the anomaly profile and recent history. If the shortfall is severe or unexplained, call `run_fault_triage`. An inconclusive verdict is acceptable: report the shortfall without a cause rather than guessing. For zero or near-zero output, say "check the inverter" first |
| 6 | `watch_model_validity` | If `check_net_position` flags a period trending toward export, surface it as a warning. The savings model depends on it |
| 7 | `persist_and_notify` | Respect `check_alert_state`: no re-sends unless severity escalates, and always send a recovery notice. Write the record before delivering |
| 8 | `deliver` | Finish with `deliver_report`. The narrative must stay under about 45 seconds of speech. Speak figures exactly as tools formatted them (e.g. "8.2 kilowatt-hours", "$1.37"). A rejected report returns the offending numerals: correct it and call again |

### Triage agent
| P | Name | Instruction (summary) |
|---|---|---|
| 1 | `evidence_only` | Every claim cites the tool result it came from. Figures you want reported go in `figures` |
| 2 | `diagnose` | Separate the causes in order: inverter fault (zero or flat-lined output, abrupt cutoffs), weather (irradiance explains the shortfall), shading (a repeatable time-of-day notch on clear days), degradation (a slow multi-week slide on clear days), sensor fault (Emporia unavailable, or implausible values) |
| 3 | `inverter_first` | For zero or near-zero output, the leading hypothesis is the inverter. Panels very rarely fail as a group |
| 4 | `finish` | End with `submit_verdict`. Choose `inconclusive` rather than guessing |

---

## 8. A — Actions

All tools are registered with `@register_tool`. Injected dependencies (`_ha`, `_openmeteo`, `_run_store`, `_records`, `_delivery`, …) are hidden from the model. Tags determine which tools each agent gets.

### Tags
| Tag | Meaning |
|---|---|
| `source` | One tool per external data source; validates and stores |
| `analysis` | Deterministic computations over stored data |
| `telemetry` | Detailed Home Assistant reads, for triage |
| `weather` | Detailed Open-Meteo reads, for triage |
| `memory_read` | Reads persistent history |
| `memory_write` | Writes the nightly record |
| `notify` | Alert state and outbound messages (through HAL's delivery service) |
| `delegate` | Invokes the triage sub-agent |
| `report` | The guarded delivery tool |
| `verdict` | Triage's terminal tool |
| `inverter_diagnostics` | RS-485 (follow-on). Registered but empty |
| `system` | `terminate` |

### Solar agent toolkit
Tags: `source, analysis, memory_read, memory_write, notify, delegate, report`. That is 13 tools.

| Tool | Tag | Behaviour |
|---|---|---|
| `get_home_energy(date)` | `source` | **Home Assistant.** Reads three Emporia entities from HA statistics (hourly, plus 5-minute for the anomaly profile): solar, grid import and grid export. Applies `SOLAR_SCALE` to statistics before `SOLAR_FIXED_AT` and the 5-minute artifact guard, derives household load (import + solar − export), and runs the sensor-sanity check (solar ≥ export in every hour, otherwise `sensor_fault`). Stores everything and returns the day total, peak, coverage, gaps and sanity result. `HAUnavailable` is not recoverable, so the run goes to the fallback |
| `get_weather(date)` | `source` | **Open-Meteo HTTP.** Hourly global tilted irradiance for tilt 33° and Open-Meteo azimuth 0 (south; this is **not** the compass 180), plus GHI, cloud cover and temperature. Stores it and returns the daily insolation and a clear/partly/overcast summary. `WeatherUnavailable` is recoverable: the band becomes "unassessed" and the report says so |
| `assess_performance()` | `analysis` | Expected production (ridge model or clear-sky bootstrap, stating which), ratio and band |
| `compute_savings()` | `analysis` | TOU-weighted savings for today's production, total and by period, plus the schedule version. `TariffNotConfigured` is recoverable |
| `check_net_position()` | `analysis` | Yesterday's finalized full day, per TOU period, with any trend flag |
| `get_hourly_anomaly_profile()` | `analysis` | Late start, midday dropout, early cutoff, uniform shortfall, or flat zero, with its figures stored |
| `get_recent_history(days)` | `memory_read` | Trailing N final records |
| `run_fault_triage(reason)` | `delegate` | Starts triage with a handover assembled from the run store. Merges `verdict.figures` into the run store and rolls triage cost into this run's ledger. Returns the verdict; on triage failure or budget exhaustion, returns `inconclusive` |
| `check_alert_state(alert_key)` | `notify` | Open? Severity? Age? |
| `record_alert(alert_key, severity, text)` | `notify` | Updates dedup state |
| `send_telegram(message, severity)` | `notify` | Through HAL's delivery service; at most 3 per run |
| `write_nightly_record()` | `memory_write` | Builds today's provisional record from the run store (no model-supplied numbers), validates it against the schema, and appends it |
| `deliver_report(narrative)` | `report`, **terminal on success** | Runs the completeness check (production, performance, and savings or a reason must be present) and the numeric guard against the run store. On pass it hands the report to HAL and ends the loop. On fail it returns the offending numerals (not terminal). A second failure makes the environment send the Jinja2 report |

`terminate` is **not** in the solar toolkit. The only way to finish is a guarded delivery. If the model answers in prose instead, HAL's completeness check (step 11) catches it.

### Triage agent toolkit
Tags: `analysis` (a subset, by name), `telemetry, weather, memory_read, verdict, inverter_diagnostics`. It has **no `source`, `notify`, `memory_write`, `delegate` or `report`**.

| Tool | Tag | Behaviour |
|---|---|---|
| `read_handover()` | `analysis` | The solar agent's stored figures and the reason for triage |
| `get_intraday_curve(date, resolution_min)` | `telemetry` | 5- or 15-minute AC power from HA |
| `get_sensor_history(entity_id, start, end)` | `telemetry` | Drill-down, limited to an allow-list of entity IDs in `site.yaml` |
| `get_irradiance_curve(date)` | `weather` | Hourly plane-of-array irradiance and cloud cover for any past date |
| `compare_to_clearsky(date)` | `analysis` | Clear-sky expected curve against actual, with the residual by hour |
| `get_recent_history(days)` | `memory_read` | Shared with the solar agent |
| `find_similar_days(date, n)` | `memory_read` | Past days with similar irradiance, compared on ratio |
| `submit_verdict(...)` | `verdict`, terminal | Uses a strict JSON schema (OpenAI strict function calling, `parameters_override`) and is validated again in Python |

### Tools that deliberately do not exist
- No generic Home Assistant service call or state write. The only HA write is HAL's TTS announcement, made in coordinator code and never by a model-facing tool.
- No `run_python` or eval tool, no generic HTTP fetch, and no `write_file`.
- No analysis tool accepts numeric arrays from the model.

---

## 9. M — Memory

**Working memory** is in-run and uses the framework's `Memory`.
- **Briefing seed.** The first entry holds yesterday's final summary, open alerts and the 7-day trend.
- **Result compaction.** A `process_new_memories` capability replaces any tool result over about 2 KB with a summary and a run-store handle. The full values stay in the run store for the guard.
- **Triage** starts fresh (`call_agent` semantics). Its first `read_handover()` loads the solar agent's stored figures verbatim, so nothing is paraphrased between the two agents.

**Persistent memory** is SQLite at `data/hal9000.db`.

| Table | Written by | Notes |
|---|---|---|
| `nightly_records` | `write_nightly_record` (provisional, at 20:00) and HAL's finalize step (final, the next evening) | Append-only and schema-validated. Each date has a provisional version, then a final one; history reads use the latest. Fields: date, version kind, production, expected, ratio, band, savings (total and by period, or reason absent), net position (final only), tariff schedule version, calibration date, verdict summary, alerts raised and cleared, renderer used (`llm` or `fallback`), degraded flag |
| `hourly_production` / `hourly_consumption` | the source tool and finalize step | Training data for the baseline model and the net-position input |
| `alerts` | `record_alert` | Dedup, escalation and recovery state |
| `ledger` | cost metering | Per-call usage, per-run and month-to-date totals |
| `meter_net` | the Green Button import script | Utility-meter `import_kwh`, `export_kwh`. The name says "net" on purpose, so export is never mistaken for generation |
| `run_store` | tools, during a run | Figures by key for the guard, kept 30 days for replay |

---

## 10. E — Environment

`SolarEnvironment` subclasses the framework's `PythonEnvironment`.
- **Injection.** It supplies the HA and Open-Meteo clients, target date, tariff config, run ID, run store, stores, HAL's delivery service, the shadow flag and the usage sink.
- **Side-effect budgets.** `send_telegram` is capped at 3 per run and `deliver_report` success at 1. Going over a cap returns a normal tool error.
- **Structured errors.** Every failure returns `{"tool_executed": false, "error", "error_type", "recoverable", "guidance"}`.
- **Guard fall-through.** A second `deliver_report` rejection renders the Jinja2 report and ends the loop.
- **Loop hygiene.** Repeated identical tool calls get a nudge injected. At the iteration cap or wall-clock limit, the run goes to the fallback path.

`ReplayEnvironment` serves tool results from a JSON fixture, so a whole agent run is reproducible offline at zero cost. Every production run's transcript and tool results are saved, so any evening can become a fixture.

`ShadowEnvironment` is used during the dry run. Telegram sends go to a **private test chat**, and the announcement is logged rather than played. Everything else, including record writes, runs for real.

### The numeric guard
`find_unbacked_numerals(narrative, run_store)`:
1. Extracts every numeral from the narrative, including numbers written as words, percentages and currency.
2. Normalizes them.
3. Requires an **exact** match with a display-formatted value in the run store.

Values are stored already formatted for speech (kWh to 1 decimal, dollars to cents, percent as an integer). The model speaks those exact values, so a strict guard is also natural to listen to.

### The deterministic floor
`solar/reporting/fallback.py` renders `templates/report.j2` from whatever the run store holds, with no branching beyond present or absent fields. HAL uses it on:
- any exception, the iteration cap or wall-clock limit
- a cost cap
- two guard failures
- the monthly cap being reached before the run
- a failed completeness check

---

## 11. Delegation — the triage sub-agent

**Why triage is a sub-agent:**
- The two loops need separate iteration caps: 12 for the evening routine, 24 for a hard fault.
- Eight reporting goals and four diagnostic goals would dilute each other in one prompt.
- The blast radius stays small: triage failing costs the solar agent one tool result, not the report.

**Handover.** `run_fault_triage` writes the solar agent's run-store keys and the stated reason under a handover key. The triage task string names that key.

**Return contract** (`submit_verdict`, strict schema):
```json
{
  "verdict": "inverter_suspected | weather_explained | shading | degradation_trend | sensor_fault | inconclusive",
  "confidence": "high | medium | low",
  "evidence": [{"observation": "str", "source_tool": "str", "supports": "str"}],
  "recommended_alert": {"severity": "info | warn | critical", "text": "str"},
  "figures": {"<key>": "<display-formatted value>"}
}
```

**`figures` is load-bearing.** `run_fault_triage` merges them into the run store *before* returning, so the solar agent can speak them and still pass the guard. `recommended_alert` is only a recommendation: the solar agent still applies dedup, escalation and the send budget.

**Failure handling.** If triage errors or exhausts its budget, the solar agent gets `inconclusive`, with the reason recorded as evidence.

**Trigger thresholds** (initial values in `config/thresholds.yaml`, tuned during the dry run):

| Band | ratio = actual ÷ expected | Action |
|---|---|---|
| normal | ≥ 0.85 | none |
| low | 0.60 – 0.85 | anomaly profile and history; triage only if unexplained |
| severe | < 0.60 | triage |
| zero | ≤ 0.05, or production data flat-lined during daylight | triage; inverter-first alert (critical) |
| unassessed | weather unavailable | no band; report production only |
| sensor_fault | solar < grid export in any hour that has solar data | no band and no triage; report the sensor problem (low-severity alert, deduplicated) |
| partial_data | daylight hours with too few solar buckets (fewer than 10 of 12 five-minute buckets) hide more than 40% of the day's expected energy | no band and no triage; say the data was incomplete. Below 40%, the missing hours are left out of both actual and expected and the report says so |
| no_data | Home Assistant unreachable or no solar statistics at all | report that production data is missing; never zero |

---

## 12. Model provider and cost control

**Provider: OpenAI, through LiteLLM** (`LLM_PROVIDER=openai`, using the key already in `.env`).

| Agent | Tier | Model (verify before committing) | Reasoning effort |
|---|---|---|---|
| Solar | fast | a current small tier, e.g. `openai/gpt-5.6-luna` | low |
| Triage | powerful | a current mid tier, e.g. `openai/gpt-5.6-terra` | medium |

Model strings and $/M rates live in `config.py` and `.env`, and nowhere else. The first build task copies the current names and prices from OpenAI's own pricing page into config, with the date checked.

### Four layers, cheapest first
1. **Prepaid credits, auto-reload off.** This is account-level and needs no code. It is the hard ceiling that no bug, loop or bad cron entry can get past.
2. **Per-run ledger.** A metered `generate_response` wrapper records usage from every call, across both agents, because they share the injected callable.
   - **Soft cap ($0.25):** a system nudge to deliver now.
   - **Hard cap ($1.00):** the wrapper raises `BudgetExceeded`, HAL renders the fallback, and a low-severity notice records the truncation.
3. **Monthly ledger ($10).** HAL checks it before the loop.
   - At 80% of the monthly cap, triage is switched off for the rest of the month, and each report says so.
   - Above the cap, the agent is skipped and the deterministic report goes out.
4. **Tiering and caching.** OpenAI caches automatically once a prompt prefix reaches about 1,024 tokens. Cached reads are billed at about 10% of the input price, and caches last about 30 minutes.
   - Goals, then tool schemas, form the stable prefix, followed by the briefing and the run.
   - `prompt_cache_key` is `hal-solar` or `hal-triage`.
   - The cache pays off *within* a run, not from one evening to the next.

**Expected cost** (to confirm during the dry run): a few cents per evening on the small tier, with triage evenings costing more. The caps are deliberately loose placeholders, to be tightened from the ledger after the dry run.

---

## 13. Green Button — what it is for

The utility meter is bidirectional. The Green Button CSV (meter 05203979, 15-minute intervals, 2025-08-07 → 2026-09-07) splits its net flow into two one-directional channels:

| CSV column | Actual meaning | Stored as |
|---|---|---|
| `Consumption` | **grid import**: load minus production, when positive | `import_kwh` |
| `Generation` | **grid export**: production minus load, when positive | `export_kwh` |
| `Net` | `Consumption − Generation`, exact in every row | not stored (derived) |

Evidence from the inspection run:
- **The columns never move together.** In each interval one of them is zero, apart from the hours where the direction changes.
- **Export stops while the sun is still up.** On the best day (2026-05-10), export ends in the 16:00 hour, although a south-facing array keeps producing until about 19:00. Over the same hours, import climbs smoothly (0.145 → 0.260 → 0.370 → 0.455 kWh). That is the fading solar arc hidden under a steady load. Morning shows the mirror image: import falls from 0.395 to 0.085 kWh between 05:00 and 08:00, before any export appears.
- **Annual export is 2,021 kWh/yr, 56% of the ~3,580 kWh/yr expected gross.** The rest is consumed on site and never reaches the meter.
- **Import is zero in 62.7% of daylight intervals**, as expected whenever production exceeds load.
- **The "hard ceiling" near 2.08 kW** is the array's peak minus a steady midday base load. It is not a gross-production ceiling, so the script's gross verdict for that test is a false positive.

**Green Button is never a production source.** The baseline model trains on Emporia data only.

What Green Button *is* for:
- **Checking the Emporia sensor.** Export is a lower bound on gross production: in any interval, Emporia solar kWh must be ≥ meter export. Any historical interval that breaks this points to an Emporia or HA data problem. On overlapping dates, gross − export also gives self-consumption.
- **Tariff ground truth.** Replay historical intervals through the tariff engine and reconcile kWh and dollars per TOU period against the bills actually issued, for every downloaded month. This covers season transitions and the 2026-05-01 change.
- **TOU calendar ground truth (kWh).** For each of the 12 billing cycles in Appendix A, Green Button import − export, bucketed by the TOU calendar, must equal the bill's NEM net kWh per period. This validates the calendar, including the 2026-05-01 change, with no rates needed.
- **Net position (settled).** Single days can be net-export in super off-peak (2026-05-10 was). But the bill nets each TOU period over the whole cycle, and every period of every cycle from Sep 2025 to Aug 2026 was net import (Appendix A). The savings model holds; the net-position report tracks the margin going forward.
- **Validating winter TOU period boundaries** against winter bills, before the winter rates arrive.
- **Outage history.** Sunny days with anomalously high net import are likely past inverter outages, which is useful context for triage.

It is imported **offline** by `scripts/import_greenbutton.py` into `meter_net (meter, start_utc, import_kwh, export_kwh)`, and the import checks that `Net == Consumption − Generation` row by row. The overnight intervals with zero import (16 of 6,352) are treated as data gaps and listed in the import report. Green Button is never an agent tool.

---

## 14. Deployment, scheduling and watchdog

| Stage | Host | How it runs |
|---|---|---|
| Build and manual runs | MacBook Pro | The CLI directly (no server needed), or `uv run uvicorn hal9000.coordinator.api:app` in a terminal when testing the API |
| 21-day dry run | MacBook Pro | The API under a launchd `KeepAlive` agent. `pmset repeat wakeorpoweron MTWRFSU 19:55:00` keeps the Mac awake for the run; it must be on power |
| Production | Raspberry Pi running Home Assistant **Container (Docker)** | A second container from an arm64 image (`deploy/docker/`), with `data/` on a volume and `restart: unless-stopped` |

### Manual execution (first)
```bash
uv run python -m hal9000 run                      # today, live: announces on Sonos
uv run python -m hal9000 run --shadow             # today, no Sonos; Telegram to the test chat (or log only)
uv run python -m hal9000 run --date 2026-09-26    # any past date (weather from the Open-Meteo archive)
uv run python -m hal9000 run --no-llm             # deterministic report only, zero API cost
uv run python -m hal9000 run --replay tests/fixtures/replay_zero_output.json
uv run python -m hal9000 cost                     # ledger report
```
- Each run prints the tool-by-tool trace (the framework's existing trace output), the report text, the renderer used, the cost, and where the transcript was saved.
- A manual run is identical to a scheduled one: same runner, same guard, same ledger, same records.
- **Telegram is optional for manual runs.** With no chat configured, alerts are printed and logged instead of sent.
- A run for a past date re-runs that date. The records are append-only, so the latest version wins and nothing is lost.

### Orchestration framework decision
HAL's orchestration stays in hal_9000's own GAME loop plus plain-Python coordinator code; **no LangGraph**. The evening run is a fixed pipeline with one agent and one sub-agent. The GAME loop already provides the agent loop, tool registry, delegation and capability hooks. LangGraph would duplicate them, add a dependency tree on the Pi, and it doesn't schedule anything unless you also run its server platform. Revisit when HAL becomes an LLM router across several agents with long-running or resumable, human-in-the-loop workflows (§19); GAME agents can then be wrapped as graph nodes.

### Scheduling (later, before the dry run)
- **n8n** is a thin shell with one schedule: at 20:00 it sends `POST /runs/evening`. On a non-2xx response or a timeout (10 minutes) it sends a Telegram message. That is three nodes, exported to `deploy/n8n/`.
- **FastAPI endpoints (HAL):**
  - `POST /runs/evening {date?}`
  - `POST /runs/replay {fixture}`
  - `GET /health` (last success, status, degraded flag, month-to-date spend)
  - `GET /cost`
  - Endpoints are plain `def`, because the loop is synchronous. They are LAN-only and protected by a bearer token.
- **CLI:** `python -m hal9000` with `--date`, `--shadow`, `--replay FIXTURE`, `--cost-report`, `--no-announce`.
- **Watchdog.** A Home Assistant REST sensor polls `/health`. An automation alerts if the last success is more than 26 hours old, or the sensor has been unavailable for 30 minutes. It alerts through HA's own notifier, separate from HAL's Telegram path.
  - During the dry run, HA (on the Pi) and HAL (on the Mac) are separate failure domains.
  - **In production they share the Pi**, so cutover adds an external dead-man ping that HAL calls after each success.

---

## 15. Testing strategy

All tests run with **no API key and no network**, using `FakeLLM`, `ReplayEnvironment` and injected client fakes. A separate, opt-in `live` marker runs real-model evaluations, with a cost ceiling.

| Suite | Asserts |
|---|---|
| **Tariff properties** (Hypothesis) | Savings are ≥ 0, monotonic in generation, and ≤ Σhourly × max rate. Per-period kWh sum to the total. Boundary hours (09:59/10:00, 13:59/14:00, 15:59/16:00, 20:59/21:00). Weekends and holidays. The 2026-05-01 change. Raises `TariffNotConfigured` for any unset date |
| **Tariff vs bills** | Every downloaded month reconciles per TOU period within tolerance (kWh ±1%, dollars ±$0.05 per period) |
| **Source tools** | HA and Open-Meteo responses are validated, units normalized, gaps reported. Open-Meteo requests use azimuth 0 for due south. DST days yield 23 or 25 hourly rows |
| **Numeric guard** (adversarial) | Rounded figures, numbers in words, derived percentages, currency formats, and figures present only in triage's `figures` → exact-match behaviour holds |
| **Fallback** | Renders from a full store, one without savings, one without weather, one with production only, and one without production (headline: data missing) |
| **Replay runs** | Fixtures: `clear_day`, `cloudy_day`, `zero_output`, `midday_dropout`, `slow_degradation`, `tariff_unset`, `sensor_unavailable`, `weather_unavailable`, `net_export_warning`, `triage_timeout`, `budget_exceeded`, `prose_instead_of_tool` |
| **Finalize** | The next evening appends a final record for yesterday, with production tail and full-day consumption; net position uses it; history reads return the final version |
| **Delegation** | The triage registry contains no `source`, `notify`, `memory_write`, `delegate` or `report` tool (checked on the built registry). Triage props contain no delivery or record store. Triage figures survive the guard. Triage failure produces `inconclusive` plus a delivered report. Triage cost lands in the parent ledger |
| **Side-effect budget** | A looping script cannot send a 4th Telegram message or deliver twice |
| **Cost caps** | Soft cap produces the nudge; hard cap produces fallback plus notice; monthly cap skips the agent; 80% disables triage |
| **Completeness** | A run ending in prose with no delivery or no record still produces both and announces |
| **Live evals** (opt-in) | Real model × replay fixtures, 3 runs each (handoff WP12) |

---

## 16. Phases

| Phase | Scope | Gate |
|---|---|---|
| **0 — Data availability** | Done: Green Button channels, HA history depth, bills, entity IDs, HA access. **Remaining: fix the Emporia solar channel (§2.1)**; OpenAI key and credits | A sunny post-fix day shows solar ≥ export in every hour |
| **1 — Repository conversion** | Rename to `hal9000`, delete the greeter domain, F1–F4, contracts, config skeletons, ADR 0002 | CI green; a `FakeLLM` stub agent terminates |
| **2 — Deterministic domain** | TOU calendar, tariff, savings, net position, clear-sky bootstrap, expectation model, anomaly profile | Property tests green; a report renders from fixtures with no LLM |
| **3 — Green Button** | Import; the net-position report per billing cycle (first); bill reconciliation | Net position answered; every downloaded month with rates reconciles |
| **4 — Clients, source tools and storage** | HA and Open-Meteo clients, the two source tools, SQLite stores, run store, finalize step | Live read for today matches the HA history UI; no-network test passes |
| **5 — The floor** | Fallback renderer, numeric guard, completeness check, cost metering and caps, side-effect budgets | Nothing can be delivered unguarded; every failure path lands in the fallback |
| **6 — Solar agent** | Goals, toolkit, briefing, compaction, Solar, Replay and Shadow environments | Replay suite green |
| **7 — Triage** | Triage agent, tools, verdict, handover, figures merge, cost rollup | Zero-output replay runs solar → triage → verdict → guarded report, offline |
| **8 — HAL coordinator and manual execution** | Runner, delivery (Sonos TTS, Telegram optional), CLI, FastAPI, Docker image | Mike runs HAL by hand on the Mac, in shadow and then live, for real dates, and approves the output |
| **8b — Scheduling and watchdog** *(when Mike is ready)* | n8n flow, launchd and pmset, HA REST sensor and watchdog | A shadow run fires on its own at 20:00, and a deliberately skipped run trips the watchdog |
| **9 — Dry run (21 days)** | Shadow mode nightly on the Mac | §16.1 criteria met |
| **10 — Cutover** | Move to the Pi container, external dead-man ping, shadow off, Sonos live | The watchdog fires when the run is deliberately skipped |

### 16.1 Dry-run exit criteria
- 21 consecutive evenings completed, with no unexplained fallbacks, no run hitting the iteration cap, and no cost cap reached.
- Every run finishes within 5 minutes of 20:00.
- Hand-computed savings for 3 sampled days agree to the cent.
- The dry-run month reconciles against its bill, consistent with the Phase 3 reconciliation.
- Every narrative has been read:
  - no reassuring language on a real shortfall
  - triage fired when it should and stayed quiet on ordinary cloudy days
- Real caps are set from the ledger.

---

## 17. Outstanding inputs

| # | Input | Needed by | Status |
|---|---|---|---|
| 1 | **Winter rates** | Nov 1 | **Provisional** (§2), derived from the summer bill rates and CEA's published winter generation rates. Confirm with the first winter bill (November) |
| 2 | **TOU period definitions** | Phase 2 | **Resolved** from SDG&E and CEA tariffs (§2). Only the holiday list needs a quick confirmation |
| 3 | Whether the summer marginal rates include PCIA | Phase 2 | **Resolved: yes.** The published components sum to the bill rates only with PCIA included (§2) |
| 4 | Green Button channel meaning | Phase 0 | **Resolved:** `Consumption` = grid import, `Generation` = grid export (§13) |
| 5 | Historical bills | Phase 3 | **Resolved.** February and August 2026 bills give dollar detail; the August NEM summary gives net kWh per TOU period for all 12 cycles (Appendix A) |
| 6 | Home Assistant URL and token | Phase 4 | **Resolved.** In `.env` on the Mac (git-ignored); never in docs or git |
| 7 | HA entity IDs | Phase 4 | **Resolved** (§2). "Michael's Room" for testing; the production speaker is a one-line `.env` change |
| 8 | HA history depth | Phase 0 | **Resolved: statistics from mid-November 2025** (about 10 months). Enough for the ridge model once `solar_scale` corrects the history (§2.1); no backfill needed |
| 15 | Fix the Emporia solar channel | Performance figures, baseline training | **Applied** (×2.0 multiplier, 2026-09-27 18:35). **Confirm on the first sunny day** with `check_solar_sensor.py` |
| 16 | OpenAI API key copied into the hal_9000 `.env` | Phase 6 | Open (a placeholder line is there) |
| 9 | Telegram | Phase 8 | **Resolved.** Bot token and chat ID are in `.env`; one chat, with dry-run messages prefixed. The bot must be a member of the chat (it's a supergroup/channel ID) |
| 10 | **OpenAI**: prepaid credits funded, **auto-reload off**, current model names and prices | Phase 1 (names), Phase 6 (credits) | Key is present in `.env` |
| 11 | **n8n**: where it runs and its URL | Phase 8b | **Deferred.** Manual execution comes first |
| 12 | **External dead-man ping** service and URL | Phase 10 | Open |
| 13 | Crediting of a net-export TOU period | — | **Not needed.** No period was net-export in any billing cycle (Appendix A) |

Already recorded: site coordinates, tilt (33°) and azimuth (180°) in `config/site.yaml`.

---

## 18. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Model answers in prose, so no report or record | High | Terminal-only-on-success delivery; HAL's completeness check; replay fixture `prose_instead_of_tool` |
| Emporia solar channel scale wrong | Medium | Multiplier fixed at the source; `SOLAR_SCALE` corrects history; the nightly sanity check flags `sensor_fault` whenever solar < export; the artifact guard drops impossible 5-minute jumps |
| Net-consumer margin erodes | Low | Verified for 12 cycles; the net-position report and daily check watch the super off-peak margin |
| Forward winter rates are wrong | Low | Current delivery + CEA's published winter generation, flagged provisional; the November bill confirms them; February 2026 actuals cover history |
| Tariff bucketing wrong at a season boundary | High | Reconciliation across every downloaded month |
| Too little history for the baseline | High until Phase 0 | The clear-sky bootstrap ships regardless and doubles as the recovery path after data loss |
| Open-Meteo azimuth convention (0 = south) confused with compass 180 | Medium | A dedicated test on the request parameters; clear-day fixture comparison |
| Consumption incomplete at 20:00 | Low | Net position uses yesterday's finalized day; the provisional record is marked provisional |
| MacBook asleep at 20:00 during the dry run | Medium | `pmset` wake schedule, on power; the n8n failure branch and the HA watchdog both alert |
| HAL and the watchdog share the Pi in production | Medium | External dead-man ping at cutover |
| Runaway spend | High | Prepaid credits with auto-reload off; per-run and monthly caps degrade to the fallback well before the ceiling |
| Triage spend escapes the ledger | Medium | Shared metered callable; tested |
| Triage figures spoken but missing from the run store | Medium | Merged in `run_fault_triage` before it returns; tested |
| 13 tools on a small model | Medium | The method goal fixes the order; replay and live evals measure tool-selection errors |
| Narrative drifts toward reassurance on a real fault | Medium | Severity comes from the bands; alert routing keys off the band, not the prose |
| Triage fires on ordinary cloudy days | Low | Irradiance-adjusted expectation; thresholds tuned in the dry run |
| Model string changed to a pricier tier | Medium | Model names live in config only; the monthly ledger catches it within days |
| Green Button mistaken for a generation source | Medium | `meter_net` naming; the baseline only reads `hourly_production` |
| Reasoning tokens exhaust `max_tokens` | Medium | `AGENT_MAX_TOKENS=4096`; low effort on the solar tier; a truncated response routes to the fallback |

---

## 19. Follow-on scope (not in this build)

- **HAL as an LLM coordinator + Telegram query agent:** HAL gains an LLM front door on Telegram, routing questions ("how did last week go?") to a read-only query agent. The query agent has no `notify` tag. Needs an inbound webhook.
- **Monthly tariff audit agent:** reconciles each month's computed savings against the bill, read-only. It is also the place to model the NEM 1.0 one-time ~1 kW expansion, whose case is weakened by post-May-2026 midday pricing.
- **RS-485 inverter diagnostics:**
  - Read the SOLIVIA protocol first, then prototype against `robertklep/esphome-delta-solivia` and `htvekov/solivia_esphome`. Support for the NA G4 TL is unconfirmed, so treat it as protocol adaptation.
  - Register `get_inverter_diagnostics` under `inverter_diagnostics`. Triage picks it up with no loop changes.
  - Priority: isolation resistance, given the transformerless inverter and the coastal site.
- **Other home domains** registered under HAL (the reason HAL owns the registry, delivery and ledgers now).

---

## Appendix A — Billing ground truth

**Net kWh per TOU period, per billing cycle** (August 2026 bill, NEM summary; meter 05203979; relevant period starting 2025-09-27, true-up 2026-09-28). Every value is positive, meaning net import.

| Bill date | On-peak | Off-peak | Super off-peak | Total |
|---|---|---|---|---|
| 2025-09-30 | 15 | 4 | 118 | 137 |
| 2025-10-28 | 130 | 73 | 738 | 941 |
| 2025-11-26 | 154 | 242 | 516 | 912 |
| 2025-12-29 | 267 | 205 | 530 | 1,002 |
| 2026-01-28 | 212 | 247 | 505 | 964 |
| 2026-02-27 | 104 | 5 | 771 | 880 |
| 2026-03-30 | 110 | 67 | 711 | 888 |
| 2026-04-29 | 89 | 34 | 723 | 846 |
| 2026-05-29 | 87 | 36 | 634 | 757 |
| 2026-06-29 | 87 | 35 | 849 | 971 |
| 2026-07-29 | 126 | 106 | 359 | 591 |
| 2026-08-27 | 213 | 151 | 703 | 1,067 |

**Line items for dollar reconciliation:**

| Bill (period) | Delivery UDC $/kWh (on / off / super off) | CEA generation $/kWh (on / off / super off) | PCIA 2023 | Other per-kWh | Fixed |
|---|---|---|---|---|---|
| February 2026 (1/29–2/27, 30 days, winter) | 0.32322 / 0.32322 / 0.03676 | 0.24430 / 0.15782 / 0.05187 | 0.02823 | Wildfire 0.00591 · IPCA 0.00006 · State Reg 0.001 · Clean Impact Plus 0.001 · Franchise 1.10% | Base Services $0.79343/day |
| August 2026 (7/30–8/27, 29 days, summer) days 1–2 | 0.31711 / 0.31711 / 0.04114 | 0.51684 / 0.15975 / 0.04961 | 0.02828 | as above | as above |
| August 2026 days 3–29 | 0.31218 / 0.31218 / 0.04114 | as above | 0.02613 | as above | as above |

Billed kWh by period: February 104 / 5 / 771 (880 kWh; delivery $119.22, CEA $67.32); August 18 kWh at "Rate 1" (8 / 6 / 4) plus 1,049 kWh at "Rate 2" (205 / 145 / 699), for 1,067 kWh (delivery $203.09, CEA $170.14).

