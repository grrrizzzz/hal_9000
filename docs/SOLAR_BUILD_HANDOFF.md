# HAL 9000 Solar Monitoring — Build Handoff

This is the brief for an **orchestrating agent** that builds HAL 9000's solar monitoring by dispatching sub-agents. It is self-contained. Together with [`SOLAR_AGENT_PLAN.md`](SOLAR_AGENT_PLAN.md), which is the design and the source of truth for *what* to build, it is everything a fresh agent needs.

**Repository:** `hal_9000`, the root of this repo, on Mike's MacBook Pro at `~/AI/hal_9000`. Build in place; do not fork. The Python package becomes `hal9000`.
**Owner and approver:** Mike. He supplies the inputs, reviews the dry run and approves cutover.

**What gets built** (plan §4):
- **HAL 9000 coordinator.** Deterministic Python: service, scheduler entry point, budgets, delivery, completeness and fallback.
- **Solar agent.** A GAME agent on a small model that runs every evening at 20:00 and announces on Sonos as soon as it finishes.
- **Triage agent.** A read-only GAME sub-agent on a mid-tier model.

---

## 1. How to use this document

1. **Orchestrator:** read this whole document and `SOLAR_AGENT_PLAN.md`, then §8 (inputs). Build the dependency waves in §5. Dispatch one sub-agent per work package (WP) using the brief template in §6. Have every WP checked by an independent reviewer (§3.3) before merging it. Stop at the human gates (§7).
2. **Builder sub-agent:** you receive one WP brief. Your WP's **Acceptance criteria** define "done". Nothing else does. When every criterion has evidence, hand back the report in §6.2 and stop.
3. **Reviewer sub-agent:** you receive a WP brief and the builder's report. Re-run every acceptance check yourself and trust no claim. Return PASS or FAIL with evidence.

---

## 2. Rules for every agent

| # | Rule |
|---|---|
| R1 | **The plan is the source of truth.** Do not change a design decision in `SOLAR_AGENT_PLAN.md`. If one looks wrong, stop and report it to the orchestrator, which escalates to Mike. |
| R2 | **Never guess a rate, date, entity ID, threshold marked as an input, or credential.** Missing values raise errors (`TariffNotConfigured`, `ConfigMissing`) or use a fixture. Report `BLOCKED(<input #>)`. |
| R3 | **No network and no API key in tests.** `uv run pytest` must pass offline. Only tests marked `live` may touch the network, and they are opt-in. |
| R4 | **No secrets in git.** Credentials live in `.env` (git-ignored). `.env.example` documents every variable with a placeholder. |
| R5 | **Deterministic code owns every number.** No tool accepts a figure or array from the model that is later reported. Figures flow through the run store. |
| R6 | **Stay in your lane.** Edit only the paths your WP owns. Changing `contracts.py`, `core/` or another WP's files needs the orchestrator's approval. |
| R7 | **Tests first for `solar/domain/`, `solar/reporting/guard.py` and `cost/`.** Write the failing test, then the code. |
| R8 | **Every tool docstring is a prompt.** It says what the tool does, when to call it, and what comes back. Every error string tells the agent what to do next. |
| R9 | **Keep the framework lean.** Don't enable optional capabilities (plan_first, progress_tracking, time_aware) and don't add tools the plan doesn't list. |
| R10 | **Branch per WP** (`wp/<id>-<slug>`), with small commits. The orchestrator merges to `main` only after a reviewer PASS. |

### Repo-wide gates (every WP must leave these green)
```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest            # offline; coverage gate ≥ 85% (pyproject addopts)
```

---

## 3. Roles

### 3.1 Orchestrator
**Responsibilities:**
- Plan the waves.
- Write each WP brief (§6.1) with the relevant plan sections quoted or linked.
- Dispatch builders.
- Dispatch reviewers.
- Merge on PASS.
- Keep `docs/BUILD_STATUS.md` current.
- Escalate blocks and design questions to Mike.

**Must not:** write product code itself (except merge-conflict fixes); merge without a reviewer PASS; change the plan.

**The orchestrator is finished when:**
- [ ] WP1–WP11 are merged with reviewer PASS, except WP10b, which waits for Mike (§7, G3).
- [ ] WP12's offline evals and, once input 10 is complete, its live evals meet their thresholds.
- [ ] `docs/BUILD_STATUS.md` shows every WP state, every open input, and every deviation or escalation with Mike's decision.
- [ ] The manual-run steps (WP10) and the dry-run runbook (WP12) are handed to Mike, **and the build is paused at human gate G2** (§7).

After Mike's manual runs and G3, the orchestrator resumes with WP10b. The dry run itself and cutover (WP13) resume only on Mike's instruction.

### 3.2 Builder sub-agent
**Responsibilities:** implement exactly one WP; meet its acceptance criteria; produce the §6.2 report.
**Finished when:** every acceptance criterion is checked with evidence (command output, test names, file paths), the repo-wide gates are green on the WP branch, and the report is returned. If blocked, it returns `BLOCKED` with the input number and whatever partial work is safely committed.

### 3.3 Reviewer sub-agent (fresh context, never the builder)
**Responsibilities:**
- Check out the WP branch and run the repo-wide gates.
- Run every acceptance check independently.
- Read the diff for violations of R1–R10 and for scope creep.
- Confirm that tests actually assert the behaviour. A test that can't fail is a FAIL.

**Finished when:** it returns `PASS` or `FAIL`, with a line per criterion, the evidence, and for each FAIL the minimal fix required.

---

## 4. Work packages

Each WP lists what it owns, its dependencies, its tasks, and **acceptance criteria**, which are its definition of done. Paths are relative to `src/hal9000/` unless they start with a top-level folder.

### WP0 — Phase 0 data checks *(builder writes the scripts; Mike runs them against live systems)*
- **Owns:** `scripts/ha_history_depth.py`, `scripts/inspect_greenbutton.py` (a copy of Mike's existing inspection script), `docs/PHASE0_RESULTS.md`
- **Depends on:** none (may run in parallel with WP1)
- **Tasks:**
  - Write a script that reports how many days of hourly statistics exist for a given HA entity, and the date range with gaps.
  - Place the Green Button inspection script in `scripts/`.
  - Write `PHASE0_RESULTS.md`, pre-filled with the Green Button result.
- **Acceptance:**
  - [ ] `ha_history_depth.py --entity <id>` prints the first and last dates, the day count, and any gaps longer than 1 hour. It has unit tests against a faked HA response.
  - [ ] `inspect_greenbutton.py <csv>` runs on a fixture CSV and prints its five tests.
  - [ ] `PHASE0_RESULTS.md` records inputs 4, 5, 6, 7 and 8 as resolved (plan §2, §17, Appendix A) and the solar-channel finding (plan §2.1).
  - [ ] `scripts/check_solar_sensor.py --start <date> --end <date>` compares hourly Emporia solar with Emporia grid export from HA statistics. It prints the hours where solar < export, the daily ratio against clear-sky, and a suggested `solar_scale` fitted from post-fix days. It has unit tests on faked statistics. Mike runs it after the fix (input 15).

### WP1 — Repository conversion and contracts
- **Owns:** everything under `src/` at conversion time, `pyproject.toml`, `.github/`, `README.md`, `.env.example`, `docs/ADR/0002-*.md`, the move into `docs/template/`, `contracts.py`, `config/*.yaml` skeletons
- **Depends on:** none
- **Tasks:**
  - Carry out the plan §6 conversion: rename `hello_agent` → `hal9000`, delete the greeter domain, move the template docs, create the plan §6 layout as empty packages.
  - Apply framework changes F1–F4.
  - Set the OpenAI tiers in config: fetch the current model names and $/M rates from OpenAI's official pricing page and record the date checked.
  - Write the pydantic contracts listed in plan §6, plus a `Delivery` protocol (`announce(text)`, `send_telegram(text, severity)`) with a fake implementation for tests.
  - Write the config skeletons:
    - `site.yaml`: coordinates 33.25 / −117.30, tilt 33, azimuth 180 (compass), DC 2.47 kW, run time 20:00. Entity IDs are read from `.env` (`HA_ENTITY_SOLAR`, `HA_ENTITY_GRID_IMPORT`, `HA_ENTITY_GRID_EXPORT`, `HA_ENTITY_SPEAKER`). `SOLAR_SCALE` (2.0) and `SOLAR_FIXED_AT` (2026-09-27T18:35-07:00) are also read from `.env`, and `site.yaml` documents them.
    - `tariff.yaml`: the TOU calendar, seasons and holidays from plan §2 (including the pre-2026-05-01 March/April rule), and **effective-dated rate components** from plan Appendix A (UDC, PCIA and CEA per period; per-kWh adders; franchise %). Forward winter rates are marked `provisional: true`. Each value has its source bill or tariff.
    - `thresholds.yaml`: plan §11 bands.
  - Write ADR 0002.
- **Acceptance:**
  - [ ] `grep -r hello_agent src tests` returns nothing. `python -m hal9000 --help` works.
  - [ ] The greeter files are gone. `docs/template/` holds the template docs.
  - [ ] F1: a test shows `Agent(action_tier="fast")` requests the fast tier on every loop call, and the default is still powerful.
  - [ ] F2: a test with a fake `completion` shows the usage sink receives prompt, cached, completion and reasoning tokens plus the model; `parallel_tool_calls=False` and `prompt_cache_key` are passed; the return type is still `str`.
  - [ ] F3: `LLM_PROVIDER=openai` resolves both tiers from config. `AGENT_MAX_TOKENS` defaults to 4096. `HalFeatures.from_env()` reads `AGENT_ENABLE_SHADOW_MODE`, `…_TRIAGE_ENABLED` and `…_VERBOSE_LOGGING`.
  - [ ] F4: a helper `triage_props(parent_props)` drops `delivery`, `record_store`, `alert_store` and `ledger_admin`, with a test.
  - [ ] The contracts round-trip JSON. `Verdict` rejects an unknown `verdict` value and a missing `figures` field.
  - [ ] Loading `tariff.yaml` exposes the `provisional` flag and source per rate. Deleting any rate yields `None`, never 0. The marginal rates computed from the components equal plan §2 to within $0.00001 (summer now, February 2026 winter, forward winter).
  - [ ] A `FakeLLM` stub agent with only `terminate` runs and exits 0.
  - [ ] The repo-wide gates pass.

### WP2 — Deterministic domain
- **Owns:** `solar/domain/*`, `tests/domain/*`
- **Depends on:** WP1. Uses inputs 1–3 (works without them: unset values raise).
- **Tasks:**
  - `tou_calendar` (period for any timestamp; weekends and holidays from config).
  - `tariff` (the schedule in effect by date; raises `TariffNotConfigured`).
  - `savings` (hourly value-weighting).
  - `net_position` (per period, for a full day; also over an arbitrary billing cycle for WP3).
  - `clearsky` (plane-of-array clear-sky irradiance for site tilt and compass azimuth; `pvlib` is allowed).
  - `expectation` (ridge regression in numpy on ≥ 90 days of site history against plane-of-array irradiance; bootstrap = clear-sky POA × system factor × a fixed degradation factor of 0.945 when history is under 90 days).
  - `anomaly` (profile classes: late start, midday dropout, early cutoff, uniform shortfall, flat zero).
  - `numerals` (extraction and normalization for the guard).
- **Acceptance:**
  - [ ] Hypothesis (≥ 500 examples each):
    - savings ≥ 0, monotonic in generation, ≤ Σhourly × max rate
    - per-period kWh sum to the total within 1e-9
    - classification bands are exhaustive and non-overlapping
  - [ ] Boundary tests at 09:59, 10:00, 13:59, 14:00, 15:59, 16:00, 20:59 and 21:00 on a weekday, a weekend and a holiday, and on the dates either side of 2026-05-01.
  - [ ] Any date with an unset rate raises `TariffNotConfigured`, naming the season and the missing key.
  - [ ] Calendar: weekday 11:00 on 2026-04-15 and 2026-06-15 is super off-peak; weekday 11:00 on 2026-01-15 (before the change, outside March/April) is off-peak; Saturday 13:00 is super off-peak and Saturday 15:00 off-peak; Thanksgiving follows the weekend pattern.
  - [ ] Clear-sky:
    - on the summer solstice, the modelled daily energy is between 11 and 16 kWh for 2.47 kW DC at 33°/180° (sanity band)
    - output is zero before sunrise and after sunset
    - modelled production after 20:00 is < 1% of the daily total on every day of the year (this validates the 20:00 run time)
  - [ ] Expectation: with synthetic history generated as irradiance × k + noise, the fitted model recovers k within 2%. The bootstrap path is used, and flagged, when history is under 90 days.
  - [ ] Anomaly: each class has a synthetic curve that classifies correctly, and a clear-day curve classifies as normal.
  - [ ] Numerals: extracts "8.2", "eight point two", "$1.37", "37%", "1,024" and "twelve"; normalization is idempotent.
  - [ ] Nothing in `solar/domain/` imports `clients`, `storage`, `tools`, `agents` or `coordinator` (enforced by a test).

### WP3 — Green Button import, net position and tariff reconciliation
- **Owns:** `scripts/import_greenbutton.py`, `tests/reconciliation/*`, `tests/greenbutton/*`, `docs/NET_POSITION_HISTORY.md`
- **Depends on:** WP2 (`tou_calendar`, `net_position`, `tariff`) and WP5 (the `meter_net` table). All inputs are available (plan Appendix A). The Green Button CSV is on Mike's Mac; ask him for its path.
- **Tasks:**
  - Import the CSV into `meter_net`, mapping `Consumption` → `import_kwh` and `Generation` → `export_kwh`. `Net` is checked, not stored.
  - **First deliverable:** the net-position report per TOU period per billing cycle (plan §13).
  - Encode the bills as `tests/greenbutton/bills.yaml` (kWh and $ per TOU period, per month).
  - Write the reconciliation test.
  - Write the `emporia_ge_export(start, end)` check.
- **Acceptance:**
  - [ ] The import checks `Net == Consumption − Generation` on every row (to 1e-6) and fails on any mismatch. It reports the interval count (expected: 38,112), the date range and the zero-import overnight intervals (data gaps).
  - [ ] **kWh reconciliation, all 12 cycles:** Green Button import − export, bucketed by `tou_calendar` over each billing cycle, matches plan Appendix A's net kWh per period within ±2 kWh or ±1%, whichever is larger. Cycle boundaries are inferred from the bill dates, and the rule used is documented. A mismatch concentrated around 2026-05-01 or a season boundary is reported as a calendar bug, not tolerated.
  - [ ] **Dollar reconciliation, February and August 2026:** the energy charges computed from the Appendix A components reproduce each bill's delivery and CEA per-period charges within $0.05 per line.
  - [ ] `docs/NET_POSITION_HISTORY.md` shows, per cycle and period, import, export, net and the margin. It confirms no period was net-export in any cycle, which matches the bill.
  - [ ] Nothing reads `meter_net` as production (a grep-based test).
  - [ ] `emporia_ge_export` flags any interval where Emporia solar kWh < meter export, and is tested on synthetic data.

### WP4 — Clients and source tools
- **Owns:** `solar/clients/*`, `solar/tools/sources.py`, `tests/clients/*`, `tests/fixtures/http/*`
- **Depends on:** WP1 (contracts), WP5 (run store). Needs inputs 6 and 7 for the live check only.
- **Tasks:**
  - **HA client:** long-term and 5-minute statistics, history reads, and a TTS service call (used only by HAL's delivery service).
  - **Open-Meteo client:** hourly `global_tilted_irradiance` (`tilt=33`, **`azimuth=0`**, Open-Meteo's convention for south), `shortwave_radiation`, `cloud_cover` and `temperature_2m`. Uses the forecast endpoint for today and the archive or historical endpoint for past dates.
  - **Source tools**, one per source (plan §8):
    - `get_home_energy(date)`: HA; Emporia solar, grid import and grid export (entity IDs from `.env`). Applies `SOLAR_SCALE` before `SOLAR_FIXED_AT` and the artifact guard, derives household load, and runs the sanity check (solar ≥ export per hour, otherwise `sensor_fault`).
    - `get_weather(date)`: Open-Meteo.
    - Each validates, stores to the run store with display formatting, and returns a summary.
  - Every client is injectable, with recorded fixtures and typed return models.
- **Acceptance:**
  - [ ] Every client method has a test against a recorded fixture. `test_no_network.py` covers each client.
  - [ ] Timeouts and non-2xx responses become typed errors (`HAUnavailable` not recoverable, `WeatherUnavailable` recoverable) carrying a `recoverable` flag. The source tools surface them as structured tool errors.
  - [ ] The Open-Meteo request test asserts `tilt=33&azimuth=0`, and a fixture comparison shows that a compass-180 request would be rejected.
  - [ ] Timestamps come back tz-aware in `America/Los_Angeles`. DST fixtures (2026-03-08, 2026-11-01) yield 23 and 25 hourly rows.
  - [ ] `get_home_energy` at a 20:00 cutoff returns production, import and export through 20:00, flagged `partial_consumption=true`.
  - [ ] Sanity check: a fixture copied from HA on 2026-05-10, unscaled (solar 7.92 kWh, export 12.07 kWh), yields `sensor_fault=true`. With `SOLAR_SCALE=2.0` applied it yields `false`. The scale applies only before `SOLAR_FIXED_AT`.
  - [ ] Artifact guard: a fixture of 2026-09-27 including the +3.26 kWh 5-minute jump at 18:35 flags `data_artifact`, excludes the jump, and marks the day `transition` so baseline training skips it.
  - [ ] The HA client reads daily-resetting `total` sensors through long-term statistics `change`, not raw state differences, so resets at midnight and restarts don't create negative or double counts.
  - [ ] The source tools' schemas expose only `date`: no injected parameter leaks into the model-facing schema.
  - [ ] Live check (`pytest -m live`, with HA reachable at `HA_BASE_URL`): yesterday's HA solar, import and export kWh equal the HA UI values within 0.01 kWh. Emporia export matches Green Button export within 3% on overlapping dates.

### WP5 — Storage and run store
- **Owns:** `storage/*`, `tests/storage/*`
- **Depends on:** WP1
- **Tasks:** SQLite schema and migrations for `nightly_records` (provisional and final versions), `hourly_production`, `hourly_consumption`, `alerts`, `ledger`, `meter_net` and `run_store`; a run-store API keyed by (run_id, key) holding display-formatted values; retention jobs.
- **Acceptance:**
  - [ ] A malformed `NightlyRecord` is rejected and nothing is written.
  - [ ] A date can hold a provisional record, then a final one. History reads return the latest; the older row is kept.
  - [ ] Alert dedup: open → no resend; escalation → resend; clear → a recovery notice is generated once.
  - [ ] The ledger gives correct per-run and month-to-date totals across a month boundary (a test at 23:59 → 00:01 on the 1st).
  - [ ] Concurrent writes from two threads don't corrupt the database (WAL mode; test).
  - [ ] The run store returns exactly the formatted strings stored. Rows expire after 30 days.

### WP6 — Reporting floor
- **Owns:** `solar/reporting/*`, `tests/reporting/*`
- **Depends on:** WP1, WP2 (numerals)
- **Tasks:** Jinja2 fallback renderer and template; `find_unbacked_numerals`; completeness check.
- **Acceptance:**
  - [ ] The fallback renders from five stores (full, no savings, no weather, production only, no production) with no exception, and the no-production case leads with "production data missing".
  - [ ] Adversarial guard suite, ≥ 25 cases, all pass:
    - exact values pass
    - rounded, derived or reworded values fail
    - numbers in words match only their exact value
    - dates, the time "8 pm" and TOU hours listed in config as allowed tokens pass
  - [ ] The completeness check fails when production, performance, or savings-or-reason is missing.
  - [ ] The fallback template output, read aloud at 150 words per minute, is ≤ 45 seconds for a full store (word-count test).

### WP7 — Cost metering and caps
- **Owns:** `cost/*`, `tests/cost/*`
- **Depends on:** WP1 (F2), WP5 (ledger)
- **Tasks:**
  - `metered(generate_response, ledger, run_id)` wrapper that turns usage into dollars using config rates.
  - Soft and hard per-run caps; `BudgetExceeded`.
  - Monthly precheck; auto-disable triage at 80% of the monthly cap.
  - `--cost-report`.
- **Acceptance:**
  - [ ] With a fake usage stream, dollars equal the hand calculation to 1e-6, cached tokens included.
  - [ ] Crossing the soft cap sets a flag the environment turns into a nudge. Crossing the hard cap raises `BudgetExceeded` on the next call.
  - [ ] A sub-agent sharing the wrapped callable accrues to the same run (test with two agents and one ledger).
  - [ ] Over the monthly cap, `precheck()` returns `skip_agent`. At 80% it returns `triage_enabled=False`.
  - [ ] `--cost-report` prints month-to-date spend, per-run average, and spend by agent and model.

### WP8 — Solar agent
- **Owns:** `solar/agents/solar.py`, `solar/agents/briefing.py`, `solar/tools/analysis.py`, `solar/tools/history.py`, `solar/tools/alerts.py`, `solar/tools/records.py`, `solar/tools/report.py`, `solar/environments/*`, `config/report_policy.txt`, `tests/agent/*`, `tests/fixtures/replay_*.json`
- **Depends on:** WP2, WP4, WP5, WP6, WP7
- **Tasks:**
  - The goals from plan §7.
  - The toolkit from plan §8, using WP4's source tools.
  - Briefing seed and compaction capability.
  - `SolarEnvironment` (budgets, structured errors, guard fall-through, repeat-call nudge).
  - `ReplayEnvironment`, `ShadowEnvironment`.
  - All replay fixtures from plan §15.
  - Notify tools call the injected `Delivery` protocol (the fake in tests).
- **Acceptance:**
  - [ ] The built registry is exactly the 13 tools in plan §8. `terminate` is absent.
  - [ ] Every replay fixture, with a scripted `FakeLLM` and the agent run directly, ends with `deliver_report` success (or the environment's two-strike fallback), `write_nightly_record` called, and ≤ 12 iterations.
  - [ ] `tariff_unset`: the report omits savings and states the gap; a single low-severity config alert is recorded and is not repeated the next evening.
  - [ ] `weather_unavailable`: the band is "unassessed", the report states it, and there is no triage.
  - [ ] No analysis tool accepts a numeric or array argument (schema test across the registry).
  - [ ] Side-effect budget: a script trying 4 Telegram sends gets an error on the 4th, and exactly 3 go out.
  - [ ] Shadow mode: sends go to the test chat target, and records are written.
  - [ ] Every prompt the fake LLM receives starts with the same goals and tools prefix, byte-identical across iterations (the cache prerequisite).

### WP9 — Triage sub-agent
- **Owns:** `solar/agents/triage.py`, `solar/tools/triage.py`, `solar/tools/diagnostics.py`, `tests/delegation/*`
- **Depends on:** WP8
- **Tasks:**
  - Triage goals and toolkit (plan §7, §8).
  - `submit_verdict` with a strict schema.
  - `run_fault_triage`: handover, filtered props, figures merge, cost rollup, failure → `inconclusive`.
  - `inverter_diagnostics` tag registered and empty.
- **Acceptance:**
  - [ ] The built triage registry contains no tool tagged `source`, `notify`, `memory_write`, `delegate` or `report` (assertion on the registry object).
  - [ ] Triage's `action_context` has no `delivery`, `record_store`, `alert_store`, `ledger_admin` or `agent_registry`.
  - [ ] `zero_output` replay runs solar → triage → verdict `inverter_suspected` → the solar narrative names the inverter first → guard passes, fully offline.
  - [ ] Figures merge: the verdict carries a figure the solar agent never fetched, the solar agent speaks it, and `deliver_report` accepts it. The test fails if the merge is removed.
  - [ ] `triage_timeout`: triage exhausts 24 iterations, the solar agent receives `inconclusive`, and the report is delivered.
  - [ ] A malformed verdict is rejected at `submit_verdict` with a corrective error, and never returned to the solar agent.
  - [ ] Triage token spend appears in the run's ledger total.
  - [ ] Triage runs on the powerful tier and the solar agent on the fast tier (asserted via `FakeLLM.tiers`).

### WP10 — HAL coordinator, delivery and manual execution
- **Owns:** `coordinator/*`, `deploy/README.md` (the manual-run section), `tests/service/*`
- **Depends on:** WP8, WP9
- **Tasks:**
  - `runner.py`: precheck → finalize yesterday → brief → dispatch solar agent → completeness check → fallback if needed → announce → ledger → health (plan §5).
  - `delivery.py`: Sonos TTS through HA, Telegram (production and test chat), with a Telegram fallback if TTS fails.
  - The CLI from plan §14 (`run`, `--date`, `--shadow`, `--no-llm`, `--replay`, `cost`), printing the trace, report, renderer, cost and transcript path.
  - FastAPI endpoints (plan §14) with a bearer token.
  - Delivery degrades gracefully: with no Telegram configured, alerts are printed and logged. When the test chat is unset or equals the main chat, shadow-mode messages are prefixed `[DRY RUN]`.
  - `--speaker <entity_id>` overrides `HA_ENTITY_SPEAKER` for one run.
- **Acceptance:**
  - [ ] Every WP8 and WP9 replay fixture, run through `runner`, produces a `RunResult` with `report_delivered=true`, `record_written=true`, `announced=true` (or logged in shadow mode), and the renderer matching the fixture (`llm`, or `fallback` for `budget_exceeded`, `sensor_unavailable` and two-strike guard cases).
  - [ ] `prose_instead_of_tool`: the model replies in prose on iteration 2, and the runner still delivers (fallback), writes the record and announces.
  - [ ] Finalize: a two-evening replay appends a final record for evening 1 with the full-day consumption; evening 2's `check_net_position` uses it.
  - [ ] Monthly cap exceeded → the agent is not called; the fallback is announced; one notice is sent.
  - [ ] Announcement happens once per run, immediately after completion. A TTS failure sends the report to Telegram. Shadow mode logs rather than plays.
  - [ ] API tests with the FastAPI TestClient and fakes:
    - `POST /runs/evening` returns 200 with a `RunResult`
    - a degraded run returns 200 with `degraded:true`
    - a config error returns 422
    - nothing delivered returns 500
    - a missing or wrong bearer token returns 401
  - [ ] `/health` reports last success, status, degraded flag and month-to-date spend.
  - [ ] CLI tests: `run --replay` with a fake LLM prints a trace, report, renderer and cost and exits 0; `--no-llm` makes zero LLM calls; `--date` for a past date uses the weather archive client; a failed run exits non-zero with a one-line cause.
  - [ ] With no Telegram configuration, a run that raises an alert completes and prints the alert. A shadow run with a single chat prefixes `[DRY RUN]`.
  - [ ] `--speaker` announces on the given entity. Changing `HA_ENTITY_SPEAKER` alone moves announcements (no code change).
  - [ ] `deploy/README.md` has copy-paste steps for a first manual run: fill in `.env`, `uv sync`, `run --no-llm`, then `run --shadow`, then live. Mike follows them in G2.

### WP10b — Scheduling and watchdog *(dispatched only when Mike asks, at G3)*
- **Owns:** `deploy/launchd/*`, `deploy/n8n/*`, `deploy/ha/*`, `deploy/README.md` (the scheduling section)
- **Depends on:** WP10; input 11
- **Tasks:** launchd plist for the API; `pmset` wake instructions; n8n flow (20:00 → `POST /runs/evening`, failure → Telegram); HA REST sensor and watchdog automation YAML.
- **Acceptance:**
  - [ ] The n8n flow JSON has ≤ 3 nodes and imports cleanly (schema check). Its HTTP timeout is 10 minutes.
  - [ ] The watchdog YAML's template evaluates correctly against sample `/health` payloads: stale → alert, fresh → quiet.
  - [ ] `deploy/README.md` gives copy-paste steps for launchd, `pmset repeat wakeorpoweron MTWRFSU 19:55:00`, n8n import and HA config.
  - [ ] A shadow run fires on its own at 20:00 on the Mac, and a deliberately skipped evening trips the watchdog (Mike confirms).

### WP11 — Production container
- **Owns:** `deploy/docker/*`
- **Depends on:** WP10
- **Tasks:**
  - arm64 Dockerfile (uv, non-root, `data/` volume, healthcheck on `/health`).
  - `compose.yaml` to run next to HA Container on the Pi.
  - An external dead-man ping hook (URL from env; no-op if unset).
- **Acceptance:**
  - [ ] `docker buildx build --platform linux/arm64` succeeds. The image is under 400 MB.
  - [ ] The container starts, `/health` answers, and the data persists across a restart (local test).
  - [ ] The ping hook is called after each successful run when `DEADMAN_PING_URL` is set (unit test).

### WP12 — Agent evaluation and dry-run readiness
- **Owns:** `tests/evals/*`, `docs/DRY_RUN_RUNBOOK.md`, `docs/BUILD_STATUS.md` (with the orchestrator)
- **Depends on:** WP8–WP10
- **Tasks:** the offline and live evaluation suites below; the dry-run runbook and daily checklist.
- **Acceptance (offline, required):**
  - [ ] The full replay matrix passes 3 consecutive times (no flakiness).
  - [ ] Coverage is ≥ 85% overall and ≥ 95% for `solar/domain/`, `solar/reporting/guard.py` and `cost/`.
- **Acceptance (live; run once input 10 is complete, capped at $2 total):**
  - [ ] Real model × each replay fixture × 3 runs. Required pass rates:
    - report delivered by the LLM renderer on ≥ 90% of runs (not the fallback)
    - the solar agent calls both source tools before any analysis tool in 100% of runs
    - triage invoked on `zero_output` and `midday_dropout` in 3/3 runs
    - triage invoked on `clear_day` and `cloudy_day` in 0/3 runs
    - inverter named first on `zero_output` in 3/3 runs
    - no run over 12 solar or 24 triage iterations
    - wall-clock ≤ 5 minutes per run
    - mean evening cost (no triage) recorded in `BUILD_STATUS.md`
  - [ ] Guard rejection rate recorded. If it is above 20%, the orchestrator escalates to Mike with sample narratives before G3.
- [ ] `DRY_RUN_RUNBOOK.md` contains the daily 5-minute review checklist (plan §16.1 criteria), how to replay any evening, and how to read the ledger.

### WP13 — Cutover *(only after gate G4; Mike-driven)*
- **Tasks:**
  - Deploy the WP11 container on the Pi.
  - Point n8n and the HA sensor at the Pi.
  - Set `DEADMAN_PING_URL`.
  - Turn shadow mode off.
  - Skip one evening deliberately to prove the watchdog fires.
- **Acceptance:**
  - [ ] The first live evening is announced on Sonos within 5 minutes of 20:00.
  - [ ] The deliberate skip triggers both the HA watchdog and the external ping alert.
  - [ ] `BUILD_STATUS.md` is marked complete.

---

## 5. Dependency waves

```
Wave 0:  WP0 (scripts; Mike runs)  ║  WP1
Wave 1:  WP2 · WP5                  (parallel, after WP1)
Wave 2:  WP4 (after WP5) · WP6 (after WP2) · WP7 (after WP5) · WP3 net-position report (after WP2+WP5)
Wave 3:  WP8 (after WP2,4,5,6,7)   ║  WP3 bill reconciliation (after G1)
Wave 4:  WP9 (after WP8)
Wave 5:  WP10 (after WP9) · WP11 (after WP10)
Wave 6:  WP12 → G2 manual runs (Mike) → G3 → WP10b scheduling → 21-day dry run (Mike) → G4 → WP13
```
WP3 is off the critical path, but **G4 requires WP3 PASS**. The solar-channel fix (input 15) doesn't block building. It blocks trusting performance figures, so G2 manual runs before the fix will show `sensor_fault`, and that is expected.

---

## 6. Templates

### 6.1 WP brief (orchestrator → builder)
```
WP: <id> — <name>
Branch: wp/<id>-<slug>
Read first: docs/SOLAR_AGENT_PLAN.md §<sections>; this handoff §2 (rules), §4 <WP>
You own: <paths>        Do not edit: everything else
Inputs available: <list, with values or "use fixture">
Acceptance criteria: <copied verbatim from §4>
Hand back: the report in §6.2. Stop when every criterion has evidence, or when BLOCKED.
```

### 6.2 Builder report (builder → orchestrator → reviewer)
```
WP: <id>   Status: DONE | BLOCKED(<input #s>) | PARTIAL
Branch / last commit: <sha>
Gates: ruff ✔/✘  format ✔/✘  pytest ✔/✘ (N passed, coverage X%)
Acceptance:
  [x] <criterion> — evidence: <test name / command + output excerpt / file>
  [ ] <criterion> — why not
Deviations from plan: none | <what, why — needs orchestrator approval>
New questions for Mike: none | <question>
```

### 6.3 Reviewer verdict
```
WP: <id>   Verdict: PASS | FAIL
Gates re-run: <results>
Per criterion: <PASS/FAIL — evidence>
Rule violations (R1–R10): none | <list>
Required fixes (if FAIL): <minimal list>
```

---

## 7. Human gates (Mike)

| Gate | When | Mike does | Unblocks |
|---|---|---|---|
| **G1** | After WP0 | Runs `check_solar_sensor.py` on the first sunny post-fix day and confirms `SOLAR_SCALE=2.0` (or records the fitted value) | Trustworthy performance figures; ridge training on corrected history |
| **G2** | After WP10 | Supplies inputs 6, 7 and 10 (9 optional); follows `deploy/README.md`; **runs HAL by hand** (`--no-llm`, then `--shadow`, then live) for as many evenings and past dates as he likes | WP12 live evals |
| **G3** | After WP12 and Mike's manual runs | Reviews the eval results, manual output and cost; supplies input 11; says "schedule it" | WP10b, then the 21-day dry run |
| **G4** | After 21 evenings | Confirms the plan §16.1 exit criteria, confirms WP3 PASS, supplies input 12 | WP13 cutover |

Escalations outside the gates: a WP3 kWh mismatch that points at the TOU calendar, a guard rejection rate above 20% (from WP12), and any R1 design question.

---

## 8. Outstanding inputs

| # | Input | Needed by |
|---|---|---|
| 1 | Winter rates: **provisional** values in plan §2; confirm with the first winter bill | WP3 (after the November bill) |
| 2 | ~~TOU period definitions~~ **Resolved** (plan §2). Holiday list: confirm only | WP2 |
| 3 | ~~PCIA included?~~ **Resolved: yes** | done |
| 4 | ~~Green Button channel meaning~~ **Resolved:** `Consumption` = grid import, `Generation` = grid export | done |
| 5 | ~~Bills~~ **Resolved:** plan Appendix A (12 cycles of kWh; February and August dollar detail) | done |
| 6 | ~~HA URL and token~~ **Resolved:** in the git-ignored `.env` on the Mac | done |
| 7 | ~~HA entity IDs~~ **Resolved** (plan §2). The speaker is configurable; "Michael's Room" for testing | done |
| 8 | ~~HA history depth~~ **Resolved:** statistics from mid-November 2025 | done |
| 9 | ~~Telegram~~ **Resolved:** bot token and chat ID in `.env`; one chat, with dry-run messages prefixed | done |
| 10 | OpenAI prepaid credits funded with **auto-reload off**; current model names and prices (WP1 fetches the prices). The key is already in `.env` | WP1, G2 |
| 11 | n8n host and URL — **deferred** until after manual runs | G3 (WP10b) |
| 12 | External dead-man ping service and URL | G4 |
| 13 | ~~Net-export crediting~~ **Not needed:** no net-export period in any cycle | done |
| 15 | Emporia solar channel: **fix applied** 2026-09-27 18:35 (×2.0). Confirm on the first sunny day with `check_solar_sensor.py` | G1 |
| 16 | OpenAI API key copied into the hal_9000 `.env` | G2 |

Already provided:
- site coordinates 33.25, −117.30; tilt 33°; azimuth 180° (compass)
- summer marginal rates and PCIA (plan §2)
- Green Button channel meaning (plan §13)
- run time 20:00, with the announcement on completion
- hosts: MacBook for the build and dry run; Raspberry Pi running HA Container for production
- provider: OpenAI
