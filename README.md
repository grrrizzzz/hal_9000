# hal9000

A working LLM agent you can read in one sitting, shaped as a GitHub template.

It is "hello, world" taken literally: you tell it about a visitor in your own
words, and it greets them and signs the guestbook.

```
$ python -m hal9000 "Dr. Maria Chen is here -- marine biologist, flew in from Osaka, second visit this year."

[greeter] extract_visitor_details  -> {"name": "Dr. Maria Chen", "role": "marine biologist", "origin": "Osaka", "previous_visits": "second...
[greeter] compose_greeting         -> Welcome back, Dr. Chen -- Osaka's loss is our gain twice over this year. The reef samples you mentio...
[greeter] sign_guestbook           -> {"status": "signed", "entry_id": 1, "name": "Dr. Maria Chen", "signed_at": "2026-09-21T23:54:37"}
[greeter] terminate                -> Welcome back, Dr. Chen -- Osaka's loss is our gain twice over this year. The reef samples you mentio...

Welcome back, Dr. Chen -- Osaka's loss is our gain twice over this year.
The reef samples you mentioned last spring are still the talk of the lab.

Signed the guestbook as entry #1.
```

The domain is trivial on purpose. The architecture underneath it is not: this
repository is a complete, tested implementation of the GAME framework — Goals,
Actions, Memory, Environment — with a tool-decorator registry, dependency
injection that the model never sees, a ten-hook capability lifecycle,
self-prompting, multi-agent delegation, and environmental safety. See
[`docs/COVERAGE.md`](docs/COVERAGE.md) for the pattern-to-file map.

---

## Quickstart (two minutes)

You need [`uv`](https://docs.astral.sh/uv/) and one API key.

```bash
git clone <your-new-repo> && cd <your-new-repo>
uv sync                                  # ~20s, one runtime dependency: litellm

export ANTHROPIC_API_KEY=sk-ant-...      # or: export OPENAI_API_KEY=sk-...

uv run python -m hal9000 "Dr. Maria Chen is here -- marine biologist, flew in from Osaka."
```

That is the whole setup. The guestbook is created on first write at
`./guestbook.json`, and is git-ignored.

> **About `.env`.** `.env.example` documents every variable this template reads,
> and `cp .env.example .env` is a good way to keep your key out of your shell
> history — but **nothing in the code loads `.env` for you**. Either `export` the
> variable, or let your runner do it:
> `uv run --env-file .env python -m hal9000 "..."`.

Run the tests, which need no key and no network:

```bash
uv run pytest        # FakeLLM drives every test; see "Why this is testable" below
uv run ruff check .
```

---

## The lean default is a feature, not an omission

`build_greeter_agent()` with no arguments gives you **five tools, zero
capabilities**, a plain `PythonEnvironment` and the function-calling language:

```
extract_visitor_details   compose_greeting   sign_guestbook   read_guestbook   terminate
```

Sixteen tools are registered in the catalogue. Eleven of them are reachable by
the greeter. Five are in the default run. That gap is the design.

Every advanced pattern in the corpus this template implements — planning,
progress tracking, time awareness, logging, sub-agents, staged human review,
transactional rollback, dynamic personas — **ships complete, tested, and off**.
Turning one on is one flag. Turning them all on is `Features.all_on()`.

Shipping everything on would cost you three things at once:

1. **Accuracy.** A model choosing between eleven tools picks worse than a model
   choosing between five. Crowding the tool list is the most common way to make
   an agent unreliable, and it does not announce itself — it just degrades.
2. **Money and latency.** `progress_tracking` alone is one extra LLM call *per
   iteration*. `plan_first` is one per run, and the plan it writes lengthens
   every prompt after it.
3. **Readability.** The default path is four files deep and reads start to
   finish. That is what makes this a teaching artifact rather than a framework
   you have to trust.

So the showcase lives in tests, docs and flags — never in the default run.

---

## The flags, and what each one costs

All off by default. Each has a CLI flag and an environment variable; the flag
wins when both are given.

| Flag | `AGENT_ENABLE_*` | What it adds | What it costs |
|---|---|---|---|
| `--time-aware` | `TIME_AWARE` | `TimeAwareCapability`: the current time seeded into memory and re-injected into **every** prompt | No extra LLM calls. A few tokens per turn. Timezone from `AGENT_TIMEZONE` (default `America/Chicago`). |
| `--plan-first` | `PLAN_FIRST` | `PlanFirstCapability`: a numbered plan written before the first move | **One extra LLM call per run**, at init. The plan then sits in memory and lengthens every later prompt. |
| `--progress-tracking` | `PROGRESS_TRACKING` | `ProgressTrackingCapability`: a progress report at the end of each tracked iteration | **One extra LLM call per tracked iteration** — with the default `track_frequency=1`, that is every iteration. The most expensive flag here. Each report also stays in memory. |
| `--verbose-logging` | `VERBOSE_LOGGING` | `LoggingCapability`: one structured line per loop event, on stderr | No extra LLM calls. Noisy by design. Goes last in the capability list so it logs what the model actually received. |
| `--delegation` | `DELEGATION` | `call_agent` (a 6th tool), a translator sub-agent, an `AgentRegistry`, and a 4th goal telling the greeter when to use it | **A whole sub-agent run** when the visitor named a language — its own loop, its own LLM calls (capped at 3 iterations). Plus one more tool to choose between and one more goal in every prompt. |
| `--staged-review` | `STAGED_REVIEW` | Environmental safety (ruling R12): `StagedActionEnvironment` holds `sign_guestbook` for review, **and** `undo_last_guestbook_entry` joins the toolkit (6 tools) | With the default `human` reviewer, the run **pauses for a y/N answer on stdin** — and **refuses every write when stdin is not a terminal**. That is fail-closed behaviour, not a bug. With `--reviewer llm`, one powerful-tier call per staged write instead. |
| `--dynamic-experts` | `DYNAMIC_EXPERTS` | The whole `prompts` tag: **four** tools — `prompt_llm`, `prompt_llm_for_json`, `prompt_expert`, `create_and_consult_expert` (9 tools) | Four more tools in the choice set. `create_and_consult_expert` costs **three LLM calls** every time it is called: one to invent the persona, one to write the consultation, one to ask. |

The name `--dynamic-experts` suggests one tool. It adds four — the flag switches
on a tag, not a function.

Verified composition, flag by flag:

| flag | tools | capabilities | environment | goals |
|---|---|---|---|---|
| *(none)* | 5 | — | `PythonEnvironment` | 3 |
| `time_aware` | 5 | Time Awareness | `PythonEnvironment` | 3 |
| `plan_first` | 5 | Plan First Capability | `PythonEnvironment` | 3 |
| `progress_tracking` | 5 | Progress Tracking | `PythonEnvironment` | 3 |
| `verbose_logging` | 5 | Logging | `PythonEnvironment` | 3 |
| `delegation` | 6 | — | `PythonEnvironment` | 4 |
| `staged_review` | 6 | — | `StagedActionEnvironment` | 3 |
| `dynamic_experts` | 9 | — | `PythonEnvironment` | 3 |
| all seven | 11 | all four | `StagedActionEnvironment` | 4 |

```bash
uv run python -m hal9000 --plan-first --verbose-logging "Ada Lovelace, mathematician, visiting from London."
AGENT_ENABLE_TIME_AWARE=1 uv run python -m hal9000 "Ada Lovelace, visiting."
AGENT_REVIEWER=llm uv run python -m hal9000 --staged-review "Ada Lovelace, visiting."
```

---

## Command line

```
python -m hal9000 [-h] [--version] [--quiet|-q] [--reviewer {human,llm}]
                      [--time-aware] [--plan-first] [--progress-tracking]
                      [--verbose-logging] [--delegation] [--staged-review]
                      [--dynamic-experts] [visitor]
```

Omit `visitor` to read from stdin, or to be prompted at a terminal. `--quiet`
prints only the agent's final message, without the tool-by-tool trace.

**Exit codes.**

| Code | Meaning |
|---|---|
| `0` | The agent ran and printed a final message. |
| `2` | Something you can fix: no provider configured, no visitor text, or `--staged-review` with the visitor's details piped in (the human reviewer needs stdin to answer on). One line on stderr prefixed `hal9000: `, never a traceback. |
| `1` | The agent ran but produced no final message — usually the iteration ceiling. Raise `AGENT_MAX_ITERATIONS`, or re-run with `--verbose-logging`. |

**Environment variables.** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`LLM_PROVIDER`, `AGENT_MODEL_POWERFUL`, `AGENT_MODEL_FAST`, `AGENT_MAX_TOKENS`,
`AGENT_MAX_ITERATIONS`, `AGENT_MAX_DURATION_SECONDS`, `AGENT_GUESTBOOK_PATH`,
`AGENT_TIMEZONE`, `AGENT_REVIEWER`, and the seven `AGENT_ENABLE_*` flags.
`.env.example` documents all of them with their defaults; `--help` lists them
too.

Provider resolution, in order: `LLM_PROVIDER` → `ANTHROPIC_API_KEY` present →
`OPENAI_API_KEY` present → a clean error naming both variables.

| Provider | powerful tier | fast tier |
|---|---|---|
| `anthropic` | `anthropic/claude-opus-5` | `anthropic/claude-haiku-4-5` |
| `openai` | `openai/gpt-4o` | `openai/gpt-4o-mini` |

Two tiers, because MATE's *model efficiency* is a thing you do, not a thing you
say: `extract_visitor_details` runs on the fast tier, `compose_greeting` on the
powerful one.

---

## Where do I change things

| I want to change... | Edit |
|---|---|
| **What the agent is for** — its instructions, its voice, its output format | `src/hal9000/agents/greeter.py` → **`greeter_goals()`** |
| What it can do | `src/hal9000/tools/` — decorate a function with `@register_tool(tags=[...])` |
| Which tools the agent gets | `agents/greeter.py` → `DEFAULT_TAGS` and `build_greeter_action_registry()` |
| Which optional patterns run | `Features` in `config.py`, a `--flag`, or `AGENT_ENABLE_*` |
| Where data is stored | `src/hal9000/storage/guestbook.py` |
| House rules, with no code change | `config/greeting_policy.txt` — read fresh on every `compose_greeting` |
| Models, tiers, limits, paths | `src/hal9000/config.py` (or the env vars above) |
| How the loop itself behaves | Add a `Capability` in `src/hal9000/capabilities/` — **not** `core/agent.py` |
| Prompt format / how tools are presented | `src/hal9000/core/agent_language.py` |
| How actions execute (review, sandboxing, remote) | Subclass `Environment` — see `src/hal9000/safety/staged.py` |

The full layout:

```
src/hal9000/
├── __main__.py        CLI: flags in, greeting out. Decides nothing about behaviour.
├── config.py          provider resolution, model tiers, Features flags
├── llm.py             generate_response(Prompt, tier) -> str   <- the keystone
├── core/              the framework: goals, memory, actions, registry,
│                      action_context, environment, agent_language,
│                      capabilities, agent (the loop)
├── tools/             decorators + the tool catalogue
├── capabilities/      time_aware, plan_first, progress_tracking, logging
├── safety/            reversible, staged, review  (MATE's E)
├── storage/           guestbook.py — JSON file, path-confined
└── agents/            greeter.py (THE agent) · translator.py (a sub-agent)
```

---

## Using this template

Click **Use this template**, then change three things. In this order:

**1. The goals — `src/hal9000/agents/greeter.py`, `greeter_goals()`.**
This is the entry point for building your own agent. Three `Goal` objects by
default (four with `delegation`): the mission, the method, and how to deliver
the answer. They are prose. Rewrite them for your domain and you have a
different agent without touching a line of framework code.

**2. The tools — `src/hal9000/tools/`.** A tool is a decorated function; its
name, docstring and type hints become the JSON Schema the model sees. Add yours,
tag it, and add the tag to `DEFAULT_TAGS`:

```python
@register_tool(tags=["greeting"])
def check_badge(visitor_name: str, _badge_service=None) -> dict:
    """Check whether a visitor has a badge waiting at the desk."""
    ...
```

**3. The flags — `Features` / the CLI.** Decide which of the seven patterns your
agent actually needs. Start with none. Add one when you can say what it buys you.

Then delete what you do not need: the guestbook is an **example, not a
commitment**. [`docs/EXTENDING.md`](docs/EXTENDING.md) has a worked "replace the
domain entirely" diff for exactly this.

Housekeeping for your copy: update this README's title and description,
`[project]` in `pyproject.toml`, the `LICENSE` copyright line, and
`.github/ISSUE_TEMPLATE/` if you want different issue forms.

---

## Why this is testable

`generate_response(prompt: Prompt, tier: str = "powerful") -> str`.

One signature, returning a **string**. When the prompt carries tools and the
model calls one, that string is `{"tool": name, "args": {...}}`. Every
provider-shape difference is absorbed in `llm.py` and nowhere else.

The consequence: a complete LLM stand-in is a list of strings.

```python
llm = FakeLLM([tool_call("terminate", message="done")])
agent = build_greeter_agent(generate_response=llm)
```

No mocking library, no HTTP interception, no recorded cassettes. The whole suite
runs green with **no API key and no network**, which is also how CI runs it — no
secrets are configured for the workflow, on purpose. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#why-generate_response-returns-a-string).

---

## Verification

Everything that can be checked without an API key is checked in CI on Python
3.11, 3.12 and 3.13: `ruff check`, `ruff format --check`, and the full test
suite against `FakeLLM`.

```bash
uv sync
uv run pytest
uv run pytest --cov=hal9000
uv run ruff check . && uv run ruff format --check .
```

### Live verification (needs an API key)

Everything above runs with no key. These four checks are the ones that need a
real model, and they are **not** performed by the build — the template was
assembled in an environment with no API keys. Run whichever provider you have;
each takes under a minute.

```bash
# 1. A real Anthropic run, lean default. Writes to a temp guestbook.
AGENT_GUESTBOOK_PATH=$(mktemp -d)/guestbook.json \
  ANTHROPIC_API_KEY=sk-ant-... \
  uv run python -m hal9000 "Ada Lovelace, mathematician, visiting from London."

# 2. The same run on OpenAI. LLM_PROVIDER pins it, so a machine with both
#    keys exported still exercises the OpenAI path rather than Anthropic twice.
AGENT_GUESTBOOK_PATH=$(mktemp -d)/guestbook.json \
  OPENAI_API_KEY=sk-... LLM_PROVIDER=openai \
  uv run python -m hal9000 "Ada Lovelace, mathematician, visiting from London."

# 3. The live smoke tests, as tests. Skips any provider whose key is absent,
#    so this is safe to run with only one key exported.
ANTHROPIC_API_KEY=sk-ant-... uv run pytest -m live -v

# 4. Three flags against a real model. Each changes what the agent does, not
#    just how it is assembled, so a fake cannot fully stand in for this.
ANTHROPIC_API_KEY=sk-ant-... AGENT_GUESTBOOK_PATH=$(mktemp -d)/guestbook.json \
  uv run python -m hal9000 --time-aware --plan-first --delegation \
  "Dr. Maria Chen, marine biologist, visiting from Osaka. She prefers Japanese."
```

**What each should show.** (1) and (2): the four-line tool trace, a greeting,
and `Signed the guestbook as entry #1.` (3): `2 passed`, or `1 passed 1 skipped`
with a single key. (4): a `create_plan` step before the first tool call, the
current time in the agent's reasoning, and a `call_agent` line delegating to
`translator_agent` — the greeting arrives in Japanese.

> **Do not add `--staged-review` to a piped or CI command.** Its default
> reviewer reads stdin; with no terminal it refuses every write and the
> guestbook is never created. That is deliberate fail-closed behaviour. For
> non-interactive use set `AGENT_REVIEWER=llm`.

---

## Documentation

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The GAME walkthrough, the loop and its capability hooks, the DI mechanism, and why `generate_response` returns a string |
| [`docs/EXTENDING.md`](docs/EXTENDING.md) | Worked diffs: add a tool, a capability, a sub-agent; swap the agent language; swap storage for a database; **replace the domain entirely**. Plus the five traps found in the source material |
| [`docs/COVERAGE.md`](docs/COVERAGE.md) | Every pattern in the source corpus, and the file that implements it |
| [`docs/ADR/0001-litellm.md`](docs/ADR/0001-litellm.md) | Why one provider layer |
| [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) · [`docs/ORCHESTRATOR_RULINGS.md`](docs/ORCHESTRATOR_RULINGS.md) | The build's historical record: the plan as written, and the nineteen rulings made where the source notes were wrong or disagreed with each other |

## License

MIT — see [`LICENSE`](LICENSE).
