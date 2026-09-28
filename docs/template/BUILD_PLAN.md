# Build Plan: `hal9000` — A GAME-Framework Agent Template

**Status:** Ready for handoff to an orchestrating agent.
**Author:** AI architect pass over `coursera_ai_agent_content` (36 notes, all read).
**Target:** `/Users/mike/AI/coursera/agent-template/` — new git repo, to be published as a GitHub template repo.

> **First action for the orchestrator:** copy this document to `agent-template/docs/BUILD_PLAN.md` in the new repo so it travels with the code, then drive the task waves in §7.

---

## 1. Context

`coursera_ai_agent_content` is a notes repository from the Coursera AI Agent Developer Specialization. It describes — in prose and Python fragments — a complete, coherent architecture for LLM agents: the GAME framework (Goals/Actions/Memory/Environment), a tool-decorator registry, dependency injection via `ActionContext`, a capability lifecycle, self-prompting (structured extraction + persona experts), multi-agent delegation, and environmental-safety patterns (reversible actions, transactions, staged review).

Nothing in that repo is executable. Every pattern exists as an illustrative snippet with implied-but-absent surrounding code.

**The need:** a working, runnable reference implementation of the *entire* corpus, shaped as a GitHub template repo. Someone clicks "Use this template," supplies one API key, and has a running agent in under two minutes — then builds their real agent by editing goals, adding tools, and flipping feature flags, never by re-deriving the framework.

**The intended outcome:** a repo where every pattern in the notes is implemented, tested, and documented; where the *default* run is small enough to read in one sitting; and where enabling any advanced pattern costs one config line, not an afternoon.

### 1.1 Decisions already made (do not revisit)

| Decision | Choice |
|---|---|
| Location | New sibling repo at `/Users/mike/AI/coursera/agent-template/`; notes repo untouched |
| LLM layer | **LiteLLM only** — one dependency, both providers via model strings (course-faithful) |
| Default runtime | **Lean core.** Everything else ships complete + tested, enabled by one flag |
| Tooling | `uv` + `pyproject.toml` + `src/` layout + `pytest`, Python 3.11+ |
| CI / tests | Must run green with **zero API keys** via a deterministic `FakeLLM` |

---

## 2. What the agent does

**`hal9000` is a guestbook greeter.** It is "hello, world" taken literally: you tell it about a visitor in free-form prose, and it greets them and records the visit.

```
$ python -m hal9000 "Dr. Maria Chen is here — marine biologist, flew in from Osaka, second visit this year."

[greeter] extract_visitor_details  → {"name": "Dr. Maria Chen", "role": "marine biologist", "origin": "Osaka", ...}
[greeter] compose_greeting          → "Welcome back, Dr. Chen — Osaka's loss is our gain twice over this year..."
[greeter] sign_guestbook            → {"status": "signed", "entry_id": 7}
[greeter] terminate

Welcome back, Dr. Chen — Osaka's loss is our gain twice over this year.
The reef samples you mentioned last spring are still the talk of the lab.

Signed the guestbook as entry #7.
```

This task is deliberately trivial in domain and rich in architecture. It exercises, in the default path alone:

- an **agent loop** that must sequence three dependent tool calls and stop
- **structured extraction** from unstructured prose (self-prompting + fixed JSON schema)
- a **persona expert** consultation (greeting composition)
- a **side-effecting, specific, reversible tool** (`sign_guestbook` — not `write_file`)
- **dependency injection** (storage handle + two LLM tiers arrive via `ActionContext`, invisible to the model)
- **memory** carrying extraction results forward to composition

And with flags flipped: capabilities, planning, progress tracking, multi-agent delegation, staged human/LLM review, and transactional rollback.

### 2.1 Why a guestbook

- **Zero external services.** One JSON file. No API beyond the LLM.
- **Genuinely stateful.** The guestbook persists across runs, which makes reversibility, transactions, and staged review *mean* something rather than being decorative.
- **Naturally multi-agent.** "Greet them in their own language" is an obvious, honest reason to delegate to a translator sub-agent — not a contrived one.
- **Policy-as-prompt has a home.** `config/greeting_policy.txt` holds house rules in plain English, read at call time. Editing that file changes agent behavior with no code change — the `purchasing_rules.txt` pattern from the notes.

---

## 3. Corpus coverage map

Every source note maps to an implementing module. The build is not done until `docs/COVERAGE.md` reproduces this table with every row satisfied.

| Pattern | Source note(s) | Implemented in | Default run? |
|---|---|---|---|
| Agent loop, prompt/parse/execute/remember | `agentic_ai_concepts/first_agent.md`, `agent_env_interface.md`, `agent_feedback_and_memory.md` | `core/agent.py` | ✅ |
| Stateless LLM / memory as messages | `agentic_ai_concepts/giving_agents_memory.md` | `core/memory.py` | ✅ |
| Programmatic prompting, ChatML roles | `programmatic_prompting_2.md`, `_3.md` | `llm.py`, `core/prompt.py` | ✅ |
| Function calling | `agent_tools_actions_language/function_calling.md`, `agent_loop_with_function_calling.md` | `core/agent_language.py`, `llm.py` | ✅ |
| JSON-Schema tool metadata, naming | `agent_tools.md`, `tool_design_best_practices.md` | `tools/decorators.py` | ✅ |
| Instructions injected into error messages | `tool_design_best_practices.md` | `tools/guestbook.py`, `tools/greeting.py` | ✅ |
| `@register_tool` decorator + introspection | `agent_tool_management/python_decorators.md` | `tools/decorators.py` | ✅ |
| Tag-based tool organization + registries | `tool_organization.md`, `refactored_readme_agent.md` | `core/registry.py`, `core/actions.py` | ✅ |
| GAME: Goal / Action / Memory / Environment | `game_framework/modular_agent_design.md`, `game_framework.md` | `core/goals.py`, `actions.py`, `memory.py`, `environment.py` | ✅ |
| Agent class + swappable components | `agent_loop_customization.md`, `game_in_code.md` | `core/agent.py` | ✅ |
| AgentLanguage (function-calling + JSON-block) | `agent_language.md` | `core/agent_language.py` | ✅ (FC); JSON-block shipped |
| `ActionContext` dependency injection | `dependency_injection_for_tools/clean_tools_with_dep_injection.md` | `core/action_context.py` | ✅ |
| Env-level DI, `_`-prefix params, schema hiding | `dep_injection_with_environment.md` | `core/environment.py`, `tools/decorators.py` | ✅ |
| `prompt_llm_for_json` + retry | `extending_with_self_prompting/structured_data_extraction.md` | `tools/self_prompting.py` | ✅ |
| Specialized fixed-schema extractor | `structured_data_extraction.md`, `invoice_processing_agent.md` | `tools/greeting.py` | ✅ |
| `prompt_expert` / persona pattern | `persona_pattern.md`, `persona_abstraction.md` | `tools/self_prompting.py`, `tools/greeting.py` | ✅ |
| Dynamic persona creation | `persona_abstraction.md` | `tools/self_prompting.py` (`create_and_consult_expert`) | flag |
| Policy-as-prompt read from disk | `invoice_processing_with_experts.md` | `config/greeting_policy.txt` + `tools/greeting.py` | ✅ |
| Separation of reasoning vs. execution | `separation_of_reasoning.md` | architecture-wide; `docs/ARCHITECTURE.md` | ✅ |
| Capability lifecycle (10 hooks) | `improving_agent_reasoning/capability_architecture_pattern.md` | `core/capabilities.py`, `core/agent.py` | ✅ (hooks live, list empty) |
| `TimeAwareCapability` | `capability_architecture_pattern.md` | `capabilities/time_aware.py` | flag |
| `PlanFirstCapability` + `create_plan` | `plan_first_capability.md` | `capabilities/plan_first.py`, `tools/planning.py` | flag |
| `ProgressTrackingCapability` + `track_progress` | `progress_tracking_capability.md` | `capabilities/progress_tracking.py`, `tools/planning.py` | flag |
| Logging capability | `capability_architecture_pattern.md` | `capabilities/logging.py` | flag |
| `AgentRegistry` + `call_agent` | `multi-agent_systems/agent-to-agent-communication.md` | `core/registry.py`, `tools/delegation.py` | flag |
| Memory reflection / handoff / selective share | `agent_interaction_with_memory.md`, `advanced_agent_interaction.md` | `tools/delegation.py` (all four patterns) | flag |
| `ReversibleAction`, `ActionTransaction` | `agent_design_principles/environment_safety.md` | `safety/reversible.py` | flag |
| Staged execution with review | `environment_safety.md` | `safety/staged.py` | flag |
| Single-safe-tool over multiple risky tools | `environment_safety.md` | `tools/guestbook.py` + `docs/EXTENDING.md` | ✅ |
| MATE — **M**odel efficiency (tiered LLMs) | `MATE_design_principle.md` | `llm.py`, `config.py`, used in `tools/greeting.py` | ✅ |
| MATE — **A**ction specificity | `MATE_design_principle.md` | `tools/guestbook.py` + `docs/EXTENDING.md` | ✅ |
| MATE — **T**oken efficiency | `MATE_design_principle.md` | terse tool prompts, `Memory.get_memories(limit)` | ✅ |
| MATE — **E**nvironmental safety | `MATE_design_principle.md` | `safety/`, path confinement in `storage/` | ✅ |

---

## 4. Repository layout

```
agent-template/
├── README.md                     # the template's front door — 2-minute quickstart
├── pyproject.toml                # uv, deps, ruff, pytest config
├── uv.lock
├── .env.example
├── .gitignore                    # incl. guestbook.json, .env
├── .github/workflows/ci.yml      # ruff + pytest, no secrets
├── config/
│   └── greeting_policy.txt       # human-readable house rules, read at call time
├── docs/
│   ├── BUILD_PLAN.md             # this document
│   ├── ARCHITECTURE.md           # GAME walkthrough + loop diagram
│   ├── EXTENDING.md              # "add a tool / agent / capability in N minutes"
│   ├── COVERAGE.md               # §3 table, filled in and verified
│   └── ADR/0001-litellm.md       # why one provider layer
├── src/hal9000/
│   ├── __init__.py
│   ├── __main__.py               # CLI entrypoint
│   ├── config.py                 # provider detect, model tiers, Features flags
│   ├── llm.py                    # generate_response(Prompt) -> str via LiteLLM
│   ├── core/
│   │   ├── goals.py              # Goal
│   │   ├── prompt.py             # Prompt
│   │   ├── memory.py             # Memory
│   │   ├── action_context.py     # ActionContext
│   │   ├── actions.py            # Action, ActionRegistry, PythonActionRegistry
│   │   ├── registry.py           # global tools/tools_by_tag, AgentRegistry
│   │   ├── environment.py        # Environment, PythonEnvironment (DI)
│   │   ├── agent_language.py     # AgentLanguage + FunctionCalling + JsonAction
│   │   ├── capabilities.py       # Capability base (10 hooks)
│   │   └── agent.py              # the loop
│   ├── tools/
│   │   ├── __init__.py           # imports every module → triggers registration
│   │   ├── decorators.py         # register_tool, get_tool_metadata, get_json_type
│   │   ├── self_prompting.py     # prompt_llm, prompt_llm_for_json, prompt_expert, …
│   │   ├── greeting.py           # extract_visitor_details, compose_greeting
│   │   ├── guestbook.py          # sign_guestbook, read_guestbook, undo_last_entry
│   │   ├── planning.py           # create_plan, track_progress
│   │   ├── delegation.py         # call_agent (+3 memory-sharing variants)
│   │   └── system.py             # terminate
│   ├── capabilities/
│   │   ├── time_aware.py · plan_first.py · progress_tracking.py · logging.py
│   ├── safety/
│   │   ├── reversible.py         # ReversibleAction, ActionTransaction
│   │   └── staged.py             # StagedActionEnvironment
│   ├── storage/
│   │   └── guestbook.py          # JSON-file store, path-confined
│   └── agents/
│       ├── greeter.py            # build_greeter_agent(features) — THE agent
│       └── translator.py         # build_translator_agent() — sub-agent
└── tests/
    ├── conftest.py · fakes.py    # FakeLLM + fixtures
    ├── test_decorators.py · test_registry.py · test_actions.py
    ├── test_environment_di.py · test_agent_language.py · test_agent_loop.py
    ├── test_memory.py · test_capabilities.py · test_self_prompting.py
    ├── test_greeting.py · test_guestbook.py · test_storage.py
    ├── test_safety.py · test_delegation.py · test_config.py
    ├── test_e2e_fake.py          # full run, all flag permutations, no network
    └── test_live_smoke.py        # @pytest.mark.live — skipped without a key
```

---

## 5. Frozen interface contracts

**These signatures are binding.** Sub-agents working in parallel must code against them exactly. Any change requires the orchestrator's approval and a broadcast to in-flight agents.

### 5.1 The LLM contract — the keystone

Everything hangs off this one signature. `generate_response` takes a `Prompt` and returns a **string**. When the agent language is function-calling, that string is a JSON object `{"tool": name, "args": {...}}`. This is what makes `FakeLLM` trivial and CI key-free.

```python
# core/prompt.py
@dataclass
class Prompt:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict]] = None
    metadata: Dict = field(default_factory=dict)

# llm.py
def generate_response(prompt: Prompt, tier: str = "powerful") -> str:
    """Call the LLM via LiteLLM. Returns a tool-call JSON string, or plain text."""
    # completion(model=config.model_for(tier), messages=prompt.messages,
    #            tools=prompt.tools or None, max_tokens=config.MAX_TOKENS)
    # if prompt.tools and msg.tool_calls:
    #     tc = msg.tool_calls[0]
    #     return json.dumps({"tool": tc.function.name,
    #                        "args": json.loads(tc.function.arguments)})
    # return msg.content
```

`AgentFunctionCallingActionLanguage.parse_response` does `json.loads(response)` and, on failure, falls back to `{"tool": "terminate", "args": {"message": response}}` — exactly as in `game_framework/agent_language.md`.

### 5.2 Core primitives

```python
@dataclass(frozen=True)
class Goal:
    priority: int; name: str; description: str

class Action:
    def __init__(self, name, function, description, parameters, terminal=False): ...
    def execute(self, **args) -> Any: ...

class ActionRegistry:
    def register(self, action: Action) -> None: ...
    def get_action(self, name: str) -> Optional[Action]: ...
    def get_actions(self) -> List[Action]: ...

class PythonActionRegistry(ActionRegistry):
    def __init__(self, tags: List[str] = None, tool_names: List[str] = None): ...
    # pulls from the global registry populated by @register_tool

class Memory:
    def add_memory(self, memory: dict) -> None: ...
    def get_memories(self, limit: int = None) -> List[Dict]: ...
    def copy_without_system_memories(self) -> "Memory": ...

class ActionContext:
    def __init__(self, properties: Dict = None): ...
    def get(self, key, default=None): ...
    def get_memory(self) -> Memory: ...
    def get_action_registry(self) -> ActionRegistry: ...
    def get_agent_registry(self) -> "AgentRegistry": ...
```

### 5.3 Environment and DI

```python
class Environment:
    def execute_action(self, agent, action_context, action, args) -> dict: ...
    def format_result(self, result) -> dict:
        return {"tool_executed": True, "result": result,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

class PythonEnvironment(Environment):
    """Injects `action_context` and any `_prefixed` ActionContext property
    whose name matches a parameter of the tool function."""
```

Injection rules (must match `dep_injection_with_environment.md`):
- a parameter named `action_context` receives the `ActionContext`
- a parameter named `_foo` receives `action_context.get("foo")`
- **both are excluded from the JSON schema the model sees** (`get_tool_metadata` skips `action_context`, `action_agent`, and any `_`-prefixed name)

### 5.4 Tool registration

```python
def register_tool(tool_name=None, description=None, parameters_override=None,
                  terminal=False, tags=None):
    """Registers into core.registry.tools and core.registry.tools_by_tag."""
```

Type mapping in `get_json_type`: `str→string`, `int→integer`, `float→number`, `bool→boolean`, `list/List[...]→array`, `dict/Dict[...]→object`; unknown → `string`. Docstring becomes the description; missing type hint defaults to `str`; no default value ⇒ required.

### 5.5 Agent

```python
class Agent:
    def __init__(self, goals, agent_language, action_registry,
                 generate_response, environment,
                 capabilities: List[Capability] = None,
                 max_iterations: int = 10,
                 max_duration_seconds: int = 180): ...

    def run(self, user_input: str, memory=None,
            action_context_props: dict = None) -> Memory: ...
```

`run` must invoke, in order, every `Capability` hook from `capability_architecture_pattern.md`: `init` → loop{`start_agent_loop` → `process_prompt` → `process_response` → `process_action` → `process_result` → `process_new_memories` → `end_agent_loop` → `should_terminate`} → `terminate`. `process_*` hooks compose left-to-right via `reduce`. **These hooks exist and fire in the default run even though the default capability list is empty** — that is what makes enabling one a single-line change.

### 5.6 Agent builders

```python
def build_greeter_agent(features: Features = None,
                        generate_response: Callable = None) -> Agent: ...
def build_translator_agent(generate_response: Callable = None) -> Agent: ...
```

Both accept an injected `generate_response` so tests pass a `FakeLLM` and never touch the network.

### 5.7 Configuration

```python
@dataclass
class Features:
    time_aware: bool = False
    plan_first: bool = False
    progress_tracking: bool = False
    verbose_logging: bool = False
    delegation: bool = False          # adds translator sub-agent + call_agent
    staged_review: bool = False       # StagedActionEnvironment
    dynamic_experts: bool = False     # create_and_consult_expert
    @classmethod
    def from_env(cls) -> "Features": ...   # AGENT_ENABLE_TIME_AWARE=1 etc.
```

Provider resolution, in order: `LLM_PROVIDER` env → `ANTHROPIC_API_KEY` present → `OPENAI_API_KEY` present → raise a `ConfigError` naming both variables and pointing at `.env.example`.

| Provider | `powerful` tier | `fast` tier |
|---|---|---|
| `anthropic` | `anthropic/claude-opus-5` | `anthropic/claude-haiku-4-5` |
| `openai` | `openai/gpt-4o` | `openai/gpt-4o-mini` |

Overridable via `AGENT_MODEL_POWERFUL` / `AGENT_MODEL_FAST`. Both tiers land in `ActionContext` as `llm` (powerful) and `fast_llm`, which is how `tools/greeting.py` demonstrates MATE model efficiency: extraction uses `fast_llm`, composition uses `llm`.

### 5.8 The FakeLLM contract

```python
class FakeLLM:
    """Deterministic scripted generate_response. Records every Prompt it saw."""
    def __init__(self, responses: List[str]): ...
    def __call__(self, prompt: Prompt, tier: str = "powerful") -> str: ...
    calls: List[Prompt]           # for asserting on prompt construction
```

Helper: `tool_call(name, **args) -> str` returns `json.dumps({"tool": name, "args": args})`.

---

## 6. Tool inventory

| Tool | Tags | Terminal | Default set | Notes |
|---|---|---|---|---|
| `extract_visitor_details(document_text)` | `greeting` | no | ✅ | Fixed schema, `fast_llm`, retry-wrapped |
| `compose_greeting(visitor, _greeting_policy)` | `greeting` | no | ✅ | `prompt_expert` concierge persona; policy injected from disk |
| `sign_guestbook(name, greeting, note="")` | `guestbook` | no | ✅ | `_guestbook` injected; validates; returns `entry_id` |
| `read_guestbook(limit=5)` | `guestbook` | no | ✅ | Empty → error message that *instructs* the agent what to do next |
| `terminate(message)` | `system` | **yes** | ✅ | |
| `undo_last_guestbook_entry()` | `safety` | no | flag | Pairs with `ReversibleAction` |
| `prompt_llm(prompt)` | `prompts` | no | flag | Agent-facing self-dialog |
| `prompt_llm_for_json(schema, prompt)` | `prompts` | no | flag | 3-attempt retry, ```json fence stripping |
| `prompt_expert(description_of_expert, prompt)` | `prompts` | no | flag | Persona pattern |
| `create_and_consult_expert(expertise_domain, problem)` | `prompts` | no | flag | Dynamic persona |
| `create_plan()` | `planning` | no | flag | Used by `PlanFirstCapability` |
| `track_progress()` | `planning` | no | flag | Used by `ProgressTrackingCapability` |
| `call_agent(agent_name, task)` | `delegation` | no | flag | Message passing |
| `call_agent_with_reflection(...)` | `delegation` | no | flag | Memory reflection |
| `hand_off_to_agent(...)` | `delegation` | no | flag | Memory handoff |
| `call_agent_with_selected_context(...)` | `delegation` | no | flag | LLM-selected memories |

**Default registry:** `PythonActionRegistry(tags=["greeting", "guestbook", "system"])` — five tools. Flags append tags.

Note the deliberate MATE/action-specificity choice, which `docs/EXTENDING.md` must call out: there is no `write_file`. `sign_guestbook` is the single safe tool that owns validation, ordering, and the storage path — the "Single Safe Tool vs Multiple Risky Tools" pattern.

---

## 7. Task waves

Each task below is a unit of work for one sub-agent. **Every task is done only when its acceptance criteria pass and its own tests are green.** Tests are written by the task that owns the code — not deferred.

### Wave A — Foundation (1 agent, blocking)

**T1 · Scaffold, config, LLM layer, fake**
Files: `pyproject.toml`, `.env.example`, `.gitignore`, `.github/workflows/ci.yml`, `src/hal9000/config.py`, `llm.py`, `core/prompt.py`, `tests/fakes.py`, `tests/conftest.py`, `tests/test_config.py`.

Acceptance:
- `uv sync` succeeds; `uv run python -c "import hal9000"` succeeds.
- `Features.from_env()` reads all seven flags; unset ⇒ all `False`.
- Provider resolution covers all four cases (override / anthropic / openai / neither), and the "neither" case raises an error naming both env vars.
- `FakeLLM` and `tool_call()` exist and match §5.8.
- `uv run pytest tests/test_config.py` green **with no API keys in the environment**.
- `ruff check` clean.

### Wave B — GAME core (1 agent, blocking)

**T2 · Core primitives + agent loop**
Files: `core/goals.py`, `memory.py`, `action_context.py`, `actions.py`, `registry.py`, `environment.py`, `agent_language.py`, `capabilities.py`, `agent.py`; tests `test_actions.py`, `test_memory.py`, `test_environment_di.py`, `test_agent_language.py`, `test_agent_loop.py`, `test_capabilities.py`.

Acceptance:
- All §5.2/5.3/5.5 signatures present and exact.
- `PythonEnvironment` injects `action_context` and `_`-prefixed props; a tool that declares neither receives neither. Tool exceptions are caught and returned as `{"tool_executed": False, "error": ..., "traceback": ...}` — the loop never crashes.
- Both agent languages implemented; `AgentJsonActionLanguage` round-trips a ```action fenced block; function-calling language falls back to `terminate` on unparseable output.
- `Capability` exposes all ten hooks; a spy capability asserts every hook fires exactly when expected, in order, across a 3-iteration run.
- Loop honors `max_iterations` **and** `max_duration_seconds`, and terminates on a terminal action.
- Tests drive the loop entirely with `FakeLLM`. No network.

### Wave C — Tool system (1 agent, blocking)

**T3 · Decorator, global registry, self-prompting**
Files: `tools/decorators.py`, `tools/__init__.py`, `tools/self_prompting.py`, `tools/system.py`; tests `test_decorators.py`, `test_registry.py`, `test_self_prompting.py`.

Acceptance:
- `@register_tool` derives name from `__name__`, description from `__doc__`, and a JSON schema from signature + type hints; `parameters_override` wins when supplied.
- Schema **excludes** `action_context`, `action_agent`, and `_`-prefixed params — assert this directly.
- `tools_by_tag` index correct; `PythonActionRegistry(tags=[...])` selects the right subset; unknown tag yields an empty registry, not an error.
- `prompt_llm_for_json` retries 3×, strips ```json fences, raises on final failure; test with a FakeLLM that returns garbage twice then valid JSON.
- `prompt_expert` puts the persona in a `system` message and the task in a `user` message.
- Importing `hal9000.tools` registers every tool exactly once; double-import does not duplicate.

### Wave D — Domain + capabilities (2 agents, parallel)

**T4 · Storage, guestbook tools, greeting tools**
Files: `storage/guestbook.py`, `tools/guestbook.py`, `tools/greeting.py`, `config/greeting_policy.txt`; tests `test_storage.py`, `test_guestbook.py`, `test_greeting.py`.

Acceptance:
- Store is a JSON file at a configurable path, created on first write, **confined** to its configured directory — a traversal attempt raises.
- `sign_guestbook` validates input, returns a monotonic `entry_id`, and is safe to call concurrently-ish (read-modify-write under a lock or atomic replace).
- `read_guestbook` on an empty book returns an error string that *instructs the agent what to do next* (the injected-instruction pattern), not a bare failure.
- `extract_visitor_details` uses `fast_llm`; `compose_greeting` uses `llm` — assert the tier argument, this is the MATE demo.
- `compose_greeting` reads `config/greeting_policy.txt` at call time; a missing file degrades gracefully; **editing the file changes the prompt** (assert the policy text appears in the captured prompt).
- Every tool has a docstring good enough to be its own model-facing description.

**T5 · Capabilities**
Files: `capabilities/time_aware.py`, `plan_first.py`, `progress_tracking.py`, `logging.py`, `tools/planning.py`; tests extend `test_capabilities.py`.

Acceptance:
- `TimeAwareCapability` seeds memory in `init` and refreshes the time in **every** `process_prompt`; timezone comes from `ActionContext`, default `America/Chicago`.
- `PlanFirstCapability` calls `create_plan` once, stores it in memory with the configured type; a second `init` does not re-plan.
- `ProgressTrackingCapability` honors `track_frequency` (assert: with `frequency=2`, 4 iterations ⇒ 2 reports).
- `LoggingCapability` writes structured lines and mutates nothing.
- Each capability tested in isolation with `FakeLLM`, and at least one test composes two capabilities and asserts the `reduce` ordering.

### Wave E — Safety + multi-agent (2 agents, parallel)

**T6 · Environmental safety**
Files: `safety/reversible.py`, `safety/staged.py`, `undo_last_guestbook_entry` in `tools/guestbook.py`; tests `test_safety.py`.

Acceptance:
- `ReversibleAction.run` records args/result/timestamp; `undo` reverses; `undo` without a prior run raises.
- `ActionTransaction`: on mid-sequence failure, everything already executed is reversed **in reverse order**, and the error propagates. Assert with a 3-action transaction failing on the third.
- `StagedActionEnvironment` stages configured tools instead of executing them, submits them for review, and executes only on approval. Two reviewers: `human` (stdin y/N) and `llm` (uses the powerful tier). Rejection returns a structured refusal the agent can read and react to.
- End-to-end: with `staged_review` on and the reviewer rejecting, the guestbook file is **unchanged**.

**T7 · Multi-agent**
Files: `tools/delegation.py`, `agents/translator.py`, `AgentRegistry` in `core/registry.py`; tests `test_delegation.py`.

Acceptance:
- All four memory-sharing patterns implemented and distinguishable by test: message passing (fresh memory, last item returned), reflection (callee memories copied back, tagged with source), handoff (caller's memory object passed through), selective (LLM picks memory IDs; reasoning preserved in caller memory).
- `call_agent` does **not** pass the agent registry down — assert no infinite recursion is possible.
- Unknown agent name returns a structured error, not an exception escaping the loop.
- `build_translator_agent` runs standalone under `FakeLLM` and returns a translated greeting.

### Wave F — Assembly (1 agent, blocking)

**T8 · The agent, the CLI, the flags**
Files: `agents/greeter.py`, `__main__.py`; tests `test_e2e_fake.py`.

Acceptance:
- `build_greeter_agent()` with no args produces the lean default: 5 tools, empty capability list, `PythonEnvironment`, function-calling language.
- Each of the seven flags, set individually, changes the assembled agent in an asserted way (tool count, capability list, or environment class) **and** the agent still completes a scripted run.
- All-flags-on completes a scripted run.
- CLI: `python -m hal9000 "text"` runs; `--help` documents every flag and env var; no args ⇒ reads stdin or prompts; a clean error (exit 2, no traceback) when no API key is configured.
- `test_e2e_fake.py` runs the full three-tool happy path start to finish on `FakeLLM`, asserts the guestbook file contents, and asserts the final terminate message reaches stdout.

### Wave G — Docs + verification (2 agents, parallel)

**T9 · Verification pass**
- `tests/test_live_smoke.py`, marked `@pytest.mark.live`, auto-skipped without a key; one test per provider.
- Coverage gate wired into `pyproject.toml` (`--cov=hal9000 --cov-fail-under=85`).
- CI workflow runs `ruff check`, `ruff format --check`, `pytest` on 3.11/3.12/3.13, with **no secrets configured**.
- Fresh-clone rehearsal documented and performed: clone to a temp dir, `uv sync`, run with each provider key, record timings in the task report.

**T10 · Documentation + template polish**
- `README.md`: what it is, the 2-minute quickstart, the feature-flag table, a "where do I change things" map, the guestbook example transcript.
- `docs/ARCHITECTURE.md`: GAME walkthrough, the loop with capability hooks, the DI mechanism, why `generate_response` returns a string.
- `docs/EXTENDING.md`: add a tool (decorate + tag), add a capability, add a sub-agent, swap the agent language, replace guestbook storage with a DB — each with a worked diff. Must explain the action-specificity choice (why no `write_file`).
- `docs/COVERAGE.md`: the §3 table, every row pointing at a real file, verified by a script or a test that asserts each named path exists.
- `docs/ADR/0001-litellm.md`.
- GitHub template hygiene: `.github/ISSUE_TEMPLATE/`, a `LICENSE`, and a `README` "Using this template" section telling the user the three things to change first (goals in `agents/greeter.py`, tools in `tools/`, flags in `config.py`).

---

## 8. Global definition of done

The orchestrator reports completion only when **all ten** hold:

1. `uv run pytest` — green, **with no API keys in the environment**, no network access.
2. `uv run pytest --cov=hal9000` — ≥ 85% line coverage.
3. `ruff check .` and `ruff format --check .` — clean.
4. `ANTHROPIC_API_KEY=… uv run python -m hal9000 "Ada Lovelace, mathematician, visiting from London."` — completes, prints a greeting, appends exactly one guestbook entry.
5. Same run with **only** `OPENAI_API_KEY` set — completes identically in shape.
6. Every one of the seven feature flags individually enabled: run completes (verified under `FakeLLM` in CI; at least `time_aware`, `plan_first`, and `delegation` verified live).
7. `docs/COVERAGE.md` accounts for every row in §3, each pointing at a file that exists.
8. Fresh-clone rehearsal: clone → `uv sync` → export one key → run, in under two minutes, with no step not written in the README.
9. No file in `src/` exceeds ~200 lines; no function exceeds ~50. This is a teaching artifact; density is a defect.
10. `git log` is a coherent history — one commit per task, message naming the task ID and the patterns it implements.

---

## 9. Orchestration notes

- **Wave discipline.** A/B/C/F are single-agent blocking waves; D/E/G run two agents in parallel. Do not start a wave until the prior wave's acceptance criteria pass — §5's contracts only hold parallel agents together if the contracts are already on disk.
- **The contracts in §5 are the coordination mechanism.** Give every sub-agent §5 in full, plus its own task block and the relevant §3 rows. A sub-agent that wants to change a frozen signature must escalate, not improvise.
- **Point sub-agents at the source notes.** Each task block's patterns map to specific files in `/Users/mike/AI/coursera/coursera_ai_agent_content/`. Those notes are the specification; where this plan and a note disagree on a detail, the note wins and the orchestrator records the deviation.
- **Model IDs.** `claude-opus-5` / `claude-haiku-4-5` are current and correct as written — do not "correct" them to dated variants. `.env.example` must show both providers.
- **The lean default is a feature, not an omission.** If a sub-agent proposes enabling something by default "to show it off," decline. The showcase lives in tests, docs, and flags.

## 10. Risks

| Risk | Mitigation |
|---|---|
| LiteLLM's tool-call response shape drifts between providers | Isolate all shape-handling in `llm.py`; one live smoke test per provider catches drift immediately |
| Parallel agents diverge on interfaces | §5 frozen contracts written to disk in Wave A–C before any parallel wave starts |
| Coverage becomes a checkbox, not a demonstration | `docs/COVERAGE.md` is test-verified, and every flag has a behavioral assertion, not just an import |
| Template accumulates guestbook-specific assumptions | `docs/EXTENDING.md` T10 must include a worked "replace the domain entirely" diff |
| 36 notes' worth of patterns bloats a "hello world" | Hard limits in DoD #9; lean default keeps the readable path short |
