# Orchestrator Rulings

Binding decisions made during the build, where BUILD_PLAN.md §5 needed
interpretation or the source notes disagreed with each other. Read this
alongside §5 — these rulings amend it.

---

## R1 · ActionContext LLM tier keys — `llm` is canonical, `powerful_llm` is an alias

**Raised by:** T1. **Affects:** T2, T4, T7, T8, T10.

The notes corpus is internally inconsistent:

| Key | Occurrences | Where |
|---|---|---|
| `llm` | 4 | `persona_pattern.md`, `structured_data_extraction.md`, `persona_abstraction.md`, `clean_tools_with_dep_injection.md` |
| `powerful_llm` | 1 | `MATE_design_principle.md` (alongside `fast_llm`) |

"Where plan and note disagree, the note wins" cannot resolve this, because the
notes disagree with each other.

**Ruling.** `ActionContext` carries **three** keys, two of which are the same object:

- `llm` — the powerful tier. **Canonical.** Every tool reads this.
- `powerful_llm` — an alias bound to the identical callable as `llm`.
- `fast_llm` — the fast tier.

Rationale: `llm` is what all four self-prompting tools we actually implement
use, and §5.7 already froze it. The alias costs one dictionary entry and makes
the MATE note's example run verbatim, so both notes are honoured rather than
one being declared a typo.

`docs/COVERAGE.md` (T10) must note the alias so a reader coming from
`MATE_design_principle.md` is not confused.

---

## R2 · Env-derived values are functions, not import-time constants

**Raised by:** T1 (accepted). **Affects:** all tasks.

An import-time constant reading an env var makes the test suite depend on the
developer's shell. House rule:

- **Env-derived ⇒ a callable**: `config.max_tokens()`, `max_iterations()`,
  `max_duration_seconds()`, `guestbook_path()`, `timezone()`, `model_for()`,
  `resolve_provider()`.
- **Fixed ⇒ a constant**: `DEFAULT_MAX_TOKENS`, `DEFAULT_MAX_ITERATIONS`,
  `DEFAULT_MAX_DURATION_SECONDS`.

The commented sketch in §5.1 shows `config.MAX_TOKENS`; that sketch is
superseded. Frozen *signatures* in §5 are unaffected.

`Agent.__init__` keeps its §5.5 literal defaults; the T8 call site passes
`config.max_iterations()` / `config.max_duration_seconds()` so tests can set
limits directly.

---

## R3 · `readme` key deferred to T10

**Raised by:** T1. **Affects:** T10.

Hatchling fails `uv sync` when `[project] readme = "README.md"` points at a
nonexistent file, so T1 omitted the key rather than create T10's deliverable.
**T10 must add `readme = "README.md"` back to `[project]` when it writes the
README, and re-run `uv sync` to confirm.**

---

## R4 · Markdown excluded from ruff

**Raised by:** T1 (accepted). **Affects:** T10.

`extend-exclude = ["*.md"]` in `[tool.ruff]`. Ruff 0.16 formats Python inside
Markdown fences and wanted to rewrite the code samples in `BUILD_PLAN.md`.
Excluding Markdown keeps the plan byte-stable and lets docs show abbreviated,
illustrative code without fighting the formatter.

---

## R5 · `start_agent_loop` is a veto, not the note's `reduce`

**Raised by:** T2. **Accepted — verified independently by the orchestrator.**
**Affects:** T5, T8.

`capability_architecture_pattern.md` line 24 shows:

```python
can_start_loop = reduce(lambda a, c: c.start_agent_loop(self, action_context),
                        self.capabilities, False)
```

The lambda discards its accumulator `a`, so this expression returns the *last*
capability's value — and with the default **empty** capability list it returns
the seed, `False`. Verified:

```
empty caps, literal note form -> False
two caps,  literal note form  -> True
```

A literal implementation would mean the lean-default agent — zero capabilities,
which is our headline configuration — never runs a single iteration.

**Ruling.** This is an error in the note, not a design. The loop calls
`start_agent_loop` on every capability and proceeds unless **any** returns
`False`, which honours the hook's own documented contract ("Called at the start
of each iteration... return True to continue"). Implemented as
`all([c.start_agent_loop(...) for c in self.capabilities])`, which is
vacuously `True` for an empty list.

This is the one place in the build where "the note wins" is overridden. It is
overridden because the note contradicts itself: the docstring says return True
to continue, and the reduce makes that impossible to honour.

T5: a capability that wants to halt the run returns `False` from
`start_agent_loop`. Do not expect accumulator semantics.

---

## R6 · `create_and_consult_expert` takes `problem_description`

**Raised by:** T3. **Affects:** T8, T10.

BUILD_PLAN §6's tool-inventory table says
`create_and_consult_expert(expertise_domain, problem)`.
`persona_abstraction.md` says `problem_description`. **The note wins**; the
§6 table row is wrong. T10 must correct it when writing `docs/COVERAGE.md`.

---

## R7 · `terminate` returns its message unchanged

**Raised by:** T3. **Affects:** T8.

Another R1-shaped conflict — two notes disagree and neither is "the plan", so
"note wins" cannot arbitrate:

| Note | Behaviour |
|---|---|
| `refactored_readme_agent.md` | returns `f"{message}\nTerminating..."` |
| `game_in_code.md` | returns `message` |

**Ruling: return `message` unchanged.** BUILD_PLAN §2's transcript shows the
composed greeting printed verbatim to the user, and T8's e2e criterion asserts
the terminate message reaches stdout. A `\nTerminating...` suffix would append
framework chatter to the agent's actual product. T8 gets exactly what the model
wrote.

---

## R8 · LiteLLM's import-time network fetch is pinned off in tests and CI

**Raised by:** T3 as escalation E1. **Fixed by the orchestrator.** **Affects:** T9.

`import litellm` fetches its model-price map from raw.githubusercontent.com at
import time, with three retries and 2s backoffs. Measured: **4 outbound
connection attempts** per import. It degrades gracefully to a bundled copy, so
the suite passed — but only because the machine had internet. DoD #1 requires
*no network*, and a sandboxed runner would eat the full retry latency.

`LITELLM_LOCAL_MODEL_COST_MAP=True` is now set in two places:

- `tests/conftest.py`, at module scope via `os.environ.setdefault` — it must
  precede the first `litellm` import.
- `.github/workflows/ci.yml`, on the Test step.

Deliberately **not** set in `src/hal9000/llm.py`: in production a current
price map is worth the fetch. This is a test/CI guarantee, not a library
default.

Verified with `socket.socket.connect` hard-blocked across the whole suite:
`183 passed, exit 0, outbound connection attempts: 0`.

**T9:** keep this property. Add a regression test that fails if any future
import reaches the network, and do not remove either setting.

---

## R9 · `TimeAwareCapability.process_prompt` must preserve `tools`

**Raised by:** T5. **Accepted — verified independently.** **Affects:** T8, T10.

`capability_architecture_pattern.md` line 271 ends `process_prompt` with:

```python
return Prompt(messages=messages)
```

`Prompt` carries `messages`, `tools` and `metadata`. Reconstructing it from
`messages` alone **discards every tool definition**. Under
`AgentFunctionCallingActionLanguage` that means the moment time-awareness is
enabled, the model stops being offered any tools at all — it would be handed a
prompt describing a task it has no way to act on.

The note's version also mutates the caller's `messages[0]` in place.

**Ruling.** `process_prompt` returns
`Prompt(messages=<new list>, tools=prompt.tools, metadata=prompt.metadata)`.
This is the second outright error found in
`capability_architecture_pattern.md` (see R5); that file's code should be read
as illustrative, not as a reference implementation.

**General rule for any future capability:** a `process_prompt` hook must
preserve every field of the `Prompt` it did not deliberately change. T10 should
state this in `docs/EXTENDING.md`, because it is the single easiest way for
someone adding a capability to break their agent silently.

---

## R10 · Planning tools are tagged `planning`, not `prompts`

**Raised by:** T5. **Affects:** T10.

The notes disagree with each other:

| Note | Tag on its tool |
|---|---|
| `plan_first_capability.md` | `create_plan` → `["planning"]` |
| `progress_tracking_capability.md` | `track_progress` → `["prompts"]` |

**Ruling: both are tagged `planning`**, per BUILD_PLAN §6.

This is the one place the plan beats a note, and the reason is behavioural, not
cosmetic: the `prompts` tag is what `Features.dynamic_experts` switches on.
Tagging `track_progress` as `prompts` would mean enabling dynamic experts
silently hands the model the progress-tracking tool as well — a flag with an
invisible second effect. One tag, one coherent group.

---

## R11 · Planning tools take `_memory` and `_action_registry`

**Raised by:** T5. **Accepted — verified independently.** **Affects:** T10.

The notes are internally inconsistent about dependency naming:

| Note | Signature |
|---|---|
| `plan_first_capability.md` | `create_plan(action_context, memory, action_registry)` |
| `progress_tracking_capability.md` | `track_progress(action_context, _memory, action_registry)` |

Under §5.3's injection rules only `_`-prefixed names are injected and hidden.
An unprefixed `memory` or `action_registry` becomes a **required object
parameter in the model-facing JSON Schema** — the model would be asked to
invent a `Memory` object as a tool argument.

`progress_tracking_capability.md` already uses `_memory`, so the corpus
contains both readings. Only the prefixed one works.

**Ruling:** both tools take `_memory` and `_action_registry`. Both schemas are
asserted to be exactly `{"type": "object", "properties": {}, "required": []}`.

---

## R12 · `Features.staged_review` also adds the `safety` tag

**Raised by:** T6, correctly, as an unresolved gap. **Affects:** T8, T10.

BUILD_PLAN §6 lists `undo_last_guestbook_entry` with tag `safety` and default
column "flag" — but **names no flag**, and §5.7's `Features` has no `safety`
field. T6 did not invent one, which was the right call: `Features` is frozen.

**Ruling.** `staged_review` is the environmental-safety flag and governs both.
When it is set, T8 adds `"safety"` to the registry tag list *and* swaps in
`StagedActionEnvironment`:

```python
tags = ["greeting", "guestbook", "system"]
if features.staged_review:
    tags.append("safety")
    environment = StagedActionEnvironment()
```

Rationale: staging and undo are one concern — preventing bad writes and
reversing them. Splitting them would need an eighth flag and a change to a
frozen contract, for no gain. The flag's documented meaning in the README and
`--help` is "environmental safety", not "staged review" alone.

**Do not stage `undo_last_guestbook_entry` itself.** `DEFAULT_STAGED_TOOLS`
stays `("sign_guestbook",)`. Undo has no registered reverser, so staging it
would create an approved-but-unrollbackable action. `sign_guestbook` is the
only tool in the template that writes — which is the single-safe-tool payoff.

---

## R13 · `ActionTransaction` is synchronous

**Raised by:** T6. **Accepted — verified independently.** **Affects:** T10.

`environment_safety.md` declares `async def execute` / `async def rollback`,
and `rollback` does `await action.undo()` — but `ReversibleAction.undo` is
declared `def undo(self)` in the same file, thirty lines earlier. The literal
code raises `TypeError: object NoneType can't be used in 'await' expression`.
The `async` is illustrative, not functional; the note contradicts itself, so
"the note wins" cannot arbitrate.

**Ruling: synchronous.** Nothing else in the codebase is async, `Agent.run` and
`Environment.execute_action` are both sync, and an async transaction would
force `asyncio.run()` inside a sync method — which raises if a caller ever
embeds the agent in an existing event loop. A sync transaction is also directly
callable from `StagedActionEnvironment.execute_action`, which must honour the
frozen `Environment.execute_action` signature.

T10: note this in `docs/EXTENDING.md` for anyone porting the note's code.

---

## R14 · LLM approval parsing fails closed

**Raised by:** T6. **Accepted — verified independently.** **Affects:** T10.

`environment_safety.md` line 148 decides whether an LLM reviewer approved a
staged action with:

```python
return "approved" in response.lower()
```

Measured against realistic reviewer replies:

```
True   <- read as APPROVED  |  'approved'
True   <- read as APPROVED  |  'This is not approved - unsafe'
False                       |  'I would reject this'
False                       |  'Do not approve'
```

A reviewer that says **"This is not approved - unsafe"** is recorded as having
approved. The check fails *open*, in the one component whose entire job is to
fail closed. This is the most consequential error found in the corpus.

**Ruling.** Rejection words are checked first (`not approved`, `reject`,
`deny`, `unsafe`, `do not`), and anything ambiguous, empty, or unparseable is a
**refusal**. Same principle governs `HumanReviewer`: no tty, a cancelled
prompt, or any answer other than `y`/`yes` — including an empty line — is a
refusal.

T10 must call this out in `docs/EXTENDING.md`. Anyone copying the note's
one-liner into their own reviewer inherits a safety check that approves things
it was told not to approve.

---

## R15 · The greeter gets `call_agent` only, not all four delegation tools

**Raised by:** T7 as a recommendation. **Accepted.** **Affects:** T8, T10.

All four memory-sharing patterns are implemented, tagged `delegation`, and
tested. But when `Features.delegation` is on, the greeter receives **only
`call_agent`**.

The greeter's honest use case is "translate this greeting into the visitor's
`language`" — a self-contained sub-task where it wants the answer and nothing
else. That is message passing, exactly. The other three are actively worse
here:

- **reflection** pastes the translator's whole transcript into the greeter's
  memory on every later turn, for reasoning the greeter cannot use
- **handoff** lets the translator write into the greeter's memory and see the
  entire guestbook conversation — wrong shape, since the translator is being
  consulted, not taking over
- **selective** spends an extra powerful-tier call choosing from a four-item
  memory

Four tools bolted onto a five-tool agent also doubles the surface the model
must choose between to do one job — the crowded-tool-list problem the
lean-default rule exists to prevent.

`PythonActionRegistry(tags=["delegation"])` still yields all four in one line
for anyone who wants them, and `docs/EXTENDING.md` is where T10 should show the
other three.

---

## R16 · Delegation results are unwrapped from the framework envelope

**Raised by:** T7. **Affects:** T10.

The multi-agent notes all end at `result_memory.items[-1]["content"]`. In this
framework that last memory item is
`json.dumps({"tool_executed": true, "result": "...", "timestamp": ...})`,
because `Agent.update_memory` writes tool results back into memory as JSON.
The note's literal form therefore hands the calling agent a string describing
the framework's own plumbing instead of the sub-agent's answer.

The note's framework has the same property, so this is an unexamined detail
rather than a design choice.

**Ruling.** Delegation tools unwrap exactly that envelope and pass anything
else through untouched, so `call_agent(...)["result"]` is the answer the
sub-agent produced.

---

## R17 · Two T7 commits, deliberately not squashed

**Affects:** T9, T10 (DoD #10).

T7 amended a commit while T6 was committing concurrently and its amend landed
on T6's commit. T7 caught it immediately and repaired with
`git reset --mixed b65be1d`. **Verified by the orchestrator:** `b65be1d`
contains exactly T6's six files, no T7 file appears in it, `git fsck` is
silent, and the tree is clean.

The repair left T7's work in two commits (`676b501`, `5648bf6`) with T6's
commit between them. Squashing would require rewriting T6's commit, which is a
worse trade than two commits sharing a task ID. History stands as is.

DoD #10 asks for a coherent history naming each task — that is satisfied.

---

## R18 · `tool_call` gains a positional-only marker

**Raised by:** T8. **Authorized — this amends frozen §5.8.** **Affects:** T9.

`tests/fakes.py` defines `tool_call(name, **args)`. Its first parameter is
`name`, which collides with any tool that has a `name` argument — including
`sign_guestbook`, the template's central write:

```
tool_call("sign_guestbook", name="Ada")
TypeError: tool_call() got multiple values for argument 'name'
```

T8 correctly refused to change a frozen helper that 358 tests depend on, and
worked around it with literal `json.dumps`. The orchestrator hit the same trap
independently while verifying the CLI, which is evidence enough that it will
catch template users too.

**Ruling.** T9 changes the signature to:

```python
def tool_call(name, /, **args):
```

The positional-only marker frees `name` for use as a tool argument and
**requires zero call-site changes** — every existing call already passes the
tool name positionally. T9 must confirm all tests still pass and add a case
covering `tool_call("sign_guestbook", name="Ada")`.

This is the only amendment to a frozen signature in the entire build, and it is
purely additive to what callers can express.

---

## R19 · Live verification is deferred to the user — no API keys in this session

**Affects:** T9, and the final handoff.

Neither `ANTHROPIC_API_KEY` nor `OPENAI_API_KEY` is set in the build
environment, and there is no `.env`. **DoD items #4, #5, and the live portion
of #6 therefore cannot be verified during this build.** They are not satisfied
and must not be reported as satisfied.

What T9 delivers instead:

- `tests/test_live_smoke.py`, marked `@pytest.mark.live`, one test per
  provider, **auto-skipping when its key is absent** — so the suite stays green
  and the tests are ready to run the moment a key exists.
- A fresh-clone rehearsal covering everything that does not need a key: clone
  to a temp dir, `uv sync`, `uv run pytest`, the exit-2 no-provider path, and a
  `FakeLLM`-driven end-to-end run. Timings recorded.
- An explicitly labelled list of the exact commands the user must run to close
  #4, #5 and #6, placed in the README's verification section.

The orchestrator reports these three items as **outstanding, pending user
keys** in the final handoff. Everything verifiable without a key is verified.
