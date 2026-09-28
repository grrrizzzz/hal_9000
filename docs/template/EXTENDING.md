# Extending this template

Six worked changes, then the traps. Every diff below is against the repository
as it stands; read [`ARCHITECTURE.md`](ARCHITECTURE.md) first if the words
`ActionContext` or "capability hook" are new.

| I want to... | Section |
|---|---|
| Give the agent a new ability | [Add a tool](#1-add-a-tool) |
| Change how the loop itself behaves | [Add a capability](#2-add-a-capability) |
| Split work across two agents | [Add a sub-agent](#3-add-a-sub-agent) |
| Stop using provider function calling | [Swap the agent language](#4-swap-the-agent-language) |
| Put the data in a real database | [Replace the storage](#5-replace-guestbook-storage-with-a-database) |
| Build something that is not a guestbook | [Replace the domain](#6-replace-the-domain-entirely) |
| Not break my own agent | [Five traps](#five-traps-found-in-the-source-material) |

---

## 1. Add a tool

A tool is a decorated function. Its name, docstring and type hints become the
JSON Schema the model sees, so there is exactly one source of truth.

```diff
--- a/src/hal9000/tools/guestbook.py
+++ b/src/hal9000/tools/guestbook.py
@@
+@register_tool(tags=["guestbook"])
+def find_previous_visits(name: str, _guestbook: GuestbookStore | None = None) -> dict:
+    """Find earlier visits by this person, so you can greet a returning guest properly.
+
+    Call this before compose_greeting when the visitor may have been here before.
+
+    Args:
+        name: The visitor's name, spelled as they gave it.
+
+    Returns:
+        ``{"visits": [...], "count": N}``, or ``{"visits": [], "count": 0,
+        "note": ...}`` explaining that this is their first visit.
+    """
+    store = _store(_guestbook)
+    needle = name.strip().casefold()
+    matches = [e for e in store.entries() if e.get("name", "").casefold() == needle]
+    if not matches:
+        return {"visits": [], "count": 0,
+                "note": f"No earlier visit by {name!r}. Greet them as a first-time visitor."}
+    return {"visits": matches, "count": len(matches)}
```

It is registered the moment the module is imported, and `tools/__init__.py`
already imports every tool module. Because the tag is `guestbook`, which is in
`DEFAULT_TAGS`, the greeter now has six tools. A new tag needs one more line:

```diff
--- a/src/hal9000/agents/greeter.py
+++ b/src/hal9000/agents/greeter.py
-DEFAULT_TAGS: tuple[str, ...] = ("greeting", "guestbook", "system")
+DEFAULT_TAGS: tuple[str, ...] = ("greeting", "guestbook", "system", "badges")
```

...and if the tool should be optional, gate the tag on a `Features` field
instead, the way `dynamic_experts` gates `prompts`.

**The docstring is a prompt.** It is the only thing the model reads about your
tool. Say what the tool does, when to call it, and what comes back. "Call this
before compose_greeting" is instruction the model will follow; "Finds visits"
is not.

**Errors are prompts too.** Every failure string in this template tells the
agent what to do next — `_EMPTY_BOOK_INSTRUCTION` says "this is not a failure,
do not call again, carry on and sign". Copy that habit. Two styles, chosen per
situation:

- **Invalid arguments raise.** The environment turns the exception into
  `{"tool_executed": False, "error": ..., "traceback": ...}` and the agent
  reads it and retries.
- **A legitimate empty state returns data** carrying an instruction. Emptiness
  is not a malfunction.

**Injected parameters.** Anything the model should not supply goes in as
`action_context`, `action_agent`, or `_name` — all three are stripped from the
schema. Add the resource in `greeter_context_props()` and the tool receives it:

```diff
 props: dict = {"guestbook": guestbook if guestbook is not None else GuestbookStore()}
+props["badge_service"] = BadgeService(api_key=os.environ["BADGE_API_KEY"])
```

A tool declaring `_badge_service` now gets it, and the model never learns it
exists.

**Do not add `write_file`.** See [trap 1](#trap-1-there-is-deliberately-no-write_file).

**Tests.** Tools are ordinary functions — call them directly. Where a tool calls
the LLM, pass a `FakeLLM` through a hand-made `ActionContext`:

```python
ctx = ActionContext({"llm": FakeLLM(['{"visits": []}']), "guestbook": store})
```

---

## 2. Add a capability

A capability changes how the loop *runs* — seed memory, rewrite every prompt,
veto an action, stop early. It is the alternative to growing an `if` in
`core/agent.py` per feature. Override only the hooks you care about; the other
nine pass their value through.

```diff
--- /dev/null
+++ b/src/hal9000/capabilities/budget.py
+"""Stop the run once it has spent more iterations than the task is worth."""
+
+from hal9000.core.action_context import ActionContext
+from hal9000.core.capabilities import Capability
+from hal9000.core.prompt import Prompt
+
+
+class BudgetCapability(Capability):
+    """Warn the model as the budget runs down, and halt when it is gone."""
+
+    def __init__(self, max_actions: int = 6):
+        super().__init__(name="Budget", description="Caps the number of actions per run.")
+        self.max_actions = max_actions
+        self.used = 0
+
+    def init(self, agent, action_context: ActionContext) -> None:
+        self.used = 0                      # instances are reused across runs; reset
+
+    def process_prompt(self, agent, action_context: ActionContext, prompt: Prompt) -> Prompt:
+        left = self.max_actions - self.used
+        note = {"role": "system", "content": f"You have {left} tool calls left. Finish within them."}
+        return Prompt(
+            messages=[note] + list(prompt.messages),
+            tools=prompt.tools,             # <- see trap 2. Dropping this disarms the agent.
+            metadata=prompt.metadata,
+        )
+
+    def end_agent_loop(self, agent, action_context: ActionContext) -> None:
+        self.used += 1
+
+    def should_terminate(self, agent, action_context: ActionContext, response: str) -> bool:
+        return self.used >= self.max_actions
```

Wire it to a flag — two lines in `config.py`, two in `greeter.py`:

```diff
--- a/src/hal9000/config.py
+++ b/src/hal9000/config.py
     dynamic_experts: bool = False  # create_and_consult_expert
+    budget: bool = False           # BudgetCapability: cap actions per run
```
```diff
--- a/src/hal9000/agents/greeter.py
+++ b/src/hal9000/agents/greeter.py
     if features.verbose_logging:
         capabilities.append(LoggingCapability())
+    if features.budget:
+        capabilities.insert(0, BudgetCapability())   # before Logging: see below
     return capabilities
```

`Features.from_env` and the CLI pick the new flag up automatically — both
iterate the dataclass fields — but `__main__.FLAG_HELP` needs a matching entry
or `--help` construction fails. That coupling is deliberate: a flag with no
documented cost is not a finished flag.

**Order is the contract.** `process_*` hooks compose left to right, so the last
capability decides what the loop actually gets. `LoggingCapability` goes last so
it records the prompt the model really received. Anything that *rewrites* should
go before it.

**Capabilities are stateful.** `build_greeter_capabilities()` constructs fresh
instances per agent because all four count iterations on themselves. If yours
holds state, reset it in `init`.

**Halting.** Return `False` from `start_agent_loop` (checked with `all()`), or
`True` from `should_terminate` (checked with `any()`). Do not expect accumulator
semantics — see [trap 5](#trap-5-the-notes-reduce-in-start_agent_loop-never-starts).

---

## 3. Add a sub-agent

A sub-agent is not a new kind of thing: it is a tool whose implementation is
another agent loop. Nothing in `core/` changes.

```diff
--- /dev/null
+++ b/src/hal9000/agents/badger.py
+BADGE_AGENT_NAME = "badge_agent"
+
+def build_badge_agent(generate_response=None):
+    if generate_response is None:
+        from hal9000.llm import generate_response as generate_response_default
+        generate_response = generate_response_default
+    return Agent(
+        goals=[Goal(priority=1, name="persona", description=_PERSONA),
+               Goal(priority=2, name="method", description=_METHOD)],
+        agent_language=AgentFunctionCallingActionLanguage(),
+        action_registry=PythonActionRegistry(tool_names=["issue_badge", "terminate"]),
+        generate_response=generate_response,
+        environment=PythonEnvironment(),
+        max_iterations=3,           # a sub-agent that needs ten turns misunderstood the task
+    )
```
```diff
--- a/src/hal9000/agents/greeter.py
+++ b/src/hal9000/agents/greeter.py
     if features.delegation:
         registry = AgentRegistry()
         registry.register_agent(TRANSLATOR_AGENT_NAME, build_translator_agent(generate_response).run)
+        registry.register_agent(BADGE_AGENT_NAME, build_badge_agent(generate_response).run)
         props["agent_registry"] = registry
```

Register the **bound `run` method**, not the `Agent` — that is the whole callee
contract, and sharing `generate_response` means one `FakeLLM` covers both
agents. Then tell the caller it exists, in a goal; a tool the goals never
mention is a tool the model will not reach for.

Three rules worth keeping:

- **Give a sub-agent the smallest toolkit that can do its job.** The translator
  gets `terminate` and nothing else. It is the cheapest reliability measure
  available.
- **The registry never travels.** `NOT_INHERITED` in `tools/delegation.py`
  strips `agent_registry` (plus `memory`, `action_registry` and the LLM keys)
  from what a callee inherits. A callee with no directory cannot delegate, so
  two agents cannot call each other forever. Everything else *is* forwarded,
  which is how an auth token reaches a sub-agent.
- **A sub-agent failure is news, not death.** `_delegate` catches it and returns
  `{"success": False, "error": ...}` for the caller to read.

### The other three delegation tools

`Features.delegation` gives the greeter **only `call_agent`** (ruling R15),
because its one honest sub-task — "translate this greeting" — is message
passing: send a self-contained request, get an answer back. All four patterns
from the notes are implemented, tagged and tested; the other three are actively
worse *for the greeter*, and four delegation tools bolted onto a five-tool agent
doubles the choice the model must get right to do one job.

| Tool | Memory down | Memory back | Use it when |
|---|---|---|---|
| `call_agent` | fresh | last item only | You want the answer and not the working. An email. |
| `call_agent_with_reflection` | fresh | *everything*, tagged `{agent}_thought` | The reasoning matters to you. Costs a longer memory on every turn afterwards. |
| `hand_off_to_agent` | **the caller's own `Memory` object** | written in place | A specialist is taking over, not being consulted. |
| `call_agent_with_selected_context` | an LLM picks the relevant items | the callee's items plus the selection reasoning | History is long, or some of it is none of the callee's business. Costs one extra powerful-tier call. |

One line gives you all four:

```python
PythonActionRegistry(tags=["delegation"])
```

Or, inside the greeter's builder, extend the by-name list:

```diff
-DELEGATION_TOOL = "call_agent"
+DELEGATION_TOOLS = ["call_agent", "call_agent_with_reflection", "hand_off_to_agent"]
```

Whichever you pick, say in the goals *which* one to use when. A model handed
four near-synonymous tools with no guidance will pick by name similarity.

---

## 4. Swap the agent language

The default is provider function calling. The alternative asks for a fenced
```` ```action ```` block, which puts the model's reasoning into memory before
the decision — often what you want while debugging *why* an agent chose what it
chose, and necessary on any model without tool-calling support.

```diff
--- a/src/hal9000/agents/greeter.py
+++ b/src/hal9000/agents/greeter.py
-from hal9000.core.agent_language import AgentFunctionCallingActionLanguage
+from hal9000.core.agent_language import AgentJsonActionLanguage
@@
-        agent_language=AgentFunctionCallingActionLanguage(),
+        agent_language=AgentJsonActionLanguage(),
```

That is the entire change; the loop, tools and memory are untouched. What
changes underneath:

|  | Function calling | JSON block |
|---|---|---|
| `Prompt.tools` | populated with JSON Schemas | `None` — tools are described in a system message |
| Reply | structured by the provider | prose, then a fenced block |
| Unparseable | falls back to `terminate` with the prose as the message | **raises `ValueError`** |

That last row is the one to plan for. The JSON language refuses to guess, on
purpose: a missing block means the model ignored the required format, and
silently terminating would hide it. Budget an extra iteration or two for
reformatting, and note that scripted `FakeLLM` responses must now be fenced
blocks rather than bare JSON.

Writing a third language means subclassing `AgentLanguage` and implementing
`construct_prompt` and `parse_response`. `format_goals` and `format_memory` are
inherited and usually fine.

---

## 5. Replace guestbook storage with a database

`storage/guestbook.py` is the only module that knows the data is a JSON file.
The tools use five methods — `append`, `read`, `entries`, `remove_entry`,
`remove_last` — so any object with those works.

```diff
--- /dev/null
+++ b/src/hal9000/storage/sqlite_guestbook.py
+"""A SQLite guestbook. Same five methods; nothing above this file changes."""
+
+import sqlite3, threading, time
+
+
+class SqliteGuestbookStore:
+    def __init__(self, path: str = "guestbook.db"):
+        self._lock = threading.Lock()
+        self._db = sqlite3.connect(path, check_same_thread=False)
+        self._db.row_factory = sqlite3.Row
+        self._db.execute(
+            "CREATE TABLE IF NOT EXISTS entries ("
+            " entry_id INTEGER PRIMARY KEY AUTOINCREMENT,"   # never reused, as before
+            " signed_at TEXT, name TEXT, greeting TEXT, note TEXT)"
+        )
+
+    def append(self, entry: dict) -> dict:
+        with self._lock, self._db:
+            cur = self._db.execute(
+                "INSERT INTO entries (signed_at, name, greeting, note) VALUES (?,?,?,?)",
+                (time.strftime("%Y-%m-%dT%H:%M:%S"), entry["name"],
+                 entry.get("greeting", ""), entry.get("note", "")),
+            )
+        return self.entries_by_id(cur.lastrowid)
+
+    def entries(self, limit: int | None = None) -> list[dict]:
+        sql = "SELECT * FROM entries ORDER BY entry_id"
+        rows = self._db.execute(sql).fetchall()
+        return [dict(r) for r in (rows[-limit:] if limit else rows)]
+
+    def read(self) -> dict:
+        rows = self.entries()
+        return {"entries": rows, "next_entry_id": (rows[-1]["entry_id"] + 1) if rows else 1}
+
+    def remove_entry(self, entry_id: int) -> dict | None:
+        row = self.entries_by_id(entry_id)
+        with self._lock, self._db:
+            self._db.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
+        return row
+
+    def remove_last(self) -> dict | None:
+        rows = self.entries()
+        return self.remove_entry(rows[-1]["entry_id"]) if rows else None
```

Swap it in at the one place the store is constructed:

```diff
--- a/src/hal9000/agents/greeter.py
+++ b/src/hal9000/agents/greeter.py
-    props: dict = {"guestbook": guestbook if guestbook is not None else GuestbookStore()}
+    props: dict = {"guestbook": guestbook if guestbook is not None else SqliteGuestbookStore()}
```

Four things to carry over, because they are load-bearing rather than decorative:

1. **Entry ids are never reused.** Undoing entry 3 means the next signing is 4.
   `AUTOINCREMENT` gives you that; `MAX(entry_id)+1` does not.
2. **`append` returns the stored row, including `entry_id`.** The reverser in
   `safety/reversible.py` reads `record["result"]["entry_id"]` to undo a
   signing. Break that return shape and rollback breaks silently.
3. **Serialise writes.** The JSON store holds a per-path lock and replaces the
   file atomically. Keep the equivalent.
4. **Refuse untrusted paths.** `GuestbookStore` confines the file to a root
   directory and raises `GuestbookPathError` on traversal. If a path in your
   version can ever come from a model or a user, confine it the same way. In the
   template it cannot — which is itself the point of
   [trap 1](#trap-1-there-is-deliberately-no-write_file).

For a fully swappable backend, add a `Protocol` with those five methods and type
`_guestbook` against it. The template does not, because one implementation and
an injected object is the smaller thing that works.

---

## 6. Replace the domain entirely

**The guestbook is an example, not a commitment.** It exists because it is
stateful, needs no external service, and makes reversibility mean something. If
you are building a ticket triager, nothing about greeting visitors should
survive.

Here is the whole change, as a checklist. Nothing in `core/` appears in it.

**Delete:** `tools/greeting.py`, `tools/guestbook.py`, `storage/guestbook.py`,
`agents/translator.py`, `config/greeting_policy.txt`, and their tests.

**Keep, untouched:** all of `core/`, `tools/decorators.py`,
`tools/self_prompting.py`, `tools/planning.py`, `tools/delegation.py`,
`tools/system.py`, all of `capabilities/`, `safety/reversible.py`,
`safety/review.py`, `safety/staged.py`, `config.py`, `llm.py`,
`tests/fakes.py`.

**Write:** your storage module, your tools, your goals.

```diff
--- a/src/hal9000/agents/greeter.py
+++ b/src/hal9000/agents/triager.py
-_MISSION = """You are the greeter at the front desk of this building. ..."""
+_MISSION = """\
+You are the first responder on a support queue. A ticket has just arrived. It is
+handled when it has been classified, an owning team has been assigned, and the
+decision is recorded in the tracker -- all three, or none of them."""

-_METHOD = """Use your tools in this order: 1. extract_visitor_details ..."""
+_METHOD = """\
+1. classify_ticket -- pass the ticket text verbatim.
+2. choose_owner -- pass the classification, whole and unchanged.
+3. record_triage -- write the decision. This is the only tool that writes.
+4. terminate -- deliver the result exactly as your final goal describes."""

-DEFAULT_TAGS = ("greeting", "guestbook", "system")
+DEFAULT_TAGS = ("triage", "tracker", "system")

-def greeter_goals(features=None): ...
+def triager_goals(features=None): ...
```

Then update the three references to the builder — `__main__.py`, `agents/__init__.py`,
`tests/test_e2e_fake.py` — plus `FLAG_HELP`, `.env.example`, and this README.

What you inherit for free, and would otherwise spend a week rebuilding: the
loop, memory, the tool decorator and registry, dependency injection with schema
hiding, the ten capability hooks and all four capabilities, both agent
languages, tiered models, delegation with four memory-sharing patterns, staged
review with two reviewers, reversible actions, transactions, a key-free test
harness, and CI.

Two things to preserve as you go:

- **Keep one tool that writes.** Whatever your domain, the single-safe-tool
  pattern is what made staging, reversal and validation cheap here.
- **Keep the default lean.** Start with the tools your happy path needs and
  nothing else. Every tool you add is a choice the model can get wrong.

---

# Five traps found in the source material

This template implements a corpus of course notes. Building it turned up five
places where following the notes literally produces a broken agent — and four of
them fail *silently*. They are recorded in full in
[`ORCHESTRATOR_RULINGS.md`](ORCHESTRATOR_RULINGS.md); they are repeated here
because this is the file you will read while writing the code that could hit
them.

## Trap 1: there is deliberately no `write_file`

The most common shape for a first agent is a `write_file(path, content)` tool.
It looks general and reusable. It is neither.

`write_file` gives the model three ways to be wrong — the wrong path, the wrong
format, a clobbered file — and gives you no way to check any of them. It also
forces the *model* to know your storage format, which means the format is now
part of the prompt and drifts whenever the model improvises.

`sign_guestbook` is the alternative: **action specificity** (MATE's A) and the
**single safe tool** pattern in one function. It takes a name, a greeting and an
optional note. The path comes from configuration, the format from the storage
module, the ordering from a lock, and every one of the three arguments is
validated with an error message that tells the agent how to fix it.

The payoff compounds, and this is the part that is easy to miss:

- **Staging is a one-item list.** `DEFAULT_STAGED_TOOLS = ("sign_guestbook",)`,
  because there is exactly one tool that writes.
- **Reversal is one registered function.** `DEFAULT_REVERSERS` has one entry.
- **The path is never model-supplied**, so a traversal check is a belt on
  braces rather than the only defence.
- **The review prompt is readable.** A human approving "sign the guestbook for
  Dr. Chen" can actually judge it. Nobody can judge a 4 KB `write_file` payload.

Generic tools push the cost onto every safety feature you later want. Specific
tools pay for themselves the first time you need one.

## Trap 2: a `process_prompt` hook must preserve every `Prompt` field

**Ruling R9. The easiest way to break your own agent.**

`capability_architecture_pattern.md` ends its `process_prompt` with:

```python
return Prompt(messages=messages)          # WRONG
```

`Prompt` carries `messages`, `tools` **and** `metadata`. Rebuilding it from
`messages` alone **discards every tool definition**. Under the function-calling
language that means: the moment you enable that capability, the model is offered
no tools at all. It is handed a task it has no way to act on, answers in prose,
and `parse_response` turns the prose into `terminate`. The agent does not crash.
It just stops doing anything, politely.

```python
def process_prompt(self, agent, action_context, prompt):
    messages = list(prompt.messages)                    # copy; do not mutate the caller's
    messages[0] = {**messages[0], "content": header + messages[0]["content"]}
    return Prompt(
        messages=messages,
        tools=prompt.tools,          # <- both of these
        metadata=prompt.metadata,    # <-
    )
```

**Rule: a `process_prompt` hook preserves every field it did not deliberately
change.** The note's version also mutates `messages[0]` in place, which
corrupts the prompt for the next capability in the chain and for any test
holding the original. Copy, then modify the copy. `TimeAwareCapability` in
`capabilities/time_aware.py` is the reference.

## Trap 3: never write `"approved" in response.lower()`

**Ruling R14. The most consequential error in the corpus.**

`environment_safety.md` decides whether an LLM reviewer approved a staged action
with one line:

```python
return "approved" in response.lower()      # WRONG
```

Measured against replies a real reviewer produces:

| Reviewer said | That line reads it as |
|---|---|
| `"approved"` | approved |
| `"This is not approved - unsafe"` | **approved** |
| `"I would reject this"` | not approved |
| `"Do not approve"` | not approved |

It fails **open**, in the one component whose entire job is to fail closed.

`safety/review.py` checks rejection words first and treats anything ambiguous,
empty or unparseable as a refusal:

```python
def _reads_as_approval(response: str) -> bool:
    text = (response or "").strip().lower()
    if not text:
        return False
    if any(w in text for w in ("not approved", "reject", "deny", "unsafe", "do not")):
        return False
    return "approve" in text or text.startswith("yes")
```

The same principle governs `HumanReviewer`: no tty, a cancelled prompt, or any
answer other than `y`/`yes` — **including an empty line** — is a refusal. That
is why `--staged-review` refuses every write in CI: there is nobody to ask, so
the answer is no.

**Writing your own reviewer?** It is a callable taking a `ReviewRequest` and
returning a `ReviewDecision` — a rule engine, a Slack approval, a second agent.
Whatever it is, make the *default* branch a refusal, and make substring matching
your last resort rather than your parser.

## Trap 4: transactions here are synchronous

**Ruling R13.** `environment_safety.md` declares `async def execute` /
`async def rollback`, and `rollback` does `await action.undo()` — but
`ReversibleAction.undo` is declared `def undo(self)` thirty lines earlier in the
same file. Copied literally it raises
`TypeError: object NoneType can't be used in 'await' expression` on the first
rollback, which is the worst possible moment to discover a type error.

`safety/reversible.py` is synchronous throughout. `Agent.run` and
`Environment.execute_action` are sync, so an async transaction would force
`asyncio.run()` inside a sync method — which raises outright if a caller ever
embeds the agent in an existing event loop.

If you are porting the note's code: drop the `async`/`await`. If you genuinely
need async execution, make it async all the way up — `Environment`,
`Agent.run`, and the delegation tools that call sub-agents — rather than at one
layer.

What `ActionTransaction` guarantees, and what to keep if you rewrite it: on a
mid-sequence failure, **everything already executed is reversed in reverse
order** and the error propagates. A tool with no registered reverser gets one
that raises, loudly, rather than a no-op — a rollback that silently skips an
irreversible step is worse than no rollback, because it reports success.

## Trap 5: the note's `reduce` in `start_agent_loop` never starts

**Ruling R5.** `capability_architecture_pattern.md`:

```python
can_start_loop = reduce(lambda a, c: c.start_agent_loop(self, action_context),
                        self.capabilities, False)     # WRONG
```

The lambda discards its accumulator `a`, so the expression returns the *last*
capability's answer — and with an **empty** capability list it returns the seed,
`False`. The default agent, which is the headline configuration, would never run
a single iteration.

`core/agent.py` uses a veto instead, which is vacuously `True` for an empty
list and honours the hook's own documented contract:

```python
if not all([c.start_agent_loop(self, action_context) for c in self.capabilities]):
    break
```

The list comprehension is not incidental: it collects every vote before
deciding, so no capability is short-circuited. `should_terminate` uses `any()`
the same way. **When you write a capability that wants to halt a run, return
`False` from `start_agent_loop` or `True` from `should_terminate`, and do not
expect accumulator semantics.**

### A sixth, smaller one: unwrap the delegation envelope

**Ruling R16.** The multi-agent notes all end at
`result_memory.items[-1]["content"]`. In this framework that last memory item is
`json.dumps({"tool_executed": true, "result": ..., "timestamp": ...})`, because
`Agent.update_memory` writes tool results back into memory as JSON. Returned
literally, the calling agent receives a description of the framework's own
plumbing instead of the sub-agent's answer. `_last_content` in
`tools/delegation.py` opens exactly that envelope and passes anything else
through untouched, so `call_agent(...)["result"]` is the answer.

---

## Before you ship your version

```bash
uv run pytest            # no key, no network — keep it that way
uv run ruff check . && uv run ruff format --check .
```

- Every new tool has a docstring good enough to be its own model-facing
  description, and every error message tells the agent what to do next.
- Every new flag has an entry in `FLAG_HELP` stating **what it costs**.
- Every side-effecting tool has a reverser in `DEFAULT_REVERSERS`, or is kept
  out of transactions.
- The default run is still the smallest thing that does the job.
