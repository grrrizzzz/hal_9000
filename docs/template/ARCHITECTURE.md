# Architecture

How this template is put together, and why. Four ideas carry the whole design:

1. **GAME** — an agent is Goals, Actions, Memory and an Environment, and nothing
   else. Swap a component, get a different agent; the loop never changes.
2. **`generate_response` returns a string.** The keystone. It is why a complete
   LLM stand-in is a list of strings and why CI needs no API key.
3. **Dependency injection by parameter name**, stripped from the schema the
   model sees. Tools get storage handles and LLM tiers; the model never knows.
4. **Capabilities, not `if` statements.** Ten hooks into the loop, so the loop
   stays the same size forever.

---

## GAME

```
                  ┌──────────────── Agent.run ────────────────┐
   Goals ─────────┤                                           │
   Actions ───────┤   prompt → decide → execute → remember    ├──→ Memory
   Memory ────────┤                                           │
   Environment ───┤                                           │
                  └───────────────────────────────────────────┘
```

### G — Goals · `core/goals.py`, `agents/greeter.py`

```python
@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str
```

Goals are **prose**, rendered into the system message by the agent language.
That is not a limitation dressed up as a feature: the model is a reader, and the
most direct way to change what an agent does is to change what it was told.

The greeter has three (four with `delegation`), and they are separate because
they change for different reasons: `welcome_and_record` (what a handled visit
*is*), `method` (the tool order and the rules it must not break),
`greet_in_their_language` (only present when `call_agent` is in the toolkit —
an agent told to use a tool it does not have will try), and `deliver` (the
output contract: everything goes in `terminate`'s message).

**This is the file you edit to build your own agent.** `greeter_goals()` in
`agents/greeter.py`.

### A — Actions · `core/actions.py`, `core/registry.py`, `tools/decorators.py`

A tool is a decorated function. Registration is a side effect of import:

```python
@register_tool(tags=["guestbook"])
def sign_guestbook(name: str, greeting: str, note: str = "",
                   _guestbook: GuestbookStore | None = None) -> dict:
    """Record a visit in the guestbook. Call this once per visitor, ..."""
```

`@register_tool` derives everything the model needs from the function itself —
`__name__` is the tool name, `__doc__` is the description, the signature plus
type hints are the JSON Schema — so there is one source of truth and nothing to
keep in step. It writes into two module-level dicts in `core/registry.py`:
`tools` (name → metadata) and `tools_by_tag` (tag → names).

`PythonActionRegistry(tags=[...], tool_names=[...])` then slices that catalogue
into one agent's toolkit. Assembling an agent's Actions is a list of tags:

```python
PythonActionRegistry(tags=["greeting", "guestbook", "system"])   # the lean default: 5 tools
PythonActionRegistry(tool_names=["terminate"])                    # the translator: 1 tool
```

An `Action` is the name, function, description, parameter schema, and a
`terminal` flag. Exactly one tool in this template is terminal: `terminate`.

### M — Memory · `core/memory.py`

The LLM is stateless. Everything it "knows" on turn three is in the prompt,
which means memory is just a list of `{"type": ..., "content": ...}` items that
the agent language renders back into messages.

Each iteration appends two items: what the model decided (`assistant`) and what
came back (`user`, carrying the result envelope as JSON). Types matter:
`copy_without_system_memories()` is what lets a hand-off give another agent the
conversation without the first agent's private framing — plans and progress
reports are stored as `system` for exactly that reason.

### E — Environment · `core/environment.py`

The Environment owns three things the loop must not know about:

1. **Execution** — calling the function.
2. **Dependency injection** — reading the signature and supplying what it asked
   for (below).
3. **Error containment** — a tool that raises produces a *result*, not a crash:

```python
{"tool_executed": True,  "result": ..., "timestamp": ...}
{"tool_executed": False, "error": "...", "traceback": "..."}
```

Both shapes go straight back into the prompt, which is why there is no `try` in
the loop. An agent that reads "that failed, because X" can retry. An agent that
dies cannot. The traceback is included deliberately — it is the most useful
thing you can hand a model that just called a tool wrongly.

Because execution is a component, changing *how* actions happen is a swap, not a
rewrite: `StagedActionEnvironment` (`safety/staged.py`) holds `sign_guestbook`
for review and executes it only on approval. Goals, tools, memory and loop are
untouched.

---

## The loop, and where capabilities hook in

`core/agent.py`, `Agent.run`. One iteration in full:

```
construct_prompt    goals + tools + memory   ->  Prompt
generate_response   Prompt                   ->  str
parse_response      str                      ->  {"tool": ..., "args": ...}
execute_action      Action + args            ->  result dict
update_memory       response + result        ->  two memory items
```

With the hooks shown:

```
init                                  once, before the first iteration
┌─ start_agent_loop                   top of each iteration; any False stops the run
│  construct_prompt
│    process_prompt                   prompt built, not yet sent
│  generate_response
│    process_response                 raw text back from the LLM
│  parse_response
│    process_action                   parsed {"tool", "args"} — may be redirected
│  execute_action
│    process_result                   action executed
│  update_memory
│    process_new_memories             memories drafted, not yet stored
│  end_agent_loop                     bottom of each iteration
└─ should_terminate                   any True stops the run
terminate                             once, after the loop, however it ended
```

The loop also stops on a terminal action, on `max_iterations` (default 10), and
on `max_duration_seconds` (default 180).

**All ten hooks fire on every run, even though the default capability list is
empty.** That is the point: `capabilities=[TimeAwareCapability()]` is a genuine
one-line change, not an integration.

The five `process_*` hooks compose left to right via `functools.reduce`: with
`capabilities=[A, B]` the loop uses `B(A(value))`. Order in the list is
therefore meaningful — which is why `build_greeter_capabilities()` always puts
`LoggingCapability` last, so it logs the prompt the model actually receives,
after `TimeAwareCapability` has injected the clock into it.

`start_agent_loop` and `should_terminate` are **votes, not folds**:
`all([...])` and `any([...])` over the list, collected before deciding, so no
capability is short-circuited. The source note implements `start_agent_loop`
with a `reduce` whose lambda discards its accumulator — which returns the seed
`False` for an empty capability list, meaning the lean-default agent would never
run a single iteration. That is ruling R5, the one place in this build where a
note was overruled rather than followed.

The four shipped capabilities live in `capabilities/` and are one flag each;
`docs/EXTENDING.md` shows how to write one.

---

## Dependency injection

The problem: `sign_guestbook` needs a storage handle. Three bad answers —
import the agent (welds the tool to one agent), make it a parameter (the model
has to invent a file handle), or use a global (untestable).

The answer here is a per-run bag of resources, `ActionContext`, injected by
`PythonEnvironment` into exactly the tools that ask for them, **by parameter
name**:

| Parameter declared | Receives |
|---|---|
| `action_context` | the `ActionContext` itself |
| `action_agent` | the running `Agent` |
| `_foo` | `action_context.get("foo")` |

```python
def sign_guestbook(name: str, greeting: str, note: str = "", _guestbook=None): ...
#                  └──── the model supplies these ────┘         └── injected ──┘
```

And the half that makes it work: **`@register_tool` strips all three shapes from
the JSON Schema.** The model never sees `_guestbook`, never has to supply it,
and cannot forge one. A tool that declares none of them is called with exactly
the arguments the model chose.

```python
registry.tools["sign_guestbook"]["parameters"]
# {"type": "object",
#  "properties": {"name": {...}, "greeting": {...}, "note": {...}},
#  "required": ["name", "greeting"]}
```

What is in the `ActionContext` on every run (`core/action_context.py`):

| Key | Value |
|---|---|
| `memory` | the run's `Memory` |
| `llm` | `generate_response`, powerful tier. **Canonical** — every tool reads this |
| `powerful_llm` | an alias bound to the identical callable (ruling R1) |
| `fast_llm` | `generate_response` pinned to the fast tier — MATE's model efficiency |
| `action_registry` | the agent's `ActionRegistry` |
| `agent_registry` | an `AgentRegistry` — **only** when delegation is on |
| *anything else* | whatever you pass to `Agent.run(action_context_props=...)` |

`agent_registry`'s absence is load-bearing. A sub-agent that cannot see the
registry cannot delegate, so two agents cannot call each other forever — and
that same mechanism is why delegation is off by default: the key simply is not
there. `tools/delegation.py` also refuses to forward it downward
(`NOT_INHERITED`).

---

## Why `generate_response` returns a string

```python
def generate_response(prompt: Prompt, tier: str = "powerful") -> str: ...
```

Everything hangs off this one signature. When the prompt carries tools and the
model calls one, the returned string is `{"tool": name, "args": {...}}`;
otherwise it is the model's plain text. Provider objects — `tool_calls`,
`message.content`, argument strings that may or may not be pre-parsed — are
normalised inside `llm.py` and **appear nowhere else in the codebase**.

Three consequences, and they are the reason the design is shaped this way.

**1. A complete LLM is a list of strings.** `tests/fakes.py` is fifty lines with
no mocking library, no HTTP interception, and no recorded cassettes:

```python
llm = FakeLLM([
    tool_call("extract_visitor_details", document_text=VISITOR_TEXT),
    json.dumps(VISITOR),                                  # the tool's own inner call
    tool_call("compose_greeting", visitor=VISITOR),
    GREETING,                                             # the tool's own inner call
    json.dumps({"tool": "sign_guestbook", "args": {...}}),
    tool_call("terminate", message=FINAL),
])
main([VISITOR_TEXT], generate_response=llm)
```

Because `FakeLLM` also records every `Prompt` it saw, tests assert on prompt
*construction* — that the policy text reached the model, that extraction ran on
the fast tier, that a capability preserved `prompt.tools`. Those are the
assertions that catch real regressions, and they are only possible because the
boundary is this narrow.

**2. CI needs no secrets.** The workflow runs `ruff` and `pytest` on 3.11/3.12/
3.13 with no keys configured and no network reachable. A test suite that needs a
key is a test suite people skip.

**3. Provider drift is a one-file fix.** When LiteLLM's response shape changes
for one provider, `_tool_call_to_json` is the only thing that has to move. See
[`ADR/0001-litellm.md`](ADR/0001-litellm.md).

The cost is honest: a JSON string is weaker typing than a parsed object, and the
function-calling language's `parse_response` has to `json.loads` and fall back
to `terminate` when the model answers with prose instead of a tool call. That
fallback is not a failure mode — a model that stops calling tools has decided it
is finished, and this turns that into a clean stop rather than an error.

---

## Agent language

`core/agent_language.py` decides how goals, tools and memory become a `Prompt`,
and how the reply becomes `{"tool": ..., "args": ...}`. Two ship:

| | `AgentFunctionCallingActionLanguage` (default) | `AgentJsonActionLanguage` |
|---|---|---|
| Tools | in the API's `tools` field, as JSON Schema | described in prose in a system message |
| Reply | already structured by the provider | a ```` ```action ```` fenced JSON block |
| Unparseable reply | falls back to `terminate` with the prose as the message | raises — a missing block means the model ignored the format, and hiding that helps nobody |
| Why you would use it | provider validates arguments; nothing to parse | the model thinks out loud *before* the block, so its reasoning lands in memory — valuable when debugging why an agent chose what it chose |

Swapping is one constructor argument; `docs/EXTENDING.md` has the diff.

---

## Separation of reasoning and execution

Worth naming explicitly, because it is a property of the whole structure rather
than of any one file. The model decides; it never acts. Every effect in this
template passes through an `Action`, executed by an `Environment`, which
validates, injects, contains errors and — optionally — asks permission first.

That is what makes the safety layer cheap. Staged review is an environment
swap. Reversibility is a registered reverser. Rollback is a transaction around
the same `Action` objects. None of it required the model to cooperate, and none
of it touched the loop.

The other half of the same idea is **action specificity** (MATE's A): there is
deliberately no `write_file` here. `sign_guestbook` is the single tool that
writes, and it owns validation, entry numbering, the storage path and the
atomic write. One tool to stage, one reversal to register, one place for a
traversal check. `docs/EXTENDING.md` explains why in full — it is the design
decision most worth copying into your own agent.
