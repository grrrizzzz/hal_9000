# ADR 0001 — One provider layer: LiteLLM

**Status:** Accepted
**Date:** 2026-09
**Deciders:** the build's architecture pass, per BUILD_PLAN §1.1 ("decisions
already made; do not revisit")

## Context

The template must reach at least two model providers — Anthropic and OpenAI —
because a reader arriving with either key should be running in two minutes, not
reading a porting guide. At the same time, DoD #1 requires the whole test suite
to pass with **no API keys and no network**, and DoD #9 caps files at ~200 lines
because this is a teaching artifact.

Those three pull against each other. Provider support is exactly the kind of
requirement that grows a `providers/` package, a factory, an abstract base
class and a mocking layer — and a reader who came to learn the agent loop then
has to read all of that first.

## Decision

**One runtime dependency, `litellm`, called from exactly one file,
`src/hello_agent/llm.py`, through exactly one function.**

```python
def generate_response(prompt: Prompt, tier: str = "powerful") -> str: ...
```

Providers are selected by *model string* — `anthropic/claude-opus-5`,
`openai/gpt-4o` — resolved in `config.py` from a `{provider: {tier: model}}`
table plus two override environment variables. There is no provider class, no
adapter interface, and no factory.

Two consequences fall out of that and are load-bearing:

1. **`generate_response` returns a `str`.** Tool calls are normalised to
   `{"tool": name, "args": {...}}` inside `llm.py`; plain replies pass through as
   text. Provider response objects appear nowhere else in the codebase.
2. **Adding a provider is a table entry.** Two lines in `config.MODELS`, for
   anything LiteLLM already speaks.

This is also the course-faithful choice: the source notes use LiteLLM's
`completion()` directly, so a reader moving between the notes and this repo sees
the same call.

## Alternatives considered

| Option | Why not |
|---|---|
| **Each provider's own SDK, behind our own interface** | Two SDKs, two response shapes, and an abstraction we would have to design, document and test — for a template whose subject is the agent loop, not HTTP clients. It also puts the provider difference in *our* code, so drift becomes our maintenance. |
| **LangChain / an agent framework** | Would supply the very thing this repository exists to demonstrate. The reader would learn the framework's opinions instead of the GAME framework's mechanics, and the dependency tree is orders of magnitude larger than one package. |
| **OpenAI SDK against an OpenAI-compatible proxy** | One SDK, but it makes every non-OpenAI provider someone else's infrastructure problem and requires the reader to run a proxy before they can say hello. Fails the two-minute quickstart. |
| **No abstraction — pick one provider** | Halves the audience and hard-codes a vendor into a template meant to be reused. |

## Consequences

**Good.**

- The network boundary is one function in a 60-line file. Faking it is a list of
  strings (`tests/fakes.py`), so the suite runs with no key, no network and no
  mocking library — and CI has no secrets configured at all.
- Provider drift is a one-file fix. When a provider changes its tool-call shape,
  `_tool_call_to_json` is the only thing that moves, and the live smoke tests
  (one per provider, auto-skipped without a key) are what catch it.
- Tiered models (MATE's *model efficiency*) cost nothing extra: the tier is an
  argument, the mapping is a dict. `extract_visitor_details` runs on the fast
  tier and `compose_greeting` on the powerful one, in the default run.
- The reader can ignore `llm.py` entirely and still understand the agent.

**Costs, accepted.**

- A dependency we do not control sits on the hot path. The mitigation is the
  narrowness of the contract: `Prompt -> str`. Replacing LiteLLM with raw SDK
  calls would be a rewrite of one file, and no other module would notice.
- LiteLLM is a large package with a broad surface. We use one function from it.
- A `str` return is weaker typing than a parsed object. It is what makes the
  fake trivial, and `parse_response` handles the one failure mode (prose instead
  of a tool call) by turning it into a clean `terminate`.
- **`import litellm` fetches its model-price map over the network at import
  time** — measured at four outbound connection attempts, with retries and
  backoff, degrading gracefully to a bundled copy. That is acceptable in
  production, where a current price map is worth the fetch, and unacceptable in
  a suite that claims to need no network. `LITELLM_LOCAL_MODEL_COST_MAP=True` is
  therefore set in `tests/conftest.py` (at module scope, before the first
  import) and on the CI test step — and deliberately **not** in `llm.py`.
  Ruling R8; a regression test guards it.
- The import is slow, so `llm.py` imports `litellm` inside the function rather
  than at module scope. Nothing that never calls a real model pays for it.

## Revisit if

- LiteLLM's maintenance stalls, or its response normalisation starts differing
  per provider in ways `_tool_call_to_json` cannot absorb.
- The template needs streaming, or token-level cost accounting, which would push
  real logic into `llm.py` and make the thinness of this layer a fiction.
- A target provider is not supported by LiteLLM at all.

In any of those cases the replacement is scoped to one file and one function
signature, which is the whole point of writing it down this way.
