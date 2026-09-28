# Corpus coverage

This template is a reference implementation of a specific body of material: the
35 notes of the Coursera *AI Agent Developer* specialization, collected in
`coursera_ai_agent_content`. Nothing in those notes is executable — every
pattern there is an illustrative fragment with implied-but-absent surrounding
code. This repository is the surrounding code.

The table below is [`BUILD_PLAN.md`](BUILD_PLAN.md) §3, filled in against what
was actually built. Where the plan and the code disagree, **the code is right
and the plan is stale**; every such correction is called out under
[Corrections](#corrections-to-the-plan).

**Default run?** — `default` means the pattern is exercised by
`python -m hal9000 "..."` with no flags. `flag` means it ships complete and
tested but switched off. That split is the template's central design decision;
[the README](../README.md#the-lean-default-is-a-feature-not-an-omission)
explains why.

Every implementation path in this document is verified to exist — see
[Verification](#verification) for the check and how to re-run it.

---

## Pattern → implementation

| Pattern | Source note(s) | Implemented in | Default run? |
|---|---|---|---|
| Agent loop: prompt / parse / execute / remember | `agentic_ai_concepts/first_agent.md`, `agentic_ai_concepts/agent_env_interface.md`, `agentic_ai_concepts/agent_feedback_and_memory.md` | `src/hal9000/core/agent.py` | default |
| Stateless LLM; memory as messages | `agentic_ai_concepts/giving_agents_memory.md` | `src/hal9000/core/memory.py` | default |
| Programmatic prompting, ChatML roles | `agentic_ai_concepts/programmatic_prompting_2.md`, `agentic_ai_concepts/programmatic_prompting_3.md` | `src/hal9000/llm.py`, `src/hal9000/core/prompt.py` | default |
| Function calling | `agent_tools_actions_language/function_calling.md`, `agent_tools_actions_language/agent_loop_with_function_calling.md` | `src/hal9000/core/agent_language.py`, `src/hal9000/llm.py` | default |
| JSON-Schema tool metadata, tool naming | `agent_tools_actions_language/agent_tools.md`, `agent_tools_actions_language/tool_design_best_practices.md` | `src/hal9000/tools/decorators.py` | default |
| Instructions injected into error messages | `agent_tools_actions_language/tool_design_best_practices.md` | `src/hal9000/tools/guestbook.py`, `src/hal9000/tools/greeting.py` | default |
| `@register_tool` decorator + introspection | `agent_tool_management/python_decorators.md` | `src/hal9000/tools/decorators.py` | default |
| Tag-based tool organization + registries | `agent_tool_management/tool_organization.md`, `agent_tool_management/refactored_readme_agent.md` | `src/hal9000/core/registry.py`, `src/hal9000/core/actions.py` | default |
| GAME: Goal / Action / Memory / Environment | `game_framework/modular_agent_design.md`, `game_framework/game_framework.md` | `src/hal9000/core/goals.py`, `src/hal9000/core/actions.py`, `src/hal9000/core/memory.py`, `src/hal9000/core/environment.py` | default |
| Agent class + swappable components | `game_framework/agent_loop_customization.md`, `game_framework/game_in_code.md` | `src/hal9000/core/agent.py` | default |
| Assembling an agent from the components | `game_framework/example_readme_agent.md` | `src/hal9000/agents/greeter.py`, `src/hal9000/agents/translator.py` | default |
| AgentLanguage: function-calling **and** JSON-block | `game_framework/agent_language.md` | `src/hal9000/core/agent_language.py` | default (function calling); JSON-block shipped and tested |
| `ActionContext` dependency injection | `dependency_injection_for_tools/clean_tools_with_dep_injection.md` | `src/hal9000/core/action_context.py` | default |
| Env-level DI, `_`-prefix params, schema hiding | `dependency_injection_for_tools/dep_injection_with_environment.md` | `src/hal9000/core/environment.py`, `src/hal9000/tools/decorators.py` | default |
| `prompt_llm_for_json` + retry + fence stripping | `extending_with_self_prompting/structured_data_extraction.md` | `src/hal9000/tools/self_prompting.py` | default (called inside `extract_visitor_details`) |
| Specialized fixed-schema extractor | `extending_with_self_prompting/structured_data_extraction.md`, `extending_with_self_prompting/invoice_processing_agent.md` | `src/hal9000/tools/greeting.py` | default |
| `prompt_expert` / persona pattern | `extending_with_self_prompting/persona_pattern.md`, `extending_with_self_prompting/persona_abstraction.md` | `src/hal9000/tools/self_prompting.py`, `src/hal9000/tools/greeting.py` | default (called inside `compose_greeting`) |
| Dynamic persona creation | `extending_with_self_prompting/persona_abstraction.md` | `src/hal9000/tools/self_prompting.py` — `create_and_consult_expert` | flag: `dynamic_experts` |
| Policy-as-prompt, read from disk at call time | `extending_with_self_prompting/invoice_processing_with_experts.md` | `config/greeting_policy.txt`, `src/hal9000/tools/greeting.py` | default |
| Separation of reasoning and execution | `extending_with_self_prompting/separation_of_reasoning.md` | architecture-wide; described in `docs/ARCHITECTURE.md` | default |
| Capability lifecycle, ten hooks | `improving_agent_reasoning/capability_architecture_pattern.md` | `src/hal9000/core/capabilities.py`, `src/hal9000/core/agent.py` | default — **hooks fire; the list is empty** |
| `TimeAwareCapability` | `improving_agent_reasoning/capability_architecture_pattern.md` | `src/hal9000/capabilities/time_aware.py` | flag: `time_aware` |
| `PlanFirstCapability` + `create_plan` | `improving_agent_reasoning/plan_first_capability.md` | `src/hal9000/capabilities/plan_first.py`, `src/hal9000/tools/planning.py` | flag: `plan_first` |
| `ProgressTrackingCapability` + `track_progress` | `improving_agent_reasoning/progress_tracking_capability.md` | `src/hal9000/capabilities/progress_tracking.py`, `src/hal9000/tools/planning.py` | flag: `progress_tracking` |
| Logging capability | `improving_agent_reasoning/capability_architecture_pattern.md` | `src/hal9000/capabilities/logging.py` | flag: `verbose_logging` |
| `AgentRegistry` + `call_agent` | `multi-agent_systems/agent-to-agent-communication.md` | `src/hal9000/core/registry.py`, `src/hal9000/tools/delegation.py` | flag: `delegation` |
| Memory reflection / handoff / selective share | `multi-agent_systems/agent_interaction_with_memory.md`, `multi-agent_systems/advanced_agent_interaction.md` | `src/hal9000/tools/delegation.py` — all four patterns | flag: `delegation` gives the greeter `call_agent` only (R15); the other three are one line away |
| `ReversibleAction`, `ActionTransaction` | `agent_design_principles/environment_safety.md` | `src/hal9000/safety/reversible.py` | flag: `staged_review` |
| Staged execution with review (human + LLM) | `agent_design_principles/environment_safety.md` | `src/hal9000/safety/staged.py`, `src/hal9000/safety/review.py` | flag: `staged_review` |
| Single safe tool vs. multiple risky tools | `agent_design_principles/environment_safety.md` | `src/hal9000/tools/guestbook.py`, `docs/EXTENDING.md` | default |
| MATE — **M**odel efficiency (tiered LLMs) | `agent_design_principles/MATE_design_principle.md` | `src/hal9000/llm.py`, `src/hal9000/config.py`, used in `src/hal9000/tools/greeting.py` | default |
| MATE — **A**ction specificity | `agent_design_principles/MATE_design_principle.md` | `src/hal9000/tools/guestbook.py`, `docs/EXTENDING.md` | default |
| MATE — **T**oken efficiency | `agent_design_principles/MATE_design_principle.md` | terse tool descriptions throughout `src/hal9000/tools/`; `Memory.get_memories(limit)` in `src/hal9000/core/memory.py` | default |
| MATE — **E**nvironmental safety | `agent_design_principles/MATE_design_principle.md` | `src/hal9000/safety/`, path confinement in `src/hal9000/storage/guestbook.py` | default (confinement); flag: `staged_review` (staging, undo) |

All 35 source notes are accounted for. `README.md` in the notes repository is an
index, not a pattern note.

---

## Tool inventory, corrected

BUILD_PLAN §6's table is stale in three ways. This is what is actually
registered — **16 tools**, of which 5 are in the default run and 11 are
reachable by the greeter.

| Tool | Tag | Terminal | In the greeter when | Notes |
|---|---|---|---|---|
| `extract_visitor_details(document_text)` | `greeting` | no | always | Fixed schema; runs on `fast_llm`; retry-wrapped |
| `compose_greeting(visitor)` | `greeting` | no | always | `prompt_expert` concierge persona; `_greeting_policy` injected |
| `sign_guestbook(name, greeting, note="")` | `guestbook` | no | always | `_guestbook` injected; validates; returns `entry_id`. **The only tool that writes** |
| `read_guestbook(limit=5)` | `guestbook` | no | always | Empty book returns an instruction, not a bare failure |
| `terminate(message)` | `system` | **yes** | always | Returns `message` unchanged (R7) |
| `undo_last_guestbook_entry()` | `safety` | no | `staged_review` | Pairs with `ReversibleAction`; never itself staged (R12) |
| `prompt_llm(prompt)` | `prompts` | no | `dynamic_experts` | Agent-facing self-dialog |
| `prompt_llm_for_json(schema, prompt)` | `prompts` | no | `dynamic_experts` | 3-attempt retry, ```` ```json ```` fence stripping |
| `prompt_expert(description_of_expert, prompt)` | `prompts` | no | `dynamic_experts` | Persona pattern |
| `create_and_consult_expert(expertise_domain, problem_description)` | `prompts` | no | `dynamic_experts` | Dynamic persona. **Three LLM calls per invocation** |
| `call_agent(agent_name, task)` | `delegation` | no | `delegation` | Message passing |
| `call_agent_with_reflection(agent_name, task)` | `delegation` | no | never, by default (R15) | Memory reflection |
| `hand_off_to_agent(agent_name, task)` | `delegation` | no | never, by default (R15) | Memory handoff |
| `call_agent_with_selected_context(agent_name, task)` | `delegation` | no | never, by default (R15) | LLM-selected memories |
| `create_plan()` | `planning` | no | never — called by the capability | `PlanFirstCapability` calls it directly in `init` |
| `track_progress()` | `planning` | no | never — called by the capability | `ProgressTrackingCapability` calls it in `end_agent_loop` |

Default registry: `PythonActionRegistry(tags=["greeting", "guestbook", "system"])`
— five tools. Flags append tags, except `delegation`, which adds one tool by
name.

---

## Corrections to the plan

Recorded here because BUILD_PLAN and ORCHESTRATOR_RULINGS are the build's
historical record and are not edited after the fact.

**1. `create_and_consult_expert` takes `problem_description`, not `problem`
(R6).** §6's row is wrong; `persona_abstraction.md` is right, and the note wins.
The implemented signature is
`create_and_consult_expert(expertise_domain, problem_description)`.

**2. `dynamic_experts` adds four tools, not one.** §6 lists
`create_and_consult_expert` as the flag's effect, and the flag's *name* suggests
one tool. It switches on the whole `prompts` tag: `prompt_llm`,
`prompt_llm_for_json`, `prompt_expert` **and** `create_and_consult_expert`. Five
tools become nine. Say so plainly when you document your own flags — a flag
whose name understates its effect is worse than no flag.

**3. `staged_review` is the *environmental safety* flag (R12).** §6 tags
`undo_last_guestbook_entry` "flag" but names no flag, and §5.7's frozen
`Features` has no `safety` field. `staged_review` governs both: it adds the
`safety` tag to the registry **and** swaps in `StagedActionEnvironment`. Its
documented meaning in the README and in `--help` is "environmental safety", not
"staged review" alone. Staging a write and being able to reverse one are one
concern.

**4. `ActionContext` carries an `llm` / `powerful_llm` alias (R1).** The notes
are internally inconsistent: four of them read `llm`, while
`MATE_design_principle.md` reads `powerful_llm`. Both keys exist and are bound
to the **identical callable**. `llm` is canonical and every tool in this
template reads it; `powerful_llm` costs one dictionary entry and lets the MATE
note's examples run verbatim. If you arrived from that note, you are not
confused and nothing is shadowed — they are the same object. `fast_llm` is the
same callable pinned to the cheap tier.

**5. `safety/` has a fifth file, `safety/review.py`.** §4's layout shows only
`reversible.py` and `staged.py`. Reviewers — `HumanReviewer`, `LLMReviewer`,
`ReviewRequest`, `ReviewDecision`, `reviewer_for` — earned their own module,
because "who gets asked, and how their answer is read" is a separate concern
from "what gets held back", and because that is where R14's fail-closed approval
parsing lives. The package is `__init__.py`, `reversible.py`, `review.py`,
`staged.py`.

**6. Registered-tool count.** §6's inventory lists 16 tools and the default
five; it is worth stating the numbers together, because they are the design:
**16 registered · 11 reachable by the greeter · 5 in the default run.** The five
that the greeter never receives are the three alternative delegation tools
(R15) and the two planning tools, which their capabilities call directly rather
than offering to the model.

Five further rulings shaped the code rather than this table, and are documented
where a reader will hit them —
[`EXTENDING.md`](EXTENDING.md#five-traps-found-in-the-source-material) covers
R5, R9, R13, R14 and R16.

---

## Verification

Every implementation path named in this file exists. The check is mechanical,
and re-running it after an edit takes a second:

```bash
# from the repository root
grep -o '`[A-Za-z0-9_./-]\+`' docs/COVERAGE.md | tr -d '`' \
  | grep -E '^(src|docs|config|tests|\.github)/' | sort -u \
  | while read -r p; do [ -e "$p" ] || echo "MISSING: $p"; done; echo "checked."
```

Silence (beyond `checked.`) means every row points at something real. The same
check against the source column, run from the notes repository, confirms every
cited note exists:

```bash
# from coursera_ai_agent_content/
grep -o '`[A-Za-z0-9_-]\+/[A-Za-z0-9_.-]\+\.md`' ../agent-template/docs/COVERAGE.md | tr -d '`' \
  | grep -v '^docs/' | sort -u \
  | while read -r p; do [ -e "$p" ] || echo "MISSING NOTE: $p"; done; echo "checked."
```
