# ADR 0002 — HAL is a deterministic coordinator; no LangGraph

**Status:** Accepted
**Date:** 2026-09

## Context

The evening run has one fixed shape: check the budget, finalize yesterday, let
the solar agent gather and judge the day, guarantee a report and a record, speak
it, and write the books. Only the middle of that needs judgment. Everything
around it is bookkeeping that must happen the same way every night, including
when the model misbehaves, overspends or is unreachable.

LangGraph was considered for orchestration and scheduling. It models agent
workflows as graphs of nodes with checkpointed state, and it works with agents
not built on LangChain.

## Decision

HAL 9000 is plain Python (`coordinator/runner.py`), not an LLM and not a graph
framework. The two agents run on the repository's own GAME loop:

- The **solar agent** (fast tier) decides which tools to call and writes the
  narrative.
- The **triage sub-agent** (powerful tier) is a tool the solar agent may call.
  It has its own tool registry, so it can't deliver reports, send alerts or
  write records.

The coordinator owns every guarantee: the report and record exist, the numbers
spoken were produced by tools, the cost caps hold, and delivery falls back from
the speaker to Telegram.

Scheduling is external: run the CLI at 20:00 from cron or launchd now, and from
n8n calling `hal9000 serve` later.

## Why not LangGraph

- **The graph is a straight line.** There is one conditional edge (triage),
  and the agent already makes that call as a tool. A graph adds a second
  control-flow model without adding a path.
- **The guarantees must not depend on the orchestrator.** The fallback report,
  the numeric guard and the cost caps are ordinary code that runs whatever the
  agent did. Moving them into graph nodes would make them depend on the
  framework's state handling.
- **Checkpointing isn't needed.** A run takes under a minute and is idempotent
  per date; a failed run is simply re-run. Durable history is already in SQLite.
- **Fewer moving parts on the Raspberry Pi.** The target host is the Pi that
  also runs Home Assistant. The GAME loop has no dependencies beyond LiteLLM.

## Consequences

- Adding a step to the evening means editing `run_evening`, which is short and
  tested end to end by the replay suite.
- If HAL later grows several independent agents with branching hand-offs, or
  runs that span hours and must resume, this decision should be revisited.
- n8n (or cron) decides *when*; HAL decides *what*. The HTTP API
  (`/runs/evening`, `/health`, `/cost`) is the seam between them.
