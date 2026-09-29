# ADR 0003 — The house knowledge base: one repo, SQLite, a pipeline, our own UI

**Status:** Accepted (plan approved at gate G-K0)
**Date:** 2026-09
**Source:** [`KNOWLEDGE_PLAN.md`](../KNOWLEDGE_PLAN.md) §3–4, which holds the full reasoning

## Context

HAL gains a second domain: answering questions about the house from its
accumulated documents (inspections, estimates, invoices, manuals), with a
citation to the exact page for every answer. It runs on the Raspberry Pi 5
beside Home Assistant, is reachable only on the home network, and is cost
capped. The chunking and retrieval settings are to be *chosen by measurement*
(plan §8), so the design must keep every retrieval step visible and timed.

## Decisions

1. **One repository.** The knowledge base is `src/hal9000/knowledge/` plus a
   `web/` UI inside `hal_9000`, not a new repository. HAL is the single entry
   point for home agents, and the LLM boundary (`llm.py`), cost ledger and
   caps (`cost/`), config, CLI, CI coverage gate and Pi target already exist
   and are tested. The domains stay apart through a separate package and
   database (`data/knowledge.db`), a separate process (`hal9000 kb serve`), an
   import-boundary test (only `solar.domain.numerals` may be shared), and
   optional `uv` dependency groups. *Revisit* if the UI becomes a general
   household app: then `web/` moves out and this API stays its backend.

2. **SQLite + FTS5 + sqlite-vec, in one file.** A household corpus is
   thousands to tens of thousands of chunks, where exact KNN takes
   milliseconds. FTS5 gives BM25; `sqlite-vec` gives cosine KNN with metadata
   filters applied inside the search. One file backs up with the rest of
   `data/` and adds no server to the Pi. Each index profile gets its own
   `fts_<id>` and `vec_<id>` tables, so BM25 statistics are never pooled
   across chunk sizes. A `VectorIndex` protocol keeps Qdrant or LanceDB as
   drop-in alternatives. `sqlite-vec` is pinned (0.1.9) to the version
   verified on x86_64 and aarch64 (gate G-K0b).

3. **A fixed pipeline, not an agent.** Every question runs the same stages:
   condense → embed query → vector + BM25 → fuse → rerank → select → abstain
   check → generate (streamed) → verify. There is no decision for an agent to
   make, and a pipeline is faster, cheaper, measurable per stage and
   evaluable. Every stage is wrapped in the shared `stage()` timer with names
   from one closed enum (plan §9.2). The same retrieval is also offered to
   HAL's future agents as a GAME tool (`search_house_docs`).

4. **A custom UI** (React, Vite, TypeScript) served by the same FastAPI
   process. Off-the-shelf chat UIs (Open WebUI) bring their own RAG pipeline,
   which would hide the very choices the evaluation measures. The UI needs
   folder upload, a gold-set review screen, judge calibration and a query
   inspector that no packaged UI has. The API's OpenAPI schema, committed at
   `web/src/api/openapi.json`, is the contract between the two; a test fails
   when it drifts.

5. **Prometheus and Grafana for observability.** `kb_*` metrics (stage
   latency, time to first token, cost, abstentions, grounding failures,
   ingestion health) are exposed at `/metrics`; Grafana dashboards are
   provisioned as code and also read the `traces` table through a read-only
   SQLite datasource. OpenTelemetry export is optional. Langfuse (Postgres,
   ClickHouse, Redis, S3) is too heavy for the Pi; Arize Phoenix's licence
   is not OSI open source.

## Consequences

- LangChain and LlamaIndex are not used: each step is a few dozen lines here,
  and a framework would obscure per-stage timing.
- OpenAI is the only paid provider (embeddings, chat, judge); every call goes
  through the metered wrapper (`Meter.stream`, `Meter.embed`, `Meter.wrap`)
  with agent `knowledge` or `knowledge-eval`, each with its own monthly cap.
  The solar cap does not count knowledge spend, so neither domain can switch
  the other off.
- Heavy candidates (Docling, fastembed rerankers) live in their own `uv`
  groups and are imported lazily; they ship only if the evaluation shows they
  earn their weight on the Pi.
- `tiktoken` encodings are vendored (`vendor/tiktoken/`) so token counting
  never touches the network, in tests or on the Pi.
- No house data enters git: synthetic fixtures in tests, `data/` ignored, and
  a denylist check in a pre-commit hook and in CI.
