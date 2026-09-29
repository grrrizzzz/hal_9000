# House knowledge base: build status

**Integration branch:** `knowledge` (cut from `solar-agent`) · **Plan:**
[`KNOWLEDGE_PLAN.md`](KNOWLEDGE_PLAN.md) · **Handoff:**
[`KNOWLEDGE_BUILD_HANDOFF.md`](KNOWLEDGE_BUILD_HANDOFF.md)

## Work packages

Every package was built by a builder agent, checked by a separate reviewer agent (re-running
the gates, mutation-testing the key tests and probing edge cases), fixed until it passed, then
merged into `knowledge`.

| KWP | Title | State | Review rounds |
|---|---|---|---|
| KWP0 | Foundation, contracts and API contract | merged | FAIL → fixed |
| KWP1 | Parsing, cleaning, segmentation, fixture corpus | merged (plus 1b real-bundle generalisation, 1c labels) | FAIL → fixed; 1b FAIL → fixed |
| KWP2 | Chunkers and index profiles | merged | PASS (+ follow-ups) |
| KWP3 | Embeddings, index and ingestion pipeline | merged | FAIL → fixed |
| KWP4 | Retriever: condense, hybrid search, rerank, select, abstain | merged | FAIL → fixed |
| KWP5 | Evaluation harness (part A core, part B sweep/latency/decide/test) | merged | A FAIL → fixed; B FAIL → fixed |
| KWP6 | Answer pipeline and HTTP API | merged | FAIL (security) → fixed |
| KWP7 | Web UI (phase 1 mocks, phase 2 real API) | merged | 1 FAIL → fixed; 2 PASS |
| KWP8 | Observability (library, live checks, Grafana/Prometheus provisioning) | merged | 1 FAIL → fixed; 2 FAIL → fixed |
| KWP9 | The setup evaluation with Mike | **waiting for Mike** (runbook §2–§6) | — |
| KWP10 | HAL integration (`search_house_docs`, backup/restore, stats) | merged | FAIL → fixed |
| KWP11 | Pi deployment (image, compose, runbook) | merged; arm64 build on the Pi pending | FAIL → fixed |
| KWP12 | Acceptance | after G-K2 (G-K3 checklist) | — |

## Verification at hand-off (knowledge branch)

| Check | Result |
|---|---|
| Offline suite | 2,108 passed, 51 opt-in skipped; coverage 97.1%; ruff clean |
| Web | lint, typecheck, Vitest 68/68, build |
| Browser end to end | Playwright 18/18 against the real service (`npm run e2e:real`) and 18/18 against mocks |
| Real inspection bundle (`pytest -m house`, local only) | 89 pages with text; document boundaries 11/11 and types 11/11 without the LLM; no running headers left |
| Real bundle end to end (fake embedder) | ingests in about 18 s: 11 documents, 89 pages, 225 chunks |
| House-data denylist | 481 tracked files clean |
| Container | built on a stand-in base; runs offline, healthy, non-root, 672 MB; arm64 build is a Pi step |
| Not yet verified (needs OpenAI, Hugging Face or the Pi) | live answers and embeddings; LLM segmentation; rerankers and Docling; Grafana/Prometheus containers; S6 latency on the Pi |

## Gates

| Gate | State |
|---|---|
| G-K0 plan approved | done |
| G-K0b sqlite-vec check | done on Linux x86_64 and aarch64; the Mac is checked by `kb check` (runbook §2.3) |
| G-K1 question set | next: runbook §2 |
| G-K1b sweep, judge labels, S5 | runbook §3 |
| G-K2 choose | runbook §5–§6 |
| G-K3 acceptance | after a week of use |

## Contracts fixed by KWP0 (changes need orchestrator approval, rule K4)

- `src/hal9000/knowledge/contracts.py`: domain models and the API shapes.
- `web/src/api/openapi.json`: regenerate with `uv run python scripts/kb_export_openapi.py`;
  a test fails on drift. The web CI job expects generated types at
  `web/src/api/schema.d.ts` (`npx openapi-typescript src/api/openapi.json -o src/api/schema.d.ts`).
- `src/hal9000/knowledge/obs/timing.py`: the `Stage` enum and `stage()`; metrics go in
  `timing.REGISTRY`.
- `knowledge.db` schema: `src/hal9000/knowledge/storage/schema.py` (append-only migrations).
  `vec0` metadata columns take no NULL in sqlite-vec 0.1.9: store `''` for an unknown
  `doc_type`, `collection` or `doc_date`.
- Ledger agents: `knowledge` (cap `KB_MONTHLY_CAP_USD`) and `knowledge-eval`
  (cap `KB_EVAL_CAP_USD`); build a `Meter` with `monthly_caps=knowledge.config.monthly_caps()`.
- CLI: each KWP replaces the `run(args)` stub in its own `knowledge/cli/<module>.py`.

## Deviations

| # | What | Why | Approved |
|---|---|---|---|
| D1 | Work is committed on local `kwp/*` branches and merged locally; nothing is pushed to GitHub | GitHub refuses pushes from the build session (handoff §2.1 assumed pushes) | orchestrator |
| D2 | G-K0b satisfied by the orchestrator on x86_64 and aarch64 Linux with `sqlite-vec==0.1.9`; the Mac check is deferred to `hal9000 kb check` at G-K1 | The Pi-equivalent (aarch64 Linux) host was verified in the build environment; the Mac only serves the evaluation | orchestrator |
| D3 | `tiktoken` encodings vendored from the `tiktoken-rs` 0.6.0 crate, with the official sha256 hashes verified | `openaipublic.blob.core.windows.net` is unreachable from the build environment; `scripts/kb_fetch_tiktoken.py` stays as a fallback | orchestrator |
| D4 | `.github/workflows/ci.yml` now syncs the knowledge groups (all groups except `knowledge-docling` and `knowledge-rerank`) | Without them the existing CI job fails: the no-network test imports every `hal9000` module and coverage counts the new package | orchestrator (KWP0 review) |
| D5 | `monthly_precheck` (the solar cap) no longer counts `knowledge`/`knowledge-eval` spend; `Database.month_cost_for` added | Plan §13 gives the knowledge agents their own caps; otherwise knowledge spend could switch off the evening report. Solar-only ledgers behave exactly as before | orchestrator (KWP0 review) |
| D6 | Pipeline order is parse → segment (raw pages) → clean(pages, documents) → chunk | Running headers are found per sub-document; plan §5.1 listed cleaning before segmentation | orchestrator (KWP1b) |
| D7 | Collections filter includes sub-folders; new file versions supersede old ones in search | Orchestrator decisions D1/D2 in the KWP3 review | orchestrator |
| D8 | Question set target is about 120 items (60/40 stratified dev/test); setup sweep cap $10 metered as `knowledge-eval`, separate from the $10 monthly chat cap | Plan review: 60 items cannot separate close configurations | Mike (plan) |
| D9 | Grafana mounts `data/` read-only with its default uid | Verified live: a read-write mount lets Grafana create WAL files hal-kb then cannot write | orchestrator (KWP8b/KWP11) |
| D10 | Default first-ingest profile `pdfium.structure-768.hdr.te3s-1536` in `config/knowledge.yaml` | Needed so G-K1 has a real index before question generation; replaced at G-K2 by `kb eval choose` | orchestrator (KWP11 review) |
| D11 | Regression checks never use a query transform, and run once per drained ingest queue as their own job | Cost and determinism | orchestrator |

