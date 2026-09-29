# House knowledge base: build status

**Integration branch:** `knowledge` (cut from `solar-agent`) · **Plan:**
[`KNOWLEDGE_PLAN.md`](KNOWLEDGE_PLAN.md) · **Handoff:**
[`KNOWLEDGE_BUILD_HANDOFF.md`](KNOWLEDGE_BUILD_HANDOFF.md)

## Work packages

| KWP | Title | Branch | State |
|---|---|---|---|
| KWP0 | Foundation, contracts and API contract | `kwp/0-foundation` | built, awaiting review |
| KWP1 | Parsing, cleaning, segmentation and the fixture corpus | | pending |
| KWP2 | Chunkers and index profiles | | pending |
| KWP3 | Embeddings and index | | pending (after G-K0b) |
| KWP4 | Query processing, reranking and abstention | | pending |
| KWP5 | Evaluation harness | | pending |
| KWP6 | Answer pipeline and API | | pending |
| KWP7 | Web UI | | pending |
| KWP8 | Observability | | pending |
| KWP9 | The setup evaluation (with Mike) | | pending |
| KWP10 | HAL integration | | pending |
| KWP11 | Pi deployment | | pending |
| KWP12 | Acceptance and documentation | | pending |

## Gates

| Gate | State |
|---|---|
| G-K0 plan approved | done |
| G-K0b sqlite-vec check | Linux x86_64 and aarch64 done (orchestrator); Mac deferred to `kb check` at G-K1 |
| G-K1, G-K1b, G-K2, G-K3 | pending |

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
| D4 | `.github/workflows/ci.yml` now syncs the knowledge groups (all groups except `knowledge-docling` and `knowledge-rerank`) | Without them the existing CI job fails: the no-network test imports every `hal9000` module and coverage counts the new package | pending (KWP0 report) |
| D5 | `monthly_precheck` (the solar cap) no longer counts `knowledge`/`knowledge-eval` spend; `Database.month_cost_for` added | Plan §13 gives the knowledge agents their own caps; otherwise knowledge spend could switch off the evening report. Solar-only ledgers behave exactly as before | pending (KWP0 report) |
