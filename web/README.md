# HAL house knowledge: web UI (KWP7)

React 19 + Vite + TypeScript, Tailwind CSS v4, shadcn-style components on Radix
(`radix-ui`), TanStack Query, react-dropzone, react-pdf (pdf.js), Recharts.
Plan: `docs/KNOWLEDGE_PLAN.md` §10–11. Handoff: `docs/KNOWLEDGE_BUILD_HANDOFF.md` KWP7.

## Scripts

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server; `/api`, `/health`, `/metrics` proxied to `hal9000 kb serve` on `KB_PORT` (default 8910) or `KB_DEV_BACKEND` |
| `npm run dev:mock` | Vite dev server with the API answered in the browser by MSW (`src/mocks/`), no backend needed |
| `npm run build` | Type-checks, then builds to **`web/dist/`** |
| `npm run gen:api` | Regenerates `src/api/schema.d.ts` from `src/api/openapi.json` (CI fails on drift) |
| `npm run lint` / `typecheck` / `test` | ESLint, `tsc --noEmit`, Vitest (offline; MSW in Node) |
| `npm run e2e` | Playwright (see below) |
| `npm run fixtures:pdf` | Rewrites the synthetic PDF used by the mocks |

## Serving the build from FastAPI

`npm run build` writes a static site to `web/dist/` (`index.html`, `favicon.svg`,
`assets/`). KWP6 mounts it at `/`:

- serve `web/dist/assets/*` and the root files as static files;
- for any other non-`/api`, non-`/health`, non-`/metrics` GET, return
  `web/dist/index.html` (history-API fallback), because the UI uses real paths
  such as `/traces/<id>` (linked from Grafana) and `/c/<conversation id>`.

`dist/` and `node_modules/` are git-ignored. Nothing from `src/mocks/` is in the
build: MSW is imported only when `import.meta.env.DEV && VITE_MOCK === "1"`, and
the service worker is served by a dev-only Vite middleware, never from `public/`.

Optional build-time setting: `VITE_GRAFANA_URL` adds Grafana links to Traces and
the Query Inspector.

## Tests

- **Vitest** (`src/**/*.test.ts(x)`): library code (SSE parser, chat stream
  reducer, citation splitter, fuzzy highlight matcher, the folder-drop walker with
  fake `FileSystemDirectoryEntry`/`FileSystemFileEntry` objects, pre-upload
  summary, upload pool) and page tests that render the app against the same MSW
  handlers the browser mocks use.
- **Playwright** lives in **`web/e2e/`** (not `tests/e2e/`) so it shares
  `web/node_modules`. Config: `web/playwright.config.ts`. Two projects:
  - `flows`: sign in, folder picker (`setInputFiles` with a directory), dropped
    loose files, summary with skipped images, jobs reaching "Indexed", streamed
    answer, forced reset, `[S1]` → viewer at the cited page with a highlight
    (including the letter-spaced heading), thumbs down + comment, "Add to
    question set", keyboard approval, judge labelling, charts, profile switch,
    degraded banners, Query Inspector.
  - `visual`: every page at 390 px and 1,440 px in light and dark; asserts no
    horizontal scroll and that every visible control is reachable with Tab;
    screenshots go to `web/e2e-screenshots/` (git-ignored).

  Against the mocks (default, Playwright starts `npm run dev:mock` on port 5174):

  ```bash
  PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers npm run e2e   # in the build container
  ```

  Against a real server (e.g. `hal9000 kb serve --fake`):

  ```bash
  E2E_BASE_URL=http://127.0.0.1:8910 E2E_PASSCODE=<passcode> npm run e2e
  ```

  `@playwright/test` is pinned to 1.56.1, whose Chromium revision (1194)
  matches the pre-installed browsers; never run `playwright install` there.

### Pending contract

`src/api/pending-contract.ts` holds the contract additions approved for KWP6 but
not yet in `openapi.json`: `MessageOut.grounding`/`issues`, optional `rating`
with `add_to_gold` (feedback is one row per message; a later POST updates it),
`citations` on judge-label items, `GET /api/eval/runs/{id}/results` (paged),
`POST /api/jobs/{id}/retry` and `TraceSummary.question`. The UI and the MSW
mocks use them only through that file; when `openapi.json` is regenerated,
swap each type for the generated one and delete the file.

### Fake-mode triggers

The mocks (and, to pass the same suite, `kb serve --fake`) react to phrases in
the question (`src/mocks/data.ts` `TRIGGERS`, `e2e/fixtures.ts` `QUESTIONS`):

| Phrase in the question | Behaviour |
|---|---|
| `force a reset` | streams a draft with a wrong figure, sends `reset`, streams the corrected answer, `verify: verified` |
| `retrieval only` | `sources` with `retrieval_only: true` and a banner; no tokens; `outcome: retrieval_only` |
| `embedding down` | normal answer; `sources.banner` says semantic search is unavailable (BM25 only) |
| `unverified` | `reset`, then the same answer again with `verify: failed` and issues |
| `swimming pool` | abstains |

Mock passcode: `pod-bay-doors`. File names containing `corrupt` fail at the
parse step (to show the error row and Retry).

All mock and fixture data is synthetic ("12 Example Lane"); no house data.
