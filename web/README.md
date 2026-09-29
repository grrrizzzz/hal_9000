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
| `npm run e2e` | Playwright against the MSW dev server, or `E2E_BASE_URL` (see below) |
| `npm run e2e:real` | Builds `dist/`, then Playwright starts `uv run hal9000 kb serve --fake` itself and runs the suite against it |
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
  `web/node_modules`. Config: `web/playwright.config.ts`. Three projects:
  - `flows`: sign in, folder picker (`setInputFiles` with a directory), dropped
    loose files, summary with skipped images, jobs reaching "Indexed", streamed
    answer, forced reset, `[S1]` → viewer at the cited page with a highlight
    (including the letter-spaced heading), thumbs down + comment, "Add to
    question set", keyboard approval, judge labelling, charts, profile switch,
    degraded banners, Query Inspector.
  - `visual`: every page at 390 px and 1,440 px in light and dark; asserts no
    horizontal scroll and that every visible control is reachable with Tab;
    screenshots go to `web/e2e-screenshots/` (git-ignored).
  - `session`: Account → "Sign out everywhere" (confirm, then
    `DELETE /api/session?everywhere=true`); on a real server a second browser
    loses its session too. It depends on the other two, so it runs last: it
    revokes every session on the server.

  Against the mocks (default, Playwright starts `npm run dev:mock` on port 5174):

  ```bash
  PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers npm run e2e   # in the build container
  ```

  Against the real API in fake mode, one command (needs `uv` and the synced
  Python environment; the server gets a fresh temporary library, port
  `E2E_REAL_PORT`, default 8932, and passcode `KB_PASSCODE` or `pod-bay-doors`).
  CI runs this in the `web` job:

  ```bash
  PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers npm run e2e:real
  ```

  Against a server you started yourself (it must serve a current `dist/`):

  ```bash
  E2E_BASE_URL=http://127.0.0.1:8910 E2E_PASSCODE=<passcode> npm run e2e
  ```

  The suite passes against a shared server in any order and on a re-run: each
  run's uploads carry a run id (the server would call repeated bytes
  "Already in library"), upload fixtures avoid the chat questions' subjects,
  and gold items are followed by id (`data-gold-id`). Direct API calls from
  Playwright's `request` context must send `Sec-Fetch-Site: same-origin` (or a
  matching `Origin`): the server refuses other non-GET `/api` requests.

  `@playwright/test` is pinned to 1.56.1, whose Chromium revision (1194)
  matches the pre-installed browsers; never run `playwright install` there.

### API types

Every request and response type comes from `src/api/schema.d.ts`, generated from
`src/api/openapi.json` (`npm run gen:api`; CI fails on drift), through the typed
client in `src/api/client.ts`. The MSW handlers are typed the same way and follow
the server's behaviour (one feedback row per message, retry re-queues the same
job, `DELETE /api/session` needs a session and takes `?everywhere=true`).

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
