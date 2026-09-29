/**
 * Playwright for the web UI (KWP7).
 *
 * - Default: runs against the Vite dev server with MSW mocks (`npm run dev:mock`),
 *   which Playwright starts on port 5174.
 * - `npm run e2e:real` (E2E_REAL=1): builds web/dist, then Playwright starts
 *   `uv run hal9000 kb serve --fake` itself (from the repo root, port
 *   E2E_REAL_PORT, default 8932) and runs the same suite against it. Needs uv
 *   and the synced Python environment.
 * - Against a server you started: E2E_BASE_URL=http://127.0.0.1:8910 E2E_PASSCODE=… npm run e2e
 *   (e.g. `hal9000 kb serve --fake`); no server is started then.
 *
 * Browsers: this uses @playwright/test 1.56.1, whose Chromium revision (1194)
 * matches the pre-installed browsers in /opt/pw-browsers. Set
 * PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers there (never `playwright install`).
 */
import { defineConfig, devices } from "@playwright/test";

const external = process.env.E2E_BASE_URL;
const real = !external && process.env.E2E_REAL === "1";
const port = Number(process.env.E2E_PORT ?? 5174);
const realPort = Number(process.env.E2E_REAL_PORT ?? 8932);
const baseURL = external ?? (real ? `http://127.0.0.1:${realPort}` : `http://127.0.0.1:${port}`);
if (real) {
  // The server's passcode (KB_PASSCODE, default "pod-bay-doors") and the one the tests type must agree.
  process.env.E2E_PASSCODE ??= process.env.KB_PASSCODE || "pod-bay-doors";
}

const mockServer = {
  command: `npm run dev:mock -- --port ${port} --strictPort --host 127.0.0.1`,
  url: baseURL,
  reuseExistingServer: !process.env.CI,
  timeout: 60_000,
};

/** `kb serve --fake`: a fresh temp library seeded with the synthetic corpus, serving web/dist. */
const realServer = {
  command: `uv run --locked hal9000 kb serve --fake --host 127.0.0.1 --port ${realPort}`,
  cwd: "..",
  url: `${baseURL}/health`,
  // Always a fresh server: the suite changes its state (uploads, approvals, the active profile).
  reuseExistingServer: false,
  // Seeding the synthetic library takes a while on a cold start.
  timeout: 180_000,
  env: { KB_PASSCODE: process.env.E2E_PASSCODE ?? "" },
  // SIGTERM lets the server delete its temporary data directory.
  gracefulShutdown: { signal: "SIGTERM" as const, timeout: 10_000 },
  stdout: "ignore" as const,
  stderr: "pipe" as const,
};

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",
  fullyParallel: true,
  workers: process.env.CI ? 2 : undefined,
  retries: process.env.CI ? 1 : 0,
  timeout: 45_000,
  expect: { timeout: 10_000 },
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    { name: "flows", testIgnore: /(visual|session)\.spec\.ts/, use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "visual", testMatch: /visual\.spec\.ts/, use: { ...devices["Desktop Chrome"] } },
    // Revokes every session on the server, so it runs last.
    { name: "session", testMatch: /session\.spec\.ts/, dependencies: ["flows", "visual"], use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
  ],
  webServer: external ? undefined : real ? realServer : mockServer,
});
