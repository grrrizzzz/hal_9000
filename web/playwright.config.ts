/**
 * Playwright for the web UI (KWP7).
 *
 * - Default: runs against the Vite dev server with MSW mocks (`npm run dev:mock`),
 *   which Playwright starts on port 5174.
 * - Against a real server: E2E_BASE_URL=http://127.0.0.1:8910 E2E_PASSCODE=… npm run e2e
 *   (e.g. `hal9000 kb serve --fake`); no dev server is started then.
 *
 * Browsers: this uses @playwright/test 1.56.1, whose Chromium revision (1194)
 * matches the pre-installed browsers in /opt/pw-browsers. Set
 * PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers there (never `playwright install`).
 */
import { defineConfig, devices } from "@playwright/test";

const external = process.env.E2E_BASE_URL;
const port = Number(process.env.E2E_PORT ?? 5174);
const baseURL = external ?? `http://127.0.0.1:${port}`;

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
    { name: "flows", testIgnore: /visual\.spec\.ts/, use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "visual", testMatch: /visual\.spec\.ts/, use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: external
    ? undefined
    : {
        command: `npm run dev:mock -- --port ${port} --strictPort --host 127.0.0.1`,
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
      },
});
