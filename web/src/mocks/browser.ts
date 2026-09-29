/** Starts MSW in the browser for `npm run dev:mock` (and the Playwright mock runs). */
import { setupWorker } from "msw/browser";
import { handlers } from "./handlers";
import { db } from "./db";

export async function startMockWorker(): Promise<void> {
  // A reload keeps you signed in, like the real 30-day cookie.
  const KEY = "hal.mock.signedIn";
  try {
    db.signedIn = sessionStorage.getItem(KEY) === "1";
    db.health.retrievalOnly = localStorage.getItem("hal.mock.retrievalOnly") === "1";
  } catch {
    /* storage blocked */
  }
  const worker = setupWorker(...handlers);
  worker.events.on("response:mocked", ({ request }) => {
    if (new URL(request.url).pathname === "/api/session") {
      try {
        sessionStorage.setItem(KEY, db.signedIn ? "1" : "0");
      } catch {
        /* storage blocked */
      }
    }
  });
  // Handy in the console and for Playwright: window.__halMock.db.health.retrievalOnly = true
  (window as unknown as { __halMock: unknown }).__halMock = { db };
  await worker.start({ onUnhandledRequest: "bypass", quiet: true });
}
