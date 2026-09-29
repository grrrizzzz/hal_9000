/**
 * K12 visual checks: every page at 390 px and 1,440 px, light and dark.
 * Asserts no horizontal scroll and that every visible control is reachable
 * with the Tab key, and writes screenshots to web/e2e-screenshots/ (git-ignored;
 * synthetic fixture data only).
 */
import { expect, test, type Page } from "@playwright/test";
import { signIn } from "./fixtures";

const PAGES: [string, string][] = [
  ["chat", "/"],
  ["library", "/library"],
  ["document", "/library/1"],
  ["upload", "/upload"],
  ["eval-review", "/evaluation?tab=review"],
  ["eval-judge", "/evaluation?tab=judge"],
  ["eval-runs", "/evaluation?tab=runs"],
  ["eval-profile", "/evaluation?tab=profile"],
  ["traces", "/traces"],
];

const INTERACTIVE =
  'a[href], button:not([disabled]), input:not([type="hidden"]):not([disabled]), select:not([disabled]), textarea:not([disabled]), [role="radio"], [role="tab"], summary, [tabindex]:not([tabindex="-1"])';

/** Tab through the page and return the visible interactive elements that never got focus. */
async function unreachable(page: Page): Promise<string[]> {
  await page.evaluate(() => {
    const w = window as unknown as { __focused: Set<Element> };
    w.__focused = new Set();
    document.addEventListener("focusin", (e) => w.__focused.add(e.target as Element), true);
    (document.activeElement as HTMLElement | null)?.blur();
  });
  const total = await page.evaluate((sel) => document.querySelectorAll(sel).length, INTERACTIVE);
  for (let i = 0; i < total + 12; i++) await page.keyboard.press("Tab");
  return page.evaluate((sel) => {
    const w = window as unknown as { __focused: Set<Element> };
    const groups = new Set<Element>();
    for (const el of w.__focused) {
      const g = el.closest('[role="radiogroup"], [role="tablist"]');
      if (g) groups.add(g);
    }
    const missing: string[] = [];
    for (const el of document.querySelectorAll<HTMLElement>(sel)) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height || getComputedStyle(el).visibility === "hidden" || el.closest("[aria-hidden=true], [inert]")) continue;
      if (el.matches('input[type="file"]')) continue; // hidden pickers, opened by their buttons
      // Radio groups and tab lists are one tab stop; the arrow keys move within them.
      const g = el.closest('[role="radiogroup"], [role="tablist"]');
      if (g && groups.has(g)) continue;
      if (!w.__focused.has(el)) missing.push(el.outerHTML.slice(0, 140));
    }
    return missing;
  }, INTERACTIVE);
}

for (const width of [390, 1440]) {
  for (const scheme of ["light", "dark"] as const) {
    test.describe(`${width}px ${scheme}`, () => {
      test.use({ viewport: { width, height: width < 600 ? 844 : 900 }, colorScheme: scheme });

      test("pages fit, are keyboard reachable, and are captured", async ({ page }) => {
        test.setTimeout(150_000);
        await signIn(page);
        for (const [name, path] of PAGES) {
          await page.goto(path);
          // (not "networkidle": the job stream stays open) — wait for skeletons to go.
          await page.waitForFunction(
            () => !!document.querySelector("main h1, main textarea") && !document.querySelector('.animate-pulse, [role="status"] > .animate-spin'),
            undefined,
            { timeout: 20_000 },
          );
          await page.waitForTimeout(400); // charts and fonts settle
          const [sw, cw] = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
          expect(sw, `${name} scrolls horizontally at ${width}px`).toBeLessThanOrEqual(cw);
          // Chart tick labels must sit inside their chart (e.g. "Cross-doc" at 390 px).
          const clipped = await page.evaluate(() =>
            [...document.querySelectorAll<SVGTextElement>("text.recharts-cartesian-axis-tick-value")]
              .filter((t) => {
                const svg = t.closest("svg")!.getBoundingClientRect();
                const r = t.getBoundingClientRect();
                return r.width > 0 && (r.left < svg.left - 1 || r.right > svg.right + 1);
              })
              .map((t) => t.textContent),
          );
          expect(clipped, `${name}: clipped chart labels at ${width}px`).toEqual([]);
          // Every tab is fully on screen (no tab hidden past the edge of a scrolling strip).
          const offscreenTabs = await page.evaluate(() =>
            [...document.querySelectorAll<HTMLElement>('[role="tab"]')]
              .filter((t) => {
                const r = t.getBoundingClientRect();
                const strip = t.parentElement!.getBoundingClientRect();
                return r.width > 0 && (r.left < 0 || r.right > window.innerWidth || r.left < strip.left - 1 || r.right > strip.right + 1);
              })
              .map((t) => t.textContent),
          );
          expect(offscreenTabs, `${name}: tabs out of view at ${width}px`).toEqual([]);
          await page.screenshot({ path: `e2e-screenshots/${name}-${width}-${scheme}.png`, fullPage: true });
          if (scheme === "light") expect(await unreachable(page), `${name} at ${width}px`).toEqual([]);
        }

        // A streamed answer and the viewer, the two views that matter most.
        await page.goto("/");
        await page.getByLabel("Ask about the house").fill("How old is the furnace?");
        await page.keyboard.press("Enter");
        await expect(page.getByTestId("assistant-message")).toBeVisible();
        await page.screenshot({ path: `e2e-screenshots/chat-answer-${width}-${scheme}.png`, fullPage: true });
        await page.getByTestId("assistant-message").getByRole("button", { name: /^Source 1:/ }).click();
        await expect(page.locator("mark[data-hl]").first()).toBeVisible({ timeout: 20_000 });
        await page.waitForTimeout(500);
        await page.screenshot({ path: `e2e-screenshots/viewer-${width}-${scheme}.png` });
        await page.keyboard.press("Escape");

        // The Account menu and the "Sign out everywhere" confirmation (cancelled here).
        if (width < 1024) await page.getByRole("button", { name: "Open menu" }).click();
        await page.getByRole("button", { name: "Account" }).click();
        const everywhere = page.getByRole("button", { name: "Sign out everywhere…" });
        await expect(everywhere).toBeVisible();
        await page.screenshot({ path: `e2e-screenshots/account-menu-${width}-${scheme}.png` });
        await everywhere.click();
        const confirm = page.getByRole("alertdialog", { name: "Sign out everywhere?" });
        await expect(confirm).toBeVisible();
        const box = (await confirm.boundingBox())!;
        expect(box.x >= 0 && box.x + box.width <= width, `confirm dialog fits at ${width}px`).toBe(true);
        await page.screenshot({ path: `e2e-screenshots/account-confirm-${width}-${scheme}.png` });
        await confirm.getByRole("button", { name: "Cancel" }).click();
        await expect(confirm).toBeHidden();

        await page.goto("/sign-in");
        await expect(page.getByLabel("Passcode")).toBeVisible();
        await page.screenshot({ path: `e2e-screenshots/sign-in-${width}-${scheme}.png` });
      });
    });
  }
}
