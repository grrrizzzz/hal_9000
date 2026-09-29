/**
 * Shared setup for the end-to-end suite. It runs against the MSW-mocked dev
 * server by default, or any real server via E2E_BASE_URL (e.g. `kb serve --fake`).
 * The fake-mode trigger phrases below must match the server's fake mode.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { expect, type Page } from "@playwright/test";
import { syntheticPdf } from "./pdf";

export const PASSCODE = process.env.E2E_PASSCODE ?? "pod-bay-doors";

export const QUESTIONS = {
  plain: "How old is the furnace?",
  reset: "How much was the termite repair? (force a reset)",
  retrievalOnly: "Where is the furnace filter? (retrieval only)",
  embeddingDown: "Where is the furnace? (embedding down)",
  unverified: "When was the furnace installed? (unverified)",
};

export async function signIn(page: Page, next = "/"): Promise<void> {
  await page.goto(`/sign-in?next=${encodeURIComponent(next)}`);
  await page.getByLabel("Passcode").fill(PASSCODE);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(new RegExp(`${next.replace(/[?]/g, "\\?")}$`));
}

/** Write synthetic files (never house data) under `root`. */
export function writeTree(root: string, files: Record<string, string | Buffer>): void {
  for (const [rel, body] of Object.entries(files)) {
    const p = join(root, rel);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, body);
  }
}

/**
 * Makes this run's uploads new to the server: a real server reports bytes it
 * already holds as "Already in library", so a re-run (or a retry) against the
 * same `kb serve --fake` would otherwise never see "Indexed".
 */
export const RUN = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const tinyPdf = syntheticPdf(["Synthetic inspection summary", "Property: 12 Example Lane, Testville", "The roof covering is in fair condition.", `Fixture run ${RUN}`]);
const tinyJpg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46, 0, 1, 0xff, 0xd9]);

/**
 * A nested folder of 6 indexable fixture files plus 2 images (handoff KWP6/KWP7).
 * On a real server these stay in the library for the rest of the run, so they
 * avoid the chat questions' subjects (furnace, termite repair) and figures.
 */
export const NESTED_FOLDER: Record<string, string | Buffer> = {
  "House/inspection-summary.pdf": tinyPdf,
  "House/Appliances/Dishwasher/manual.md": `# Dishwasher\n\nRinse the spray arms every month.\n\nFixture run ${RUN}.\n`,
  "House/Appliances/Dishwasher/warranty.txt": `Synthetic warranty, ten years. Fixture run ${RUN}.\n`,
  "House/Appliances/Water heater/notes.html": `<html><body><p>50 gallon gas unit.</p><p>Fixture run ${RUN}.</p></body></html>`,
  "House/Receipts/2023/gutter-invoice.txt": `Invoice 42: gutter cleaning, $310.00\nFixture run ${RUN}.\n`,
  "House/Receipts/2023/pest-report.md": `Pest report: no activity.\n\nFixture run ${RUN}.\n`,
  "House/Photos/front.jpg": tinyJpg,
  "House/Photos/back.jpg": tinyJpg,
};

/** Drop loose files onto an element, the way a browser drop does (DataTransfer + drop event). */
export async function dropFiles(page: Page, selector: string, files: { name: string; body: string; type?: string }[]): Promise<void> {
  const dt = await page.evaluateHandle((list) => {
    const d = new DataTransfer();
    for (const f of list) d.items.add(new File([f.body], f.name, { type: f.type ?? "" }));
    return d;
  }, files);
  await page.dispatchEvent(selector, "dragenter", { dataTransfer: dt });
  await page.dispatchEvent(selector, "dragover", { dataTransfer: dt });
  await page.dispatchEvent(selector, "drop", { dataTransfer: dt });
}
