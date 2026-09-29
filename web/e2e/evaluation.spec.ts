import { expect, test } from "@playwright/test";
import { signIn } from "./fixtures";

test.beforeEach(async ({ page }) => {
  await signIn(page, "/evaluation?tab=review");
});

test("approve a gold item with the keyboard", async ({ page }) => {
  const question = page.getByTestId("review-question");
  await expect(question).toBeVisible();
  const first = (await question.textContent())!;
  // Tracked by id: on a real server other tests (or runs) can add a draft with the same question.
  const id = (await page.getByTestId("review-card").getAttribute("data-gold-id"))!;
  const inList = page.getByRole("list", { name: "Questions" }).locator(`li[data-gold-id="${id}"]`);

  // J/K move through the queue.
  await page.keyboard.press("j");
  await expect(question).not.toHaveText(first);
  await page.keyboard.press("k");
  await expect(question).toHaveText(first);
  await expect(page.getByTestId("review-card")).toHaveAttribute("data-gold-id", id);

  await page.keyboard.press("a");
  await expect(page.getByTestId("review-flash")).toContainText(`Approved: “${first}”`);
  await expect(inList).toHaveCount(0);

  await page.getByRole("radio", { name: "Approved" }).click();
  await expect(inList).toBeVisible();
  await expect(inList).toContainText(first);
});

test("label an answer on the Judge calibration tab", async ({ page }) => {
  await page.getByRole("tab", { name: "Judge calibration" }).click();
  const card = page.locator('[data-testid="judge-card"][data-labelled="false"]').first();
  await expect(card).toBeVisible();
  const before = await page.getByTestId("judge-progress").textContent();

  // The answer's [S1] is a citation chip that opens the cited page.
  await card.getByRole("button", { name: /^Source 1:/ }).first().click();
  await expect(page.getByTestId("doc-viewer").locator("mark[data-hl]").first()).toBeVisible({ timeout: 20_000 });
  await page.keyboard.press("Escape");

  await card.getByRole("radio", { name: "Correct", exact: true }).click();
  await card.getByRole("radiogroup", { name: /faithful/ }).getByRole("radio", { name: "Yes" }).click();
  await card.getByRole("radiogroup", { name: /citations correct/ }).getByRole("radio", { name: "Yes" }).click();
  await card.getByRole("button", { name: "Save labels" }).click();

  // Once saved, the judge's labels are revealed next to yours.
  const labelled = page.locator('[data-testid="judge-card"][data-labelled="true"]').filter({ has: page.getByTestId("judge-comparison") });
  await expect(labelled).toHaveCount(2);
  await expect(page.getByTestId("judge-progress")).not.toHaveText(before!);
});

test("sweep charts and the profile switch", async ({ page }) => {
  await page.getByRole("tab", { name: "Sweep runs" }).click();
  for (const id of ["chart-recall-curve", "chart-pareto", "chart-categories"]) {
    await expect(page.getByTestId(id).locator("svg.recharts-surface").first()).toBeVisible();
  }
  // Every chart has a table view.
  await page.getByTestId("chart-pareto").getByRole("radio", { name: "Table" }).click();
  await expect(page.getByTestId("chart-pareto").getByRole("table")).toBeVisible();

  // A run opens with its per-question drill-down, paged.
  await page.getByRole("button", { name: /Structure-aware · 512/ }).first().click();
  const results = page.getByTestId("run-results");
  await expect(results.getByText(/^1–20 of \d+$/)).toBeVisible();
  await results.getByRole("button", { name: "Next questions" }).click();
  await expect(results.getByText(/^21–40 of \d+$/)).toBeVisible();
  await page.keyboard.press("Escape");

  await page.getByRole("tab", { name: "Index profile" }).click();
  await page.getByRole("button", { name: "Switch to this profile" }).first().click();
  await page.getByRole("button", { name: "Switch and re-index" }).click();
  await expect(page.getByText(/Re-indexing in the background|Re-index finished/)).toBeVisible();
});

test("re-select evidence by highlighting text on the page", async ({ page }) => {
  const card = page.getByTestId("review-card");
  await expect(card).toBeVisible();
  await card.getByRole("button", { name: "Re-select" }).first().click();

  const viewer = page.getByTestId("doc-viewer");
  const span = viewer.locator(".textLayer span", { hasText: "flexible gas connector" });
  await expect(span).toBeVisible({ timeout: 20_000 });
  const use = page.getByRole("button", { name: "Use selection as evidence" });
  await expect(use).toBeDisabled();

  // Select the line in the pdf.js text layer, as a drag-select would.
  await span.evaluate((el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection()!;
    sel.removeAllRanges();
    sel.addRange(range);
  });
  await expect(use).toBeEnabled();
  await use.click();

  await expect(page.getByTestId("review-flash")).toContainText("Evidence updated");
  await expect(card.locator("blockquote").first()).toHaveText("A flexible gas connector is routed through the cabinet wall.");
});
