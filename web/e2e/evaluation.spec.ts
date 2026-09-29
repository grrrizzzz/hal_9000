import { expect, test } from "@playwright/test";
import { signIn } from "./fixtures";

test.beforeEach(async ({ page }) => {
  await signIn(page, "/evaluation?tab=review");
});

test("approve a gold item with the keyboard", async ({ page }) => {
  const question = page.getByTestId("review-question");
  await expect(question).toBeVisible();
  const first = (await question.textContent())!;

  // J/K move through the queue.
  await page.keyboard.press("j");
  await expect(question).not.toHaveText(first);
  await page.keyboard.press("k");
  await expect(question).toHaveText(first);

  await page.keyboard.press("a");
  await expect(page.getByTestId("review-flash")).toContainText(`Approved: “${first}”`);
  await expect(page.getByRole("list", { name: "Questions" }).getByText(first)).toHaveCount(0);

  await page.getByRole("radio", { name: "Approved" }).click();
  await expect(page.getByRole("list", { name: "Questions" }).getByText(first)).toBeVisible();
});

test("label an answer on the Judge calibration tab", async ({ page }) => {
  await page.getByRole("tab", { name: "Judge calibration" }).click();
  const card = page.locator('[data-testid="judge-card"][data-labelled="false"]').first();
  await expect(card).toBeVisible();
  const before = await page.getByTestId("judge-progress").textContent();

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

  await page.getByRole("tab", { name: "Index profile" }).click();
  await page.getByRole("button", { name: "Switch to this profile" }).first().click();
  await page.getByRole("button", { name: "Switch and re-index" }).click();
  await expect(page.getByText(/Re-indexing in the background|Re-index finished/)).toBeVisible();
});
