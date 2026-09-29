import { expect, test } from "@playwright/test";
import { dropFiles, NESTED_FOLDER, signIn, writeTree } from "./fixtures";

test("pick a nested folder: summary skips the images, jobs reach indexed", async ({ page }, info) => {
  const root = info.outputPath("pick");
  writeTree(root, NESTED_FOLDER);
  await signIn(page, "/upload");

  // Playwright ≥ 1.45 uploads a whole directory into <input webkitdirectory>.
  await page.getByTestId("folder-input").setInputFiles(`${root}/House`);

  const summary = page.getByTestId("summary-text");
  await expect(summary).toHaveText("8 files found: 6 will be added, 2 skipped (.jpg ×2)");
  await expect(page.getByText("Images are skipped")).toBeVisible();

  // The expandable list keeps each file's relative path.
  await page.getByText("Show all 8 files").click();
  await expect(page.getByTestId("upload-summary").getByText("House/Appliances/Furnace/manual.md")).toBeVisible();
  await expect(page.getByTestId("upload-summary").getByText("House/Photos/front.jpg")).toBeVisible();

  await page.getByRole("button", { name: "Upload 6 files" }).click();
  const rows = page.getByTestId("upload-row");
  await expect(rows).toHaveCount(6);
  await expect(page.getByTestId("upload-row").filter({ hasText: "House/Receipts/2023/roof-invoice.txt" })).toBeVisible();
  // Every file goes through parse … index on the job stream and ends "Indexed".
  await expect(page.locator('[data-testid="upload-row"][data-status="Indexed"]')).toHaveCount(6, { timeout: 30_000 });
  await expect(page.getByText("6 of 6 indexed")).toBeVisible();

  // The new documents are in the Library, grouped by their folder collection.
  await page.getByRole("link", { name: "See them in the Library" }).click();
  await expect(page.getByRole("heading", { name: "House/Appliances/Furnace" })).toBeVisible();
});

test("drop loose files onto the drop zone", async ({ page }) => {
  await signIn(page, "/upload");
  await dropFiles(page, '[data-testid="drop-zone"]', [
    { name: "boiler-notes.txt", body: "Synthetic boiler notes." },
    { name: "deck.md", body: "# Deck\n\nStained in 2024." },
    { name: "photo.jpg", body: "x", type: "image/jpeg" },
  ]);
  await expect(page.getByTestId("summary-text")).toHaveText("3 files found: 2 will be added, 1 skipped (.jpg ×1)");
  await page.getByRole("button", { name: "Upload 2 files" }).click();
  await expect(page.locator('[data-testid="upload-row"][data-status="Indexed"]')).toHaveCount(2, { timeout: 30_000 });
});
