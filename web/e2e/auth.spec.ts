import { expect, test } from "@playwright/test";
import { PASSCODE } from "./fixtures";

test("signing in sets the session and returns to the page you asked for", async ({ page }) => {
  await page.goto("/library");
  await expect(page).toHaveURL(/\/sign-in\?next=%2Flibrary$/);

  await page.getByLabel("Passcode").fill("not-the-passcode");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page).toHaveURL(/\/sign-in/);

  await page.getByLabel("Passcode").fill(PASSCODE);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/library$/);
  await expect(page.getByRole("heading", { name: "Library" })).toBeVisible();

  // Signing out sends you back to Sign in.
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/sign-in/);
});
