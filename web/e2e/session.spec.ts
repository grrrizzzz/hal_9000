/**
 * "Sign out everywhere" revokes every session on the server, so this spec runs
 * in its own Playwright project after all the others (see playwright.config.ts):
 * on a real server it would sign the parallel tests out too.
 */
import { expect, test } from "@playwright/test";
import { signIn } from "./fixtures";

const REAL_SERVER = !!process.env.E2E_BASE_URL;

test("sign out everywhere asks first, then revokes every session", async ({ page, browser }) => {
  await signIn(page, "/library");
  // A second, independent browser (another device) signed in at the same time.
  const other = await browser.newContext();
  const otherPage = await other.newPage();
  await signIn(otherPage, "/library");

  const openConfirm = async () => {
    await page.getByRole("button", { name: "Account" }).click();
    await page.getByRole("button", { name: "Sign out everywhere…" }).click();
    return page.getByRole("alertdialog", { name: "Sign out everywhere?" });
  };

  // Cancel keeps you signed in.
  let dialog = await openConfirm();
  await expect(dialog).toContainText("Every signed-in browser and phone");
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toBeHidden();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Library" })).toBeVisible();

  // Confirm: DELETE /api/session?everywhere=true, then back to Sign in.
  dialog = await openConfirm();
  const revoke = page.waitForRequest((r) => r.method() === "DELETE" && new URL(r.url()).pathname === "/api/session" && new URL(r.url()).searchParams.get("everywhere") === "true");
  await dialog.getByRole("button", { name: "Sign out everywhere" }).click();
  await revoke;
  await expect(page).toHaveURL(/\/sign-in/);
  await page.goto("/library");
  await expect(page).toHaveURL(/\/sign-in\?next=%2Flibrary$/);

  // The other browser's session is gone too. (The mocks keep state per page,
  // so only a real server can show this.)
  if (REAL_SERVER) {
    await otherPage.reload();
    await expect(otherPage).toHaveURL(/\/sign-in\?next=%2Flibrary$/);
  }
  await other.close();
});
