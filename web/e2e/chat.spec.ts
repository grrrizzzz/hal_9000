import { expect, test, type Page } from "@playwright/test";
import { QUESTIONS, REAL_SERVER, signIn } from "./fixtures";

async function ask(page: Page, q: string) {
  await page.getByLabel("Ask about the house").fill(q);
  await page.getByRole("button", { name: "Send" }).click();
}

/**
 * Record every version of the answer text as it changes: the streaming bubble
 * (`final: false`) and then the finished answer (`final: true`), so the
 * regenerated text is always among the snapshots.
 */
async function recordAnswerText(page: Page) {
  await page.evaluate(() => {
    const w = window as unknown as { __answers: { text: string; final: boolean }[] };
    w.__answers = [];
    new MutationObserver(() => {
      const streaming = document.querySelector('[data-testid="assistant-streaming"] [data-testid="answer-text"]');
      const done = document.querySelector('[data-testid="assistant-message"] [data-testid="answer-text"]');
      const el = streaming ?? done;
      const last = w.__answers.at(-1);
      if (el?.textContent && (last?.text !== el.textContent || last.final !== !streaming)) w.__answers.push({ text: el.textContent, final: !streaming });
    }).observe(document.body, { subtree: true, childList: true, characterData: true });
  });
  return () => page.evaluate(() => (window as unknown as { __answers: { text: string; final: boolean }[] }).__answers);
}

test.beforeEach(async ({ page }) => {
  await signIn(page);
});

test("answers stream in, and [S1] opens the viewer at the cited page with a highlight", async ({ page }) => {
  const snapshots = await recordAnswerText(page);
  await ask(page, QUESTIONS.plain);
  const answer = page.getByTestId("assistant-message");
  await expect(answer).toContainText("installed in 2009");
  // Tokens arrived one by one: the streaming bubble showed growing partial text.
  const seen = (await snapshots()).filter((s) => !s.final).map((s) => s.text);
  expect(seen.length).toBeGreaterThan(3);
  expect(seen[0].length).toBeLessThan(seen.at(-1)!.length);
  await expect(answer.getByTestId("badge-verified")).toBeVisible();

  await answer.getByRole("button", { name: /^Source 1:/ }).click();
  const viewer = page.getByTestId("doc-viewer");
  await expect(viewer).toHaveAttribute("data-page", "2");
  await expect(viewer.getByText("Page 2 of 3")).toBeVisible();
  const marks = viewer.locator("mark[data-hl]");
  await expect(marks.first()).toBeVisible({ timeout: 20_000 });
  await expect(viewer).not.toHaveAttribute("data-highlights", /^(0)?$/);
  const texts = (await marks.allInnerTexts()).join(" ");
  expect(texts).toContain("forced-air gas unit");
  // The letter-spaced heading ("D E S C R I P T I O N") is matched too.
  expect(texts.replace(/\s+/g, "")).toContain("DESCRIPTION");
  await expect(viewer.getByTestId("highlight-status")).toHaveText("Cited passage highlighted");
});

test("a forced reset shows only the regenerated answer", async ({ page }) => {
  const snapshots = await recordAnswerText(page);
  const chat = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/chat" && r.request().method() === "POST");
  await ask(page, QUESTIONS.reset);
  const answer = page.getByTestId("assistant-message");
  await expect(answer).toContainText("$1,234.00");
  await expect(answer).not.toContainText("$1,324.00");

  // The server streamed a draft with the wrong figure, then `reset`, then the regenerated answer.
  const events = (await (await chat).text())
    .split("\n")
    .filter((l) => l.startsWith("data:"))
    .map((l) => JSON.parse(l.slice(5)) as { event: string; text?: string });
  const reset = events.findIndex((e) => e.event === "reset");
  expect(reset).toBeGreaterThan(0);
  expect(events.slice(0, reset).filter((e) => e.event === "token").map((e) => e.text).join("")).toContain("1,324");
  expect(events.slice(reset).filter((e) => e.event === "token").map((e) => e.text).join("")).toContain("1,234.00");

  // On screen: once the regenerated answer appeared (at the latest as the finished answer),
  // none of the draft came back.
  const seen = (await snapshots()).map((s) => s.text);
  const firstRegenerated = seen.findIndex((s) => s.includes("1,234"));
  expect(firstRegenerated).toBeGreaterThanOrEqual(0);
  expect(seen.slice(firstRegenerated).some((s) => s.includes("1,324"))).toBe(false);
  // The mocks pace every token, so there the draft must also have been painted before the
  // reset cleared it. (A real server's events can share one network read and render once.)
  if (!REAL_SERVER) expect(seen.slice(0, firstRegenerated).some((s) => s.includes("1,324"))).toBe(true);
});

test("thumbs down with a comment, then add the answer to the question set", async ({ page }) => {
  await ask(page, QUESTIONS.plain);
  const answer = page.getByTestId("assistant-message");
  await expect(answer).toContainText("2009");

  await answer.getByRole("button", { name: "Bad answer" }).click();
  await page.getByLabel(/What was wrong/).fill("It should mention the filter size.");
  await page.getByRole("button", { name: "Send feedback" }).click();
  await expect(answer.getByRole("button", { name: "Bad answer" })).toHaveAttribute("aria-pressed", "true");
  await expect(answer.getByText("Thanks — this helps tune the answers.")).toBeVisible();

  await answer.getByRole("button", { name: "Add to question set" }).click();
  const link = answer.getByRole("link", { name: "In the review queue" });
  await expect(link).toBeVisible();
  await link.click();
  await expect(page).toHaveURL(/\/evaluation\?tab=review/);
  await expect(page.getByTestId("review-question")).toHaveText(QUESTIONS.plain);
  await expect(page.getByTestId("review-card").getByText("from chat").first()).toBeVisible();
});

test("figures that fail verification get the badge", async ({ page }) => {
  await ask(page, QUESTIONS.unverified);
  await expect(page.getByTestId("badge-unverified")).toBeVisible();
  await page.getByTestId("badge-unverified").click();
  await expect(page.getByText("Some figures couldn't be matched to the cited sources.")).toBeVisible();
});

test("degraded banners: retrieval-only and embedding unavailable", async ({ page }) => {
  await ask(page, QUESTIONS.retrievalOnly);
  await expect(page.getByTestId("banner-retrieval-only")).toBeVisible();
  await expect(page.getByTestId("assistant-message")).toHaveAttribute("data-outcome", "retrieval_only");
  // The closest passages are listed as sources with their text.
  await expect(page.getByTestId("assistant-message").getByText("Replace the air filter every 90 days", { exact: false })).toBeVisible();

  await page.getByRole("button", { name: "New chat" }).click();
  await ask(page, QUESTIONS.embeddingDown);
  await expect(page.getByTestId("banner-degraded")).toBeVisible();
  await expect(page.getByTestId("banner-degraded")).toContainText(/keyword search/i);
});

test("the answer's i button opens the Query Inspector", async ({ page }) => {
  await ask(page, QUESTIONS.plain);
  await expect(page.getByTestId("assistant-message")).toContainText("2009");
  await page.getByRole("link", { name: "Inspect this answer" }).click();
  await expect(page).toHaveURL(/\/traces\/[0-9a-f]+$/);
  await expect(page.getByRole("heading", { name: QUESTIONS.plain })).toBeVisible();
  await expect(page.getByTestId("waterfall").getByText("generate")).toBeVisible();
  // And the Traces list shows the question.
  await page.getByRole("link", { name: "Traces" }).first().click();
  await expect(page.getByRole("link", { name: QUESTIONS.plain }).first()).toBeVisible();
});
