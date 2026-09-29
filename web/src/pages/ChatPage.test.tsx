import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { db } from "@/mocks/db";
import { renderApp } from "@/test/render";

async function ask(text: string) {
  const user = userEvent.setup();
  await user.type(await screen.findByLabelText("Ask about the house"), `${text}{Enter}`);
  return user;
}

describe("Chat", () => {
  it("streams an answer with citation chips, sources and the verified badge", async () => {
    renderApp("/");
    await ask("How old is the furnace?");
    const answer = await screen.findByTestId("assistant-message", {}, { timeout: 5000 });
    expect(answer).toHaveTextContent("installed in 2009");
    expect(within(answer).getByRole("button", { name: /Source 1: Home inspection report, page 2/ })).toBeInTheDocument();
    expect(within(answer).getByTestId("badge-verified")).toBeInTheDocument();
    // The conversation now has a URL-addressable history entry.
    await waitFor(() => expect(db.conversations).toHaveLength(1));
  });

  it("after a reset only the regenerated answer is shown", async () => {
    renderApp("/");
    await ask("termite cost? force a reset");
    const answer = await screen.findByTestId("assistant-message", {}, { timeout: 5000 });
    expect(answer).toHaveTextContent("$1,234.00");
    expect(answer).not.toHaveTextContent("$1,324.00");
  });

  it("shows the retrieval-only banner and passages instead of an answer", async () => {
    renderApp("/");
    await ask("filter? retrieval only");
    const answer = await screen.findByTestId("assistant-message", {}, { timeout: 5000 });
    expect(within(answer).getByTestId("banner-retrieval-only")).toBeInTheDocument();
    expect(answer).toHaveAttribute("data-outcome", "retrieval_only");
    expect(within(answer).getByText(/Replace the air filter every 90 days/)).toBeInTheDocument();
  });

  it("sends thumbs-down feedback with a comment", async () => {
    renderApp("/");
    const user = await ask("How old is the furnace?");
    const answer = await screen.findByTestId("assistant-message", {}, { timeout: 5000 });
    await user.click(within(answer).getByRole("button", { name: "Bad answer" }));
    await user.type(await screen.findByLabelText(/What was wrong/), "Wrong year");
    await user.click(screen.getByRole("button", { name: "Send feedback" }));
    await waitFor(() => expect(db.feedback).toEqual([expect.objectContaining({ rating: -1, comment: "Wrong year", add_to_gold: false })]));
  });

  it("switching conversation mid-stream keeps the streaming turn in its own conversation", async () => {
    const now = new Date().toISOString();
    db.conversations.push({
      id: 50,
      title: "Earlier roof question",
      created_at: now,
      updated_at: now,
      messages: [
        { id: 51, role: "user", content: "Earlier roof question", citations: [], trace_id: null, outcome: null, created_at: now },
        { id: 52, role: "assistant", content: "The roof is composition shingle.", citations: [], trace_id: null, outcome: "answered", created_at: now },
      ],
    });
    renderApp("/");
    db.tokenDelayMs = 25;
    const user = await ask("How old is the furnace?");
    await screen.findByTestId("assistant-streaming");

    // Leave for the other conversation while the answer is still streaming.
    await user.click(await screen.findByRole("link", { name: /Earlier roof question/ }));
    expect(await screen.findByText("The roof is composition shingle.")).toBeInTheDocument();
    expect(screen.queryByTestId("assistant-streaming")).toBeNull();
    expect(screen.queryByText("How old is the furnace?", { selector: "[data-testid=user-message]" })).toBeNull();

    // The stream finishes in the background, in its own new conversation; we stay put.
    await waitFor(() => expect(db.conversations.find((c) => c.id !== 50)?.messages).toHaveLength(2), { timeout: 8000 });
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.getByText("The roof is composition shingle.")).toBeInTheDocument();
    expect(screen.queryByTestId("badge-verified")).toBeNull();

    // Opening that conversation shows the finished turn, badge included.
    await user.click(await screen.findByRole("link", { name: /How old is the furnace\?/ }));
    const answer = await screen.findByTestId("assistant-message");
    expect(answer).toHaveTextContent("installed in 2009");
    expect(within(answer).getByTestId("badge-verified")).toBeInTheDocument();
    expect(screen.queryByText("The roof is composition shingle.")).toBeNull();
  });

  it("add to question set after a thumbs down updates the same feedback row", async () => {
    renderApp("/");
    const user = await ask("How old is the furnace?");
    const answer = await screen.findByTestId("assistant-message");
    await user.click(within(answer).getByRole("button", { name: "Bad answer" }));
    await user.click(await screen.findByRole("button", { name: "Send feedback" }));
    await waitFor(() => expect(db.feedback).toHaveLength(1));
    await user.click(within(answer).getByRole("button", { name: "Add to question set" }));
    await within(answer).findByRole("link", { name: "In the review queue" });
    expect(db.feedback).toEqual([expect.objectContaining({ rating: -1, add_to_gold: true })]);
    expect(db.gold.filter((g) => g.origin === "feedback")).toHaveLength(1);
  });

  it("add to question set without a rating records no rating", async () => {
    renderApp("/");
    const user = await ask("How old is the furnace?");
    const answer = await screen.findByTestId("assistant-message");
    await user.click(within(answer).getByRole("button", { name: "Add to question set" }));
    await within(answer).findByRole("link", { name: "In the review queue" });
    expect(db.feedback).toEqual([expect.objectContaining({ rating: null, add_to_gold: true })]);
  });
});
