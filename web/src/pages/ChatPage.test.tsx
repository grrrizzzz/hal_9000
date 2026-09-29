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
});
