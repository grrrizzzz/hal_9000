import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { db } from "@/mocks/db";
import { renderApp } from "@/test/render";

describe("Gold review", () => {
  it("approves with A, rejects with R and moves with J/K", async () => {
    renderApp("/evaluation?tab=review");
    const q = await screen.findByTestId("review-question");
    expect(q).toHaveTextContent("How old is the furnace according to the inspection?");

    fireEvent.keyDown(window, { key: "j" });
    await waitFor(() => expect(screen.getByTestId("review-question")).toHaveTextContent("gas line"));
    fireEvent.keyDown(window, { key: "k" });
    await waitFor(() => expect(screen.getByTestId("review-question")).toHaveTextContent("furnace"));

    fireEvent.keyDown(window, { key: "a" });
    await waitFor(() => expect(db.gold.find((g) => g.id === 1)?.status).toBe("approved"));
    // The next draft is selected, and the approved one leaves the queue.
    await waitFor(() => expect(screen.getByTestId("review-question")).toHaveTextContent("gas line"));
    await waitFor(() => expect(within(screen.getByRole("list", { name: "Questions" })).queryByText(/furnace according/)).toBeNull());

    fireEvent.keyDown(window, { key: "r" });
    await waitFor(() => expect(db.gold.find((g) => g.id === 2)?.status).toBe("rejected"));
  });

  it("ignores shortcuts while typing", async () => {
    renderApp("/evaluation?tab=review");
    await screen.findByTestId("review-question");
    fireEvent.keyDown(window, { key: "e" });
    const input = await screen.findByLabelText("Question");
    fireEvent.keyDown(input, { key: "a" });
    await new Promise((r) => setTimeout(r, 50));
    expect(db.gold.find((g) => g.id === 1)?.status).toBe("draft");
  });
});
