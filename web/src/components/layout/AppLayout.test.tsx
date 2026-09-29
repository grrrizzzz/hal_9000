import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { db } from "@/mocks/db";
import { renderApp } from "@/test/render";

describe("App layout", () => {
  it("shows the retrieval-only banner when the server reports the monthly cap is reached", async () => {
    db.health.retrievalOnly = true;
    renderApp("/library");
    expect(await screen.findByTestId("banner-retrieval-only")).toHaveTextContent("Retrieval-only mode");
    expect(screen.getAllByText("$10.00").length).toBeGreaterThan(0);
  });

  it("shows no banner in normal operation", async () => {
    renderApp("/library");
    await screen.findByRole("heading", { name: "Library" });
    await screen.findAllByText("$1.84");
    expect(screen.queryByTestId("banner-retrieval-only")).toBeNull();
  });
});
