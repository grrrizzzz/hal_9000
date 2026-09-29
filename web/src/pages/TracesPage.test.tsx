import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderApp } from "@/test/render";

describe("Traces", () => {
  it("lists questions with total time and cost, linking to the inspector", async () => {
    renderApp("/traces");
    const link = await screen.findByRole("link", { name: "How old is the furnace?" });
    expect(link.getAttribute("href")).toMatch(/^\/traces\/[0-9a-f]+$/);
    const rows = screen.getAllByTestId("trace-row");
    expect(rows).toHaveLength(6);
    expect(rows[0]).toHaveTextContent("2.00 s");
    expect(rows[0]).toHaveTextContent("$0.00041");
  });
});
