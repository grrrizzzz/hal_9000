import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderApp } from "@/test/render";

describe("Library", () => {
  it("shows the file page range, the document's own page count and singular/plural counts", async () => {
    renderApp("/library");
    const invoice = (await screen.findByRole("link", { name: "Invoice 1001 — termite repair" })).closest("li")!;
    // page_start/page_end are file pages; pages is the sub-document's own count.
    expect(within(invoice).getByText("p. 3")).toBeInTheDocument();
    expect(within(invoice).getByText("1 page")).toBeInTheDocument();
    expect(within(invoice).getByText("1 chunk")).toBeInTheDocument();
    expect(invoice).not.toHaveTextContent(/ of \d/);

    const report = screen.getByRole("link", { name: "Home inspection report" }).closest("li")!;
    expect(within(report).getByText("pp. 1–2")).toBeInTheDocument();
    expect(within(report).getByText("2 pages")).toBeInTheDocument();
    expect(within(report).getByText("4 chunks")).toBeInTheDocument();
  });
});
