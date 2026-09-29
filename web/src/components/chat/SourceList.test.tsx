import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Citation } from "@/api/types";
import { SourceList } from "./SourceList";

const source = (over: Partial<Citation> = {}): Citation =>
  ({ marker: "S1", chunk_id: 1, document_id: 1, file_sha: "a".repeat(64), page: 2, title: "Home inspection report", doc_type: "inspection_report", doc_date: "2023-06-22", vendor: "Sample Inspections Co.", snippet: "", ...over }) as Citation;

describe("SourceList", () => {
  it("keeps the date and the page each on one line", () => {
    render(<SourceList sources={[source()]} onOpen={() => {}} />);
    const meta = screen.getByTestId("source-meta");
    expect(meta).toHaveTextContent("Inspection report · Sample Inspections Co. · Jun 22, 2023 · p. 2");
    const nowrap = [...meta.querySelectorAll(".whitespace-nowrap")].map((e) => e.textContent);
    expect(nowrap).toEqual(["· Jun 22, 2023", "· p. 2"]);
  });
  it("leaves out a missing vendor and date", () => {
    render(<SourceList sources={[source({ vendor: null, doc_date: null, doc_type: "manual" })]} onOpen={() => {}} />);
    expect(screen.getByTestId("source-meta")).toHaveTextContent(/^Manual · p\. 2$/);
  });
});
