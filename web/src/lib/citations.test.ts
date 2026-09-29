import { describe, expect, it } from "vitest";
import { splitCitations } from "./citations";

describe("splitCitations", () => {
  it("splits markers out of text", () => {
    expect(splitCitations("Gas furnace [S1]. Roof [S2, S3].")).toEqual([
      { kind: "text", text: "Gas furnace " },
      { kind: "cite", marker: "S1" },
      { kind: "text", text: ". Roof " },
      { kind: "cite", marker: "S2" },
      { kind: "cite", marker: "S3" },
      { kind: "text", text: "." },
    ]);
  });
  it("leaves non-markers alone", () => {
    expect(splitCitations("[S0] [x] [S12]")).toEqual([
      { kind: "text", text: "[S0] [x] " },
      { kind: "cite", marker: "S12" },
    ]);
  });
});
