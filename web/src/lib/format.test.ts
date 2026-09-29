import { describe, expect, it } from "vitest";
import { formatDocDate, formatMs, formatUsd } from "./format";

describe("formatters", () => {
  it("formats money without showing a string of zeros", () => {
    expect(formatUsd(0)).toBe("$0");
    expect(formatUsd(0.0000003)).toBe("$0.0000003");
    expect(formatUsd(0.000033)).toBe("$0.000033");
    expect(formatUsd(0.00012)).toBe("$0.00012");
    expect(formatUsd(0.00041)).toBe("$0.00041");
    expect(formatUsd(0.0041)).toBe("$0.0041");
    expect(formatUsd(0.00999)).toBe("$0.01");
    expect(formatUsd(1.844)).toBe("$1.84");
    expect(formatUsd(null)).toBe("—");
  });
  it("formats durations", () => {
    expect(formatMs(0.6)).toBe("0.60 ms");
    expect(formatMs(612.4)).toBe("612 ms");
    expect(formatMs(2003)).toBe("2.00 s");
  });
  it("shows date-only values without timezone drift", () => {
    expect(formatDocDate("2023-06-22")).toMatch(/22/);
    expect(formatDocDate(null)).toBe("");
  });
});
