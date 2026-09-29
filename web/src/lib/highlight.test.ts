import { describe, expect, it } from "vitest";
import { highlightRanges, locate, normalize, renderItem, textRanges } from "./highlight";

describe("highlight matching", () => {
  it("matches cleaned text against a letter-spaced heading split into items", () => {
    // pdf.js gives one item per glyph for a letter-spaced heading.
    const items = [..."DESCRIPTION"].map((c) => c).concat([
      "The furnace is a forced-air gas unit located in the garage.",
      "It was installed in 2009 and is near the end of its expected life.",
      "Page 2 of 3",
    ]);
    const chunk = "DESCRIPTION The furnace is a forced-air gas unit located in the garage. It was installed in 2009";
    const ranges = highlightRanges(items, [chunk]);
    // every heading glyph and both body lines are highlighted, the footer is not
    for (let i = 0; i < 11; i++) expect(ranges.get(i)).toEqual([[0, 1]]);
    expect(ranges.get(11)).toEqual([[0, items[11].length - 1]]); // up to the final "e" before "."
    expect(ranges.get(12)?.[0][0]).toBe(0);
    expect(ranges.has(13)).toBe(false);
  });

  it("tolerates small differences between cleaned and raw text", () => {
    const hay = normalize("Recom- mend evaluation by a licensed HVAC contractor. Page 2 of 3 The furnace");
    const needle = normalize("Recommend evaluation by a licensed HVAC contractor. The furnace");
    const m = locate(hay, needle);
    expect(m).not.toBeNull();
    expect(m!.score).toBeLessThan(1);
    expect(hay.slice(m!.start, m!.end)).toContain("licensedhvaccontractor");
  });

  it("returns nothing for text that isn't on the page", () => {
    expect(locate(normalize("The roof is composition shingle."), normalize("termite damage at the rear deck ledger"))).toBeNull();
  });

  it("escapes item text while marking it", () => {
    expect(renderItem("a <b> c", [[2, 5]])).toBe('a <mark data-hl="1">&lt;b&gt;</mark> c');
  });

  it("finds ranges in plain text for the text viewer", () => {
    const text = "Line one.\nThe  water heater is 50 gallon.\nEnd.";
    const [r] = textRanges(text, ["the water heater is 50 gallon"]);
    expect(text.slice(r[0], r[1])).toBe("The  water heater is 50 gallon");
  });
});
