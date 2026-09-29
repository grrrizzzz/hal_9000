import { describe, expect, it } from "vitest";
import { runPool, UPLOAD_CONCURRENCY } from "./queue";

describe("runPool", () => {
  it("never runs more than three uploads at once and runs them all", async () => {
    let inFlight = 0;
    let peak = 0;
    const done: number[] = [];
    await runPool([...Array(10).keys()], UPLOAD_CONCURRENCY, async (i) => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await new Promise((r) => setTimeout(r, 5 + (i % 3) * 3));
      inFlight--;
      done.push(i);
    });
    expect(UPLOAD_CONCURRENCY).toBe(3);
    expect(peak).toBe(3);
    expect(done.sort((a, b) => a - b)).toEqual([...Array(10).keys()]);
  });
});
