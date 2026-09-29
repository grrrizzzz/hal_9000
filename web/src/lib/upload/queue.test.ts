import { describe, expect, it } from "vitest";
import { runPool, TaskQueue, UPLOAD_CONCURRENCY } from "./queue";

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

describe("TaskQueue", () => {
  it("shares one limit across batches added at different times, and retries", async () => {
    const q = new TaskQueue(UPLOAD_CONCURRENCY);
    let inFlight = 0;
    let peak = 0;
    const task = (ms: number) => () =>
      new Promise<void>((r) => {
        inFlight++;
        peak = Math.max(peak, inFlight);
        setTimeout(() => {
          inFlight--;
          r();
        }, ms);
      });
    const first = [1, 2, 3, 4].map(() => q.run(task(15)));
    await new Promise((r) => setTimeout(r, 3));
    const second = [1, 2, 3].map(() => q.run(task(5))); // a second drop while the first is uploading
    const retry = q.run(task(5));
    expect(q.inFlight).toBe(3);
    await Promise.all([...first, ...second, retry]);
    expect(peak).toBe(3);
    expect(q.inFlight).toBe(0);
  });

  it("keeps going after a failed task", async () => {
    const q = new TaskQueue(1);
    await expect(q.run(() => Promise.reject(new Error("boom")))).rejects.toThrow("boom");
    await expect(q.run(() => Promise.resolve(7))).resolves.toBe(7);
  });
});
