import { describe, expect, it } from "vitest";
import type { EvalRun } from "@/api/types";
import { categoryRows, defaultCompare, paretoFrontier, paretoPoints, recallCurve } from "./run-data";

const ci = (v: number) => ({ value: v, low: v - 0.05, high: v + 0.05 });
function run(id: string, chunker: string, size: number, recall: number, latency: number, host = "mac"): EvalRun {
  return {
    id,
    stage: "S1a",
    split: "dev",
    config: { chunker, size },
    host,
    cost_usd: 0,
    started_at: "2026-09-20T00:00:00Z",
    finished_at: null,
    results_path: null,
    metrics: {
      n_questions: 10,
      budgeted_recall: { "2000": ci(recall) },
      full_support_hit: {},
      context_precision: {},
      latency_ms_p95: latency,
      cost_per_question_usd: 0.0001,
      by_category: { money: { "recall@2k": recall } },
    },
  };
}

describe("run shaping", () => {
  const runs = [
    run("a", "fixed", 256, 0.6, 20),
    run("b", "fixed", 512, 0.7, 30),
    run("b2", "fixed", 512, 0.72, 31), // best of the duplicates wins
    run("c", "structure", 512, 0.8, 25),
    run("p", "structure", 512, 0.79, 120, "pi-5"),
  ];

  it("builds one curve per strategy, ignoring Pi repeats", () => {
    const { rows, strategies } = recallCurve(runs, 2000);
    expect(strategies).toEqual(["fixed", "structure"]);
    expect(rows.map((r) => r.size)).toEqual([256, 512]);
    expect(rows[1].fixed).toBe(0.72);
    expect(rows[1].structure).toBe(0.8);
    expect(rows[0].structure).toBeNull();
    expect(rows[1].structure_ci).toEqual([0.75, 0.8500000000000001]);
  });

  it("marks Pi points and finds the frontier", () => {
    const pts = paretoPoints(runs);
    expect(pts.find((p) => p.id === "p")?.pi).toBe(true);
    expect(paretoFrontier(pts.filter((p) => !p.pi)).map((p) => p.id)).toEqual(["a", "c"]);
  });

  it("compares the Pi finalists by default and tabulates categories", () => {
    expect(defaultCompare(runs)).toEqual(["p"]);
    expect(categoryRows(runs.slice(0, 2), "recall@2k")).toEqual([{ category: "money", a: 0.6, b: 0.7 }]);
  });
});
