/** Shaping eval runs for the charts (plan §8.6). Pure, so it is unit-tested. */
import type { EvalRun, MetricCI } from "@/api/types";

export const STRATEGIES = ["fixed", "recursive", "structure", "parent_child"] as const;
export type Strategy = (typeof STRATEGIES)[number];
export const STRATEGY_LABEL: Record<Strategy, string> = {
  fixed: "Fixed",
  recursive: "Recursive",
  structure: "Structure-aware",
  parent_child: "Parent–child",
};

export const cfg = (r: EvalRun, key: string): unknown => (r.config as Record<string, unknown> | undefined)?.[key];
export const isPi = (r: EvalRun) => (r.host ?? "").toLowerCase().includes("pi");
export const recallAt = (r: EvalRun, budget: number): MetricCI | null =>
  (r.metrics?.budgeted_recall as Record<string, MetricCI> | undefined)?.[String(budget)] ?? null;

export function describeRun(r: EvalRun): string {
  const c = r.config as Record<string, unknown>;
  const parts: string[] = [];
  if (c.chunker) parts.push(STRATEGY_LABEL[c.chunker as Strategy] ?? String(c.chunker));
  if (c.size) parts.push(c.parent_size ? `${c.size}/${c.parent_size}` : String(c.size));
  if (c.overlap) parts.push(`overlap ${c.overlap}`);
  if (c.retrieval) parts.push(String(c.retrieval));
  if (c.rerank) parts.push(`rerank ${c.rerank}`);
  if (c.embed_model) parts.push(String(c.embed_model).replace("text-embedding-", ""));
  return parts.join(" · ") || r.id;
}

export interface CurveRow {
  size: number;
  [series: string]: number | [number, number] | null;
}

/**
 * Recall @ budget vs chunk size, one line (with a CI band) per strategy.
 * Pi runs are excluded (they repeat configurations). Where a (strategy, size)
 * has several runs, the best one is kept.
 */
export function recallCurve(runs: EvalRun[], budget: number): { rows: CurveRow[]; strategies: Strategy[] } {
  const best = new Map<string, MetricCI>();
  const sizes = new Set<number>();
  const present = new Set<Strategy>();
  for (const r of runs) {
    if (isPi(r)) continue;
    const s = cfg(r, "chunker") as Strategy;
    const size = Number(cfg(r, "size"));
    const m = recallAt(r, budget);
    if (!STRATEGIES.includes(s) || !size || !m) continue;
    const key = `${s}:${size}`;
    if (!best.has(key) || best.get(key)!.value < m.value) best.set(key, m);
    sizes.add(size);
    present.add(s);
  }
  const strategies = STRATEGIES.filter((s) => present.has(s));
  const rows = [...sizes]
    .sort((a, b) => a - b)
    .map((size) => {
      const row: CurveRow = { size };
      for (const s of strategies) {
        const m = best.get(`${s}:${size}`);
        row[s] = m ? m.value : null;
        row[`${s}_ci`] = m ? [m.low, m.high] : null;
      }
      return row;
    });
  return { rows, strategies };
}

export interface ParetoPoint {
  id: string;
  label: string;
  latency: number;
  recall: number;
  cost: number;
  pi: boolean;
}

export function paretoPoints(runs: EvalRun[], budget = 2000): ParetoPoint[] {
  const out: ParetoPoint[] = [];
  for (const r of runs) {
    const m = recallAt(r, budget);
    const latency = r.metrics?.latency_ms_p95 ?? r.metrics?.latency_ms_p50;
    if (!m || latency === null || latency === undefined) continue;
    out.push({ id: r.id, label: describeRun(r), latency, recall: m.value, cost: r.metrics?.cost_per_question_usd ?? 0, pi: isPi(r) });
  }
  return out;
}

/** Points no other point beats on both latency (lower) and recall (higher). */
export function paretoFrontier(points: ParetoPoint[]): ParetoPoint[] {
  const sorted = [...points].sort((a, b) => a.latency - b.latency || b.recall - a.recall);
  const front: ParetoPoint[] = [];
  let best = -Infinity;
  for (const p of sorted) {
    if (p.recall > best) {
      front.push(p);
      best = p.recall;
    }
  }
  return front;
}

export function categoryMetrics(runs: EvalRun[]): string[] {
  const keys = new Set<string>();
  for (const r of runs) for (const v of Object.values(r.metrics?.by_category ?? {})) for (const k of Object.keys(v)) keys.add(k);
  return [...keys].sort((a, b) => (a.includes("recall") ? -1 : 0) - (b.includes("recall") ? -1 : 0) || a.localeCompare(b));
}

export function categoryRows(runs: EvalRun[], metric: string): { category: string; [run: string]: string | number | null }[] {
  const cats = new Set<string>();
  for (const r of runs) for (const c of Object.keys(r.metrics?.by_category ?? {})) cats.add(c);
  return [...cats].map((category) => {
    const row: { category: string; [run: string]: string | number | null } = { category };
    for (const r of runs) row[r.id] = r.metrics?.by_category?.[category]?.[metric] ?? null;
    return row;
  });
}

/** Default comparison set: the Pi-measured finalists if any, else the best three by recall @ 2k. */
export function defaultCompare(runs: EvalRun[]): string[] {
  const pi = runs.filter(isPi).map((r) => r.id);
  if (pi.length) return pi.slice(0, 3);
  return [...runs]
    .sort((a, b) => (recallAt(b, 2000)?.value ?? 0) - (recallAt(a, 2000)?.value ?? 0))
    .slice(0, 3)
    .map((r) => r.id);
}
