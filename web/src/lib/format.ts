import type { DocType, Outcome } from "@/api/types";

const DOC_TYPE_LABEL: Record<DocType, string> = {
  inspection_report: "Inspection report",
  estimate: "Estimate",
  invoice: "Invoice",
  pest_report: "Pest report",
  completion_notice: "Completion notice",
  disclosure_form: "Disclosure form",
  manual: "Manual",
  warranty: "Warranty",
  permit: "Permit",
  other: "Other",
};

export const docTypeLabel = (t: DocType | null | undefined) => DOC_TYPE_LABEL[t ?? "other"] ?? "Other";

/** A date-only string (YYYY-MM-DD) shown without timezone drift: "Jun 22, 2023". */
export function formatDocDate(d: string | null | undefined): string {
  if (!d) return "";
  const [y, m, day] = d.split("-").map(Number);
  if (!y || !m) return d;
  return new Date(Date.UTC(y, m - 1, day || 1)).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

export function relativeTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "";
  const s = Math.round((now - new Date(iso).getTime()) / 1000);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const days = Math.round(h / 24);
  if (days < 7) return `${days} d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

export function formatUsd(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (v === 0) return "$0";
  if (v < 0.01) {
    // Two significant figures, however small: $0.00012, $0.000033.
    const decimals = Math.min(12, Math.max(2, 1 - Math.floor(Math.log10(v))));
    return `$${Number(v.toPrecision(2)).toFixed(decimals).replace(/(\.\d*?[1-9])0+$/, "$1")}`;
  }
  return `$${v.toFixed(2)}`;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10_240 ? 1 : 0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export const pct = (v: number | null | undefined, digits = 0) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`);

const OUTCOME_LABEL: Record<Outcome, string> = {
  answered: "Answered",
  abstained: "Not in documents",
  retrieval_only: "Sources only",
  error: "Error",
  cancelled: "Cancelled",
};
export const outcomeLabel = (o: Outcome | null | undefined) => (o ? OUTCOME_LABEL[o] : "—");
export const outcomeTone = (o: Outcome | null | undefined) =>
  o === "answered" ? "success" : o === "abstained" ? "neutral" : o === "retrieval_only" ? "warning" : o === "error" ? "danger" : "outline";

export const STAGE_LABEL: Record<string, string> = {
  condense: "Understanding the follow-up",
  embed_query: "Reading the question",
  vector: "Searching by meaning",
  bm25: "Searching by keyword",
  fuse: "Combining results",
  rerank: "Re-ranking",
  select: "Choosing passages",
  abstain: "Checking the evidence",
  generate: "Writing the answer",
  verify: "Checking figures and citations",
  screen: "Screening",
  store: "Storing",
  parse: "Reading the file",
  clean: "Cleaning text",
  segment: "Finding documents",
  chunk: "Chunking",
  embed: "Embedding",
  index: "Indexing",
  regression: "Regression check",
};

/** "1 chunk", "2 chunks". */
export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** A sub-document's file page range: "p. 3" or "pp. 3–5". */
export const pageRange = (start: number, end: number) => (start === end ? `p. ${start}` : `pp. ${start}–${end}`);
