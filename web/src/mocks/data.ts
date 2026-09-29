/**
 * Synthetic seed data for the mocks (rule K2: no house data, ever).
 * Everything here describes an imaginary house at "12 Example Lane".
 */
import type {
  DocumentSummary,
  EvalRun,
  GoldItem,
  IndexProfile,
  JudgeLabel,
  MetricCI,
  StageTiming,
  Trace,
} from "@/api/types";

export const SHA = {
  inspection: "a1".repeat(32),
  manual: "b2".repeat(32),
  warranty: "c3".repeat(32),
  permit: "d4".repeat(32),
};

export const MOCK_PASSCODE = "pod-bay-doors";

/** Fake-mode trigger phrases. Documented in web/README.md for KWP6's `kb serve --fake`. */
export const TRIGGERS = {
  reset: "force a reset",
  retrievalOnly: "retrieval only",
  embeddingDown: "embedding down",
  unverified: "unverified",
  unanswerable: "swimming pool",
};

export interface MockPage {
  text: string;
  has_text_layer: boolean;
  spans: { chunk_id: number; char_start: number; char_end: number }[];
}

function page(text: string, chunkIds: number[], has_text_layer = true): MockPage {
  // One chunk per paragraph (split on blank lines), in order.
  const spans: MockPage["spans"] = [];
  let pos = 0;
  const paras = text.split("\n\n");
  paras.forEach((p, i) => {
    const start = text.indexOf(p, pos);
    spans.push({ chunk_id: chunkIds[i] ?? chunkIds[chunkIds.length - 1], char_start: start, char_end: start + p.length });
    pos = start + p.length;
  });
  return { text, has_text_layer, spans };
}

/** Cleaned page text per file (what /api/files/{sha}/pages/{n} returns). */
export const PAGES: Record<string, MockPage[]> = {
  [SHA.inspection]: [
    page(
      "SYNTHETIC HOME INSPECTION REPORT\nProperty: 12 Example Lane, Testville\nInspection date: June 22, 2023\nInspector: Sample Inspections Co.\n\nSummary: The roof covering is composition shingle in fair condition. The water heater is a 50 gallon gas unit with a seismic strap.",
      [10, 11],
    ),
    page(
      "DESCRIPTION\nThe furnace is a forced-air gas unit located in the garage. It was installed in 2009 and is near the end of its expected life.\n\nA flexible gas connector is routed through the cabinet wall. Recommend evaluation by a licensed HVAC contractor.",
      [12, 13],
    ),
    page(
      "INVOICE 1001 - Brightline Pest Co.\nTermite damage repair at the rear deck ledger.\nTotal due: $1,234.00\nCompleted July 14, 2023.",
      [14],
    ),
  ],
  [SHA.manual]: [
    page(
      "# Furnace care\n\nReplace the air filter every 90 days, or monthly during heavy use. Use a MERV 8 filter sized 16x25x1.\n\nKeep the area around the unit clear by at least 30 inches.",
      [30, 31, 32],
    ),
    page(
      "## Troubleshooting\n\nIf the burners do not light, check that the gas valve is open and the thermostat is calling for heat.\n\nA blinking red status light with three flashes means the pressure switch did not close.",
      [33, 34, 35],
    ),
  ],
  [SHA.warranty]: [
    page(
      "Roof workmanship warranty\n\nAcme Roofing warrants the installation of the composition shingle roof at 12 Example Lane for ten years from completion on March 3, 2021.",
      [40, 41],
    ),
  ],
  [SHA.permit]: [],
};

export const MIME: Record<string, string> = {
  [SHA.inspection]: "application/pdf",
  [SHA.manual]: "text/markdown",
  [SHA.warranty]: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  [SHA.permit]: "application/pdf",
};

export function chunkText(chunkId: number): { sha: string; page: number; text: string } | null {
  for (const [sha, pages] of Object.entries(PAGES)) {
    for (let i = 0; i < pages.length; i++) {
      const s = pages[i].spans.find((sp) => sp.chunk_id === chunkId);
      if (s) return { sha, page: i + 1, text: pages[i].text.slice(s.char_start, s.char_end) };
    }
  }
  return null;
}

const ts = (d: string) => new Date(d).toISOString();

export function seedDocuments(): DocumentSummary[] {
  return [
    {
      document: {
        id: 1,
        file_sha: SHA.inspection,
        page_start: 1,
        page_end: 2,
        title: "Home inspection report",
        doc_type: "inspection_report",
        vendor: "Sample Inspections Co.",
        doc_date: "2023-06-22",
        meta_source: "auto",
      },
      original_name: "synthetic-house.pdf",
      relative_path: "Inspection/synthetic-house.pdf",
      collection: "Inspection",
      pages: 2,
      pages_without_text: 0,
      chunks: 4,
      status: "indexed",
    },
    {
      document: {
        id: 2,
        file_sha: SHA.inspection,
        page_start: 3,
        page_end: 3,
        title: "Invoice 1001 — termite repair",
        doc_type: "invoice",
        vendor: "Brightline Pest Co.",
        doc_date: "2023-07-14",
        meta_source: "user",
      },
      original_name: "synthetic-house.pdf",
      relative_path: "Inspection/synthetic-house.pdf",
      collection: "Inspection",
      pages: 1,
      pages_without_text: 0,
      chunks: 1,
      status: "indexed",
    },
    {
      document: {
        id: 3,
        file_sha: SHA.manual,
        page_start: 1,
        page_end: 2,
        title: "Furnace care and troubleshooting",
        doc_type: "manual",
        vendor: null,
        doc_date: null,
        meta_source: "auto",
      },
      original_name: "furnace-manual.md",
      relative_path: "Appliances/Furnace/furnace-manual.md",
      collection: "Appliances/Furnace",
      pages: 2,
      pages_without_text: 0,
      chunks: 6,
      status: "indexed",
    },
    {
      document: {
        id: 4,
        file_sha: SHA.warranty,
        page_start: 1,
        page_end: 1,
        title: "Roof workmanship warranty",
        doc_type: "warranty",
        vendor: "Acme Roofing",
        doc_date: "2021-03-03",
        meta_source: "auto",
      },
      original_name: "roof-warranty.docx",
      relative_path: "Warranties/roof-warranty.docx",
      collection: "Warranties",
      pages: 1,
      pages_without_text: 0,
      chunks: 2,
      status: "indexed",
    },
    {
      document: {
        id: 5,
        file_sha: SHA.permit,
        page_start: 1,
        page_end: 1,
        title: "scanned-permit.pdf",
        doc_type: "permit",
        vendor: null,
        doc_date: null,
        meta_source: "auto",
      },
      original_name: "scanned-permit.pdf",
      relative_path: "Permits/scanned-permit.pdf",
      collection: "Permits",
      pages: 4,
      pages_without_text: 4,
      chunks: 0,
      status: "failed",
    },
  ];
}

export function seedGold(): GoldItem[] {
  return [
    {
      id: 1,
      question: "How old is the furnace according to the inspection?",
      reference_answer: "It was installed in 2009, per the June 2023 inspection.",
      category: "date",
      difficulty: "easy",
      paraphrased: false,
      evidence: [{ quote: "It was installed in 2009 and is near the end of its expected life.", page: 2, file_sha: SHA.inspection }],
      status: "draft",
      split: "dev",
      reviewer_note: null,
      origin: "generated",
    },
    {
      id: 2,
      question: "Is anything wrong with the gas line near the kitchen?",
      reference_answer: "A flexible gas connector is routed through the cabinet wall; an HVAC evaluation was recommended.",
      category: "factoid",
      difficulty: "medium",
      paraphrased: true,
      evidence: [{ quote: "A flexible gas connector is routed through the cabinet wall.", page: 2, file_sha: SHA.inspection }],
      status: "draft",
      split: "dev",
      reviewer_note: null,
      origin: "generated",
    },
    {
      id: 3,
      question: "How much did the termite repair cost?",
      reference_answer: "$1,234.00 (Invoice 1001, Brightline Pest Co.).",
      category: "money",
      difficulty: "easy",
      paraphrased: false,
      evidence: [{ quote: "Total due: $1,234.00", page: 3, file_sha: SHA.inspection }],
      status: "draft",
      split: "test",
      reviewer_note: null,
      origin: "generated",
    },
    {
      id: 4,
      question: "Does the house have a swimming pool heater?",
      reference_answer: null,
      category: "unanswerable",
      difficulty: "medium",
      paraphrased: false,
      evidence: [],
      status: "draft",
      split: "dev",
      reviewer_note: null,
      origin: "generated",
    },
    {
      id: 5,
      question: "What type of roof covering does the house have?",
      reference_answer: "Composition shingle.",
      category: "factoid",
      difficulty: "easy",
      paraphrased: false,
      evidence: [{ quote: "The roof covering is composition shingle in fair condition.", page: 1, file_sha: SHA.inspection }],
      status: "approved",
      split: "dev",
      reviewer_note: null,
      origin: "generated",
    },
    {
      id: 6,
      question: "Who inspected the property?",
      reference_answer: "Sample Inspections Co.",
      category: "factoid",
      difficulty: "easy",
      paraphrased: false,
      evidence: [{ quote: "Inspector: Sample Inspections Co.", page: 1, file_sha: SHA.inspection }],
      status: "rejected",
      split: "dev",
      reviewer_note: "Too trivial",
      origin: "generated",
    },
  ];
}

export function seedJudgeLabels(): JudgeLabel[] {
  return [
    {
      id: 1,
      gold_item_id: 1,
      eval_run_id: "j-001",
      question: "How old is the furnace according to the inspection?",
      answer: "The June 2023 inspection says the furnace was installed in 2009 and is near the end of its expected life [S1].",
      human: null,
      judge: { correctness: "correct", faithful: true, citations_correct: true, abstention_correct: null },
    },
    {
      id: 2,
      gold_item_id: 3,
      eval_run_id: "j-001",
      question: "How much did the termite repair cost?",
      answer: "Invoice 1001 from Brightline Pest Co. totals $1,234.00 [S1].",
      human: null,
      judge: { correctness: "correct", faithful: true, citations_correct: true, abstention_correct: null },
    },
    {
      id: 3,
      gold_item_id: 2,
      eval_run_id: "j-001",
      question: "Is anything wrong with the gas line near the kitchen?",
      answer: "The inspection found no issues with the gas line [S1].",
      human: { correctness: "wrong", faithful: false, citations_correct: false, abstention_correct: null },
      judge: { correctness: "wrong", faithful: false, citations_correct: true, abstention_correct: null },
    },
    {
      id: 4,
      gold_item_id: 4,
      eval_run_id: "j-001",
      question: "Does the house have a swimming pool heater?",
      answer: "I couldn't find that in the house documents.",
      human: null,
      judge: { correctness: "correct", faithful: true, citations_correct: true, abstention_correct: true },
    },
  ];
}

export function seedProfiles(): IndexProfile[] {
  const base = { parser: "pypdfium2", header: true, embed_model: "text-embedding-3-small", dimensions: 1536 };
  return [
    { id: 1, name: "structure-512-h", chunker: "structure", size: 512, overlap: 0, parent_size: null, active: true, ...base },
    { id: 2, name: "recursive-384-o15-h", chunker: "recursive", size: 384, overlap: 58, parent_size: null, active: false, ...base },
    { id: 3, name: "pc-256-1500-h", chunker: "parent_child", size: 256, overlap: 0, parent_size: 1500, active: false, ...base },
  ];
}

// ------------------------------------------------------------ eval runs
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const ci = (value: number, width: number): MetricCI => ({
  value: +value.toFixed(3),
  low: +Math.max(0, value - width).toFixed(3),
  high: +Math.min(1, value + width * 0.9).toFixed(3),
});

const CATEGORIES = ["factoid", "money", "date", "list", "cross_document"];

export function seedRuns(): EvalRun[] {
  const r = rng(42);
  const runs: EvalRun[] = [];
  const sizes = [128, 256, 384, 512, 768, 1024, 1536];
  const strategies: [string, number][] = [
    ["fixed", 0],
    ["recursive", 0.025],
    ["structure", 0.05],
  ];
  let n = 0;
  const mk = (chunker: string, size: number, lift: number, host = "mac", stage = "S1a", extra: Record<string, unknown> = {}) => {
    // A curve that peaks in the middle (plan §8.3 T1).
    const peak = 0.78 + lift - Math.pow(Math.log2(size / 480), 2) * 0.05;
    const noise = (r() - 0.5) * 0.02;
    const recall2k = Math.max(0.3, Math.min(0.97, peak + noise));
    const latency = 18 + size / 60 + r() * 12 + (chunker === "parent_child" ? 9 : 0);
    const id = `${stage.toLowerCase()}-${String(++n).padStart(3, "0")}`;
    const budgeted: Record<string, MetricCI> = {
      "1000": ci(recall2k - 0.08 - size / 20000, 0.06),
      "2000": ci(recall2k, 0.05),
      "4000": ci(Math.min(0.99, recall2k + 0.05), 0.045),
    };
    if (size > 1000) delete budgeted["1000"];
    const byCat: Record<string, Record<string, number>> = {};
    for (const c of CATEGORIES) {
      const bias = c === "cross_document" ? -0.14 : c === "list" ? -0.08 : c === "money" ? 0.04 : 0;
      byCat[c] = {
        "recall@2k": +Math.max(0.2, Math.min(0.99, recall2k + bias + (r() - 0.5) * 0.06)).toFixed(3),
        mrr: +Math.max(0.2, Math.min(0.99, recall2k - 0.05 + bias + (r() - 0.5) * 0.08)).toFixed(3),
      };
    }
    runs.push({
      id,
      stage,
      split: "dev",
      config: { chunker, size, overlap: chunker === "fixed" ? Math.round(size * 0.15) : 0, retrieval: "hybrid", embed_model: "text-embedding-3-small", ...extra },
      metrics: {
        n_questions: 58,
        budgeted_recall: budgeted,
        full_support_hit: { "5": ci(recall2k - 0.1, 0.07) },
        mrr: ci(recall2k - 0.12, 0.06),
        context_precision: { "2000": ci(0.4 - size / 5000, 0.04) },
        abstain_auc: ci(0.82, 0.06),
        latency_ms_p50: +(host === "pi" ? latency * 3.2 : latency).toFixed(1),
        latency_ms_p95: +(host === "pi" ? latency * 5.1 : latency * 1.8).toFixed(1),
        cost_per_question_usd: +(0.00002 + (size / 1536) * 0.00004 + (extra.rerank ? 0 : 0)).toFixed(6),
        by_category: byCat,
      },
      host,
      cost_usd: +(0.001 + r() * 0.002).toFixed(4),
      started_at: ts(`2026-09-2${Math.min(7, 1 + Math.floor(n / 6))}T10:${String(n % 60).padStart(2, "0")}:00Z`),
      finished_at: ts(`2026-09-2${Math.min(7, 1 + Math.floor(n / 6))}T10:${String((n + 1) % 60).padStart(2, "0")}:00Z`),
      results_path: null,
    });
  };
  for (const [chunker, lift] of strategies) for (const size of sizes) mk(chunker, size, lift);
  for (const size of [128, 256, 384, 512]) mk("parent_child", size, 0.04, "mac", "S1a", { parent_size: 1500 });
  // S6: the three finalists measured on the Pi.
  mk("structure", 512, 0.05, "pi", "S6");
  mk("recursive", 384, 0.025, "pi", "S6");
  mk("parent_child", 256, 0.04, "pi", "S6", { parent_size: 1500 });
  return runs;
}

// ------------------------------------------------------------ traces
export function makeTrace(id: string, question: string, outcome: Trace["outcome"], created: Date, withGenerate = true): Trace {
  const stages: StageTiming[] = [];
  let t = 0;
  const add = (stage: StageTiming["stage"], d: number, depth = 0, parent: StageTiming["parent"] = null) => {
    stages.push({ stage, started_ms: t, duration_ms: d, depth, parent, error: null });
    t += d;
  };
  add("embed_query", 142.3);
  const vStart = t;
  add("vector", 11.8);
  add("bm25", 3.4);
  add("fuse", 0.6);
  add("select", 1.2);
  add("abstain", 0.1);
  if (withGenerate) {
    add("generate", 1840.5);
    add("verify", 2.2);
  }
  void vStart;
  return {
    id,
    created_at: created.toISOString(),
    kind: "question",
    question,
    standalone_query: question,
    profile: "structure-512-h",
    stages,
    hits: [
      { chunk_id: 12, rank: 1, score: 0.0328, dense_score: 0.612, bm25_score: 7.41, rerank_score: null, selected: true },
      { chunk_id: 31, rank: 2, score: 0.0317, dense_score: 0.587, bm25_score: 5.02, rerank_score: null, selected: true },
      { chunk_id: 13, rank: 3, score: 0.0301, dense_score: 0.544, bm25_score: null, rerank_score: null, selected: true },
      { chunk_id: 34, rank: 4, score: 0.0162, dense_score: 0.498, bm25_score: 2.2, rerank_score: null, selected: false },
      { chunk_id: 11, rank: 5, score: 0.0159, dense_score: null, bm25_score: 1.9, rerank_score: null, selected: false },
    ],
    context_tokens: 1184,
    tokens_by_model: withGenerate
      ? { "text-embedding-3-small": { prompt: 14, cached: 0, completion: 0 }, "gpt-chat-fast": { prompt: 1422, cached: 1024, completion: 96 } }
      : { "text-embedding-3-small": { prompt: 14, cached: 0, completion: 0 } },
    cost_usd: withGenerate ? 0.00041 : 0.0000003,
    outcome,
    grounding: withGenerate ? "verified" : "skipped",
    ttft_ms: withGenerate ? 612.4 : null,
    total_ms: t,
    error: null,
  };
}

export function seedTraces(): Trace[] {
  const now = Date.now();
  const qs: [string, Trace["outcome"]][] = [
    ["How old is the furnace?", "answered"],
    ["What did the June inspection say about the roof?", "answered"],
    ["Is there a swimming pool heater?", "abstained"],
    ["How much was the termite repair?", "answered"],
    ["Where is the gas shutoff?", "retrieval_only"],
    ["When was the roof warranty issued?", "answered"],
  ];
  return qs.map(([q, o], i) => {
    const id = (0xabc000 + i).toString(16).padStart(32, "0");
    return makeTrace(id, q, o, new Date(now - (i + 1) * 3_600_000 * 5), o !== "retrieval_only" && o !== "abstained");
  });
}
