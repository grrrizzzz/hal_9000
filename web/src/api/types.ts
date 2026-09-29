import type { Schemas } from "./client";

export type Citation = Schemas["Citation"];
export type Answer = Schemas["Answer"];
export type DocType = NonNullable<Schemas["DocumentMeta"]["doc_type"]>;
export type DocumentSummary = Schemas["DocumentSummary"];
export type DocumentDetail = Schemas["DocumentDetail"];
export type DocumentPatch = Schemas["DocumentPatch"];
export type PageView = Schemas["PageView"];
export type Job = Schemas["Job"];
export type UploadResult = Schemas["UploadResult"];
export type ConversationSummary = Schemas["ConversationSummary"];
export type ConversationDetail = Schemas["ConversationDetail"];
export type MessageOut = Schemas["MessageOut"];
export type GoldItem = Schemas["GoldItem"];
export type GoldPatch = Schemas["GoldPatch"];
export type EvidenceQuote = Schemas["EvidenceQuote"];
export type JudgeLabel = Schemas["JudgeLabel"];
export type RubricLabels = Schemas["RubricLabels"];
export type EvalRun = Schemas["EvalRun"];
export type EvalMetrics = Schemas["EvalMetrics"];
export type MetricCI = Schemas["MetricCI"];
export type IndexProfile = Schemas["IndexProfile"];
export type ProfileState = Schemas["ProfileState"];
export type Trace = Schemas["Trace"];
export type TraceSummary = Schemas["TraceSummary"];
export type TraceHit = Schemas["TraceHit"];
export type StageTiming = Schemas["StageTiming"];
export type Stage = Schemas["Stage"];
export type Health = Schemas["Health"];
export type SearchFilters = Schemas["SearchFilters"];
export type ChatRequest = Schemas["ChatRequest"];
export type Outcome = NonNullable<Schemas["Answer"]["outcome"]>;
export type Grounding = NonNullable<Schemas["Answer"]["grounding"]>;
export type GoldCategory = Schemas["GoldItem"]["category"];

/** Every event on the POST /api/chat stream (the `data:` line of each SSE message). */
export type ChatEvent =
  | Schemas["ChatStatusEvent"]
  | Schemas["ChatSourcesEvent"]
  | Schemas["ChatTokenEvent"]
  | Schemas["ChatResetEvent"]
  | Schemas["ChatVerifyEvent"]
  | Schemas["ChatDoneEvent"];

export const DOC_TYPES: DocType[] = [
  "inspection_report",
  "estimate",
  "invoice",
  "pest_report",
  "completion_notice",
  "disclosure_form",
  "manual",
  "warranty",
  "permit",
  "other",
];

export const GOLD_CATEGORIES: GoldCategory[] = ["factoid", "money", "date", "list", "cross_document", "unanswerable"];
