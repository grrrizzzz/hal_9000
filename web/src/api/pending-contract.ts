/**
 * Contract additions approved by the orchestrator (KWP7 review, 2026-09-28)
 * that are not in openapi.json yet. KWP6 implements them; before phase 2 the
 * orchestrator regenerates openapi.json, and each type here is replaced by
 * the generated one (then this file is deleted). Everything that uses them
 * imports from here, so the swap is one place.
 *
 * Approved:
 *  1. MessageOut.grounding + MessageOut.issues
 *  2. FeedbackRequest.rating optional when add_to_gold
 *  3. Citations on the judge-label item
 *  4. GET /api/eval/runs/{id}/results (paged per-question results)
 *  5. POST /api/jobs/{id}/retry
 *  6. TraceSummary.question
 *
 * Feedback is one row per message: a later POST for the same message_id
 * updates it (rating, comment, add_to_gold), it doesn't add a second row.
 */
import { buildUrl, requestJson } from "./client";
import type { Citation, GoldCategory, Grounding, Job, JudgeLabel, MessageOut, TraceSummary } from "./types";

/** (1) */
export type MessageOutPending = MessageOut & { grounding?: Grounding | null; issues?: string[] };

/** (2) */
export interface FeedbackRequestPending {
  message_id: number;
  rating?: -1 | 1 | null;
  comment?: string | null;
  add_to_gold?: boolean;
}

/** (3) */
export type JudgeLabelPending = JudgeLabel & { citations?: Citation[] };

/** (4) One question's result in an eval run. */
export interface EvalQuestionResult {
  gold_item_id: number;
  question: string;
  category: GoldCategory;
  /** Budgeted evidence recall keyed by token budget ("1000", "2000", "4000"); missing = N/A. */
  recall: Record<string, number>;
  /** Every evidence quote ≥ 80% covered within the top k. */
  full_support: boolean;
  /** Rank of the first chunk covering any evidence; null if none did. */
  first_hit_rank: number | null;
  abstain_score: number | null;
  abstained: boolean;
}

export interface EvalResultsPage {
  items: EvalQuestionResult[];
  total: number;
  offset: number;
  limit: number;
}

/** (6) */
export type TraceSummaryPending = TraceSummary & { question?: string | null };

export const pendingApi = {
  /** (2) */
  feedback: (body: FeedbackRequestPending) => requestJson<{ id: number; gold_item_id?: number | null }>("POST", "/api/feedback", body),
  /** (4) */
  runResults: (id: string, offset = 0, limit = 25) =>
    requestJson<EvalResultsPage>("GET", buildUrl("/api/eval/runs/{id}/results", { id }, { offset, limit })),
  /** (5) Re-run a failed job; returns the job (queued again). */
  retryJob: (id: number) => requestJson<Job>("POST", buildUrl("/api/jobs/{id}/retry", { id })),
};
