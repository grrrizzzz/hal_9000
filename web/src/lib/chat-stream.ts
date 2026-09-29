/**
 * The chat stream: POST /api/chat read with fetch + ReadableStream (not
 * EventSource), and a pure reducer that turns its events into what the
 * answer bubble shows. `reset` clears the partial answer before the
 * regeneration (plan §7.1 verify); `verify` carries the grounding result.
 */
import { notifyUnauthorized, readError } from "@/api/client";
import type { Answer, ChatEvent, ChatRequest, Citation, Grounding, Stage } from "@/api/types";
import { parseSSE } from "./sse";

export interface StreamState {
  phase: "idle" | "streaming" | "done" | "error";
  stage: Stage | null;
  stageDetail: string;
  sources: Citation[];
  text: string;
  resets: number;
  resetReason: string | null;
  grounding: Grounding | null;
  issues: string[];
  retrievalOnly: boolean;
  banner: string | null;
  answer: Answer | null;
  error: string | null;
}

export const initialStreamState: StreamState = {
  phase: "idle",
  stage: null,
  stageDetail: "",
  sources: [],
  text: "",
  resets: 0,
  resetReason: null,
  grounding: null,
  issues: [],
  retrievalOnly: false,
  banner: null,
  answer: null,
  error: null,
};

export type StreamAction = { type: "start" } | { type: "event"; event: ChatEvent } | { type: "error"; message: string };

export function streamReducer(state: StreamState, action: StreamAction): StreamState {
  switch (action.type) {
    case "start":
      return { ...initialStreamState, phase: "streaming" };
    case "error":
      return { ...state, phase: "error", error: action.message };
    case "event": {
      const e = action.event;
      switch (e.event) {
        case "status":
          return { ...state, stage: e.stage, stageDetail: e.detail ?? "" };
        case "sources":
          return {
            ...state,
            sources: e.sources,
            retrievalOnly: Boolean(e.retrieval_only),
            banner: e.banner ?? null,
          };
        case "token":
          return { ...state, text: state.text + e.text };
        case "reset":
          // Drop everything streamed so far; only the regenerated answer is shown.
          return { ...state, text: "", resets: state.resets + 1, resetReason: e.reason };
        case "verify":
          return { ...state, grounding: e.grounding, issues: e.issues ?? [] };
        case "done":
          return {
            ...state,
            phase: "done",
            answer: e.answer,
            text: e.answer.text,
            sources: e.answer.citations?.length ? mergeSources(state.sources, e.answer.citations) : state.sources,
            grounding: e.answer.grounding ?? state.grounding,
            issues: e.answer.issues?.length ? e.answer.issues : state.issues,
            retrievalOnly: state.retrievalOnly || e.answer.outcome === "retrieval_only",
          };
        default:
          return state;
      }
    }
  }
}

function mergeSources(streamed: Citation[], final: Citation[]): Citation[] {
  const byMarker = new Map(streamed.map((c) => [c.marker, c]));
  for (const c of final) byMarker.set(c.marker, c);
  return [...byMarker.values()].sort((a, b) => Number(a.marker.slice(1)) - Number(b.marker.slice(1)));
}

const KNOWN_EVENTS = new Set(["status", "sources", "token", "reset", "verify", "done"]);

/** Parse one SSE message into a chat event, or null if it isn't one. */
export function toChatEvent(eventName: string, data: string): ChatEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  // Each data line is the ChatEventEnvelope.event object; tolerate a wrapped envelope too.
  let obj = parsed as Record<string, unknown>;
  if (obj.event && typeof obj.event === "object") obj = obj.event as Record<string, unknown>;
  const name = typeof obj.event === "string" ? obj.event : eventName;
  if (!KNOWN_EVENTS.has(name)) return null;
  return { ...obj, event: name } as ChatEvent;
}

export async function streamChat(req: ChatRequest, onEvent: (e: ChatEvent) => void, signal?: AbortSignal): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    if (res.status === 401) notifyUnauthorized();
    throw await readError(res);
  }
  if (!res.body) throw new Error("The server sent no answer stream.");
  for await (const msg of parseSSE(res.body, signal)) {
    const ev = toChatEvent(msg.event, msg.data);
    if (ev) onEvent(ev);
  }
}
