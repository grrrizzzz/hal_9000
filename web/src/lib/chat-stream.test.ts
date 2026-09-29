import { describe, expect, it } from "vitest";
import type { ChatEvent, Citation } from "@/api/types";
import { initialStreamState, streamReducer, toChatEvent, type StreamState } from "./chat-stream";

const cite: Citation = {
  marker: "S1",
  chunk_id: 1,
  document_id: 1,
  file_sha: "a".repeat(64),
  page: 2,
  title: "Inspection",
  doc_type: "inspection_report",
  snippet: "",
};

const run = (events: ChatEvent[]): StreamState =>
  events.reduce((s, event) => streamReducer(s, { type: "event", event }), streamReducer(initialStreamState, { type: "start" }));

describe("streamReducer", () => {
  it("accumulates tokens after sources", () => {
    const s = run([
      { event: "status", stage: "vector", detail: "" },
      { event: "sources", sources: [cite], retrieval_only: false },
      { event: "token", text: "The furnace " },
      { event: "token", text: "is gas [S1]." },
    ]);
    expect(s.text).toBe("The furnace is gas [S1].");
    expect(s.sources).toHaveLength(1);
    expect(s.stage).toBe("vector");
  });

  it("reset clears the partial answer so only the regeneration shows", () => {
    const s = run([
      { event: "token", text: "It cost $1,324.00 [S1]." },
      { event: "reset", reason: "figure not in sources" },
      { event: "token", text: "It cost $1,234.00 [S1]." },
    ]);
    expect(s.text).toBe("It cost $1,234.00 [S1].");
    expect(s.resets).toBe(1);
    expect(s.resetReason).toBe("figure not in sources");
  });

  it("records verify and finishes on done", () => {
    const s = run([
      { event: "token", text: "x" },
      { event: "verify", grounding: "failed", issues: ["$9 not found"] },
      {
        event: "done",
        answer: { text: "x", citations: [], outcome: "answered", grounding: "failed", issues: ["$9 not found"], trace_id: "t", cost_usd: 0 },
      },
    ]);
    expect(s.phase).toBe("done");
    expect(s.grounding).toBe("failed");
    expect(s.issues).toEqual(["$9 not found"]);
  });

  it("flags retrieval-only from the sources event", () => {
    const s = run([{ event: "sources", sources: [cite], retrieval_only: true, banner: "Monthly cap reached" }]);
    expect(s.retrievalOnly).toBe(true);
    expect(s.banner).toBe("Monthly cap reached");
  });
});

describe("toChatEvent", () => {
  it("reads the event object from the data line", () => {
    expect(toChatEvent("message", '{"event":"token","text":"a"}')).toEqual({ event: "token", text: "a" });
  });
  it("falls back to the SSE event name and ignores junk", () => {
    expect(toChatEvent("reset", '{"reason":"r"}')).toEqual({ event: "reset", reason: "r" });
    expect(toChatEvent("token", "not json")).toBeNull();
    expect(toChatEvent("ping", "{}")).toBeNull();
  });
});
