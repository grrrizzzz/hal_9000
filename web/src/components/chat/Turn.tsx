import { Loader2, RotateCcw, SearchX } from "lucide-react";
import type { Citation, Grounding, Outcome } from "@/api/types";
import { Banner } from "@/components/ui/states";
import { STAGE_LABEL } from "@/lib/format";
import type { StreamState } from "@/lib/chat-stream";
import { AnswerActions } from "./AnswerActions";
import { CitationText } from "./CitationText";
import { SourceList } from "./SourceList";
import { VerifyBadge } from "./VerifyBadge";

export function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-primary-foreground" data-testid="user-message">
        {text}
      </div>
    </div>
  );
}

export interface AnswerView {
  text: string;
  citations: Citation[];
  outcome: Outcome | null | undefined;
  grounding?: Grounding | null;
  issues?: string[];
  messageId?: number | null;
  traceId?: string | null;
  banner?: string | null;
  retrievalOnly?: boolean;
}

/** A finished answer (from history or a completed stream). */
export function AnswerBlock({ a, onOpen }: { a: AnswerView; onOpen: (c: Citation) => void }) {
  const retrievalOnly = a.retrievalOnly || a.outcome === "retrieval_only";
  const cited = a.citations.filter((c) => a.text.includes(`[${c.marker}]`));
  return (
    <div className="min-w-0" data-testid="assistant-message" data-outcome={a.outcome ?? ""}>
      <DegradedBanners retrievalOnly={retrievalOnly} banner={a.banner ?? null} />
      {a.outcome === "abstained" ? (
        <div className="mb-2 inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <SearchX className="size-3.5" /> Not found in the house documents
        </div>
      ) : null}
      {a.text ? <CitationText text={a.text} citations={a.citations} onOpen={onOpen} /> : null}
      {a.grounding === "failed" || a.grounding === "verified" ? (
        <div className="mt-2">
          <VerifyBadge grounding={a.grounding} issues={a.issues ?? []} />
        </div>
      ) : null}
      <SourceList sources={retrievalOnly ? a.citations : cited.length ? cited : a.citations} onOpen={onOpen} showSnippets={retrievalOnly} />
      <AnswerActions messageId={a.messageId} traceId={a.traceId} />
    </div>
  );
}

export function DegradedBanners({ retrievalOnly, banner }: { retrievalOnly: boolean; banner: string | null }) {
  if (retrievalOnly) {
    return (
      <Banner testId="banner-retrieval-only" title="Sources only — no written answer" className="mb-3">
        {banner ?? "The knowledge budget for this month is used up. Here are the closest passages from the documents."}
      </Banner>
    );
  }
  if (banner) {
    return (
      <Banner testId="banner-degraded" title="Results may be less complete" className="mb-3">
        {banner}
      </Banner>
    );
  }
  return null;
}

/** The answer while it streams. */
export function StreamingBlock({ s, onOpen }: { s: StreamState; onOpen: (c: Citation) => void }) {
  if (s.phase === "done" && s.answer) {
    return (
      <AnswerBlock
        onOpen={onOpen}
        a={{
          text: s.text,
          citations: s.sources,
          outcome: s.answer.outcome,
          grounding: s.grounding,
          issues: s.issues,
          messageId: s.answer.message_id,
          traceId: s.answer.trace_id,
          banner: s.banner,
          retrievalOnly: s.retrievalOnly,
        }}
      />
    );
  }
  if (s.phase === "error") {
    return (
      <Banner tone="danger" title="The answer couldn't be completed">
        {s.error}
      </Banner>
    );
  }
  const regenerating = s.resets > 0 && !s.text;
  return (
    <div className="min-w-0" data-testid="assistant-streaming" aria-busy="true">
      <DegradedBanners retrievalOnly={s.retrievalOnly} banner={s.banner} />
      {s.text ? (
        <CitationText text={s.text} citations={s.sources} onOpen={onOpen} streaming />
      ) : (
        <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
          {regenerating ? <RotateCcw className="size-4 animate-spin [animation-direction:reverse]" /> : <Loader2 className="size-4 animate-spin" />}
          {regenerating ? "Double-checking figures and rewriting…" : `${STAGE_LABEL[s.stage ?? "embed_query"] ?? "Working"}…`}
        </p>
      )}
      {s.sources.length && !s.text ? <SourceList sources={s.sources} onOpen={onOpen} showSnippets={s.retrievalOnly} /> : null}
    </div>
  );
}
