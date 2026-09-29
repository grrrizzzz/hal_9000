import { useQuery, useQueryClient } from "@tanstack/react-query";
import { History } from "lucide-react";
import { Dialog as D } from "radix-ui";
import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useMatch, useNavigate } from "react-router";
import { api } from "@/api/client";
import type { MessageOutPending } from "@/api/pending-contract";
import type { Citation, SearchFilters } from "@/api/types";
import { Composer } from "@/components/chat/Composer";
import { ConversationList } from "@/components/chat/ConversationList";
import { Filters } from "@/components/chat/Filters";
import { AnswerBlock, StreamingBlock, UserBubble } from "@/components/chat/Turn";
import { Logo } from "@/components/layout/Logo";
import { Button } from "@/components/ui/button";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { ViewerDialog } from "@/components/viewer/ViewerDialog";
import { initialStreamState, streamChat, streamReducer, type StreamAction, type StreamState } from "@/lib/chat-stream";
import { docTypeLabel, formatDocDate } from "@/lib/format";

const EXAMPLES = [
  "How old is the furnace?",
  "What did the inspection say about the roof?",
  "Which repairs were invoiced, and for how much?",
  "When does the roof warranty run out?",
];

interface LocalTurn {
  key: number;
  /** The conversation view the turn belongs to: "c:<id>", or "new:<n>" until the server assigns an id. */
  view: string;
  question: string;
  state: StreamState;
}

export function ChatPage() {
  const match = useMatch("/c/:conversationId");
  const conversationId = match?.params.conversationId ? Number(match.params.conversationId) : null;
  const navigate = useNavigate();
  const location = useLocation();
  const qc = useQueryClient();
  const [filters, setFilters] = useState<SearchFilters>({});
  const [viewer, setViewer] = useState<Citation | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [turns, setTurns] = useState<LocalTurn[]>([]);
  const [fresh, setFresh] = useState(0);
  const controllers = useRef(new Map<number, AbortController>());
  const bottom = useRef<HTMLDivElement>(null);

  // "New chat" starts a new view; a stream still running elsewhere keeps going in its own conversation.
  const freshState = (location.state as { fresh?: number } | null)?.fresh;
  useEffect(() => {
    if (freshState) setFresh(freshState);
  }, [freshState]);
  const view = conversationId !== null ? `c:${conversationId}` : `new:${fresh}`;
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  }, [view]);
  useEffect(() => {
    const all = controllers.current;
    return () => all.forEach((c) => c.abort());
  }, []);

  const convo = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => api.get("/api/conversations/{id}", { params: { id: conversationId! } }),
    enabled: conversationId !== null,
  });

  const visibleTurns = turns.filter((t) => t.view === view);
  const busy = visibleTurns.some((t) => t.state.phase === "streaming");
  const serverMessages = (conversationId !== null ? (convo.data?.messages ?? []) : []) as MessageOutPending[];
  // Turns streamed on this page stay mounted (they carry the grounding result and
  // feedback state); the same messages coming back from the server are skipped.
  const localIds = new Set(turns.map((t) => t.state.answer?.message_id).filter((id): id is number => typeof id === "number"));
  const shownMessages = serverMessages.filter(
    (m, i) => !localIds.has(m.id) && !(m.role === "user" && serverMessages[i + 1] && localIds.has(serverMessages[i + 1].id)),
  );

  const lastLength = visibleTurns.at(-1)?.state.text.length ?? 0;
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [visibleTurns.length, lastLength, shownMessages.length]);

  const ask = useCallback(
    async (question: string) => {
      const key = Date.now() + Math.random();
      const startView = viewRef.current;
      const ctrl = new AbortController();
      controllers.current.set(key, ctrl);
      const update = (fn: (t: LocalTurn) => LocalTurn) => setTurns((ts) => ts.map((t) => (t.key === key ? fn(t) : t)));
      const dispatch = (a: StreamAction) => update((t) => ({ ...t, state: streamReducer(t.state, a) }));
      setTurns((ts) => [...ts, { key, view: startView, question, state: streamReducer(initialStreamState, { type: "start" }) }]);
      let assigned: number | null = null;
      try {
        const f = Object.fromEntries(Object.entries(filters).filter(([, v]) => v)) as SearchFilters;
        await streamChat(
          { conversation_id: conversationId, message: question, filters: Object.keys(f).length ? f : null },
          (event) => {
            dispatch({ type: "event", event });
            if (event.event === "done" && event.answer.conversation_id) assigned = event.answer.conversation_id;
          },
          ctrl.signal,
        );
        update((t) => (t.state.phase === "streaming" ? { ...t, state: streamReducer(t.state, { type: "error", message: "The answer stream ended early. Try asking again." }) } : t));
      } catch (e) {
        dispatch({ type: "error", message: (e as Error).name === "AbortError" ? "Stopped." : (e as Error).message });
      } finally {
        controllers.current.delete(key);
        void qc.invalidateQueries({ queryKey: ["conversations"] });
        void qc.invalidateQueries({ queryKey: ["traces"] });
        if (assigned !== null) {
          void qc.invalidateQueries({ queryKey: ["conversation", assigned] });
          if (startView.startsWith("new:")) {
            // The turn now belongs to its new conversation. Follow it only if you're still looking at it.
            update((t) => ({ ...t, view: `c:${assigned}` }));
            if (viewRef.current === startView) navigate(`/c/${assigned}`, { replace: true });
          }
        }
      }
    },
    [filters, qc, navigate, conversationId],
  );

  const stop = () => visibleTurns.forEach((t) => controllers.current.get(t.key)?.abort());
  const empty = !shownMessages.length && !visibleTurns.length && !(conversationId !== null && convo.isLoading);

  return (
    <div className="flex min-h-0 flex-1">
      <aside className="hidden w-64 shrink-0 border-r border-border md:block">
        <div className="sticky top-0 h-dvh">
          <ConversationList activeId={conversationId} />
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col" aria-label="Conversation">
        <div className="flex items-center justify-between gap-2 px-4 pt-3 md:hidden">
          <D.Root open={historyOpen} onOpenChange={setHistoryOpen}>
            <D.Trigger asChild>
              <Button size="sm" variant="ghost" className="-ml-2 text-muted-foreground">
                <History /> History
              </Button>
            </D.Trigger>
            <D.Portal>
              <D.Overlay className="fixed inset-0 z-40 bg-black/40" />
              <D.Content className="fixed inset-y-0 left-0 z-50 w-72 max-w-[85vw] bg-surface shadow-pop outline-none">
                <D.Title className="sr-only">Conversations</D.Title>
                <D.Description className="sr-only">Earlier conversations</D.Description>
                <ConversationList activeId={conversationId} onNavigate={() => setHistoryOpen(false)} />
              </D.Content>
            </D.Portal>
          </D.Root>
        </div>

        <div className="min-h-0 flex-1 px-4 sm:px-6">
          <div className="mx-auto w-full max-w-3xl py-6">
            {conversationId !== null && convo.isLoading && !visibleTurns.length ? (
              <div className="space-y-6">
                <Skeleton className="ml-auto h-10 w-2/3" />
                <Skeleton className="h-28" />
              </div>
            ) : conversationId !== null && convo.isError && !visibleTurns.length ? (
              <ErrorState error={convo.error} onRetry={() => convo.refetch()} />
            ) : empty ? (
              <div className="flex flex-col items-center pt-[8vh] text-center">
                <Logo className="[&>div]:hidden [&>svg]:size-11" />
                <h1 className="mt-4 text-2xl font-semibold tracking-tight">What would you like to know about the house?</h1>
                <p className="mt-2 max-w-md text-sm text-muted-foreground">
                  Answers come only from the documents in the library, with every claim linked to the page it came from.
                </p>
                <div className="mt-6 grid w-full max-w-xl gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => void ask(q)}
                      className="rounded-xl border border-border bg-surface px-4 py-3 text-left text-sm shadow-card transition-colors hover:border-border-strong hover:bg-surface-2"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-7">
                {shownMessages.map((m) =>
                  m.role === "user" ? (
                    <UserBubble key={m.id} text={m.content} />
                  ) : (
                    <AnswerBlock
                      key={m.id}
                      onOpen={setViewer}
                      a={{
                        text: m.content,
                        citations: m.citations ?? [],
                        outcome: m.outcome,
                        messageId: m.id,
                        traceId: m.trace_id,
                        retrievalOnly: m.outcome === "retrieval_only",
                        grounding: m.grounding ?? null,
                        issues: m.issues ?? [],
                      }}
                    />
                  ),
                )}
                {visibleTurns.map((t) => (
                  <div key={t.key} className="space-y-7">
                    <UserBubble text={t.question} />
                    <StreamingBlock s={t.state} onOpen={setViewer} />
                  </div>
                ))}
              </div>
            )}
            <div ref={bottom} />
          </div>
        </div>

        <div className="sticky bottom-0 bg-gradient-to-t from-background via-background to-background/0 px-4 pt-4 pb-4 sm:px-6">
          <div className="mx-auto w-full max-w-3xl">
            <Composer onSend={(t) => void ask(t)} onStop={stop} busy={busy}>
              <Filters value={filters} onChange={setFilters} />
            </Composer>
            <p className="mt-2 text-center text-[11px] text-muted-foreground">Check important figures against the source pages.</p>
          </div>
        </div>
      </section>

      <ViewerDialog
        target={viewer ? { fileSha: viewer.file_sha, page: viewer.page, chunkId: viewer.chunk_id, quotes: viewer.snippet ? [viewer.snippet] : [] } : null}
        title={viewer ? `${viewer.marker} · ${viewer.title}` : ""}
        description={viewer ? [`Page ${viewer.page}`, docTypeLabel(viewer.doc_type), viewer.vendor, formatDocDate(viewer.doc_date)].filter(Boolean).join(" · ") : ""}
        onClose={() => setViewer(null)}
      />
    </div>
  );
}
