import { useQuery, useQueryClient } from "@tanstack/react-query";
import { History } from "lucide-react";
import { Dialog as D } from "radix-ui";
import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useMatch, useNavigate } from "react-router";
import { api } from "@/api/client";
import type { Citation, MessageOut, SearchFilters } from "@/api/types";
import { Composer } from "@/components/chat/Composer";
import { ConversationList } from "@/components/chat/ConversationList";
import { Filters } from "@/components/chat/Filters";
import { AnswerBlock, StreamingBlock, UserBubble } from "@/components/chat/Turn";
import { Logo } from "@/components/layout/Logo";
import { Button } from "@/components/ui/button";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { ViewerDialog } from "@/components/viewer/ViewerDialog";
import { initialStreamState, streamChat, streamReducer, type StreamState } from "@/lib/chat-stream";
import { docTypeLabel, formatDocDate } from "@/lib/format";

const EXAMPLES = [
  "How old is the furnace?",
  "What did the inspection say about the roof?",
  "Which repairs were invoiced, and for how much?",
  "When does the roof warranty run out?",
];

interface LocalTurn {
  key: number;
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
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const convoRef = useRef<number | null>(conversationId);

  const convo = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => api.get("/api/conversations/{id}", { params: { id: conversationId! } }),
    enabled: conversationId !== null,
  });

  // A different conversation (or "New chat") starts from a clean slate.
  useEffect(() => {
    if (convoRef.current !== conversationId) {
      convoRef.current = conversationId;
      const streaming = turns.some((t) => t.state.phase === "streaming");
      if (!streaming) setTurns([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);
  useEffect(() => {
    if (location.state && (location.state as { fresh?: number }).fresh) {
      abort.current?.abort();
      setTurns([]);
    }
  }, [location.state]);
  useEffect(() => () => abort.current?.abort(), []);

  const busy = turns.some((t) => t.state.phase === "streaming");
  const serverMessages: MessageOut[] = conversationId !== null ? (convo.data?.messages ?? []) : [];
  // Turns streamed on this page stay mounted (they carry the grounding result and
  // feedback state); the same messages coming back from the server are skipped.
  const localIds = new Set(turns.map((t) => t.state.answer?.message_id).filter((id): id is number => typeof id === "number"));
  const shownMessages = serverMessages.filter(
    (m, i) => !localIds.has(m.id) && !(m.role === "user" && serverMessages[i + 1] && localIds.has(serverMessages[i + 1].id)),
  );
  const visibleTurns = turns;

  const lastLength = turns.at(-1)?.state.text.length ?? 0;
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [visibleTurns.length, lastLength, serverMessages.length]);

  const ask = useCallback(
    async (question: string) => {
      const key = Date.now();
      const ctrl = new AbortController();
      abort.current = ctrl;
      const update = (fn: (s: StreamState) => StreamState) => setTurns((ts) => ts.map((t) => (t.key === key ? { ...t, state: fn(t.state) } : t)));
      setTurns((ts) => [...ts, { key, question, state: streamReducer(initialStreamState, { type: "start" }) }]);
      let newConversation: number | null = null;
      try {
        const f = Object.fromEntries(Object.entries(filters).filter(([, v]) => v)) as SearchFilters;
        await streamChat(
          { conversation_id: convoRef.current, message: question, filters: Object.keys(f).length ? f : null },
          (event) => {
            update((s) => streamReducer(s, { type: "event", event }));
            if (event.event === "done" && event.answer.conversation_id && event.answer.conversation_id !== convoRef.current) {
              newConversation = event.answer.conversation_id;
            }
          },
          ctrl.signal,
        );
        update((s) => (s.phase === "streaming" ? streamReducer(s, { type: "error", message: "The answer stream ended early. Try asking again." }) : s));
      } catch (e) {
        if ((e as Error).name === "AbortError") update((s) => streamReducer(s, { type: "error", message: "Stopped." }));
        else update((s) => streamReducer(s, { type: "error", message: (e as Error).message }));
      } finally {
        void qc.invalidateQueries({ queryKey: ["conversations"] });
        void qc.invalidateQueries({ queryKey: ["traces"] });
        if (newConversation !== null) {
          convoRef.current = newConversation;
          navigate(`/c/${newConversation}`, { replace: conversationId === null });
        } else if (convoRef.current !== null) {
          void qc.invalidateQueries({ queryKey: ["conversation", convoRef.current] });
        }
      }
    },
    [filters, qc, navigate, conversationId],
  );

  const empty = !serverMessages.length && !visibleTurns.length && !(conversationId !== null && convo.isLoading);

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
            {conversationId !== null && convo.isLoading && !turns.length ? (
              <div className="space-y-6">
                <Skeleton className="ml-auto h-10 w-2/3" />
                <Skeleton className="h-28" />
              </div>
            ) : conversationId !== null && convo.isError && !turns.length ? (
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
            <Composer onSend={(t) => void ask(t)} onStop={() => abort.current?.abort()} busy={busy}>
              <Filters value={filters} onChange={setFilters} />
            </Composer>
            <p className="mt-2 text-center text-[11px] text-muted-foreground">Check important figures against the source pages.</p>
          </div>
        </div>
      </section>

      <ViewerDialog
        target={viewer ? { fileSha: viewer.file_sha, page: viewer.page, chunkId: viewer.chunk_id, quotes: viewer.snippet ? [viewer.snippet] : [] } : null}
        title={viewer ? `${viewer.marker} · ${viewer.title}` : ""}
        description={viewer ? [docTypeLabel(viewer.doc_type), viewer.vendor, formatDocDate(viewer.doc_date), `page ${viewer.page}`].filter(Boolean).join(" · ") : ""}
        onClose={() => setViewer(null)}
      />
    </div>
  );
}
