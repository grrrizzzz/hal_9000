import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MessageSquare, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { NavLink, useNavigate } from "react-router";
import { api } from "@/api/client";
import { ConfirmDialog } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";

export function ConversationList({ activeId, onNavigate }: { activeId: number | null; onNavigate?: () => void }) {
  const q = useQuery({ queryKey: ["conversations"], queryFn: () => api.get("/api/conversations") });
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [confirm, setConfirm] = useState<number | null>(null);
  const del = useMutation({
    mutationFn: (id: number) => api.delete("/api/conversations/{id}", { params: { id } }),
    onSuccess: (_r, id) => {
      void qc.invalidateQueries({ queryKey: ["conversations"] });
      if (id === activeId) navigate("/");
      setConfirm(null);
    },
  });

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="p-3">
        <Button
          className="w-full justify-start"
          onClick={() => {
            navigate("/", { state: { fresh: Date.now() } });
            onNavigate?.();
          }}
        >
          <Plus /> New chat
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        <p className="px-2 pb-1.5 text-xs font-medium tracking-wide text-muted-foreground uppercase">Recent</p>
        {q.isLoading ? (
          <div className="space-y-1.5 px-1">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-10" />
            ))}
          </div>
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => q.refetch()} className="py-6" />
        ) : !q.data?.length ? (
          <p className="px-2 py-3 text-[13px] text-muted-foreground">No conversations yet. Questions you ask are kept here.</p>
        ) : (
          <ul className="space-y-0.5">
            {q.data.map((c) => (
              <li key={c.id} className="group relative">
                <NavLink
                  to={`/c/${c.id}`}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn("flex flex-col rounded-lg px-2.5 py-2 pr-9 text-sm hover:bg-subtle", isActive && "bg-subtle")
                  }
                >
                  <span className="flex items-center gap-2 truncate font-medium">
                    <MessageSquare className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                    <span className="truncate">{c.title || "Untitled"}</span>
                  </span>
                  <span className="pl-5.5 text-xs text-muted-foreground">{relativeTime(c.updated_at)}</span>
                </NavLink>
                <button
                  type="button"
                  aria-label={`Delete conversation ${c.title}`}
                  onClick={() => setConfirm(c.id)}
                  className="absolute top-2 right-1.5 rounded-md p-1.5 text-muted-foreground [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover:opacity-100 hover:bg-border/60 hover:text-foreground [@media(hover:hover)]:focus-visible:opacity-100"
                >
                  <Trash2 className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      <ConfirmDialog
        open={confirm !== null}
        onOpenChange={(o) => !o && setConfirm(null)}
        title="Delete this conversation?"
        description="Its messages are removed. Traces and any items added to the question set are kept."
        confirmLabel="Delete"
        tone="danger"
        busy={del.isPending}
        onConfirm={() => confirm !== null && del.mutate(confirm)}
      />
    </div>
  );
}
