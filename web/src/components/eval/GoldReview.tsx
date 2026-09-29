/**
 * The question-set review queue (plan §8.1, gate G-K1): approve, edit or
 * reject each draft, with keyboard shortcuts (A approve, E edit, R reject,
 * J/K next/previous) and evidence re-selection by highlighting text on the
 * page (the server aligns the selection to cleaned text).
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, ChevronUp, FileSearch, Inbox, Pencil, Plus, Trash2, Undo2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { api } from "@/api/client";
import { GOLD_CATEGORIES, type EvidenceQuote, type GoldCategory, type GoldItem, type GoldPatch } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Input, Label, Select, Textarea } from "@/components/ui/input";
import { Kbd } from "@/components/ui/kbd";
import { Segmented } from "@/components/ui/segmented";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/ui/states";
import { ViewerDialog } from "@/components/viewer/ViewerDialog";
import { cn } from "@/lib/utils";

type StatusFilter = "draft" | "approved" | "rejected" | "all";

const CATEGORY_LABEL: Record<GoldCategory, string> = {
  factoid: "Factoid",
  money: "Money or number",
  date: "Date or age",
  list: "List",
  cross_document: "Cross-document",
  unanswerable: "Unanswerable",
};

/** Is the user typing somewhere (or in a dialog)? Then shortcuts stay off. */
export function shortcutsBlocked(e: KeyboardEvent): boolean {
  if (e.metaKey || e.ctrlKey || e.altKey) return true;
  const t = e.target as HTMLElement | null;
  if (t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName))) return true;
  return !!document.querySelector('[role="dialog"], [role="alertdialog"]');
}

interface Selecting {
  item: GoldItem;
  index: number | null; // evidence to replace; null adds
  page: number;
  fileSha: string;
}

export function GoldReview() {
  const [params, setParams] = useSearchParams();
  const status = (params.get("status") as StatusFilter) || "draft";
  const itemParam = params.get("item");
  const qc = useQueryClient();
  const list = useQuery({
    queryKey: ["gold", status],
    queryFn: () => api.get("/api/eval/gold", { query: status === "all" ? {} : { status } }),
  });
  const items = useMemo(() => list.data ?? [], [list.data]);
  const [selectedId, setSelectedId] = useState<number | null>(itemParam ? Number(itemParam) : null);
  const [editing, setEditing] = useState(false);
  const [selecting, setSelecting] = useState<Selecting | null>(null);
  const [viewing, setViewing] = useState<{ q: EvidenceQuote; item: GoldItem } | null>(null);
  const [flash, setFlash] = useState<{ text: string; undo?: () => void } | null>(null);

  const index = Math.max(0, items.findIndex((i) => i.id === selectedId));
  const current = items[index] ?? null;
  useEffect(() => {
    if (items.length && (selectedId === null || !items.some((i) => i.id === selectedId))) setSelectedId(items[Math.min(index, items.length - 1)].id ?? null);
  }, [items, selectedId, index]);

  const patch = useMutation({
    mutationFn: ({ id, body }: { id: number; body: GoldPatch }) => api.patch("/api/eval/gold/{id}", { params: { id }, body }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["gold"] }),
    onError: (e) => setFlash({ text: (e as Error).message }),
  });

  const go = useCallback(
    (delta: number) => {
      if (!items.length) return;
      const next = items[Math.min(items.length - 1, Math.max(0, index + delta))];
      setSelectedId(next.id ?? null);
    },
    [items, index],
  );

  const decide = useCallback(
    (decision: "approved" | "rejected") => {
      if (!current?.id) return;
      const id = current.id;
      const before = current.status;
      // Move on to the next item first, so a filtered list doesn't jump back to the top.
      const next = items[index + 1] ?? items[index - 1] ?? null;
      if (status !== "all" && status !== decision) setSelectedId(next?.id ?? null);
      patch.mutate(
        { id, body: { status: decision } },
        {
          onSuccess: () =>
            setFlash({
              text: `${decision === "approved" ? "Approved" : "Rejected"}: “${current.question}”`,
              undo: () => patch.mutate({ id, body: { status: before } }, { onSuccess: () => setSelectedId(id) }),
            }),
        },
      );
    },
    [current, items, index, patch, status],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (shortcutsBlocked(e)) return;
      const k = e.key.toLowerCase();
      if (k === "j") go(1);
      else if (k === "k") go(-1);
      else if (k === "a") decide("approved");
      else if (k === "r") decide("rejected");
      else if (k === "e" && current) setEditing(true);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, decide, current]);

  const setStatus = (s: StatusFilter) => {
    const p = new URLSearchParams(params);
    p.set("status", s);
    p.delete("item");
    setParams(p, { replace: true });
    setSelectedId(null);
  };

  const replaceEvidence = (sel: Selecting, quote: string, page: number) => {
    const evidence = [...sel.item.evidence!];
    const q: EvidenceQuote = { quote, page, file_sha: sel.fileSha };
    if (sel.index === null) evidence.push(q);
    else evidence[sel.index] = q;
    patch.mutate({ id: sel.item.id!, body: { evidence } }, { onSuccess: () => setFlash({ text: "Evidence updated. The server aligns it to the cleaned page text." }) });
    setSelecting(null);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented<StatusFilter>
          label="Show"
          value={status}
          onChange={setStatus}
          options={[
            { value: "draft", label: "To review" },
            { value: "approved", label: "Approved" },
            { value: "rejected", label: "Rejected" },
            { value: "all", label: "All" },
          ]}
        />
        <p className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex">
          <Kbd>A</Kbd> approve <Kbd>E</Kbd> edit <Kbd>R</Kbd> reject <Kbd>J</Kbd>/<Kbd>K</Kbd> next/previous
        </p>
      </div>

      {flash ? (
        <div role="status" className="flex items-center gap-3 rounded-xl border border-border bg-surface px-4 py-2.5 text-sm shadow-card" data-testid="review-flash">
          <span className="min-w-0 flex-1 truncate">{flash.text}</span>
          {flash.undo ? (
            <Button size="sm" variant="ghost" onClick={() => (flash.undo!(), setFlash(null))}>
              <Undo2 /> Undo
            </Button>
          ) : null}
          <button type="button" aria-label="Dismiss" onClick={() => setFlash(null)} className="rounded p-1 text-muted-foreground hover:bg-subtle">
            <X className="size-4" />
          </button>
        </div>
      ) : null}

      {list.isLoading ? (
        <SkeletonRows rows={4} />
      ) : list.isError ? (
        <Card>
          <ErrorState error={list.error} onRetry={() => list.refetch()} />
        </Card>
      ) : !items.length ? (
        <Card>
          <EmptyState icon={<Inbox />} title={status === "draft" ? "Nothing left to review" : "No items here"}>
            {status === "draft" ? "Generated questions and ones you add from chat land here as drafts." : "Try another filter."}
          </EmptyState>
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
          <Card className="max-h-64 overflow-auto lg:max-h-[70dvh]">
            <ol aria-label="Questions" className="divide-y divide-border">
              {items.map((it, i) => (
                <li key={it.id} data-gold-id={it.id}>
                  <button
                    type="button"
                    onClick={() => setSelectedId(it.id ?? null)}
                    aria-current={it.id === current?.id ? "true" : undefined}
                    className={cn(
                      "flex w-full gap-2.5 px-4 py-2.5 text-left text-[13px] hover:bg-surface-2",
                      it.id === current?.id && "bg-accent-soft/60 hover:bg-accent-soft/60",
                    )}
                  >
                    <span className="tabular w-5 shrink-0 text-muted-foreground">{i + 1}</span>
                    <span className="min-w-0 flex-1">
                      <span className="line-clamp-2">{it.question}</span>
                      <span className="mt-0.5 block text-xs text-muted-foreground">
                        {CATEGORY_LABEL[it.category]}
                        {it.origin === "feedback" ? " · from chat" : ""}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </Card>

          {current ? (
            <ReviewCard
              item={current}
              position={`${index + 1} of ${items.length}`}
              busy={patch.isPending}
              onApprove={() => decide("approved")}
              onReject={() => decide("rejected")}
              onEdit={() => setEditing(true)}
              onPrev={() => go(-1)}
              onNext={() => go(1)}
              onView={(q) => setViewing({ q, item: current })}
              onReselect={(i) => {
                const ev = current.evidence?.[i];
                if (ev?.file_sha) setSelecting({ item: current, index: i, page: ev.page, fileSha: ev.file_sha });
              }}
              onAddEvidence={() => {
                const ev = current.evidence?.find((e) => e.file_sha);
                if (ev?.file_sha) setSelecting({ item: current, index: null, page: ev.page, fileSha: ev.file_sha });
              }}
            />
          ) : null}
        </div>
      )}

      {current && editing ? <EditGoldDialog item={current} onClose={() => setEditing(false)} onSaved={(t) => setFlash({ text: t })} /> : null}

      <ViewerDialog
        target={viewing?.q.file_sha ? { fileSha: viewing.q.file_sha, page: viewing.q.page, quotes: [viewing.q.quote] } : null}
        title="Evidence"
        description={viewing ? `“${viewing.q.quote.slice(0, 80)}${viewing.q.quote.length > 80 ? "…" : ""}” · page ${viewing.q.page}` : ""}
        onClose={() => setViewing(null)}
      />
      <ViewerDialog
        target={selecting ? { fileSha: selecting.fileSha, page: selecting.page, quotes: selecting.index !== null ? [selecting.item.evidence![selecting.index].quote] : [] } : null}
        title={selecting?.index === null ? "Add evidence" : "Re-select evidence"}
        description="Highlight the passage on the page, then choose “Use selection as evidence”."
        onClose={() => setSelecting(null)}
        onUseSelection={(quote, page) => selecting && replaceEvidence(selecting, quote, page)}
      />
    </div>
  );
}

function ReviewCard({
  item,
  position,
  busy,
  onApprove,
  onReject,
  onEdit,
  onPrev,
  onNext,
  onView,
  onReselect,
  onAddEvidence,
}: {
  item: GoldItem;
  position: string;
  busy: boolean;
  onApprove: () => void;
  onReject: () => void;
  onEdit: () => void;
  onPrev: () => void;
  onNext: () => void;
  onView: (q: EvidenceQuote) => void;
  onReselect: (i: number) => void;
  onAddEvidence: () => void;
}) {
  return (
    <Card className="min-w-0" data-testid="review-card" data-status={item.status} data-gold-id={item.id}>
      <div className="flex items-center justify-between gap-2 border-b border-border px-5 py-2.5 text-xs text-muted-foreground">
        <span className="tabular">{position}</span>
        <div className="flex items-center gap-1">
          <Button size="icon-sm" variant="ghost" aria-label="Previous question (K)" onClick={onPrev}>
            <ChevronUp />
          </Button>
          <Button size="icon-sm" variant="ghost" aria-label="Next question (J)" onClick={onNext}>
            <ChevronDown />
          </Button>
        </div>
      </div>
      <div className="space-y-5 px-5 py-5">
        <div>
          <div className="mb-2 flex flex-wrap gap-1.5">
            <Badge tone="accent">{CATEGORY_LABEL[item.category]}</Badge>
            <Badge tone="outline">{item.difficulty}</Badge>
            {item.paraphrased ? <Badge tone="outline">paraphrased</Badge> : null}
            {item.split ? <Badge tone="outline">{item.split} split</Badge> : null}
            {item.origin === "feedback" ? <Badge tone="outline">from chat</Badge> : null}
            <Badge tone={item.status === "approved" ? "success" : item.status === "rejected" ? "danger" : "neutral"}>{item.status}</Badge>
          </div>
          <h2 className="text-lg leading-snug font-semibold" data-testid="review-question">
            {item.question}
          </h2>
        </div>
        <div>
          <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Reference answer</h3>
          <p className="mt-1 text-[15px] leading-relaxed">{item.reference_answer || <span className="text-muted-foreground">None — the documents don't answer this.</span>}</p>
        </div>
        <div>
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Evidence</h3>
            {item.evidence?.some((e) => e.file_sha) ? (
              <Button size="sm" variant="ghost" onClick={onAddEvidence}>
                <Plus /> Add from page
              </Button>
            ) : null}
          </div>
          {item.evidence?.length ? (
            <ul className="mt-2 space-y-2">
              {item.evidence.map((ev, i) => (
                <li key={i} className="rounded-lg border border-border bg-surface-2 px-3 py-2.5">
                  <blockquote className="border-l-2 border-mark-strong pl-3 text-sm leading-relaxed">{ev.quote}</blockquote>
                  <div className="mt-2 flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
                    <span className="mr-auto">Page {ev.page}</span>
                    {ev.file_sha ? (
                      <>
                        <Button size="sm" variant="ghost" onClick={() => onView(ev)}>
                          <FileSearch /> Show on page
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => onReselect(i)}>
                          <Pencil /> Re-select
                        </Button>
                      </>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-sm text-muted-foreground">No evidence (unanswerable item).</p>
          )}
        </div>
        {item.reviewer_note ? (
          <p className="text-sm text-muted-foreground">
            <span className="font-medium text-foreground">Note:</span> {item.reviewer_note}
          </p>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-border px-5 py-3">
        <Button variant="primary" onClick={onApprove} disabled={busy}>
          <Check /> Approve <Kbd className="ml-1 hidden border-transparent bg-primary-foreground/15 text-inherit sm:inline-grid">A</Kbd>
        </Button>
        <Button onClick={onEdit}>
          <Pencil /> Edit <Kbd className="ml-1 hidden sm:inline-grid">E</Kbd>
        </Button>
        <Button variant="ghost" className="text-danger" onClick={onReject} disabled={busy}>
          <X /> Reject <Kbd className="ml-1 hidden sm:inline-grid">R</Kbd>
        </Button>
      </div>
    </Card>
  );
}

function EditGoldDialog({ item, onClose, onSaved }: { item: GoldItem; onClose: () => void; onSaved: (text: string) => void }) {
  const [question, setQuestion] = useState(item.question);
  const [answer, setAnswer] = useState(item.reference_answer ?? "");
  const [category, setCategory] = useState<GoldCategory>(item.category);
  const [note, setNote] = useState(item.reviewer_note ?? "");
  const [evidence, setEvidence] = useState<EvidenceQuote[]>(item.evidence ?? []);
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: () =>
      api.patch("/api/eval/gold/{id}", {
        params: { id: item.id! },
        body: {
          question: question.trim(),
          reference_answer: answer.trim() || null,
          category,
          reviewer_note: note.trim() || null,
          evidence: category === "unanswerable" ? [] : evidence,
        },
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["gold"] });
      onSaved("Saved your edits.");
      onClose();
    },
  });
  const invalid = !question.trim() || (category !== "unanswerable" && !evidence.length);
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent title="Edit question" size="lg">
        <form
          className="space-y-4 p-5"
          onSubmit={(e) => {
            e.preventDefault();
            if (!invalid) save.mutate();
          }}
        >
          <div className="space-y-1.5">
            <Label htmlFor="g-q">Question</Label>
            <Input id="g-q" value={question} onChange={(e) => setQuestion(e.target.value)} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="g-a">Reference answer</Label>
            <Textarea id="g-a" value={answer} onChange={(e) => setAnswer(e.target.value)} />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="g-c">Category</Label>
              <Select id="g-c" value={category} onChange={(e) => setCategory(e.target.value as GoldCategory)}>
                {GOLD_CATEGORIES.map((c) => (
                  <option key={c} value={c}>
                    {CATEGORY_LABEL[c]}
                  </option>
                ))}
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="g-n">Note</Label>
              <Input id="g-n" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Why you changed it (optional)" />
            </div>
          </div>
          {category !== "unanswerable" ? (
            <div className="space-y-2">
              <p className="text-[13px] font-medium">Evidence quotes</p>
              {evidence.map((ev, i) => (
                <div key={i} className="flex gap-2">
                  <Textarea
                    aria-label={`Evidence ${i + 1}`}
                    className="min-h-14 flex-1"
                    value={ev.quote}
                    onChange={(e) => setEvidence((list) => list.map((x, j) => (j === i ? { ...x, quote: e.target.value } : x)))}
                  />
                  <Input
                    aria-label={`Evidence ${i + 1} page`}
                    type="number"
                    min={1}
                    className="w-20"
                    value={ev.page}
                    onChange={(e) => setEvidence((list) => list.map((x, j) => (j === i ? { ...x, page: Number(e.target.value) } : x)))}
                  />
                  <Button size="icon" variant="ghost" aria-label={`Remove evidence ${i + 1}`} onClick={() => setEvidence((list) => list.filter((_, j) => j !== i))}>
                    <Trash2 />
                  </Button>
                </div>
              ))}
              <p className="text-xs text-muted-foreground">Tip: “Re-select” on the review card lets you pick evidence by highlighting it on the page.</p>
            </div>
          ) : null}
          {save.isError ? (
            <p role="alert" className="text-[13px] text-danger">
              {(save.error as Error).message}
            </p>
          ) : invalid && category !== "unanswerable" && !evidence.length ? (
            <p className="text-[13px] text-danger">An answerable question needs at least one evidence quote.</p>
          ) : null}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" disabled={invalid || save.isPending}>
              Save
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
