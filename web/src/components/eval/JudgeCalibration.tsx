/**
 * Judge calibration (plan §8.4.1, gate G-K1b): Mike labels generated answers
 * with the judge's rubric; the judge must agree on ≥ 85% of labels. The
 * judge's own labels stay hidden until yours are saved, so they can't sway you.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Gavel, XCircle } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { api } from "@/api/client";
import type { Citation, JudgeLabel, JudgeReasons, RubricLabels } from "@/api/types";
import { CitationText } from "@/components/chat/CitationText";
import { SourceList } from "@/components/chat/SourceList";
import { ViewerDialog } from "@/components/viewer/ViewerDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Select } from "@/components/ui/input";
import { Segmented } from "@/components/ui/segmented";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/ui/states";
import { cn } from "@/lib/utils";

export const AGREEMENT_TARGET = 0.85;

type Field = keyof RubricLabels;
const FIELDS: { key: Field; label: string }[] = [
  { key: "correctness", label: "Correct" },
  { key: "faithful", label: "Faithful to sources" },
  { key: "citations_correct", label: "Citations correct" },
  { key: "abstention_correct", label: "Abstention correct" },
];

/**
 * Label-level agreement, counted exactly as S5's gate counts it (eval/judge.py
 * `agreement`): every label Mike gave; N/A against the judge's Yes or No is a
 * disagreement (only N/A on both sides is skipped); an answer the judge failed
 * to grade disagrees on every label.
 */
export function agreement(labels: JudgeLabel[]): { agree: number; total: number } {
  let agree = 0;
  let total = 0;
  for (const l of labels) {
    if (!l.human) continue;
    for (const { key } of FIELDS) {
      const h = l.human[key] ?? null;
      const j = l.judge ? (l.judge[key] ?? null) : null;
      if (key === "abstention_correct" && h === null && j === null) continue;
      total++;
      if (l.judge && h === j) agree++;
    }
  }
  return { agree, total };
}

export function JudgeCalibration() {
  const [run, setRun] = useState("");
  const [viewer, setViewer] = useState<Citation | null>(null);
  const q = useQuery({ queryKey: ["judge-labels", run], queryFn: () => api.get("/api/eval/judge-labels", { query: run ? { run } : {} }) });
  const all = useQuery({ queryKey: ["judge-labels", ""], queryFn: () => api.get("/api/eval/judge-labels"), enabled: !!run });
  const labels = useMemo(() => q.data ?? [], [q.data]);
  const runs = [...new Set((run ? all.data : q.data)?.map((l) => l.eval_run_id).filter(Boolean) as string[])];
  const done = labels.filter((l) => l.human).length;
  const { agree, total } = agreement(labels);
  const rate = total ? agree / total : null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <Card className="px-4 py-3">
          <p className="text-xs text-muted-foreground">Labelled</p>
          <p className="mt-1 text-2xl font-semibold" data-testid="judge-progress">
            {done} <span className="text-base font-normal text-muted-foreground">of {labels.length}</span>
          </p>
        </Card>
        <Card className="px-4 py-3">
          <p className="text-xs text-muted-foreground">Judge agrees with you</p>
          <p className="mt-1 text-2xl font-semibold">
            {rate === null ? "—" : `${Math.round(rate * 100)}%`}
            {rate !== null ? (
              <Badge tone={rate >= AGREEMENT_TARGET ? "success" : "warning"} className="ml-2 align-middle">
                {rate >= AGREEMENT_TARGET ? <CheckCircle2 /> : <XCircle />}
                {rate >= AGREEMENT_TARGET ? "meets" : "below"} 85%
              </Badge>
            ) : null}
          </p>
        </Card>
        <Card className="col-span-2 px-4 py-3 sm:col-span-1">
          <label htmlFor="judge-run" className="text-xs text-muted-foreground">
            Calibration run
          </label>
          <Select id="judge-run" className="mt-1.5" value={run} onChange={(e) => setRun(e.target.value)}>
            <option value="">All runs</option>
            {runs.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </Select>
        </Card>
      </div>

      {q.isLoading ? (
        <SkeletonRows rows={3} />
      ) : q.isError ? (
        <Card>
          <ErrorState error={q.error} onRetry={() => q.refetch()} />
        </Card>
      ) : !labels.length ? (
        <Card>
          <EmptyState icon={<Gavel />} title="No answers to label yet">
            Run <code className="font-mono text-xs">hal9000 kb eval judge-prep</code> after the retrieval sweep to generate calibration answers.
          </EmptyState>
        </Card>
      ) : (
        <ol className="space-y-3">
          {labels.map((l, i) => (
            <li key={l.id}>
              <LabelCard label={l} n={i + 1} onOpen={setViewer} />
            </li>
          ))}
        </ol>
      )}
      <ViewerDialog
        target={viewer ? { fileSha: viewer.file_sha, page: viewer.page, chunkId: viewer.chunk_id, quotes: viewer.snippet ? [viewer.snippet] : [] } : null}
        title={viewer ? `${viewer.marker} · ${viewer.title}` : ""}
        description={viewer ? `Page ${viewer.page}` : ""}
        onClose={() => setViewer(null)}
      />
    </div>
  );
}

const yesNo = [
  { value: "yes", label: "Yes", tone: "success" as const },
  { value: "no", label: "No", tone: "danger" as const },
];

function LabelCard({ label, n, onOpen }: { label: JudgeLabel; n: number; onOpen: (c: Citation) => void }) {
  const qc = useQueryClient();
  // an old N/A abstention label must be answered again (the judge always says Yes or No)
  const [draft, setDraft] = useState<Partial<RubricLabels>>(label.human ? { ...label.human, abstention_correct: label.human.abstention_correct ?? undefined } : {});
  const [editing, setEditing] = useState(!label.human);
  const gold = useQuery({ queryKey: ["gold-item", label.gold_item_id], queryFn: () => api.get("/api/eval/gold/{id}", { params: { id: label.gold_item_id } }), staleTime: 5 * 60_000 });
  const save = useMutation({
    mutationFn: (human: RubricLabels) => api.put("/api/eval/judge-labels/{id}", { params: { id: label.id }, body: { human } }),
    onSuccess: () => {
      setEditing(false);
      void qc.invalidateQueries({ queryKey: ["judge-labels"] });
    },
  });
  const complete =
    draft.correctness !== undefined && draft.faithful !== undefined && draft.citations_correct !== undefined && draft.abstention_correct !== undefined && draft.abstention_correct !== null;
  const bool = (v: boolean | null | undefined) => (v === true ? "yes" : v === false ? "no" : undefined);
  const g = gold.data;

  return (
    <Card data-testid="judge-card" data-labelled={label.human ? "true" : "false"}>
      <div className="space-y-4 p-5">
        <div className="min-w-0 space-y-3">
          <p className="text-xs text-muted-foreground">Answer {n}</p>
          <h3 className="font-semibold">{label.question}</h3>
          <div className="rounded-lg border border-border bg-surface-2 px-4 py-3">
            <CitationText text={label.answer} citations={label.citations ?? []} onOpen={onOpen} />
            <SourceList sources={label.citations ?? []} onOpen={onOpen} />
          </div>
          <details className="text-sm">
            <summary className="cursor-pointer text-[13px] font-medium text-muted-foreground hover:text-foreground">Reference answer and evidence</summary>
            {gold.isLoading ? (
              <p className="mt-2 text-muted-foreground">Loading…</p>
            ) : g ? (
              <div className="mt-2 space-y-2">
                <p>{g.reference_answer ?? <span className="text-muted-foreground">Unanswerable: the documents don't say.</span>}</p>
                {g.evidence?.map((e, i) => (
                  <blockquote key={i} className="border-l-2 border-mark-strong pl-3 text-[13px] text-muted-foreground">
                    {e.quote} <span className="whitespace-nowrap">(p. {e.page})</span>
                  </blockquote>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-muted-foreground">The question-set item couldn't be loaded.</p>
            )}
          </details>
        </div>

        <div className="border-t border-border pt-4">
          {editing ? (
            <form
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                if (complete) save.mutate(draft as RubricLabels);
              }}
            >
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <Rubric label="Correct?">
                  <Segmented
                    label={`Answer ${n}: correctness`}
                    value={draft.correctness}
                    onChange={(v) => setDraft((d) => ({ ...d, correctness: v }))}
                    options={[
                      { value: "correct", label: "Correct", tone: "success" },
                      { value: "partial", label: "Partial", tone: "warning" },
                      { value: "wrong", label: "Wrong", tone: "danger" },
                    ]}
                  />
                </Rubric>
                <Rubric label="Faithful to the sources?">
                  <Segmented label={`Answer ${n}: faithful`} value={bool(draft.faithful)} onChange={(v) => setDraft((d) => ({ ...d, faithful: v === "yes" }))} options={yesNo} />
                </Rubric>
                <Rubric label="Citations correct?">
                  <Segmented label={`Answer ${n}: citations correct`} value={bool(draft.citations_correct)} onChange={(v) => setDraft((d) => ({ ...d, citations_correct: v === "yes" }))} options={yesNo} />
                </Rubric>
                <Rubric label="Abstention correct?" hint="Yes if it answered a question the documents answer, or declined one they don't.">
                  <Segmented label={`Answer ${n}: abstention correct`} value={bool(draft.abstention_correct)} onChange={(v) => setDraft((d) => ({ ...d, abstention_correct: v === "yes" }))} options={yesNo} />
                </Rubric>
              </div>
              {save.isError ? (
                <p role="alert" className="text-[13px] text-danger">
                  {(save.error as Error).message}
                </p>
              ) : null}
              <div className="flex gap-2">
                <Button type="submit" variant="primary" size="sm" disabled={!complete || save.isPending}>
                  Save labels
                </Button>
                {label.human ? (
                  <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
                    Cancel
                  </Button>
                ) : null}
              </div>
              <p className="text-xs text-muted-foreground">The judge's labels appear once you've saved yours.</p>
            </form>
          ) : label.human ? (
            <Comparison human={label.human} judge={label.judge} reasons={label.judge_reasons} onEdit={() => setEditing(true)} />
          ) : null}
        </div>
      </div>
    </Card>
  );
}

function Rubric({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="space-y-1">
      <p className="text-[13px] font-medium">{label}</p>
      {children}
      {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

const show = (v: unknown) => (v === true ? "Yes" : v === false ? "No" : v === null || v === undefined ? "N/A" : String(v)[0].toUpperCase() + String(v).slice(1));

function Comparison({
  human,
  judge,
  reasons,
  onEdit,
}: {
  human: RubricLabels;
  judge: RubricLabels | null | undefined;
  reasons: JudgeReasons | null | undefined;
  onEdit: () => void;
}) {
  return (
    <div data-testid="judge-comparison" className="max-w-2xl">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="text-left text-xs text-muted-foreground">
            <th className="pb-1.5 font-medium" scope="col">
              Rubric
            </th>
            <th className="pb-1.5 font-medium" scope="col">
              You
            </th>
            <th className="pb-1.5 font-medium" scope="col">
              Judge
            </th>
          </tr>
        </thead>
        <tbody>
          {FIELDS.map(({ key, label }) => {
            const h = human[key] ?? null;
            const j = judge ? (judge[key] ?? null) : null;
            // as the gate counts it: N/A against a Yes or No disagrees
            const differs = !!judge && h !== j;
            const reason = judge ? reasons?.[key] : undefined;
            return (
              <tr key={key} className="border-t border-border align-top">
                <th scope="row" className="py-1.5 pr-2 text-left font-normal text-muted-foreground">
                  {label}
                </th>
                <td className="py-1.5 pr-3 font-medium">{show(h)}</td>
                <td className={cn("py-1.5", differs && "text-warning")}>
                  <span className={cn(differs && "font-medium")}>{judge ? show(j) : "—"}</span>
                  {differs ? <span className="sr-only"> (disagrees)</span> : null}
                  {reason ? (
                    <p data-testid="judge-reason" className={cn("mt-0.5 text-xs", differs ? "text-foreground" : "text-muted-foreground")}>
                      {reason}
                    </p>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {judge && !reasons ? (
        <p className="mt-2 text-xs text-muted-foreground">
          The judge gave no reasons for this answer (it was judged before reasons were recorded). Run{" "}
          <code className="font-mono">hal9000 kb eval judge-prep "$SWEEP" --rejudge</code> to have it judged again with reasons.
        </p>
      ) : null}
      <Button size="sm" variant="ghost" className="mt-2 -ml-2" onClick={onEdit}>
        Change my labels
      </Button>
    </div>
  );
}
