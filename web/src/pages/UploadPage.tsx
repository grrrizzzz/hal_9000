import { AlertCircle, CheckCircle2, Copy, Loader2, RotateCw } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import { api } from "@/api/client";
import type { Job, UploadResult } from "@/api/types";
import { Page, PageHeader } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress";
import { DropZone } from "@/components/upload/DropZone";
import { PreUploadSummary } from "@/components/upload/PreUploadSummary";
import { STAGE_LABEL } from "@/lib/format";
import { useJobs } from "@/lib/jobs";
import { summarize } from "@/lib/upload/classify";
import { uploadOne, uploadQueue } from "@/lib/upload/queue";
import type { PickedFile } from "@/lib/upload/walker";
import { cn } from "@/lib/utils";

interface Row {
  id: number;
  file: File;
  relativePath: string;
  phase: "waiting" | "uploading" | "sent" | "error";
  uploaded: number;
  result?: UploadResult;
  error?: string;
  /** The job as returned by a retry, until the job stream reports on it. */
  retried?: Job;
}

type Status = { label: string; tone: "neutral" | "accent" | "success" | "warning" | "danger"; progress: number; done: boolean; failed: boolean };

function statusOf(row: Row, job: Job | undefined): Status {
  if (row.phase === "waiting") return { label: "Waiting", tone: "neutral", progress: 0, done: false, failed: false };
  if (row.phase === "uploading") return { label: `Uploading ${Math.round(row.uploaded * 100)}%`, tone: "accent", progress: row.uploaded * 0.3, done: false, failed: false };
  if (row.phase === "error") return { label: row.error ?? "Upload failed", tone: "danger", progress: 1, done: true, failed: true };
  const r = row.result!;
  if (r.status === "duplicate") return { label: "Already in library", tone: "neutral", progress: 1, done: true, failed: false };
  if (r.status === "skipped") return { label: r.reason ? `Skipped: ${r.reason}` : "Skipped", tone: "warning", progress: 1, done: true, failed: false };
  if (r.status === "failed") return { label: r.reason ?? "Rejected", tone: "danger", progress: 1, done: true, failed: true };
  if (!job) return { label: "Queued", tone: "neutral", progress: 0.3, done: false, failed: false };
  if (job.status === "done") return { label: "Indexed", tone: "success", progress: 1, done: true, failed: false };
  if (job.status === "failed") return { label: job.error ?? "Indexing failed", tone: "danger", progress: 1, done: true, failed: true };
  if (job.status === "queued") return { label: "Queued", tone: "neutral", progress: 0.3, done: false, failed: false };
  return { label: `${STAGE_LABEL[job.detail] ?? "Processing"}…`, tone: "accent", progress: 0.3 + 0.7 * (job.progress ?? 0), done: false, failed: false };
}

export function UploadPage() {
  const [staged, setStaged] = useState<PickedFile[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const { jobs } = useJobs();
  const seq = useRef(0);
  const summary = useMemo(() => summarize(staged), [staged]);

  const patch = (id: number, p: Partial<Row>) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...p } : r)));

  // Every upload, from any batch or retry, goes through the one shared queue (three at a time).
  const upload = (row: Row) =>
    uploadQueue.run(async () => {
      patch(row.id, { phase: "uploading", uploaded: 0, error: undefined });
      try {
        const result = await uploadOne(row.file, row.relativePath, (f) => patch(row.id, { uploaded: f }));
        patch(row.id, { phase: "sent", uploaded: 1, result });
      } catch (e) {
        patch(row.id, { phase: "error", error: (e as Error).message });
      }
    });

  const start = () => {
    const batch: Row[] = summary.added.map((f) => ({ id: ++seq.current, file: f.file, relativePath: f.relativePath, phase: "waiting", uploaded: 0 }));
    setRows((rs) => [...batch, ...rs]);
    setStaged([]);
    batch.forEach((row) => void upload(row));
  };

  const retry = async (row: Row) => {
    const jobId = row.result?.job_id;
    if (row.phase === "sent" && jobId !== null && jobId !== undefined) {
      // The file is stored; only its processing failed. Re-run the job.
      try {
        const job = await api.post("/api/jobs/{id}/retry", { params: { id: jobId } });
        patch(row.id, { result: { ...row.result!, job_id: job.id ?? jobId }, retried: job });
      } catch (e) {
        patch(row.id, { phase: "error", error: (e as Error).message });
      }
    } else {
      patch(row.id, { phase: "waiting", uploaded: 0, error: undefined, result: undefined });
      void upload(row);
    }
  };

  const jobOf = (r: Row): Job | undefined => {
    const id = r.result?.job_id;
    const live = id === null || id === undefined ? undefined : jobs.get(id);
    if (!r.retried || !live) return live ?? r.retried;
    // After a retry, the newer of the retry response and the stream's latest event wins.
    return Date.parse(live.updated_at ?? "") >= Date.parse(r.retried.updated_at ?? "") ? live : r.retried;
  };
  const view = rows.map((r) => ({ row: r, s: statusOf(r, jobOf(r)) }));
  const running = view.some((v) => v.row.phase === "uploading" || v.row.phase === "waiting");
  const indexed = view.filter((v) => v.s.label === "Indexed").length;
  const failed = view.filter((v) => v.s.failed).length;
  const inFlight = view.filter((v) => !v.s.done).length;

  return (
    <Page>
      <PageHeader
        title="Upload"
        description="Add documents about the house. Folders keep their structure, and each folder becomes a collection you can filter by."
      />
      <div className="mt-6 space-y-4">
        <DropZone onPick={(files) => setStaged((s) => [...s, ...files])} disabled={false} />
        {staged.length ? <PreUploadSummary summary={summary} onUpload={start} onClear={() => setStaged([])} /> : null}

        {rows.length ? (
          <Card className="overflow-hidden" data-testid="upload-progress">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-5 py-3.5">
              <h2 className="text-[15px] font-semibold">Progress</h2>
              <p className="tabular text-[13px] text-muted-foreground" aria-live="polite">
                {indexed} of {rows.length} indexed
                {failed ? ` · ${failed} failed` : ""}
                {inFlight ? ` · ${inFlight} in progress` : ""}
                {running ? <Loader2 className="ml-2 inline size-3.5 animate-spin" aria-hidden /> : null}
              </p>
            </div>
            <ul className="divide-y divide-border">
              {view.map(({ row, s }) => (
                <li key={row.id} className="px-5 py-3" data-testid="upload-row" data-status={s.label}>
                  <div className="flex items-center gap-3">
                    <span className="min-w-0 flex-1 truncate font-mono text-xs" title={row.relativePath}>
                      {row.relativePath}
                    </span>
                    <Badge tone={s.tone} className={cn("max-w-[55%]", s.failed && "whitespace-normal")}>
                      {s.label === "Indexed" ? <CheckCircle2 /> : s.failed ? <AlertCircle /> : s.label === "Already in library" ? <Copy /> : null}
                      <span className="truncate">{s.label}</span>
                    </Badge>
                    {s.failed ? (
                      <Button size="sm" variant="ghost" onClick={() => void retry(row)} aria-label={`Retry ${row.relativePath}`}>
                        <RotateCw /> Retry
                      </Button>
                    ) : null}
                  </div>
                  {!s.done || s.failed ? (
                    <ProgressBar className="mt-2" value={s.progress} tone={s.failed ? "danger" : "accent"} label={`${row.relativePath} progress`} />
                  ) : null}
                </li>
              ))}
            </ul>
            {indexed && !inFlight ? (
              <div className="border-t border-border px-5 py-3 text-[13px] text-muted-foreground">
                Done. <Link className="font-medium text-accent-foreground underline-offset-4 hover:underline" to="/library">See them in the Library</Link> or{" "}
                <Link className="font-medium text-accent-foreground underline-offset-4 hover:underline" to="/">ask a question</Link>.
              </div>
            ) : null}
          </Card>
        ) : null}
      </div>
    </Page>
  );
}
