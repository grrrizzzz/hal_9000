import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layers } from "lucide-react";
import { useState } from "react";
import { api } from "@/api/client";
import type { IndexProfile } from "@/api/types";
import { ConfirmDialog } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/ui/states";
import { STAGE_LABEL } from "@/lib/format";
import { useJobs } from "@/lib/jobs";
import { STRATEGY_LABEL, type Strategy } from "./run-data";

function ProfileFacts({ p }: { p: IndexProfile }) {
  const facts = [
    ["Chunking", `${STRATEGY_LABEL[p.chunker as Strategy] ?? p.chunker}, ${p.size} tokens${p.parent_size ? ` under ${p.parent_size}` : ""}`],
    ["Overlap", p.overlap ? `${p.overlap} tokens` : "none"],
    ["Header", p.header ? "title, type, vendor, date, section" : "none"],
    ["Parser", p.parser],
    ["Embedding", `${p.embed_model} · ${p.dimensions} dims`],
  ];
  return (
    <dl className="grid gap-x-6 gap-y-1.5 text-[13px] sm:grid-cols-[auto_1fr]">
      {facts.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="font-medium break-words">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ProfilePanel() {
  const q = useQuery({ queryKey: ["profile"], queryFn: () => api.get("/api/profile") });
  const qc = useQueryClient();
  const { jobs } = useJobs();
  const [target, setTarget] = useState<IndexProfile | null>(null);
  const [jobId, setJobId] = useState<number | null>(null);
  const activate = useMutation({
    mutationFn: (name: string) => api.put("/api/profile", { body: { name } }),
    onSuccess: (state) => {
      setTarget(null);
      setJobId(state.reindex_job?.id ?? null);
      qc.setQueryData(["profile"], state);
      void qc.invalidateQueries({ queryKey: ["health"] });
    },
  });

  if (q.isLoading) return <SkeletonRows rows={3} />;
  if (q.isError)
    return (
      <Card>
        <ErrorState error={q.error} onRetry={() => q.refetch()} />
      </Card>
    );
  const state = q.data!;
  const job = jobId !== null ? (jobs.get(jobId) ?? state.reindex_job) : state.reindex_job;
  const others = (state.profiles ?? []).filter((p) => p.name !== state.active?.name);

  return (
    <div className="space-y-4">
      <Card className="p-5" data-testid="active-profile">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-[15px] font-semibold">Active profile</h3>
          {state.active ? <Badge tone="success">{state.active.name}</Badge> : <Badge tone="warning">none</Badge>}
        </div>
        {state.active ? (
          <div className="mt-3">
            <ProfileFacts p={state.active} />
          </div>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">No profile is active yet. Pick one below after the sweep.</p>
        )}
        {job && job.status !== "done" ? (
          <div className="mt-4 rounded-lg border border-border bg-surface-2 p-3" role="status">
            <p className="text-[13px] font-medium">
              Re-indexing in the background{job.status === "failed" ? " failed" : ""}: {job.status === "failed" ? job.error : `${STAGE_LABEL[job.detail] ?? job.detail}…`}
            </p>
            <ProgressBar className="mt-2" value={job.progress ?? 0} tone={job.status === "failed" ? "danger" : "accent"} label="Re-index progress" />
            <p className="mt-1.5 text-xs text-muted-foreground">Answers keep using the previous index until this finishes.</p>
          </div>
        ) : job?.status === "done" && jobId !== null ? (
          <p className="mt-4 text-[13px] text-success" role="status">
            Re-index finished. The new profile is live.
          </p>
        ) : null}
      </Card>

      <div>
        <h3 className="mb-2 text-sm font-semibold">Other profiles</h3>
        {!others.length ? (
          <Card>
            <EmptyState icon={<Layers />} title="No other profiles">
              Profiles appear here once the sweep has built them.
            </EmptyState>
          </Card>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {others.map((p) => (
              <Card key={p.name} className="flex flex-col p-5">
                <p className="font-mono text-[13px] font-medium">{p.name}</p>
                <div className="mt-3 flex-1">
                  <ProfileFacts p={p} />
                </div>
                <Button className="mt-4 self-start" onClick={() => setTarget(p)} disabled={activate.isPending || (!!job && job.status === "running")}>
                  Switch to this profile
                </Button>
              </Card>
            ))}
          </div>
        )}
      </div>
      {activate.isError ? (
        <p role="alert" className="text-[13px] text-danger">
          {(activate.error as Error).message}
        </p>
      ) : null}
      <ConfirmDialog
        open={!!target}
        onOpenChange={(o) => !o && setTarget(null)}
        title={`Switch to ${target?.name}?`}
        description="Every document is re-indexed with this profile in the background. Answers keep using the current index until it finishes; new embeddings may cost a little."
        confirmLabel="Switch and re-index"
        busy={activate.isPending}
        onConfirm={() => target && activate.mutate(target.name)}
      />
    </div>
  );
}
