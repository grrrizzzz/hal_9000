/** The Query Inspector (plan §9.4): one question's stage waterfall, retrieved chunks with scores and ranks, context and grounding. */
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ExternalLink } from "lucide-react";
import { Link, useParams } from "react-router";
import { api } from "@/api/client";
import type { StageTiming, Trace } from "@/api/types";
import { Page } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { formatDateTime, formatMs, formatUsd, outcomeLabel, outcomeTone, plural, STAGE_LABEL } from "@/lib/format";
import { cn } from "@/lib/utils";
import { GRAFANA_URL } from "./TracesPage";

export function TraceInspectorPage() {
  const id = useParams().id!;
  const q = useQuery({ queryKey: ["trace", id], queryFn: () => api.get("/api/traces/{id}", { params: { id } }) });
  return (
    <Page wide>
      <Button asChild variant="link" size="sm" className="mb-3 text-muted-foreground">
        <Link to="/traces">
          <ArrowLeft /> Traces
        </Link>
      </Button>
      {q.isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-10 w-2/3" />
          <Skeleton className="h-56" />
          <Skeleton className="h-56" />
        </div>
      ) : q.isError || !q.data ? (
        <ErrorState error={q.error ?? "Not found"} onRetry={() => q.refetch()} />
      ) : (
        <Inspector t={q.data} />
      )}
    </Page>
  );
}

function Inspector({ t }: { t: Trace }) {
  const tokens = Object.entries(t.tokens_by_model ?? {});
  const promptTokens = tokens.reduce((n, [, v]) => n + (v.prompt ?? 0), 0);
  const stats: [string, string][] = [
    ["Total", formatMs(t.total_ms)],
    ["First token", formatMs(t.ttft_ms)],
    ["Context", plural(t.context_tokens ?? 0, "token")],
    ["Prompt", plural(promptTokens, "token")],
    ["Cost", formatUsd(t.cost_usd)],
  ];
  return (
    <div className="space-y-5">
      <div>
        <p className="text-xs text-muted-foreground">
          {formatDateTime(t.created_at)} · profile {t.profile ?? "—"} · <span className="font-mono">{t.id}</span>
        </p>
        <h1 className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">{t.question || "(no question text)"}</h1>
        {t.standalone_query && t.standalone_query !== t.question ? (
          <p className="mt-1 text-sm text-muted-foreground">Rewritten as: “{t.standalone_query}”</p>
        ) : null}
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Badge tone={outcomeTone(t.outcome)}>{outcomeLabel(t.outcome)}</Badge>
          {t.grounding ? (
            <Badge tone={t.grounding === "verified" ? "success" : t.grounding === "failed" ? "warning" : "neutral"}>
              grounding {t.grounding}
            </Badge>
          ) : null}
          {t.error ? <Badge tone="danger">{t.error}</Badge> : null}
          {GRAFANA_URL ? (
            <a className="ml-auto inline-flex items-center gap-1 text-xs text-accent-foreground hover:underline" href={GRAFANA_URL} target="_blank" rel="noreferrer">
              Grafana <ExternalLink className="size-3" />
            </a>
          ) : null}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {stats.map(([k, v]) => (
          <Card key={k} className="px-4 py-3">
            <p className="text-xs text-muted-foreground">{k}</p>
            <p className="mt-0.5 text-lg font-semibold">{v}</p>
          </Card>
        ))}
      </div>

      <Waterfall stages={t.stages ?? []} total={t.total_ms ?? undefined} />

      <Card className="overflow-hidden">
        <div className="px-5 pt-4 pb-3">
          <h2 className="text-[15px] font-semibold">Retrieved chunks</h2>
          <p className="mt-0.5 text-[13px] text-muted-foreground">Ranked after fusion{t.hits?.some((h) => h.rerank_score !== null && h.rerank_score !== undefined) ? " and re-ranking" : ""}. Selected chunks went into the prompt.</p>
        </div>
        <div className="relative overflow-x-auto">
          <table className="w-full min-w-[34rem] text-[13px]">
            <thead className="bg-surface-2 text-left text-xs text-muted-foreground">
              <tr>
                <th className="py-2 pl-5 font-medium">Rank</th>
                <th className="py-2 font-medium">Chunk</th>
                <th className="py-2 text-right font-medium">Fused</th>
                <th className="py-2 text-right font-medium">Dense</th>
                <th className="py-2 text-right font-medium">BM25</th>
                <th className="py-2 text-right font-medium">Rerank</th>
                <th className="py-2 pr-5 text-right font-medium">In context</th>
              </tr>
            </thead>
            <tbody className="tabular divide-y divide-border">
              {(t.hits ?? []).map((h) => (
                <tr key={h.chunk_id} className={cn(!h.selected && "text-muted-foreground")}>
                  <td className="py-2 pl-5">{h.rank}</td>
                  <td className="py-2 font-mono">#{h.chunk_id}</td>
                  <td className="py-2 text-right">{h.score.toFixed(4)}</td>
                  <td className="py-2 text-right">{h.dense_score?.toFixed(3) ?? "—"}</td>
                  <td className="py-2 text-right">{h.bm25_score?.toFixed(2) ?? "—"}</td>
                  <td className="py-2 text-right">{h.rerank_score?.toFixed(3) ?? "—"}</td>
                  <td className="py-2 pr-5 text-right">{h.selected ? <Badge tone="accent">selected</Badge> : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {tokens.length ? (
        <Card className="p-5">
          <h2 className="text-[15px] font-semibold">Tokens by model</h2>
          <table className="tabular mt-3 w-full text-[13px]">
            <thead className="text-left text-xs text-muted-foreground">
              <tr>
                <th className="pb-1.5 font-medium">Model</th>
                <th className="pb-1.5 text-right font-medium">Prompt</th>
                <th className="pb-1.5 text-right font-medium">Cached</th>
                <th className="pb-1.5 text-right font-medium">Completion</th>
              </tr>
            </thead>
            <tbody>
              {tokens.map(([model, v]) => (
                <tr key={model} className="border-t border-border">
                  <td className="py-1.5 font-mono text-xs">{model}</td>
                  <td className="py-1.5 text-right">{v.prompt ?? 0}</td>
                  <td className="py-1.5 text-right">{v.cached ?? 0}</td>
                  <td className="py-1.5 text-right">{v.completion ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ) : null}
    </div>
  );
}

function Waterfall({ stages, total }: { stages: StageTiming[]; total?: number }) {
  const end = Math.max(total ?? 0, ...stages.map((s) => s.started_ms + s.duration_ms), 1);
  return (
    <Card className="p-5" data-testid="waterfall">
      <h2 className="text-[15px] font-semibold">Stages</h2>
      {!stages.length ? (
        <p className="mt-2 text-sm text-muted-foreground">No stage timings were recorded.</p>
      ) : (
        <ol className="mt-4 space-y-2">
          {stages.map((s, i) => {
            const left = (s.started_ms / end) * 100;
            const width = Math.max(0.4, (s.duration_ms / end) * 100);
            return (
              <li key={i} className="grid grid-cols-[7.5rem_1fr_4.5rem] items-center gap-3 text-[13px] sm:grid-cols-[11rem_1fr_5rem]">
                <span className="truncate" style={{ paddingLeft: `${(s.depth ?? 0) * 0.75}rem` }} title={STAGE_LABEL[s.stage] ?? s.stage}>
                  <span className="font-mono text-xs">{s.stage}</span>
                </span>
                <span className="relative h-2.5 rounded-full bg-subtle" aria-hidden>
                  <span
                    className={cn("absolute inset-y-0 rounded-full", s.error ? "bg-danger" : s.stage === "generate" ? "bg-series-2" : "bg-series-1")}
                    style={{ left: `${left}%`, width: `${width}%` }}
                  />
                </span>
                <span className="tabular text-right text-muted-foreground">{formatMs(s.duration_ms)}</span>
              </li>
            );
          })}
        </ol>
      )}
    </Card>
  );
}
