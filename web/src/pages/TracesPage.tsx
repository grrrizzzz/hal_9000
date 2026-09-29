import { useQuery } from "@tanstack/react-query";
import { Activity, ExternalLink } from "lucide-react";
import { Link, useNavigate } from "react-router";
import { api } from "@/api/client";
import { Page, PageHeader } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/ui/states";
import { formatDateTime, formatMs, formatUsd, outcomeLabel, outcomeTone, relativeTime } from "@/lib/format";

/** Optional: set VITE_GRAFANA_URL at build time to link the dashboards. */
export const GRAFANA_URL = (import.meta.env.VITE_GRAFANA_URL as string | undefined) || null;

export function TracesPage() {
  const q = useQuery({ queryKey: ["traces"], queryFn: () => api.get("/api/traces", { query: { limit: 100 } }) });
  const navigate = useNavigate();
  return (
    <Page wide>
      <PageHeader
        title="Traces"
        description="Recent questions with their outcome, latency and cost. Open one to see every stage in the Query Inspector."
        actions={
          GRAFANA_URL ? (
            <Button asChild>
              <a href={GRAFANA_URL} target="_blank" rel="noreferrer">
                Grafana <ExternalLink />
              </a>
            </Button>
          ) : null
        }
      />
      <Card className="mt-6 overflow-hidden">
        {q.isLoading ? (
          <SkeletonRows rows={6} className="p-4" />
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => q.refetch()} />
        ) : !q.data?.length ? (
          <EmptyState icon={<Activity />} title="No traces yet" action={<Button asChild variant="primary"><Link to="/">Ask a question</Link></Button>}>
            Every question you ask is traced here.
          </EmptyState>
        ) : (
          <div className="relative overflow-x-auto">
            <table className="w-full min-w-[36rem] text-[13px]">
              <thead className="bg-surface-2 text-left text-xs text-muted-foreground">
                <tr>
                  <th className="py-2 pl-5 font-medium">When</th>
                  <th className="py-2 font-medium">Outcome</th>
                  <th className="py-2 text-right font-medium">First token</th>
                  <th className="py-2 text-right font-medium">Total</th>
                  <th className="py-2 text-right font-medium">Cost</th>
                  <th className="py-2 pr-5 text-right font-medium">
                    <span className="sr-only">Open</span>
                  </th>
                </tr>
              </thead>
              <tbody className="tabular divide-y divide-border">
                {q.data.map((t) => (
                  <tr key={t.id} className="cursor-pointer hover:bg-surface-2" onClick={() => navigate(`/traces/${t.id}`)}>
                    <td className="py-2.5 pl-5" title={formatDateTime(t.created_at)}>
                      {relativeTime(t.created_at)}
                    </td>
                    <td className="py-2.5">
                      <Badge tone={outcomeTone(t.outcome)}>{outcomeLabel(t.outcome)}</Badge>
                    </td>
                    <td className="py-2.5 text-right">{formatMs(t.ttft_ms)}</td>
                    <td className="py-2.5 text-right">{formatMs(t.total_ms)}</td>
                    <td className="py-2.5 text-right">{formatUsd(t.cost_usd)}</td>
                    <td className="py-2.5 pr-5 text-right">
                      <Link to={`/traces/${t.id}`} className="font-medium text-accent-foreground hover:underline" onClick={(e) => e.stopPropagation()}>
                        Inspect<span className="sr-only"> trace from {formatDateTime(t.created_at)}</span>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </Page>
  );
}
