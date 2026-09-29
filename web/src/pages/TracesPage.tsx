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
            <table className="w-full text-[13px]">
              <thead className="bg-surface-2 text-left text-xs text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3 pl-4 font-medium sm:pl-5">Question</th>
                  <th className="hidden py-2 pr-3 font-medium sm:table-cell">Outcome</th>
                  <th className="hidden py-2 pr-3 text-right font-medium md:table-cell">First token</th>
                  <th className="py-2 pr-3 text-right font-medium">Total</th>
                  <th className="py-2 pr-4 text-right font-medium sm:pr-5">Cost</th>
                </tr>
              </thead>
              <tbody className="tabular divide-y divide-border">
                {q.data.map((t) => (
                  <tr key={t.id} className="cursor-pointer hover:bg-surface-2" onClick={() => navigate(`/traces/${t.id}`)} data-testid="trace-row">
                    {/* max-w-0 + w-full lets the question truncate while the numbers keep their width. */}
                    <td className="w-full max-w-0 py-2.5 pr-3 pl-4 sm:pl-5">
                      <Link
                        to={`/traces/${t.id}`}
                        className="block truncate font-medium hover:underline"
                        title={t.question ?? undefined}
                        onClick={(e) => e.stopPropagation()}
                      >
                        {t.question || <span className="text-muted-foreground">Question {t.id.slice(0, 8)}</span>}
                      </Link>
                      <span className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground" title={formatDateTime(t.created_at)}>
                        {relativeTime(t.created_at)}
                        <Badge tone={outcomeTone(t.outcome)} className="sm:hidden">
                          {outcomeLabel(t.outcome)}
                        </Badge>
                      </span>
                    </td>
                    <td className="hidden py-2.5 pr-3 sm:table-cell">
                      <Badge tone={outcomeTone(t.outcome)}>{outcomeLabel(t.outcome)}</Badge>
                    </td>
                    <td className="hidden py-2.5 pr-3 text-right whitespace-nowrap md:table-cell">{formatMs(t.ttft_ms)}</td>
                    <td className="py-2.5 pr-3 text-right whitespace-nowrap">{formatMs(t.total_ms)}</td>
                    <td className="py-2.5 pr-4 text-right whitespace-nowrap sm:pr-5">{formatUsd(t.cost_usd)}</td>
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
