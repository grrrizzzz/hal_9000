import { useState, type ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { Segmented } from "@/components/ui/segmented";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];

export function Legend({ items }: { items: { label: string; color: string; shape?: "dot" | "diamond" | "line" }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-label="Legend">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5">
          {i.shape === "diamond" ? (
            <span aria-hidden className="size-2.5 rotate-45 rounded-[1px]" style={{ background: i.color }} />
          ) : i.shape === "line" ? (
            <span aria-hidden className="h-0.5 w-4 rounded" style={{ background: i.color }} />
          ) : (
            <span aria-hidden className="size-2.5 rounded-full" style={{ background: i.color }} />
          )}
          {i.label}
        </li>
      ))}
    </ul>
  );
}

/** A chart with its table-view twin (every value readable without the chart). */
export function ChartCard({
  title,
  description,
  legend,
  chart,
  table,
  controls,
  testId,
}: {
  title: string;
  description: ReactNode;
  legend?: ReactNode;
  chart: ReactNode;
  table: ReactNode;
  controls?: ReactNode;
  testId?: string;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <Card className="min-w-0" data-testid={testId}>
      <div className="flex flex-wrap items-start justify-between gap-3 px-5 pt-4">
        <div className="min-w-0 flex-1 basis-64">
          <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
          <p className="mt-0.5 text-[13px] text-muted-foreground">{description}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {controls}
          <Segmented<"chart" | "table">
            label={`${title}: view`}
            value={view}
            onChange={setView}
            options={[
              { value: "chart", label: "Chart" },
              { value: "table", label: "Table" },
            ]}
          />
        </div>
      </div>
      {legend ? <div className="px-5 pt-3">{legend}</div> : null}
      <div className="px-2 pt-2 pb-4 sm:px-3">{view === "chart" ? chart : <div className="max-h-80 overflow-auto px-3">{table}</div>}</div>
    </Card>
  );
}

export function TooltipBox({ title, rows }: { title: ReactNode; rows: [ReactNode, ReactNode][] }) {
  return (
    <div className="max-w-64 rounded-lg border border-border bg-surface px-3 py-2 text-xs shadow-pop">
      <p className="mb-1 font-medium text-foreground">{title}</p>
      <dl className="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5">
        {rows.map(([k, v], i) => (
          <div key={i} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="tabular text-right text-foreground">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export const axisProps = {
  stroke: "var(--chart-axis)",
  tick: { fill: "var(--chart-muted)", fontSize: 11 },
  tickLine: false,
} as const;
