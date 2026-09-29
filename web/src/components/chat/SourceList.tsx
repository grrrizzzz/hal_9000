import { FileText } from "lucide-react";
import type { Citation } from "@/api/types";
import { docTypeLabel, formatDocDate } from "@/lib/format";
import { cn } from "@/lib/utils";

export function SourceList({
  sources,
  onOpen,
  showSnippets,
  className,
}: {
  sources: Citation[];
  onOpen: (c: Citation) => void;
  showSnippets?: boolean;
  className?: string;
}) {
  if (!sources.length) return null;
  return (
    <div className={cn("mt-3", className)}>
      <p className="mb-1.5 text-xs font-medium tracking-wide text-muted-foreground uppercase">Sources</p>
      <ol className={cn("grid gap-1.5", !showSnippets && "sm:grid-cols-2")}>
        {sources.map((c) => (
          <li key={c.marker}>
            <button
              type="button"
              onClick={() => onOpen(c)}
              className="group flex w-full items-start gap-2.5 rounded-lg border border-border bg-surface px-3 py-2 text-left transition-colors hover:border-border-strong hover:bg-surface-2"
            >
              <span className="mt-0.5 grid h-5 min-w-5 place-items-center rounded bg-accent-soft px-1 text-[11px] font-semibold text-accent-foreground">
                {c.marker.slice(1)}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 text-[13px] font-medium">
                  <FileText className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                  <span className="truncate">{c.title}</span>
                </span>
                <span className="mt-0.5 block text-xs text-muted-foreground" data-testid="source-meta">
                  {[docTypeLabel(c.doc_type), c.vendor].filter(Boolean).join(" · ")}
                  {/* The date and the page never break inside ("Jun 22, 2023", "p. 2"); lines break before a "·". */}
                  {c.doc_date ? <> <span className="whitespace-nowrap">· {formatDocDate(c.doc_date)}</span></> : null}{" "}
                  <span className="whitespace-nowrap">· p. {c.page}</span>
                </span>
                {showSnippets && c.snippet ? (
                  <span className="mt-1.5 line-clamp-4 block rounded-md bg-mark/40 px-2 py-1 text-[13px] leading-relaxed">{c.snippet}</span>
                ) : null}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
