import type { Citation } from "@/api/types";
import { splitCitations } from "@/lib/citations";
import { cn } from "@/lib/utils";

/** Answer text with `[S#]` markers rendered as chips that open the viewer. */
export function CitationText({
  text,
  citations,
  onOpen,
  streaming,
}: {
  text: string;
  citations: Citation[];
  onOpen: (c: Citation) => void;
  streaming?: boolean;
}) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const segments = splitCitations(text);
  return (
    <div className={cn("text-[15px] leading-relaxed break-words whitespace-pre-wrap", streaming && "stream-caret")} data-testid="answer-text">
      {segments.map((s, i) => {
        if (s.kind === "text") return <span key={i}>{s.text}</span>;
        const c = byMarker.get(s.marker);
        const n = s.marker.slice(1);
        if (!c) {
          return (
            <span key={i} className="mx-0.5 rounded bg-danger-soft px-1 text-xs text-danger" title="This source wasn't supplied">
              [{s.marker}]
            </span>
          );
        }
        return (
          <button
            key={i}
            type="button"
            onClick={() => onOpen(c)}
            aria-label={`Source ${n}: ${c.title}, page ${c.page}`}
            data-marker={s.marker}
            className="mx-0.5 inline-flex h-[1.35em] min-w-[1.6em] translate-y-[-1px] items-center justify-center rounded-md bg-accent-soft px-1 align-middle text-[11px] font-semibold text-accent-foreground ring-1 ring-accent/20 transition-colors hover:bg-accent hover:text-white"
          >
            S{n}
          </button>
        );
      })}
    </div>
  );
}
