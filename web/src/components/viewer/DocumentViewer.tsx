/**
 * The document viewer: PDFs through pdf.js at the cited page with the
 * passage highlighted; pseudo-paged formats (DOCX, HTML, Markdown, text)
 * as cleaned text with the span highlighted directly (plan §10).
 */
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Download, TextSelect, ZoomIn, ZoomOut } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, fileUrl } from "@/api/client";
import type { PageView } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { textRanges } from "@/lib/highlight";
import { cn } from "@/lib/utils";

const PdfPage = lazy(() => import("./PdfPage"));

export interface ViewerTarget {
  fileSha: string;
  page: number;
  /** Chunk whose span (from the page view) is highlighted. */
  chunkId?: number | null;
  /** Extra text to highlight (a citation snippet or evidence quotes). */
  quotes?: string[];
  /** File name, when known, decides PDF vs text view; otherwise looked up in the Library. */
  fileName?: string | null;
}

export function usePageView(sha: string, page: number) {
  return useQuery({
    queryKey: ["page", sha, page],
    queryFn: () => api.get("/api/files/{sha}/pages/{n}", { params: { sha, n: page } }),
    staleTime: 5 * 60_000,
  });
}

/** Resolve a file's name (and so whether it's a PDF) from the Library list. */
function useFileName(sha: string, known?: string | null) {
  const docs = useQuery({ queryKey: ["documents", {}], queryFn: () => api.get("/api/documents"), enabled: !known, staleTime: 60_000 });
  if (known) return { name: known, loading: false };
  const hit = docs.data?.find((d) => d.document.file_sha === sha);
  return { name: hit?.original_name ?? null, loading: docs.isLoading };
}

export function spanText(view: PageView | undefined, chunkId?: number | null): string | null {
  if (!view || chunkId === null || chunkId === undefined) return null;
  const s = view.spans?.find((sp) => sp.chunk_id === chunkId);
  return s ? view.text.slice(s.char_start, s.char_end) : null;
}

export function DocumentViewer({
  target,
  onUseSelection,
  className,
}: {
  target: ViewerTarget;
  /** When set, offers "Use selected text as evidence" (gold review re-selection). */
  onUseSelection?: (quote: string, page: number) => void;
  className?: string;
}) {
  const [page, setPage] = useState(target.page);
  const [pageCount, setPageCount] = useState<number | null>(null);
  const [highlights, setHighlights] = useState<number | null>(null);
  const [selection, setSelection] = useState("");
  const [zoom, setZoom] = useState(1);
  const area = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setPage(target.page);
    setHighlights(null);
    setZoom(1); // every document opens fitted to the width
  }, [target.fileSha, target.page, target.chunkId]);

  const view = usePageView(target.fileSha, page);
  const file = useFileName(target.fileSha, target.fileName);
  const isPdf = file.name ? /\.pdf$/i.test(file.name) : false;
  const total = view.data?.page_count ?? pageCount;

  const needles = useMemo(() => {
    if (page !== target.page) return [];
    const span = spanText(view.data, target.chunkId);
    const list = span ? [span] : [];
    for (const q of target.quotes ?? []) if (q && !list.includes(q)) list.push(q);
    return list;
  }, [page, target.page, target.chunkId, target.quotes, view.data]);

  useEffect(() => {
    if (!onUseSelection) return;
    const onSel = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !area.current) return setSelection("");
      const node = sel.anchorNode;
      setSelection(node && area.current.contains(node) ? sel.toString().replace(/\s+/g, " ").trim() : "");
    };
    document.addEventListener("selectionchange", onSel);
    return () => document.removeEventListener("selectionchange", onSel);
  }, [onUseSelection]);

  return (
    <div className={cn("flex h-full min-h-0 flex-col", className)} data-testid="doc-viewer" data-page={page} data-highlights={highlights ?? ""}>
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <div className="flex items-center gap-1">
          <Button size="icon-sm" variant="ghost" aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
            <ChevronLeft />
          </Button>
          <span className="tabular min-w-24 text-center text-[13px] text-muted-foreground" aria-live="polite">
            Page {page}
            {total ? ` of ${total}` : ""}
          </span>
          <Button size="icon-sm" variant="ghost" aria-label="Next page" disabled={!!total && page >= total} onClick={() => setPage((p) => p + 1)}>
            <ChevronRight />
          </Button>
        </div>
        {page !== target.page ? (
          <Button size="sm" variant="link" onClick={() => setPage(target.page)}>
            Back to cited page {target.page}
          </Button>
        ) : needles.length ? (
          <Badge tone={highlights === 0 ? "warning" : highlights === null ? "neutral" : "accent"} data-testid="highlight-status">
            {highlights === 0 ? "Passage not located on this page" : highlights === null ? "Locating passage…" : "Cited passage highlighted"}
          </Badge>
        ) : null}
        {view.data && !view.data.has_text_layer ? <Badge tone="warning">No text layer</Badge> : null}
        <div className="ml-auto flex items-center gap-2">
          {isPdf ? (
            <div className="flex items-center">
              <Button size="icon-sm" variant="ghost" aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom((z) => Math.max(1, z - 0.5))}>
                <ZoomOut />
              </Button>
              <button
                type="button"
                className="tabular min-w-16 rounded px-1 text-center text-xs text-muted-foreground hover:text-foreground disabled:hover:text-muted-foreground"
                onClick={() => setZoom(1)}
                disabled={zoom === 1}
                aria-label="Fit page to width"
              >
                {zoom === 1 ? "Fit width" : `${Math.round(zoom * 100)}%`}
              </button>
              <Button size="icon-sm" variant="ghost" aria-label="Zoom in" disabled={zoom >= 3} onClick={() => setZoom((z) => Math.min(3, z + 0.5))}>
                <ZoomIn />
              </Button>
            </div>
          ) : null}
          {onUseSelection ? (
            <Button size="sm" variant="primary" disabled={!selection} onClick={() => onUseSelection(selection, page)} title="Select text on the page first">
              <TextSelect /> Use selection as evidence
            </Button>
          ) : null}
          {!isPdf && file.name ? (
            <Button size="sm" variant="ghost" asChild>
              <a href={fileUrl(target.fileSha)} download={file.name}>
                <Download /> Original
              </a>
            </Button>
          ) : null}
        </div>
      </div>
      <div ref={area} className="min-h-0 flex-1 overflow-auto bg-surface-2 p-1 sm:p-5">
        {file.loading ? (
          <Skeleton className="mx-auto aspect-[8.5/11] w-full max-w-3xl" />
        ) : isPdf ? (
          <Suspense fallback={<Skeleton className="mx-auto aspect-[8.5/11] w-full max-w-3xl" />}>
            <PdfPage sha={target.fileSha} page={page} needles={needles} zoom={zoom} onPageCount={setPageCount} onHighlights={setHighlights} />
          </Suspense>
        ) : view.isLoading ? (
          <Skeleton className="mx-auto h-96 w-full max-w-3xl" />
        ) : view.isError ? (
          <ErrorState error={view.error} onRetry={() => view.refetch()} />
        ) : view.data ? (
          <TextPage text={view.data.text} needles={needles} onHighlights={setHighlights} />
        ) : null}
      </div>
    </div>
  );
}

function TextPage({ text, needles, onHighlights }: { text: string; needles: string[]; onHighlights: (n: number) => void }) {
  const ranges = useMemo(() => textRanges(text, needles), [text, needles]);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!needles.length) return;
    onHighlights(ranges.length);
    ref.current?.querySelector("mark")?.scrollIntoView({ block: "center" });
  }, [ranges, needles, onHighlights]);
  const parts: ReactNode[] = [];
  let pos = 0;
  ranges.forEach(([a, b], i) => {
    if (a < pos) return;
    parts.push(text.slice(pos, a));
    parts.push(
      <mark key={i} className="hl" data-hl="1">
        {text.slice(a, b)}
      </mark>,
    );
    pos = b;
  });
  parts.push(text.slice(pos));
  return (
    <div ref={ref} className="mx-auto max-w-3xl rounded-md bg-surface px-5 py-6 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap shadow-card ring-1 ring-border sm:px-8">
      {parts}
    </div>
  );
}
