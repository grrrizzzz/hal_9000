/**
 * One PDF page with pdf.js (react-pdf) and a highlight placed by fuzzy
 * matching the cleaned chunk text against the page's text items
 * (customTextRenderer), since chunk offsets refer to cleaned text (plan §10).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import { fileUrl } from "@/api/client";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { cn } from "@/lib/utils";
import { highlightRanges, mergeBoxes, renderItem, type Box } from "@/lib/highlight";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

const DOC_OPTIONS = { withCredentials: true, isEvalSupported: false } as const;

export interface PdfPageProps {
  sha: string;
  page: number;
  needles: string[];
  onPageCount?: (n: number) => void;
  onHighlights?: (count: number) => void;
  /** 1 = fit the width (up to 900 px). */
  zoom?: number;
}

const sameBoxes = (a: Box[], b: Box[]) =>
  a.length === b.length && a.every((x, i) => Math.abs(x.x - b[i].x) < 0.5 && Math.abs(x.y - b[i].y) < 0.5 && Math.abs(x.w - b[i].w) < 0.5 && Math.abs(x.h - b[i].h) < 0.5);

export default function PdfPage({ sha, page, needles, onPageCount, onHighlights, zoom = 1 }: PdfPageProps) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [items, setItems] = useState<string[] | null>(null);
  /** The marks' boxes merged per line, relative to the page (drawn as one highlight each). */
  const [boxes, setBoxes] = useState<Box[]>([]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(el);
    setWidth(Math.floor(el.clientWidth));
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setItems(null);
    setBoxes([]);
  }, [page, sha]);

  const ranges = useMemo(() => (items ? highlightRanges(items, needles) : null), [items, needles]);
  const renderer = useCallback(
    ({ str, itemIndex }: { str: string; itemIndex: number }) => renderItem(str, ranges?.get(itemIndex)),
    [ranges],
  );

  const file = useMemo(() => ({ url: fileUrl(sha) }), [sha]);

  return (
    <div ref={box} className="pdf-page w-full">
      <Document
        file={file}
        options={DOC_OPTIONS}
        onLoadSuccess={(pdf) => onPageCount?.(pdf.numPages)}
        loading={<Skeleton className="aspect-[8.5/11] w-full" />}
        error={<ErrorState title="Couldn't open this PDF" error="The file may be missing or damaged." />}
      >
        {width > 0 ? (
          <Page
            key={`${sha}:${page}`}
            pageNumber={page}
            width={Math.round(Math.min(width, 900) * zoom)}
            renderAnnotationLayer={false}
            loading={<Skeleton className="aspect-[8.5/11] w-full" />}
            onGetTextSuccess={(tc) => setItems(tc.items.map((it) => ("str" in it ? it.str : "")))}
            customTextRenderer={ranges ? renderer : undefined}
            // The merged highlights replace the marks' own backgrounds (see .hl-merged in index.css).
            className={cn("mx-auto overflow-hidden rounded-md shadow-card ring-1 ring-border", boxes.length > 0 && "hl-merged")}
            onRenderTextLayerSuccess={() => {
              if (!ranges) return;
              const marks = [...(box.current?.querySelectorAll<HTMLElement>("mark[data-hl]") ?? [])];
              const pageEl = box.current?.querySelector<HTMLElement>(".react-pdf__Page");
              const origin = pageEl?.getBoundingClientRect();
              const next = origin
                ? mergeBoxes(
                    marks.flatMap((m) => [...m.getClientRects()].map((r) => ({ x: r.left - origin.left, y: r.top - origin.top, w: r.width, h: r.height }))),
                  )
                : [];
              // Only a real change re-renders: the text layer (and any selection in it) must stay put.
              setBoxes((prev) => (sameBoxes(prev, next) ? prev : next));
              onHighlights?.(marks.length);
              marks[0]?.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
            }}
          >
            {boxes.map((b, i) => (
              <div key={i} aria-hidden className="hl-box" data-testid="hl-box" style={{ left: b.x, top: b.y, width: b.w, height: b.h }} />
            ))}
          </Page>
        ) : null}
      </Document>
    </div>
  );
}
