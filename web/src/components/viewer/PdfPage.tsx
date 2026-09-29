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
import { highlightRanges, renderItem } from "@/lib/highlight";

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

export default function PdfPage({ sha, page, needles, onPageCount, onHighlights, zoom = 1 }: PdfPageProps) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [items, setItems] = useState<string[] | null>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(el);
    setWidth(Math.floor(el.clientWidth));
    return () => ro.disconnect();
  }, []);

  useEffect(() => setItems(null), [page, sha]);

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
            className="mx-auto overflow-hidden rounded-md shadow-card ring-1 ring-border"
            loading={<Skeleton className="aspect-[8.5/11] w-full" />}
            onGetTextSuccess={(tc) => setItems(tc.items.map((it) => ("str" in it ? it.str : "")))}
            customTextRenderer={ranges ? renderer : undefined}
            onRenderTextLayerSuccess={() => {
              if (!ranges) return;
              const marks = box.current?.querySelectorAll("mark[data-hl]") ?? [];
              onHighlights?.(marks.length);
              marks[0]?.scrollIntoView({ block: "center", behavior: "smooth" });
            }}
          />
        ) : null}
      </Document>
    </div>
  );
}
