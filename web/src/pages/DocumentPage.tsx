import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ChevronLeft, ChevronRight, FileText } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router";
import { api } from "@/api/client";
import { StatusBadge } from "@/components/library/StatusBadge";
import { Page } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DocumentViewer, usePageView } from "@/components/viewer/DocumentViewer";
import { docTypeLabel, formatBytes, formatDocDate, pageRange, plural } from "@/lib/format";

export function DocumentPage() {
  const id = Number(useParams().id);
  const doc = useQuery({ queryKey: ["document", id], queryFn: () => api.get("/api/documents/{id}", { params: { id } }) });
  const [page, setPage] = useState<number | null>(null);

  if (doc.isLoading) {
    return (
      <Page>
        <Skeleton className="h-8 w-1/2" />
        <Skeleton className="mt-4 h-96" />
      </Page>
    );
  }
  if (doc.isError || !doc.data) {
    return (
      <Page>
        <ErrorState error={doc.error ?? "Not found"} onRetry={() => doc.refetch()} />
      </Page>
    );
  }
  const d = doc.data;
  const m = d.document;
  const current = page ?? m.page_start;

  return (
    <Page wide>
      <Button asChild variant="link" size="sm" className="mb-3 text-muted-foreground">
        <Link to="/library">
          <ArrowLeft /> Library
        </Link>
      </Button>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">{m.title}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{[docTypeLabel(m.doc_type), m.vendor, formatDocDate(m.doc_date)].filter(Boolean).join(" · ")}</p>
        </div>
        <StatusBadge status={d.status} />
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        {[
          ["File", d.file.original_name],
          ["Collection", d.collection || "Unfiled"],
          ["Pages", `${plural(d.pages, "page")} (file ${pageRange(m.page_start, m.page_end)})`],
          ["Size", formatBytes(d.file.bytes)],
          ["Chunks", String(d.chunks)],
          ["Without text layer", String(d.pages_without_text)],
          ["Labels", m.meta_source === "user" ? "Edited by you" : "Automatic"],
          ["Type", d.file.mime_type],
        ].map(([k, v]) => (
          <div key={k} className="min-w-0 rounded-lg border border-border bg-surface px-3 py-2">
            <dt className="text-xs text-muted-foreground">{k}</dt>
            <dd className="truncate font-medium" title={v}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
      {d.siblings?.length ? (
        <div className="mt-4 text-sm">
          <span className="text-muted-foreground">Also in this file: </span>
          {d.siblings.map((s, i) => (
            <span key={s.id}>
              {i ? ", " : ""}
              <Link className="text-accent-foreground underline-offset-4 hover:underline" to={`/library/${s.id}`}>
                {s.title}
              </Link>
            </span>
          ))}
        </div>
      ) : null}

      <Tabs defaultValue="text" className="mt-6">
        <TabsList>
          <TabsTrigger value="text">Extracted text</TabsTrigger>
          <TabsTrigger value="original">Original</TabsTrigger>
        </TabsList>
        <TabsContent value="text" className="mt-3">
          <ExtractedText sha={m.file_sha} page={current} first={m.page_start} last={m.page_end} onPage={setPage} />
        </TabsContent>
        <TabsContent value="original" className="mt-3">
          <Card className="h-[80dvh] overflow-hidden">
            <DocumentViewer target={{ fileSha: m.file_sha, page: current, fileName: d.file.original_name }} />
          </Card>
        </TabsContent>
      </Tabs>
    </Page>
  );
}

function ExtractedText({ sha, page, first, last, onPage }: { sha: string; page: number; first: number; last: number; onPage: (p: number) => void }) {
  const view = usePageView(sha, page);
  const v = view.data;
  const total = v?.page_count ?? last;
  // Shade alternate chunks so chunk boundaries are visible (a parsing aid, plan §10 Library).
  const parts: { text: string; chunk: number | null; i: number }[] = [];
  if (v) {
    let pos = 0;
    const spans = [...(v.spans ?? [])].sort((a, b) => a.char_start - b.char_start);
    spans.forEach((s, i) => {
      if (s.char_start > pos) parts.push({ text: v.text.slice(pos, s.char_start), chunk: null, i });
      parts.push({ text: v.text.slice(Math.max(pos, s.char_start), s.char_end), chunk: s.chunk_id, i });
      pos = Math.max(pos, s.char_end);
    });
    if (pos < v.text.length) parts.push({ text: v.text.slice(pos), chunk: null, i: -1 });
  }
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <Button size="icon-sm" variant="ghost" aria-label="Previous page" disabled={page <= 1} onClick={() => onPage(page - 1)}>
          <ChevronLeft />
        </Button>
        <span className="tabular text-[13px] text-muted-foreground">
          Page {page} of {total}
        </span>
        <Button size="icon-sm" variant="ghost" aria-label="Next page" disabled={page >= total} onClick={() => onPage(page + 1)}>
          <ChevronRight />
        </Button>
        {page < first || page > last ? <Badge tone="neutral">Outside this document</Badge> : null}
        {v && !v.has_text_layer ? <Badge tone="warning">No text layer</Badge> : null}
        <span className="ml-auto text-xs text-muted-foreground">{plural(v?.spans?.length ?? 0, "chunk")} on this page</span>
      </div>
      {view.isLoading ? (
        <Skeleton className="m-4 h-64" />
      ) : view.isError ? (
        <ErrorState error={view.error} onRetry={() => view.refetch()} />
      ) : !v?.text ? (
        <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
          <FileText className="size-4" /> No text was extracted from this page.
        </div>
      ) : (
        <pre className="max-h-[70dvh] overflow-auto px-5 py-4 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap">
          {parts.map((p, k) => (
            <span key={k} className={p.chunk !== null ? (p.i % 2 ? "bg-accent-soft/60" : "bg-subtle") : undefined} title={p.chunk !== null ? `Chunk ${p.chunk}` : undefined}>
              {p.text}
            </span>
          ))}
        </pre>
      )}
    </Card>
  );
}
