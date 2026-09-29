import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Folder, MoreHorizontal, Pencil, RefreshCw, Search, Trash2, Upload } from "lucide-react";
import { DropdownMenu } from "radix-ui";
import { useMemo, useState } from "react";
import { Link } from "react-router";
import { api } from "@/api/client";
import { DOC_TYPES, type DocType, type DocumentSummary } from "@/api/types";
import { EditDocumentForm } from "@/components/library/EditDocumentForm";
import { StatusBadge } from "@/components/library/StatusBadge";
import { Page, PageHeader } from "@/components/layout/PageHeader";
import { ConfirmDialog } from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/input";
import { EmptyState, ErrorState, SkeletonRows } from "@/components/ui/states";
import { docTypeLabel, formatDocDate } from "@/lib/format";

interface FileGroup {
  sha: string;
  name: string;
  path: string;
  docs: DocumentSummary[];
}

function group(docs: DocumentSummary[]): [string, FileGroup[]][] {
  const byCollection = new Map<string, Map<string, FileGroup>>();
  for (const d of docs) {
    const c = byCollection.get(d.collection) ?? new Map<string, FileGroup>();
    const f = c.get(d.document.file_sha) ?? { sha: d.document.file_sha, name: d.original_name, path: d.relative_path, docs: [] };
    f.docs.push(d);
    c.set(d.document.file_sha, f);
    byCollection.set(d.collection, c);
  }
  return [...byCollection.entries()]
    .sort(([a], [b]) => (a === "" ? 1 : b === "" ? -1 : a.localeCompare(b)))
    .map(([c, files]) => [c, [...files.values()].map((f) => ({ ...f, docs: f.docs.sort((x, y) => x.document.page_start - y.document.page_start) }))]);
}

export function LibraryPage() {
  const [query, setQuery] = useState("");
  const [collection, setCollection] = useState("");
  const [docType, setDocType] = useState<DocType | "">("");
  const docs = useQuery({ queryKey: ["documents", {}], queryFn: () => api.get("/api/documents") });

  const all = useMemo(() => docs.data ?? [], [docs.data]);
  const collections = useMemo(() => [...new Set(all.map((d) => d.collection))].sort(), [all]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return all.filter(
      (d) =>
        (!collection || d.collection === collection) &&
        (!docType || d.document.doc_type === docType) &&
        (!q || [d.document.title, d.document.vendor, d.relative_path].some((s) => s?.toLowerCase().includes(q))),
    );
  }, [all, query, collection, docType]);
  const groups = useMemo(() => group(filtered), [filtered]);

  return (
    <Page wide>
      <PageHeader
        title="Library"
        description={
          docs.data ? `${all.length} ${all.length === 1 ? "document" : "documents"} in ${collections.length} ${collections.length === 1 ? "collection" : "collections"}` : "Everything HAL can answer from."
        }
        actions={
          <Button asChild variant="primary">
            <Link to="/upload">
              <Upload /> Upload
            </Link>
          </Button>
        }
      />
      <div className="mt-6 flex flex-wrap gap-2">
        <div className="relative min-w-0 flex-1 basis-56">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input className="pl-9" placeholder="Search titles, vendors, paths" aria-label="Search the library" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
        <Select aria-label="Collection" className="w-auto min-w-40 flex-1 sm:flex-none" value={collection} onChange={(e) => setCollection(e.target.value)}>
          <option value="">All collections</option>
          {collections.map((c) => (
            <option key={c} value={c}>
              {c || "Unfiled"}
            </option>
          ))}
        </Select>
        <Select aria-label="Document type" className="w-auto min-w-36 flex-1 sm:flex-none" value={docType} onChange={(e) => setDocType(e.target.value as DocType | "")}>
          <option value="">All types</option>
          {DOC_TYPES.map((t) => (
            <option key={t} value={t}>
              {docTypeLabel(t)}
            </option>
          ))}
        </Select>
      </div>

      <div className="mt-5 space-y-5">
        {docs.isLoading ? (
          <SkeletonRows rows={5} />
        ) : docs.isError ? (
          <Card>
            <ErrorState error={docs.error} onRetry={() => docs.refetch()} />
          </Card>
        ) : !all.length ? (
          <Card>
            <EmptyState
              icon={<FileText />}
              title="The library is empty"
              action={
                <Button asChild variant="primary">
                  <Link to="/upload">Upload documents</Link>
                </Button>
              }
            >
              Upload inspection reports, invoices, manuals and warranties, and HAL can answer questions about them.
            </EmptyState>
          </Card>
        ) : !filtered.length ? (
          <Card>
            <EmptyState icon={<Search />} title="No matches">
              Nothing matches those filters.
            </EmptyState>
          </Card>
        ) : (
          groups.map(([c, files]) => (
            <section key={c} aria-labelledby={`col-${c}`}>
              <h2 id={`col-${c}`} className="mb-2 flex items-center gap-2 text-sm font-semibold">
                <Folder className="size-4 text-muted-foreground" aria-hidden />
                {c || "Unfiled"}
                <span className="font-normal text-muted-foreground">{files.reduce((n, f) => n + f.docs.length, 0)}</span>
              </h2>
              <Card className="divide-y divide-border overflow-hidden">
                {files.map((f) => (
                  <FileBlock key={f.sha} file={f} />
                ))}
              </Card>
            </section>
          ))
        )}
      </div>
    </Page>
  );
}

function FileBlock({ file }: { file: FileGroup }) {
  return (
    <div>
      <div className="flex items-center gap-2 bg-surface-2/70 px-4 py-2 text-xs text-muted-foreground sm:px-5">
        <FileText className="size-3.5 shrink-0" aria-hidden />
        <span className="truncate font-mono" title={file.path}>
          {file.path}
        </span>
        {file.docs.length > 1 ? <span className="shrink-0">· {file.docs.length} documents</span> : null}
      </div>
      <ul className="divide-y divide-border">
        {file.docs.map((d) => (
          <DocRow key={d.document.id} doc={d} />
        ))}
      </ul>
    </div>
  );
}

function DocRow({ doc }: { doc: DocumentSummary }) {
  const d = doc.document;
  const [editing, setEditing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const qc = useQueryClient();
  const reindex = useMutation({
    mutationFn: () => api.post("/api/documents/{id}/reindex", { params: { id: d.id! } }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["documents"] }),
  });
  const del = useMutation({
    mutationFn: () => api.delete("/api/documents/{id}", { params: { id: d.id! } }),
    onSuccess: () => {
      setConfirmDelete(false);
      void qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });
  const range = d.page_start === d.page_end ? `p. ${d.page_start}` : `pp. ${d.page_start}–${d.page_end}`;
  return (
    <li className="px-4 py-3 sm:px-5" data-testid="library-row">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <Link to={`/library/${d.id}`} className="font-medium hover:underline">
              {d.title}
            </Link>
            {d.meta_source === "user" ? (
              <Badge tone="outline" title="You edited these details; they win over automatic labels">
                <Pencil /> Edited
              </Badge>
            ) : null}
          </div>
          <p className="mt-0.5 text-[13px] text-muted-foreground">
            {[docTypeLabel(d.doc_type), d.vendor, formatDocDate(d.doc_date)].filter(Boolean).join(" · ")}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
            <StatusBadge status={doc.status} />
            <Badge tone="outline" className="tabular">
              {range} of {doc.pages}
            </Badge>
            <Badge tone="outline" className="tabular">
              {doc.chunks} chunks
            </Badge>
            {doc.pages_without_text ? (
              <Badge tone="warning" title="Pages with no text layer can't be searched">
                {doc.pages_without_text} {doc.pages_without_text === 1 ? "page" : "pages"} without text
              </Badge>
            ) : null}
          </div>
          {reindex.isError || del.isError ? (
            <p role="alert" className="mt-2 text-[13px] text-danger">
              {((reindex.error ?? del.error) as Error).message}
            </p>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {doc.status === "failed" ? (
            <Button size="sm" onClick={() => reindex.mutate()} disabled={reindex.isPending}>
              <RefreshCw /> Retry
            </Button>
          ) : null}
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button size="icon-sm" variant="ghost" aria-label={`Actions for ${d.title}`}>
                <MoreHorizontal />
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content align="end" sideOffset={4} className="z-50 min-w-44 rounded-xl border border-border bg-surface p-1 shadow-pop">
                {[
                  { label: "Edit details", icon: Pencil, onSelect: () => setEditing(true) },
                  { label: "Re-index", icon: RefreshCw, onSelect: () => reindex.mutate() },
                  { label: "Delete", icon: Trash2, onSelect: () => setConfirmDelete(true), danger: true },
                ].map(({ label, icon: Icon, onSelect, danger }) => (
                  <DropdownMenu.Item
                    key={label}
                    onSelect={onSelect}
                    className={`flex h-8 cursor-default items-center gap-2 rounded-lg px-2.5 text-sm outline-none data-[highlighted]:bg-subtle ${danger ? "text-danger" : ""}`}
                  >
                    <Icon className="size-4" aria-hidden /> {label}
                  </DropdownMenu.Item>
                ))}
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        </div>
      </div>
      {editing ? (
        <div className="mt-3">
          <EditDocumentForm doc={doc} onDone={() => setEditing(false)} />
        </div>
      ) : null}
      <ConfirmDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={`Delete “${d.title}”?`}
        description="It's removed from the library and the index. Upload the file again to restore it."
        confirmLabel="Delete"
        tone="danger"
        busy={del.isPending}
        onConfirm={() => del.mutate()}
      />
    </li>
  );
}
