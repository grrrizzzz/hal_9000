import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/api/client";
import { DOC_TYPES, type DocType, type DocumentSummary } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import { docTypeLabel } from "@/lib/format";

export function EditDocumentForm({ doc, onDone }: { doc: DocumentSummary; onDone: () => void }) {
  const d = doc.document;
  const [title, setTitle] = useState(d.title);
  const [docType, setDocType] = useState<DocType>(d.doc_type ?? "other");
  const [vendor, setVendor] = useState(d.vendor ?? "");
  const [date, setDate] = useState(d.doc_date ?? "");
  const [start, setStart] = useState(String(d.page_start));
  const [end, setEnd] = useState(String(d.page_end));
  const qc = useQueryClient();
  const idp = `edit-${d.id}`;
  const m = useMutation({
    mutationFn: () =>
      api.patch("/api/documents/{id}", {
        params: { id: d.id! },
        body: {
          title: title.trim(),
          doc_type: docType,
          vendor: vendor.trim() || null,
          doc_date: date || null,
          page_start: Number(start),
          page_end: Number(end),
        },
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["documents"] });
      void qc.invalidateQueries({ queryKey: ["document", d.id] });
      onDone();
    },
  });
  const pagesBad = Number(end) < Number(start) || Number(start) < 1;
  return (
    <form
      className="grid gap-3 rounded-xl border border-border bg-surface-2 p-4 sm:grid-cols-6"
      onSubmit={(e) => {
        e.preventDefault();
        if (!pagesBad && title.trim()) m.mutate();
      }}
      aria-label={`Edit ${d.title}`}
    >
      <div className="space-y-1.5 sm:col-span-4">
        <Label htmlFor={`${idp}-title`}>Title</Label>
        <Input id={`${idp}-title`} value={title} onChange={(e) => setTitle(e.target.value)} required />
      </div>
      <div className="space-y-1.5 sm:col-span-2">
        <Label htmlFor={`${idp}-type`}>Type</Label>
        <Select id={`${idp}-type`} value={docType} onChange={(e) => setDocType(e.target.value as DocType)}>
          {DOC_TYPES.map((t) => (
            <option key={t} value={t}>
              {docTypeLabel(t)}
            </option>
          ))}
        </Select>
      </div>
      <div className="space-y-1.5 sm:col-span-2">
        <Label htmlFor={`${idp}-vendor`}>Vendor</Label>
        <Input id={`${idp}-vendor`} value={vendor} onChange={(e) => setVendor(e.target.value)} />
      </div>
      <div className="space-y-1.5 sm:col-span-2">
        <Label htmlFor={`${idp}-date`}>Date</Label>
        <Input id={`${idp}-date`} type="date" value={date} onChange={(e) => setDate(e.target.value)} />
      </div>
      <div className="space-y-1.5 sm:col-span-1">
        <Label htmlFor={`${idp}-start`}>First page</Label>
        <Input id={`${idp}-start`} type="number" min={1} value={start} onChange={(e) => setStart(e.target.value)} />
      </div>
      <div className="space-y-1.5 sm:col-span-1">
        <Label htmlFor={`${idp}-end`}>Last page</Label>
        <Input id={`${idp}-end`} type="number" min={1} value={end} onChange={(e) => setEnd(e.target.value)} aria-invalid={pagesBad || undefined} />
      </div>
      <div className="flex flex-wrap items-center justify-end gap-2 sm:col-span-6">
        {m.isError ? (
          <p role="alert" className="mr-auto text-[13px] text-danger">
            {(m.error as Error).message}
          </p>
        ) : pagesBad ? (
          <p className="mr-auto text-[13px] text-danger">The last page must come after the first.</p>
        ) : (
          <p className="mr-auto text-xs text-muted-foreground">Your edits always win over automatic labels, even after re-indexing.</p>
        )}
        <Button variant="ghost" onClick={onDone}>
          Cancel
        </Button>
        <Button type="submit" variant="primary" disabled={m.isPending || pagesBad || !title.trim()}>
          Save
        </Button>
      </div>
    </form>
  );
}
