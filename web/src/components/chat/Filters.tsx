import { useQuery } from "@tanstack/react-query";
import { SlidersHorizontal, X } from "lucide-react";
import { api } from "@/api/client";
import { DOC_TYPES, type DocType, type SearchFilters } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label, Select } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { docTypeLabel } from "@/lib/format";

export function Filters({ value, onChange }: { value: SearchFilters; onChange: (f: SearchFilters) => void }) {
  const docs = useQuery({ queryKey: ["documents", {}], queryFn: () => api.get("/api/documents"), staleTime: 60_000 });
  const collections = [...new Set((docs.data ?? []).map((d) => d.collection).filter(Boolean))].sort();
  const active = [value.collection, value.doc_type].filter(Boolean).length;
  return (
    <>
      <Popover>
        <PopoverTrigger asChild>
          <Button size="sm" variant="ghost" className="text-muted-foreground" aria-label={active ? `Filters (${active} active)` : "Filters"}>
            <SlidersHorizontal /> <span className="hidden sm:inline">Filter</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="f-collection">Collection</Label>
            <Select id="f-collection" value={value.collection ?? ""} onChange={(e) => onChange({ ...value, collection: e.target.value || null })}>
              <option value="">All collections</option>
              {collections.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="f-type">Document type</Label>
            <Select id="f-type" value={value.doc_type ?? ""} onChange={(e) => onChange({ ...value, doc_type: (e.target.value || null) as DocType | null })}>
              <option value="">All types</option>
              {DOC_TYPES.map((t) => (
                <option key={t} value={t}>
                  {docTypeLabel(t)}
                </option>
              ))}
            </Select>
          </div>
        </PopoverContent>
      </Popover>
      {value.collection ? (
        <Badge tone="accent" className="max-w-40">
          <span className="truncate">{value.collection}</span>
          <button type="button" aria-label="Clear collection filter" onClick={() => onChange({ ...value, collection: null })}>
            <X />
          </button>
        </Badge>
      ) : null}
      {value.doc_type ? (
        <Badge tone="accent">
          {docTypeLabel(value.doc_type)}
          <button type="button" aria-label="Clear type filter" onClick={() => onChange({ ...value, doc_type: null })}>
            <X />
          </button>
        </Badge>
      ) : null}
    </>
  );
}
