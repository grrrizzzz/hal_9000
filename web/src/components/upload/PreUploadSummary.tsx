import { ChevronDown, ImageOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { formatBytes } from "@/lib/format";
import { SKIP_LABEL, type Summary } from "@/lib/upload/classify";

export function PreUploadSummary({ summary, onUpload, onClear }: { summary: Summary; onUpload: () => void; onClear: () => void }) {
  return (
    <Card className="overflow-hidden" data-testid="upload-summary">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
        <div className="min-w-0">
          <p className="font-medium" data-testid="summary-text">
            {summary.text}
          </p>
          {summary.hasImages ? (
            <p className="mt-1 flex items-center gap-1.5 text-[13px] text-muted-foreground">
              <ImageOff className="size-3.5 shrink-0" aria-hidden />
              Images are skipped: HAL reads text, not pictures. Scans should be PDFs with a text layer.
            </p>
          ) : null}
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={onClear}>
            Clear
          </Button>
          <Button variant="primary" onClick={onUpload} disabled={!summary.added.length}>
            Upload {summary.added.length} {summary.added.length === 1 ? "file" : "files"}
          </Button>
        </div>
      </div>
      <details className="group border-t border-border">
        <summary className="flex cursor-pointer list-none items-center gap-1.5 px-5 py-2.5 text-[13px] font-medium text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
          <ChevronDown className="size-4 transition-transform group-open:rotate-180" aria-hidden />
          Show all {summary.total} files
        </summary>
        <ul className="max-h-80 divide-y divide-border overflow-auto border-t border-border text-[13px]">
          {[...summary.added, ...summary.skipped].map((f) => (
            <li key={f.relativePath} className="flex items-center gap-3 px-5 py-2">
              <span className="min-w-0 flex-1 truncate font-mono text-xs" title={f.relativePath}>
                {f.relativePath}
              </span>
              <span className="tabular hidden text-xs text-muted-foreground sm:inline">{formatBytes(f.file.size)}</span>
              {f.skip ? <Badge tone="neutral">{SKIP_LABEL[f.skip]}</Badge> : <Badge tone="accent">Will be added</Badge>}
            </li>
          ))}
        </ul>
      </details>
    </Card>
  );
}
