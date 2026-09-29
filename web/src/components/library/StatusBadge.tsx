import { AlertCircle, CheckCircle2, Clock, Loader2 } from "lucide-react";
import type { DocumentSummary } from "@/api/types";
import { Badge } from "@/components/ui/badge";

export function StatusBadge({ status }: { status: DocumentSummary["status"] }) {
  switch (status) {
    case "indexed":
      return (
        <Badge tone="success">
          <CheckCircle2 /> Indexed
        </Badge>
      );
    case "indexing":
      return (
        <Badge tone="accent">
          <Loader2 className="animate-spin" /> Indexing
        </Badge>
      );
    case "queued":
      return (
        <Badge tone="neutral">
          <Clock /> Queued
        </Badge>
      );
    case "failed":
      return (
        <Badge tone="danger">
          <AlertCircle /> Failed
        </Badge>
      );
    default:
      return null;
  }
}
