import { ShieldAlert, ShieldCheck } from "lucide-react";
import type { Grounding } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export function VerifyBadge({ grounding, issues }: { grounding: Grounding | null | undefined; issues: string[] }) {
  if (grounding === "failed") {
    return (
      <Popover>
        <PopoverTrigger asChild>
          <button type="button" className="rounded-full" data-testid="badge-unverified">
            <Badge tone="warning">
              <ShieldAlert /> Figures not verified
            </Badge>
          </button>
        </PopoverTrigger>
        <PopoverContent align="start" className="text-sm">
          <p className="font-medium">Some figures couldn't be matched to the cited sources.</p>
          <p className="mt-1 text-muted-foreground">Check them against the documents before relying on them.</p>
          {issues.length ? (
            <ul className="mt-2 list-disc space-y-0.5 pl-4 text-[13px] text-muted-foreground">
              {issues.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          ) : null}
        </PopoverContent>
      </Popover>
    );
  }
  if (grounding === "verified") {
    return (
      <Badge tone="success" data-testid="badge-verified" title="Every figure and citation was found in the cited sources">
        <ShieldCheck /> Figures checked
      </Badge>
    );
  }
  return null;
}
