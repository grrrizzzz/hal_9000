import type * as React from "react";
import { cn } from "@/lib/utils";

export function Kbd({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        "inline-grid h-5 min-w-5 place-items-center rounded border border-border-strong bg-surface-2 px-1 font-mono text-[11px] font-medium text-muted-foreground",
        className,
      )}
      {...props}
    />
  );
}
