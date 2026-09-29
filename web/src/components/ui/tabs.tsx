import { Tabs as T } from "radix-ui";
import type * as React from "react";
import { cn } from "@/lib/utils";

export const Tabs = T.Root;
export const TabsContent = T.Content;

export function TabsList({ className, ...props }: React.ComponentProps<typeof T.List>) {
  return (
    <T.List
      className={cn(
        // Wraps rather than scrolls: a tab hidden past the edge is a tab nobody finds.
        "flex max-w-full flex-wrap gap-1 rounded-xl border border-border bg-surface-2 p-1",
        className,
      )}
      {...props}
    />
  );
}

export function TabsTrigger({ className, ...props }: React.ComponentProps<typeof T.Trigger>) {
  return (
    <T.Trigger
      className={cn(
        "inline-flex h-8 shrink-0 items-center justify-center gap-1.5 rounded-lg px-3 text-[13px] font-medium text-muted-foreground transition-colors",
        "hover:text-foreground data-[state=active]:bg-surface data-[state=active]:text-foreground data-[state=active]:shadow-card",
        "[&_svg]:size-3.5",
        className,
      )}
      {...props}
    />
  );
}
