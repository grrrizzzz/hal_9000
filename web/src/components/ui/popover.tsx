import { Popover as P } from "radix-ui";
import type * as React from "react";
import { cn } from "@/lib/utils";

export const Popover = P.Root;
export const PopoverTrigger = P.Trigger;
export const PopoverClose = P.Close;

export function PopoverContent({ className, ...props }: React.ComponentProps<typeof P.Content>) {
  return (
    <P.Portal>
      <P.Content
        sideOffset={6}
        collisionPadding={12}
        className={cn(
          "z-50 w-72 max-w-[calc(100vw-1.5rem)] rounded-xl border border-border bg-surface p-3 shadow-pop outline-none",
          className,
        )}
        {...props}
      />
    </P.Portal>
  );
}
