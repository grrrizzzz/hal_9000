import { Tooltip as T } from "radix-ui";
import type * as React from "react";

export const TooltipProvider = T.Provider;

/** A small label on hover or focus. Never the only way to learn what a control does. */
export function Tip({ label, children }: { label: React.ReactNode; children: React.ReactElement }) {
  return (
    <T.Root delayDuration={350}>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content
          sideOffset={6}
          className="z-50 rounded-md bg-primary px-2 py-1 text-xs text-primary-foreground shadow-pop"
        >
          {label}
        </T.Content>
      </T.Portal>
    </T.Root>
  );
}
