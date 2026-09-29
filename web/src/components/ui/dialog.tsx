import { X } from "lucide-react";
import { Dialog as D } from "radix-ui";
import type * as React from "react";
import { cn } from "@/lib/utils";

export const Dialog = D.Root;
export const DialogTrigger = D.Trigger;
export const DialogClose = D.Close;

export function DialogContent({
  className,
  children,
  title,
  description,
  size = "md",
  ...props
}: React.ComponentProps<typeof D.Content> & {
  title: React.ReactNode;
  description?: React.ReactNode;
  size?: "md" | "lg" | "full";
}) {
  return (
    <D.Portal>
      <D.Overlay className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] data-[state=open]:animate-in" />
      <D.Content
        className={cn(
          "fixed z-50 flex flex-col bg-surface shadow-pop outline-none",
          "inset-x-0 bottom-0 max-h-[92dvh] rounded-t-2xl border-t border-border",
          "sm:inset-auto sm:top-1/2 sm:left-1/2 sm:max-h-[88dvh] sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-2xl sm:border",
          size === "md" && "sm:w-[min(32rem,calc(100vw-2rem))]",
          size === "lg" && "sm:w-[min(48rem,calc(100vw-2rem))]",
          size === "full" && "sm:h-[88dvh] sm:w-[min(72rem,calc(100vw-2rem))]",
          className,
        )}
        {...props}
      >
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-3.5">
          <div className="min-w-0">
            <D.Title className="truncate text-[15px] font-semibold tracking-tight">{title}</D.Title>
            {description ? (
              <D.Description className="mt-0.5 truncate text-[13px] text-muted-foreground">{description}</D.Description>
            ) : (
              <D.Description className="sr-only">Dialog</D.Description>
            )}
          </div>
          <D.Close
            className="-mr-1.5 rounded-md p-1.5 text-muted-foreground hover:bg-subtle hover:text-foreground"
            aria-label="Close"
          >
            <X className="size-4" />
          </D.Close>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">{children}</div>
      </D.Content>
    </D.Portal>
  );
}
