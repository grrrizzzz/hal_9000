import { AlertDialog as A } from "radix-ui";
import type * as React from "react";
import { Button } from "./button";

/** A confirm dialog for destructive or expensive actions (delete, switch profile). */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  tone = "primary",
  onConfirm,
  busy,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: React.ReactNode;
  confirmLabel: string;
  tone?: "primary" | "danger";
  onConfirm: () => void;
  busy?: boolean;
}) {
  return (
    <A.Root open={open} onOpenChange={onOpenChange}>
      <A.Portal>
        <A.Overlay className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px]" />
        <A.Content className="fixed top-1/2 left-1/2 z-50 w-[min(26rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border bg-surface p-5 shadow-pop">
          <A.Title className="text-[15px] font-semibold">{title}</A.Title>
          <A.Description className="mt-1.5 text-sm text-muted-foreground">{description}</A.Description>
          <div className="mt-5 flex justify-end gap-2">
            <A.Cancel asChild>
              <Button variant="secondary">Cancel</Button>
            </A.Cancel>
            <A.Action asChild>
              <Button variant={tone} onClick={onConfirm} disabled={busy}>
                {confirmLabel}
              </Button>
            </A.Action>
          </div>
        </A.Content>
      </A.Portal>
    </A.Root>
  );
}
