import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronsUpDown, LogOut, MonitorSmartphone, UserRound } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";
import { api } from "@/api/client";
import { ConfirmDialog } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

/**
 * The account menu: sign out of this browser, or (after a confirmation)
 * revoke every session on every device (`DELETE /api/session?everywhere=true`).
 */
export function AccountMenu({ className }: { className?: string }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState(false);

  const leave = () => {
    qc.clear();
    navigate("/sign-in", { replace: true });
  };
  // Signing out here always ends up on the sign-in page, even if the server is unreachable.
  const signOut = useMutation({ mutationFn: () => api.delete("/api/session"), onSettled: leave });
  // Everywhere must really have happened before we say so: on failure, stay and show the error.
  const everywhere = useMutation({ mutationFn: () => api.delete("/api/session", { query: { everywhere: true } }), onSuccess: leave });

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="sm" className={cn("justify-between text-muted-foreground", className)} aria-label="Account">
            <span className="flex items-center gap-2">
              <UserRound /> Account
            </span>
            <ChevronsUpDown className="opacity-60" />
          </Button>
        </PopoverTrigger>
        <PopoverContent side="top" align="start" className="w-60 p-1.5" aria-label="Account">
          <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => signOut.mutate()} disabled={signOut.isPending}>
            <LogOut /> Sign out
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start text-danger"
            onClick={() => {
              setOpen(false);
              everywhere.reset();
              setConfirm(true);
            }}
          >
            <MonitorSmartphone /> Sign out everywhere…
          </Button>
        </PopoverContent>
      </Popover>
      <ConfirmDialog
        open={confirm}
        onOpenChange={(o) => {
          // Keep the dialog up while the request runs, and after a failure so the error stays visible.
          if (!o && everywhere.isPending) return;
          setConfirm(o);
        }}
        title="Sign out everywhere?"
        description={
          <>
            Every signed-in browser and phone, including this one, will need the household passcode again.
            {everywhere.isError ? (
              <span role="alert" className="mt-2 block text-danger">
                {(everywhere.error as Error).message}
              </span>
            ) : null}
          </>
        }
        confirmLabel={everywhere.isPending ? "Signing out…" : "Sign out everywhere"}
        tone="danger"
        busy={everywhere.isPending}
        keepOpen
        onConfirm={() => everywhere.mutate()}
      />
    </>
  );
}
