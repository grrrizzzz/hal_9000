import { useMutation } from "@tanstack/react-query";
import { ArrowRight, Loader2 } from "lucide-react";
import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { api } from "@/api/client";
import { Logo } from "@/components/layout/Logo";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";

export function SignInPage() {
  const [passcode, setPasscode] = useState("");
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const next = params.get("next");
  const m = useMutation({
    mutationFn: () => api.post("/api/session", { body: { passcode } }),
    onSuccess: () => navigate(next && next.startsWith("/") && !next.startsWith("//") ? next : "/", { replace: true }),
  });

  return (
    <div className="grid min-h-dvh place-items-center bg-background px-4">
      <div className="w-full max-w-sm">
        <Logo className="mb-8 justify-center" />
        <form
          className="rounded-2xl border border-border bg-surface p-6 shadow-card"
          onSubmit={(e) => {
            e.preventDefault();
            if (passcode) m.mutate();
          }}
        >
          <h1 className="text-lg font-semibold tracking-tight">Sign in</h1>
          <p className="mt-1 text-sm text-muted-foreground">Enter the household passcode. You'll stay signed in on this device for 30 days.</p>
          <div className="mt-5 space-y-1.5">
            <Label htmlFor="passcode">Passcode</Label>
            <Input
              id="passcode"
              type="password"
              autoComplete="current-password"
              autoFocus
              value={passcode}
              onChange={(e) => setPasscode(e.target.value)}
              aria-invalid={m.isError || undefined}
              aria-describedby={m.isError ? "passcode-error" : undefined}
            />
            {m.isError ? (
              <p id="passcode-error" role="alert" className="text-[13px] text-danger">
                {(m.error as Error).message}
              </p>
            ) : null}
          </div>
          <Button type="submit" variant="primary" className="mt-5 w-full" size="lg" disabled={!passcode || m.isPending}>
            {m.isPending ? <Loader2 className="animate-spin" /> : null}
            Sign in {m.isPending ? null : <ArrowRight />}
          </Button>
        </form>
        <p className="mt-6 text-center text-xs text-muted-foreground">Answers come only from documents in the house library.</p>
      </div>
    </div>
  );
}
