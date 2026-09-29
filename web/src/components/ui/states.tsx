import { AlertTriangle, Loader2 } from "lucide-react";
import type * as React from "react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

export function Spinner({ className, label = "Loading" }: { className?: string; label?: string }) {
  return (
    <span role="status" className={cn("inline-flex items-center gap-2 text-sm text-muted-foreground", className)}>
      <Loader2 className="size-4 animate-spin" aria-hidden />
      <span>{label}…</span>
    </span>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-pulse rounded-md bg-subtle", className)} />;
}

export function SkeletonRows({ rows = 4, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn("space-y-2.5", className)} role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className={cn("h-12", i % 2 ? "w-11/12" : "w-full")} />
      ))}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  action,
  className,
}: {
  icon?: React.ReactNode;
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center px-6 py-12 text-center", className)}>
      {icon ? (
        <div className="mb-3 grid size-11 place-items-center rounded-xl bg-subtle text-muted-foreground [&_svg]:size-5">
          {icon}
        </div>
      ) : null}
      <h3 className="text-[15px] font-semibold">{title}</h3>
      {children ? <div className="mt-1 max-w-sm text-sm text-muted-foreground">{children}</div> : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function ErrorState({
  error,
  onRetry,
  className,
  title = "Something went wrong",
}: {
  error: unknown;
  onRetry?: () => void;
  className?: string;
  title?: string;
}) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className={cn("flex flex-col items-center px-6 py-10 text-center", className)}>
      <div className="mb-3 grid size-11 place-items-center rounded-xl bg-danger-soft text-danger">
        <AlertTriangle className="size-5" />
      </div>
      <h3 className="text-[15px] font-semibold">{title}</h3>
      <p className="mt-1 max-w-sm text-sm break-words text-muted-foreground">{message}</p>
      {onRetry ? (
        <Button className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      ) : null}
    </div>
  );
}

export function Banner({
  tone = "warning",
  icon,
  title,
  children,
  className,
  testId,
}: {
  tone?: "warning" | "danger" | "info";
  icon?: React.ReactNode;
  title: string;
  children?: React.ReactNode;
  className?: string;
  testId?: string;
}) {
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      data-testid={testId}
      className={cn(
        "flex gap-3 rounded-xl border px-4 py-3 text-sm [&_svg]:mt-0.5 [&_svg]:size-4 [&_svg]:shrink-0",
        tone === "warning" && "border-warning/30 bg-warning-soft text-foreground [&_svg]:text-warning",
        tone === "danger" && "border-danger/30 bg-danger-soft text-foreground [&_svg]:text-danger",
        tone === "info" && "border-accent/30 bg-accent-soft text-foreground [&_svg]:text-accent",
        className,
      )}
    >
      {icon ?? <AlertTriangle />}
      <div className="min-w-0">
        <p className="font-medium">{title}</p>
        {children ? <div className="mt-0.5 text-muted-foreground">{children}</div> : null}
      </div>
    </div>
  );
}
