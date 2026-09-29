import { cn } from "@/lib/utils";

export function ProgressBar({
  value,
  tone = "accent",
  className,
  label,
}: {
  value: number;
  tone?: "accent" | "success" | "danger";
  className?: string;
  label: string;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-subtle", className)}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-300",
          tone === "accent" && "bg-accent",
          tone === "success" && "bg-success",
          tone === "danger" && "bg-danger",
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
