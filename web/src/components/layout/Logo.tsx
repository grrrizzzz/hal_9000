import { cn } from "@/lib/utils";

export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <svg viewBox="0 0 32 32" className="size-7 shrink-0" aria-hidden>
        <circle cx="16" cy="16" r="15" fill="#1c1917" stroke="currentColor" strokeOpacity=".12" />
        <circle cx="16" cy="16" r="9" fill="#7f1d1d" />
        <circle cx="16" cy="16" r="4.5" fill="#ef4444" />
        <circle cx="14.5" cy="14.5" r="1.4" fill="#fecaca" />
      </svg>
      <div className="leading-tight">
        <p className="text-[15px] font-semibold tracking-tight">HAL</p>
        <p className="text-xs text-muted-foreground">House knowledge</p>
      </div>
    </div>
  );
}
