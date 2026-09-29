import { RadioGroup } from "radix-ui";
import { cn } from "@/lib/utils";

/** A labelled single-choice control (Radix RadioGroup), arrow-key navigable. */
export function Segmented<T extends string>({
  label,
  value,
  onChange,
  options,
  className,
  name,
}: {
  label: string;
  value: T | undefined;
  onChange: (v: T) => void;
  options: { value: T; label: string; tone?: "success" | "warning" | "danger" | "neutral" }[];
  className?: string;
  name?: string;
}) {
  return (
    <RadioGroup.Root
      aria-label={label}
      name={name}
      value={value ?? ""}
      onValueChange={(v) => onChange(v as T)}
      className={cn("inline-flex flex-wrap gap-1 rounded-lg border border-border bg-surface-2 p-0.5", className)}
    >
      {options.map((o) => (
        <RadioGroup.Item
          key={o.value}
          value={o.value}
          className={cn(
            "h-7 rounded-md px-2.5 text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground",
            "data-[state=checked]:bg-surface data-[state=checked]:text-foreground data-[state=checked]:shadow-card",
            o.tone === "success" && "data-[state=checked]:text-success",
            o.tone === "warning" && "data-[state=checked]:text-warning",
            o.tone === "danger" && "data-[state=checked]:text-danger",
          )}
        >
          {o.label}
        </RadioGroup.Item>
      ))}
    </RadioGroup.Root>
  );
}
