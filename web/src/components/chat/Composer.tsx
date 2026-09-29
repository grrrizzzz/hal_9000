import { ArrowUp, Square } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";

/** ChatRequest.message maxLength in openapi.json. */
export const MAX_QUESTION = 4000;

export function Composer({
  onSend,
  onStop,
  busy,
  children,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  children?: ReactNode;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);
  const submit = () => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  };
  return (
    <form
      className="rounded-2xl border border-border bg-surface p-2 shadow-card focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/20"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <label htmlFor="question" className="sr-only">
        Ask about the house
      </label>
      <textarea
        id="question"
        ref={ref}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
        maxLength={MAX_QUESTION}
        aria-describedby={text.length > MAX_QUESTION - 500 ? "question-count" : undefined}
        placeholder="Ask about the house…"
        className="block max-h-[200px] w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed outline-none placeholder:text-muted-foreground/80"
      />
      <div className="mt-1 flex items-center gap-2">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">{children}</div>
        {text.length > MAX_QUESTION - 500 ? (
          <span id="question-count" className="tabular shrink-0 text-xs text-muted-foreground" aria-live="polite">
            {text.length} / {MAX_QUESTION}
          </span>
        ) : null}
        {busy ? (
          <Button size="icon-sm" variant="secondary" aria-label="Stop" onClick={onStop}>
            <Square className="fill-current" />
          </Button>
        ) : (
          <Button type="submit" size="icon-sm" variant="primary" aria-label="Send" disabled={!text.trim()}>
            <ArrowUp />
          </Button>
        )}
      </div>
    </form>
  );
}
