import { useMutation } from "@tanstack/react-query";
import { Check, Info, ListPlus, ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export function AnswerActions({ messageId, traceId }: { messageId: number | null | undefined; traceId: string | null | undefined }) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const [downOpen, setDownOpen] = useState(false);
  const [comment, setComment] = useState("");
  const [goldId, setGoldId] = useState<number | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const send = useMutation({
    mutationFn: (body: { rating: 1 | -1; comment?: string | null; add_to_gold?: boolean }) =>
      api.post("/api/feedback", { body: { message_id: messageId!, add_to_gold: false, ...body } }),
  });

  if (!messageId) return null;

  const rate = (r: 1 | -1, c?: string) =>
    send.mutate(
      { rating: r, comment: c || null },
      {
        onSuccess: () => {
          setRating(r);
          setNote(r === 1 ? "Thanks — noted." : "Thanks — this helps tune the answers.");
        },
      },
    );

  return (
    <div className="mt-2 flex flex-wrap items-center gap-1 text-muted-foreground" data-testid="answer-actions">
      <Tip label="Good answer">
        <Button size="icon-sm" variant="ghost" aria-label="Good answer" aria-pressed={rating === 1} className={cn(rating === 1 && "text-success")} onClick={() => rate(1)} disabled={send.isPending}>
          <ThumbsUp />
        </Button>
      </Tip>
      <Popover open={downOpen} onOpenChange={setDownOpen}>
        <PopoverTrigger asChild>
          <Button size="icon-sm" variant="ghost" aria-label="Bad answer" aria-pressed={rating === -1} className={cn(rating === -1 && "text-danger")}>
            <ThumbsDown />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              rate(-1, comment.trim());
              setDownOpen(false);
            }}
          >
            <label htmlFor={`fb-${messageId}`} className="text-[13px] font-medium">
              What was wrong? <span className="font-normal text-muted-foreground">(optional)</span>
            </label>
            <Textarea id={`fb-${messageId}`} className="mt-2" value={comment} onChange={(e) => setComment(e.target.value)} placeholder="e.g. It quoted the wrong invoice" />
            <div className="mt-2 flex justify-end">
              <Button type="submit" size="sm" variant="primary">
                Send feedback
              </Button>
            </div>
          </form>
        </PopoverContent>
      </Popover>
      {goldId ? (
        <Button size="sm" variant="ghost" asChild className="text-success">
          <Link to={`/evaluation?tab=review&status=draft&item=${goldId}`}>
            <Check /> In the review queue
          </Link>
        </Button>
      ) : (
        <Button
          size="sm"
          variant="ghost"
          disabled={send.isPending}
          onClick={() =>
            send.mutate(
              { rating: rating ?? 1, comment: comment.trim() || null, add_to_gold: true },
              { onSuccess: (r) => setGoldId(r.gold_item_id ?? -1) },
            )
          }
        >
          <ListPlus /> Add to question set
        </Button>
      )}
      {traceId ? (
        <Tip label="Inspect retrieval and timing">
          <Button size="icon-sm" variant="ghost" asChild>
            <Link to={`/traces/${traceId}`} aria-label="Inspect this answer">
              <Info />
            </Link>
          </Button>
        </Tip>
      ) : null}
      <span role="status" className="ml-1 text-xs">
        {send.isError ? <span className="text-danger">{(send.error as Error).message}</span> : note}
      </span>
    </div>
  );
}
