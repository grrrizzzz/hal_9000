import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import type { Outcome } from "@/api/types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AnswerBlock } from "./Turn";

function renderAnswer(outcome: Outcome, grounding: "verified" | "failed" | "skipped" = "skipped") {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <TooltipProvider>
        <MemoryRouter>
          <AnswerBlock a={{ text: "The furnace is in the garage.", citations: [], outcome, grounding, messageId: 7, traceId: "t1" }} onOpen={() => {}} />
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}

describe("AnswerBlock", () => {
  it("says when an answer was stopped or failed part-way", () => {
    renderAnswer("cancelled");
    expect(screen.getByTestId("answer-incomplete")).toHaveTextContent("Stopped before the answer was finished");
  });
  it("says when an answer errored", () => {
    renderAnswer("error");
    expect(screen.getByTestId("answer-incomplete")).toHaveTextContent("The answer couldn't be completed");
  });
  it("shows no such note on a normal answer, and the badge when figures failed the check", () => {
    renderAnswer("answered", "failed");
    expect(screen.queryByTestId("answer-incomplete")).toBeNull();
    expect(screen.getByTestId("badge-unverified")).toHaveTextContent("Figures not verified");
  });
});
