import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { JudgeLabel } from "@/api/types";
import { db } from "@/mocks/db";
import { renderApp } from "@/test/render";
import { agreement } from "./JudgeCalibration";

describe("agreement", () => {
  it("compares label by label, skipping N/A and unlabelled items", () => {
    const base = { id: 1, gold_item_id: 1, question: "q", answer: "a" };
    const labels: JudgeLabel[] = [
      { ...base, human: { correctness: "correct", faithful: true, citations_correct: false, abstention_correct: null }, judge: { correctness: "correct", faithful: true, citations_correct: true, abstention_correct: true } },
      { ...base, id: 2, human: null, judge: { correctness: "wrong", faithful: false, citations_correct: false } },
    ];
    expect(agreement(labels)).toEqual({ agree: 2, total: 3 });
  });
});

describe("Judge calibration", () => {
  it("saves Mike's labels and then reveals the judge's", async () => {
    const user = userEvent.setup();
    renderApp("/evaluation?tab=judge");
    const cards = await screen.findAllByTestId("judge-card");
    const card = cards.find((c) => c.dataset.labelled === "false")!;
    expect(within(card).queryByTestId("judge-comparison")).toBeNull();

    await user.click(within(card).getByRole("radio", { name: "Partial" }));
    await user.click(within(within(card).getByRole("radiogroup", { name: /faithful/ })).getByRole("radio", { name: "Yes" }));
    await user.click(within(within(card).getByRole("radiogroup", { name: /citations correct/ })).getByRole("radio", { name: "No" }));
    await user.click(within(card).getByRole("button", { name: "Save labels" }));

    await waitFor(() =>
      expect(db.judgeLabels[0].human).toEqual({ correctness: "partial", faithful: true, citations_correct: false, abstention_correct: null }),
    );
    expect(await within(card).findByTestId("judge-comparison")).toBeInTheDocument();
  });
});
