import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { JudgeLabel } from "@/api/types";
import { db } from "@/mocks/db";
import { renderApp } from "@/test/render";
import { agreement } from "./JudgeCalibration";

describe("agreement", () => {
  it("counts labels as the S5 gate does", () => {
    const base = { id: 1, gold_item_id: 1, question: "q", answer: "a" };
    const good = { correctness: "correct", faithful: true, citations_correct: true, abstention_correct: true } as const;
    const labels: JudgeLabel[] = [
      // N/A against the judge's Yes disagrees: 2 of 4
      { ...base, human: { correctness: "correct", faithful: true, citations_correct: false, abstention_correct: null }, judge: good },
      // unlabelled by Mike: not counted
      { ...base, id: 2, human: null, judge: { correctness: "wrong", faithful: false, citations_correct: false } },
      // N/A on both sides: skipped, 3 of 3
      { ...base, id: 3, human: { ...good, abstention_correct: null }, judge: { ...good, abstention_correct: null } },
      // the judge failed: 0 of 4
      { ...base, id: 4, human: good, judge: null },
    ];
    expect(agreement(labels)).toEqual({ agree: 5, total: 11 });
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
    // abstention has no N/A: the judge always answers Yes or No, so Mike must too
    const abstention = within(card).getByRole("radiogroup", { name: /abstention correct/ });
    expect(within(abstention).queryByRole("radio", { name: "N/A" })).toBeNull();
    expect(within(card).getByRole("button", { name: "Save labels" })).toBeDisabled();
    await user.click(within(abstention).getByRole("radio", { name: "Yes" }));
    await user.click(within(card).getByRole("button", { name: "Save labels" }));

    await waitFor(() =>
      expect(db.judgeLabels[0].human).toEqual({ correctness: "partial", faithful: true, citations_correct: false, abstention_correct: true }),
    );
    const comparison = await within(card).findByTestId("judge-comparison");
    // the judge's reason for each label appears with its label
    const reasons = within(comparison).getAllByTestId("judge-reason");
    expect(reasons).toHaveLength(4);
    expect(within(comparison).getByText(/S1 says the furnace was installed in 2009/)).toBeInTheDocument();
  });

  it("says how to get reasons for answers judged before reasons were recorded", async () => {
    db.judgeLabels[1].human = { correctness: "correct", faithful: true, citations_correct: true, abstention_correct: true };
    renderApp("/evaluation?tab=judge");
    const cards = await screen.findAllByTestId("judge-card");
    const comparison = await within(cards[1]).findByTestId("judge-comparison");
    expect(within(comparison).queryAllByTestId("judge-reason")).toHaveLength(0);
    expect(within(comparison).getByText(/--rejudge/)).toBeInTheDocument();
  });
});
