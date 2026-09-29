/** The mocks follow the server where the UI could otherwise pass against them but not the server. */
import { describe, expect, it } from "vitest";
import { db } from "./db";

describe("mock contract", () => {
  it("rejects feedback with neither a rating nor add_to_gold (422), like FeedbackRequest", async () => {
    db.signedIn = true;
    const post = (body: object) => fetch(new URL("/api/feedback", location.href), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    expect((await post({ message_id: 1, comment: "no rating" })).status).toBe(422);
    expect((await post({ message_id: 1, rating: null, add_to_gold: false })).status).toBe(422);
    expect((await post({ message_id: 1, rating: -1 })).status).toBe(200);
    expect((await post({ message_id: 1, add_to_gold: true })).status).toBe(200);
  });
});
