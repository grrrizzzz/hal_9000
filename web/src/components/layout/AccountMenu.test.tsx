import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { db } from "@/mocks/db";
import { server } from "@/mocks/node";
import { renderApp } from "@/test/render";

async function openEverywhere(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: "Account" }));
  await user.click(await screen.findByRole("button", { name: "Sign out everywhere…" }));
  return screen.findByRole("alertdialog", { name: "Sign out everywhere?" });
}

describe("Account menu", () => {
  it("signs this browser out", async () => {
    const user = userEvent.setup();
    renderApp("/library");
    await screen.findByRole("heading", { name: "Library" });
    await user.click(screen.getByRole("button", { name: "Account" }));
    await user.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(await screen.findByLabelText("Passcode")).toBeInTheDocument();
    expect(db.signedIn).toBe(false);
    expect(db.revokedEverywhere).toBe(0);
  });

  it("asks before signing out everywhere; Cancel changes nothing, confirming revokes every session", async () => {
    const user = userEvent.setup();
    renderApp("/library");
    await screen.findByRole("heading", { name: "Library" });

    let dialog = await openEverywhere(user);
    expect(dialog).toHaveTextContent("Every signed-in browser and phone, including this one");
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(db.signedIn).toBe(true);
    expect(db.revokedEverywhere).toBe(0);

    dialog = await openEverywhere(user);
    await user.click(within(dialog).getByRole("button", { name: "Sign out everywhere" }));
    expect(await screen.findByLabelText("Passcode")).toBeInTheDocument();
    expect(db.revokedEverywhere).toBe(1);
    expect(db.signedIn).toBe(false);
  });

  it("stays put and shows the error when signing out everywhere fails", async () => {
    server.use(http.delete("/api/session", () => HttpResponse.json({ detail: "The server had a problem." }, { status: 500 })));
    const user = userEvent.setup();
    renderApp("/library");
    await screen.findByRole("heading", { name: "Library" });
    const dialog = await openEverywhere(user);
    await user.click(within(dialog).getByRole("button", { name: "Sign out everywhere" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("The server had a problem.");
    // Still on the page behind the (modal) dialog, still signed in.
    expect(screen.getByRole("heading", { name: "Library", hidden: true })).toBeInTheDocument();
    expect(screen.queryByLabelText("Passcode")).toBeNull();
    expect(db.signedIn).toBe(true);
  });
});
