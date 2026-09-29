import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { MOCK_PASSCODE } from "@/mocks/data";
import { renderApp } from "@/test/render";

describe("Sign in", () => {
  it("redirects to sign in on 401, shows a wrong passcode, then returns to the page", async () => {
    const user = userEvent.setup();
    renderApp("/library", { signedIn: false });
    const input = await screen.findByLabelText("Passcode");
    await user.type(input, "nope");
    await user.click(screen.getByRole("button", { name: /sign in/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("That passcode isn't right.");

    await user.clear(input);
    await user.type(input, MOCK_PASSCODE);
    await user.click(screen.getByRole("button", { name: /sign in/i }));
    expect(await screen.findByRole("heading", { name: "Library" })).toBeInTheDocument();
  });
});
