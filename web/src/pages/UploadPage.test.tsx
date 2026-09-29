import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ingestUpload } from "@/mocks/handlers";
import { renderApp } from "@/test/render";

// jsdom's FormData can't be read by MSW's XHR interceptor in Node, so the
// transport is swapped for the same server-side logic; the XHR path itself is
// covered by Playwright in a real browser.
const calls: string[] = [];
vi.mock("@/lib/upload/queue", async (orig) => {
  const real = await orig<typeof import("@/lib/upload/queue")>();
  return {
    ...real,
    uploadOne: async (file: File, rel: string, onProgress: (f: number) => void) => {
      calls.push(rel);
      onProgress(0.5);
      await new Promise((r) => setTimeout(r, 5));
      onProgress(1);
      return ingestUpload(file.name, file.size, rel);
    },
  };
});

function withPath(name: string, rel: string, body = "x") {
  const f = new File([body], name);
  Object.defineProperty(f, "webkitRelativePath", { value: rel });
  return f;
}

describe("Upload", () => {
  it("summarises a chosen folder, uploads the indexable files and follows their jobs", { timeout: 15000 }, async () => {
    const user = userEvent.setup();
    renderApp("/upload");
    const input = await screen.findByTestId("folder-input");
    fireEvent.change(input, {
      target: {
        files: [withPath("a.pdf", "Box/a.pdf"), withPath("b.md", "Box/Sub/b.md"), withPath("c.jpg", "Box/Sub/c.jpg"), withPath("d.heic", "Box/d.heic")],
      },
    });
    expect(await screen.findByTestId("summary-text")).toHaveTextContent("4 files found: 2 will be added, 2 skipped (.heic ×1, .jpg ×1)");
    await user.click(screen.getByRole("button", { name: "Upload 2 files" }));
    await waitFor(
      () => expect([...document.querySelectorAll("[data-testid=upload-row]")].map((r) => r.getAttribute("data-status"))).toEqual(["Indexed", "Indexed"]),
      { timeout: 8000 },
    );
    expect(calls.sort()).toEqual(["Box/Sub/b.md", "Box/a.pdf"]);
    expect(screen.getByText("Box/Sub/b.md")).toBeInTheDocument();
  });
});
