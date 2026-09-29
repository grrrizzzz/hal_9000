import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { db } from "@/mocks/db";
import { server } from "@/mocks/node";
import { JobsProvider, useJobs } from "./jobs";

function Show() {
  const { jobs } = useJobs();
  return <p data-testid="jobs">{[...jobs.values()].map((j) => `${j.id}:${j.status}`).join(",")}</p>;
}

describe("JobsProvider", () => {
  it("refetches the job list after the stream reconnects", { timeout: 10000 }, async () => {
    db.signedIn = true;
    let listCalls = 0;
    let streamCalls = 0;
    const now = new Date().toISOString();
    server.use(
      http.get("/api/jobs", () => {
        listCalls++;
        // The second list shows a job that changed while the stream was down.
        return HttpResponse.json(
          listCalls === 1 ? [] : [{ id: 7, kind: "ingest", status: "done", progress: 1, detail: "indexed", error: null, file_sha: null, created_at: now, updated_at: now }],
        );
      }),
      http.get("/api/jobs/stream", () => {
        streamCalls++;
        // A stream that ends at once, as when the server restarts.
        return new HttpResponse(": bye\n\n", { headers: { "Content-Type": "text/event-stream" } });
      }),
    );
    render(
      <QueryClientProvider client={new QueryClient()}>
        <JobsProvider>
          <Show />
        </JobsProvider>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(streamCalls).toBeGreaterThanOrEqual(2), { timeout: 6000 });
    expect(listCalls).toBeGreaterThanOrEqual(2);
    await waitFor(() => expect(screen.getByTestId("jobs")).toHaveTextContent("7:done"));
  });
});
