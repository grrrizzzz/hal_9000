/**
 * Live job progress: GET /api/jobs once, then GET /api/jobs/stream (SSE,
 * one `job` event per change) read with fetch, reconnecting with backoff.
 */
import { useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, notifyUnauthorized } from "@/api/client";
import type { Job } from "@/api/types";
import { parseSSE } from "./sse";

interface JobsState {
  jobs: Map<number, Job>;
  connected: boolean;
}

const JobsContext = createContext<JobsState>({ jobs: new Map(), connected: false });

export function JobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<Map<number, Job>>(new Map());
  const [connected, setConnected] = useState(false);
  const qc = useQueryClient();

  useEffect(() => {
    const ctrl = new AbortController();
    let attempt = 0;
    const upsert = (list: Job[]) =>
      setJobs((prev) => {
        const next = new Map(prev);
        let finished = false;
        for (const j of list) {
          if (j.id === null || j.id === undefined) continue;
          const before = prev.get(j.id);
          if (before && before.status !== "done" && j.status === "done") finished = true;
          next.set(j.id, j);
        }
        if (finished) {
          void qc.invalidateQueries({ queryKey: ["documents"] });
          void qc.invalidateQueries({ queryKey: ["profile"] });
        }
        return next;
      });

    const run = async () => {
      try {
        upsert(await api.get("/api/jobs", { signal: ctrl.signal }));
      } catch {
        /* the stream below retries */
      }
      while (!ctrl.signal.aborted) {
        try {
          const res = await fetch("/api/jobs/stream", { credentials: "same-origin", headers: { Accept: "text/event-stream" }, signal: ctrl.signal });
          if (res.status === 401) {
            notifyUnauthorized();
            return;
          }
          if (!res.ok || !res.body) throw new Error(`jobs stream HTTP ${res.status}`);
          setConnected(true);
          attempt = 0;
          for await (const msg of parseSSE(res.body, ctrl.signal)) {
            if (msg.event !== "job" && msg.event !== "message") continue;
            try {
              upsert([JSON.parse(msg.data) as Job]);
            } catch {
              /* ignore a malformed event */
            }
          }
        } catch (e) {
          if ((e as Error).name === "AbortError") return;
        }
        setConnected(false);
        attempt++;
        await new Promise((r) => setTimeout(r, Math.min(15_000, 500 * 2 ** attempt)));
      }
    };
    void run();
    return () => ctrl.abort();
  }, [qc]);

  const value = useMemo(() => ({ jobs, connected }), [jobs, connected]);
  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>;
}

export const useJobs = () => useContext(JobsContext);
