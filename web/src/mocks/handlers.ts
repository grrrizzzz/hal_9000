/**
 * MSW handlers implementing the plan §11 API over the synthetic mock data.
 * Response bodies are typed from the generated OpenAPI schema, so the mocks
 * can't silently drift from the contract.
 */
import { bypass, delay, http, HttpResponse, type HttpHandler } from "msw";
import type { ResponseOf } from "@/api/client";
import type {
  ChatEvent,
  ChatRequest,
  Citation,
  DocumentPatch,
  EvidenceQuote,
  GoldPatch,
  Job,
  JudgeLabel,
  UploadResult,
} from "@/api/types";
import { chunkText, makeTrace, MIME, MOCK_PASSCODE, PAGES, SHA, TRIGGERS } from "./data";
import { db, nextId } from "./db";
import pdfUrl from "./fixtures/synthetic-house.pdf?url";

const unauthorized = () => HttpResponse.json({ detail: "Not signed in" }, { status: 401 });
const notFound = (what: string) => HttpResponse.json({ detail: `${what} not found` }, { status: 404 });
const json = <T>(body: T, init?: ResponseInit) => HttpResponse.json(body as never, init);

type Resolver = Parameters<typeof http.get>[1];

/** Wrap an authenticated handler: 401 unless signed in. */
function authed(fn: Resolver): Resolver {
  return (info) => (db.signedIn ? fn(info) : unauthorized());
}

const enc = new TextEncoder();
const sse = (event: string, data: unknown) => enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

function hexId(): string {
  let s = "";
  for (let i = 0; i < 32; i++) s += Math.floor(Math.random() * 16).toString(16);
  return s;
}

function fakeSha(name: string, size: number): string {
  let h = 2166136261;
  for (const c of `${name}:${size}`) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  let out = "";
  for (let i = 0; i < 8; i++) {
    h = Math.imul(h ^ (h >>> 13), 0x5bd1e995) >>> 0;
    out += h.toString(16).padStart(8, "0");
  }
  return out.slice(0, 64);
}

// ------------------------------------------------------------ jobs
function emitJob(job: Job) {
  for (const l of db.jobListeners) l(job);
}

const STEPS = ["screen", "parse", "clean", "segment", "chunk", "embed", "index"];

function startJob(job: Job, onDone?: () => void) {
  db.jobs.unshift(job);
  emitJob(job);
  let step = -1;
  const timer = setInterval(() => {
    step++;
    const now = new Date().toISOString();
    if (step < STEPS.length) {
      Object.assign(job, { status: "running", progress: (step + 1) / (STEPS.length + 1), detail: STEPS[step], updated_at: now });
    } else {
      Object.assign(job, { status: "done", progress: 1, detail: "indexed", updated_at: now });
      clearInterval(timer);
      onDone?.();
    }
    emitJob({ ...job });
  }, db.jobTickMs);
  db.timers.push(timer);
}

// ------------------------------------------------------------ chat
function citationFor(marker: string, chunkId: number, documentId: number): Citation {
  const c = chunkText(chunkId)!;
  const doc = db.documents.find((d) => d.document.id === documentId)!;
  return {
    marker,
    chunk_id: chunkId,
    document_id: documentId,
    file_sha: c.sha,
    page: c.page,
    title: doc.document.title,
    doc_type: doc.document.doc_type ?? "other",
    doc_date: doc.document.doc_date ?? null,
    vendor: doc.document.vendor ?? null,
    snippet: c.text,
  };
}

interface Script {
  sources: Citation[];
  events: ChatEvent[];
  answer: string;
  outcome: "answered" | "abstained" | "retrieval_only";
  grounding: "verified" | "failed" | "skipped";
  issues: string[];
  retrievalOnly?: boolean;
  banner?: string | null;
}

const words = (s: string) => s.match(/\S+\s*/g) ?? [];

function scriptFor(q: string): Script {
  const lower = q.toLowerCase();
  const furnace = [citationFor("S1", 12, 1), citationFor("S2", 31, 3)];
  if (lower.includes(TRIGGERS.reset)) {
    const sources = [citationFor("S1", 14, 2)];
    const wrong = "The termite repair cost $1,324.00 [S1].";
    const right = "Invoice 1001 from Brightline Pest Co. shows the termite repair at the rear deck ledger totalled $1,234.00, completed July 14, 2023 [S1].";
    return {
      sources,
      events: [
        ...words(wrong).map((t) => ({ event: "token", text: t }) as ChatEvent),
        { event: "reset", reason: "A figure ($1,324.00) wasn't found in the cited source." },
        ...words(right).map((t) => ({ event: "token", text: t }) as ChatEvent),
        { event: "verify", grounding: "verified", issues: [] },
      ],
      answer: right,
      outcome: "answered",
      grounding: "verified",
      issues: [],
    };
  }
  if (lower.includes(TRIGGERS.retrievalOnly)) {
    return {
      sources: furnace,
      events: [],
      answer: "",
      outcome: "retrieval_only",
      grounding: "skipped",
      issues: [],
      retrievalOnly: true,
      banner: "This month's answer budget is used up, so here are the closest passages instead. Search still works.",
    };
  }
  if (lower.includes(TRIGGERS.unanswerable)) {
    const text = "I couldn't find that in the house documents. A pool equipment manual or a permit for pool work would say.";
    return {
      sources: [citationFor("S1", 11, 1)],
      events: [...words(text).map((t) => ({ event: "token", text: t }) as ChatEvent), { event: "verify", grounding: "skipped", issues: [] }],
      answer: text,
      outcome: "abstained",
      grounding: "skipped",
      issues: [],
    };
  }
  const unverified = lower.includes(TRIGGERS.unverified);
  const text = unverified
    ? "The furnace was installed in 2008 [S1] and should get a new filter every 90 days [S2]."
    : "The June 2023 inspection describes a forced-air gas furnace in the garage, installed in 2009 and near the end of its expected life [S1]. The manual says to replace the air filter every 90 days, or monthly during heavy use [S2].";
  const issues = unverified ? ["2008 does not appear in [S1]"] : [];
  return {
    sources: furnace,
    events: [
      ...words(text).map((t) => ({ event: "token", text: t }) as ChatEvent),
      ...(unverified
        ? ([
            { event: "reset", reason: "A figure (2008) wasn't found in the cited source." },
            ...words(text).map((t) => ({ event: "token", text: t }) as ChatEvent),
          ] as ChatEvent[])
        : []),
      { event: "verify", grounding: unverified ? "failed" : "verified", issues },
    ],
    answer: text,
    outcome: "answered",
    grounding: unverified ? "failed" : "verified",
    issues,
    banner: lower.includes(TRIGGERS.embeddingDown)
      ? "Semantic search is unavailable right now, so these results come from keyword search only and may be less complete."
      : null,
  };
}

function chatStream(req: ChatRequest): ReadableStream<Uint8Array> {
  const script = scriptFor(req.message);
  const now = new Date().toISOString();
  let convo = req.conversation_id ? db.conversations.find((c) => c.id === req.conversation_id) : undefined;
  if (!convo) {
    convo = { id: nextId(), title: req.message.slice(0, 60), created_at: now, updated_at: now, messages: [] };
    db.conversations.unshift(convo);
  }
  const traceId = hexId();
  (convo.messages ??= []).push({ id: nextId(), role: "user", content: req.message, citations: [], trace_id: null, outcome: null, created_at: now });
  const messageId = nextId();
  const conversationId = convo.id;
  const embeddingDown = Boolean(script.banner) && !script.retrievalOnly;

  return new ReadableStream({
    async start(controller) {
      const send = (e: ChatEvent) => controller.enqueue(sse(e.event, e));
      try {
        send({ event: "status", stage: "embed_query", detail: embeddingDown ? "embedding unavailable, using keyword search" : "" });
        await delay(db.tokenDelayMs * 3);
        send({ event: "status", stage: "vector", detail: "" });
        await delay(db.tokenDelayMs * 2);
        send({ event: "status", stage: "select", detail: "" });
        send({ event: "sources", sources: script.sources, retrieval_only: Boolean(script.retrievalOnly), banner: script.banner ?? null });
        if (script.events.length) send({ event: "status", stage: "generate", detail: "" });
        for (const e of script.events) {
          await delay(e.event === "reset" ? db.tokenDelayMs * 8 : db.tokenDelayMs);
          if (e.event === "verify") send({ event: "status", stage: "verify", detail: "" });
          send(e);
        }
        const cited = script.sources.filter((s) => script.answer.includes(`[${s.marker}]`));
        const answer = {
          text: script.answer,
          citations: script.retrievalOnly ? script.sources : cited,
          outcome: script.outcome,
          grounding: script.grounding,
          issues: script.issues,
          trace_id: traceId,
          message_id: messageId,
          conversation_id: conversationId,
          cost_usd: script.retrievalOnly ? 0 : 0.00041,
        };
        (convo!.messages ??= []).push({
          id: messageId,
          role: "assistant",
          content: script.answer,
          citations: answer.citations,
          trace_id: traceId,
          outcome: script.outcome,
          created_at: new Date().toISOString(),
        });
        convo!.updated_at = new Date().toISOString();
        const trace = makeTrace(traceId, req.message, script.outcome, new Date(), !script.retrievalOnly);
        db.traces.unshift(trace);
        send({ event: "done", answer });
      } catch {
        /* client went away */
      } finally {
        try {
          controller.close();
        } catch {
          /* already closed */
        }
      }
    },
  });
}

// ------------------------------------------------------------ handlers
export const handlers: HttpHandler[] = [
  http.get("/health", () =>
    json<ResponseOf<"/health", "get">>({
      status: db.health.retrievalOnly ? "degraded" : "ok",
      version: "0.0.0-mock",
      active_profile: db.profiles.find((p) => p.active)?.name ?? null,
      month_spend_usd: db.health.retrievalOnly ? 10 : 1.84,
      month_cap_usd: 10,
      retrieval_only: db.health.retrievalOnly,
    }),
  ),

  http.post("/api/session", async ({ request }) => {
    if (db.failedSignIns >= 10) return json({ detail: "Too many attempts. Wait a minute and try again." }, { status: 429 });
    const body = (await request.json()) as { passcode?: string };
    await delay(150);
    if (body.passcode !== MOCK_PASSCODE) {
      db.failedSignIns++;
      return json({ detail: "That passcode isn't right." }, { status: 401 });
    }
    db.signedIn = true;
    db.failedSignIns = 0;
    return json<ResponseOf<"/api/session", "post">>({ ok: true, expires_at: new Date(Date.now() + 30 * 86400_000).toISOString() });
  }),
  http.delete("/api/session", () => {
    db.signedIn = false;
    return new HttpResponse(null, { status: 204 });
  }),

  // ---------------------------------------------------------- uploads and jobs
  http.post(
    "/api/uploads",
    authed(async ({ request }) => {
      const form = await request.formData();
      const files = form.getAll("files") as File[];
      const paths = form.getAll("relative_path") as string[];
      const results: UploadResult[] = [];
      files.forEach((file, i) => {
        const relativePath = paths[i] ?? file.name;
        const ext = relativePath.split(".").pop()?.toLowerCase() ?? "";
        const parts = relativePath.split("/");
        const collection = parts.slice(0, -1).join("/");
        if (!["pdf", "docx", "txt", "md", "html"].includes(ext)) {
          results.push({ relative_path: relativePath, status: "skipped", sha256: null, collection, job_id: null, reason: `.${ext} is not a supported type` });
          return;
        }
        const sha = fakeSha(relativePath, file.size);
        if (db.uploads.has(sha)) {
          results.push({ relative_path: relativePath, status: "duplicate", sha256: sha, collection, job_id: null, reason: "already in library" });
          return;
        }
        db.uploads.set(sha, { name: file.name, relativePath, collection, bytes: file.size });
        const now = new Date().toISOString();
        const job: Job = { id: nextId(), kind: "ingest", status: "queued", progress: 0, detail: "queued", error: null, file_sha: sha, created_at: now, updated_at: now };
        const failing = file.name.toLowerCase().includes("corrupt");
        if (failing) {
          db.jobs.unshift(job);
          setTimeout(() => {
            Object.assign(job, { status: "failed", detail: "parse", error: "The PDF could not be opened (damaged file)." });
            emitJob({ ...job });
          }, db.jobTickMs * 2);
        } else {
          startJob(job, () => {
            db.documents.push({
              document: { id: nextId(), file_sha: sha, page_start: 1, page_end: 1, title: file.name.replace(/\.[^.]+$/, ""), doc_type: "other", vendor: null, doc_date: null, meta_source: "auto" },
              original_name: file.name,
              relative_path: relativePath,
              collection,
              pages: 1,
              pages_without_text: 0,
              chunks: 1,
              status: "indexed",
            });
          });
        }
        results.push({ relative_path: relativePath, status: "added", sha256: sha, collection, job_id: job.id ?? null, reason: null });
      });
      await delay(80);
      return json<ResponseOf<"/api/uploads", "post">>({ results });
    }),
  ),
  http.get("/api/jobs", authed(() => json<ResponseOf<"/api/jobs", "get">>(db.jobs))),
  http.get(
    "/api/jobs/stream",
    authed(({ request }) => {
      let listener: ((j: Job) => void) | null = null;
      let ping: ReturnType<typeof setInterval> | null = null;
      const stop = () => {
        if (listener) db.jobListeners.delete(listener);
        if (ping) clearInterval(ping);
      };
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const j of [...db.jobs].reverse()) controller.enqueue(sse("job", j));
          listener = (j) => {
            try {
              controller.enqueue(sse("job", j));
            } catch {
              stop();
            }
          };
          db.jobListeners.add(listener);
          ping = setInterval(() => {
            try {
              controller.enqueue(enc.encode(": ping\n\n"));
            } catch {
              stop();
            }
          }, 15000);
          request.signal?.addEventListener("abort", () => {
            stop();
            try {
              controller.close();
            } catch {
              /* closed */
            }
          });
        },
        cancel: stop,
      });
      return new HttpResponse(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
    }),
  ),

  // ---------------------------------------------------------- library
  http.get(
    "/api/documents",
    authed(({ request }) => {
      const url = new URL(request.url);
      const collection = url.searchParams.get("collection");
      const docType = url.searchParams.get("doc_type");
      return json<ResponseOf<"/api/documents", "get">>(
        db.documents.filter((d) => (!collection || d.collection === collection) && (!docType || d.document.doc_type === docType)),
      );
    }),
  ),
  http.get(
    "/api/documents/:id",
    authed(({ params }) => {
      const d = db.documents.find((x) => x.document.id === Number(params.id));
      if (!d) return notFound("Document");
      const upload = db.uploads.get(d.document.file_sha);
      return json<ResponseOf<"/api/documents/{id}", "get">>({
        ...d,
        file: {
          sha256: d.document.file_sha,
          original_name: d.original_name,
          relative_path: d.relative_path,
          collection: d.collection,
          mime_type: MIME[d.document.file_sha] ?? (d.original_name.endsWith(".pdf") ? "application/pdf" : "text/plain"),
          bytes: upload?.bytes ?? 2890,
          version_of: null,
          uploaded_at: "2026-09-20T18:00:00Z",
        },
        siblings: db.documents.filter((x) => x.document.file_sha === d.document.file_sha && x.document.id !== d.document.id).map((x) => x.document),
      });
    }),
  ),
  http.patch(
    "/api/documents/:id",
    authed(async ({ params, request }) => {
      const d = db.documents.find((x) => x.document.id === Number(params.id));
      if (!d) return notFound("Document");
      const patch = (await request.json()) as DocumentPatch;
      for (const [k, v] of Object.entries(patch)) if (v !== undefined) (d.document as Record<string, unknown>)[k] = v;
      d.document.meta_source = "user";
      return json<ResponseOf<"/api/documents/{id}", "patch">>(d.document);
    }),
  ),
  http.delete(
    "/api/documents/:id",
    authed(({ params }) => {
      db.documents = db.documents.filter((x) => x.document.id !== Number(params.id));
      return new HttpResponse(null, { status: 204 });
    }),
  ),
  http.post(
    "/api/documents/:id/reindex",
    authed(({ params }) => {
      const d = db.documents.find((x) => x.document.id === Number(params.id));
      if (!d) return notFound("Document");
      const now = new Date().toISOString();
      const job: Job = { id: nextId(), kind: "reindex", status: "queued", progress: 0, detail: "queued", error: null, file_sha: d.document.file_sha, created_at: now, updated_at: now };
      d.status = "indexing";
      startJob(job, () => {
        d.status = "indexed";
      });
      return json<ResponseOf<"/api/documents/{id}/reindex", "post">>(job);
    }),
  ),
  http.get(
    "/api/files/:sha",
    authed(async ({ params }) => {
      const sha = String(params.sha);
      const mime = MIME[sha];
      if (mime === "application/pdf" || (!mime && db.uploads.get(sha)?.name.endsWith(".pdf"))) {
        const res = await fetch(bypass(new URL(pdfUrl, location.href)));
        return new HttpResponse(await res.arrayBuffer(), { headers: { "Content-Type": "application/pdf" } });
      }
      const pages = PAGES[sha];
      if (!pages) return notFound("File");
      return new HttpResponse(pages.map((p) => p.text).join("\n\n"), {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": 'attachment; filename="original"',
          "X-Content-Type-Options": "nosniff",
          "Content-Security-Policy": "sandbox",
        },
      });
    }),
  ),
  http.get(
    "/api/files/:sha/pages/:n",
    authed(({ params }) => {
      const sha = String(params.sha);
      const n = Number(params.n);
      const pages = PAGES[sha] ?? (db.uploads.has(sha) ? [{ text: `Synthetic text of ${db.uploads.get(sha)!.name}.`, has_text_layer: true, spans: [] }] : null);
      if (!pages || n < 1 || n > pages.length) return notFound("Page");
      const p = pages[n - 1];
      return json<ResponseOf<"/api/files/{sha}/pages/{n}", "get">>({ file_sha: sha, page: n, page_count: pages.length, text: p.text, has_text_layer: p.has_text_layer, spans: p.spans });
    }),
  ),

  // ---------------------------------------------------------- chat
  http.post(
    "/api/chat",
    authed(async ({ request }) => {
      const req = (await request.json()) as ChatRequest;
      if (!req.message?.trim()) return json({ detail: "Ask a question first." }, { status: 422 });
      return new HttpResponse(chatStream(req), { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
    }),
  ),
  http.get(
    "/api/conversations",
    authed(() => json<ResponseOf<"/api/conversations", "get">>(db.conversations.map(({ messages: _m, ...c }) => c))),
  ),
  http.delete(
    "/api/conversations",
    authed(() => {
      db.conversations = [];
      return new HttpResponse(null, { status: 204 });
    }),
  ),
  http.get(
    "/api/conversations/:id",
    authed(({ params }) => {
      const c = db.conversations.find((x) => x.id === Number(params.id));
      return c ? json<ResponseOf<"/api/conversations/{id}", "get">>(c) : notFound("Conversation");
    }),
  ),
  http.delete(
    "/api/conversations/:id",
    authed(({ params }) => {
      db.conversations = db.conversations.filter((x) => x.id !== Number(params.id));
      return new HttpResponse(null, { status: 204 });
    }),
  ),
  http.post(
    "/api/feedback",
    authed(async ({ request }) => {
      const body = (await request.json()) as { message_id: number; rating: number; comment?: string | null; add_to_gold?: boolean };
      const id = nextId();
      db.feedback.push({ id, ...body, add_to_gold: Boolean(body.add_to_gold) });
      let goldId: number | null = null;
      if (body.add_to_gold) {
        let question = "";
        let answer = "";
        let citations: Citation[] = [];
        for (const c of db.conversations) {
          const msgs = c.messages ?? [];
          const i = msgs.findIndex((m) => m.id === body.message_id);
          if (i >= 0) {
            answer = msgs[i].content;
            citations = msgs[i].citations ?? [];
            question = msgs[i - 1]?.content ?? "";
          }
        }
        goldId = nextId();
        const evidence: EvidenceQuote[] = citations.map((c) => ({ quote: c.snippet || c.title, page: c.page, file_sha: c.file_sha }));
        db.gold.unshift({
          id: goldId,
          question: question || "(question)",
          reference_answer: answer || null,
          category: evidence.length ? "factoid" : "unanswerable",
          difficulty: "medium",
          paraphrased: false,
          evidence,
          status: "draft",
          split: null,
          reviewer_note: body.comment ?? null,
          origin: "feedback",
        });
      }
      return json<ResponseOf<"/api/feedback", "post">>({ id, gold_item_id: goldId });
    }),
  ),
  http.get(
    "/api/search",
    authed(({ request }) => {
      const q = new URL(request.url).searchParams.get("q") ?? "";
      return json<ResponseOf<"/api/search", "get">>({
        query: q,
        standalone_query: q,
        profile: "structure-512-h",
        mode: "hybrid",
        candidates: [],
        selected: [],
        context_tokens: 0,
        abstain_score: null,
        threshold: null,
        abstained: false,
      });
    }),
  ),

  // ---------------------------------------------------------- evaluation
  http.get(
    "/api/eval/gold",
    authed(({ request }) => {
      const u = new URL(request.url).searchParams;
      const status = u.get("status");
      const split = u.get("split");
      const category = u.get("category");
      return json<ResponseOf<"/api/eval/gold", "get">>(
        db.gold.filter((g) => (!status || g.status === status) && (!split || g.split === split) && (!category || g.category === category)),
      );
    }),
  ),
  http.get(
    "/api/eval/gold/:id",
    authed(({ params }) => {
      const g = db.gold.find((x) => x.id === Number(params.id));
      return g ? json<ResponseOf<"/api/eval/gold/{id}", "get">>(g) : notFound("Question");
    }),
  ),
  http.patch(
    "/api/eval/gold/:id",
    authed(async ({ params, request }) => {
      const g = db.gold.find((x) => x.id === Number(params.id));
      if (!g) return notFound("Question");
      const patch = (await request.json()) as GoldPatch;
      for (const [k, v] of Object.entries(patch)) if (v !== undefined && v !== null) (g as Record<string, unknown>)[k] = v;
      if (patch.reviewer_note === null) g.reviewer_note = null;
      return json<ResponseOf<"/api/eval/gold/{id}", "patch">>(g);
    }),
  ),
  http.get(
    "/api/eval/judge-labels",
    authed(({ request }) => {
      const run = new URL(request.url).searchParams.get("run");
      return json<ResponseOf<"/api/eval/judge-labels", "get">>(db.judgeLabels.filter((l) => !run || l.eval_run_id === run));
    }),
  ),
  http.get(
    "/api/eval/judge-labels/:id",
    authed(({ params }) => {
      const l = db.judgeLabels.find((x) => x.id === Number(params.id));
      return l ? json<ResponseOf<"/api/eval/judge-labels/{id}", "get">>(l) : notFound("Label");
    }),
  ),
  http.put(
    "/api/eval/judge-labels/:id",
    authed(async ({ params, request }) => {
      const l = db.judgeLabels.find((x) => x.id === Number(params.id));
      if (!l) return notFound("Label");
      const body = (await request.json()) as { human: JudgeLabel["human"] };
      l.human = body.human;
      return json<ResponseOf<"/api/eval/judge-labels/{id}", "put">>(l);
    }),
  ),
  http.get(
    "/api/eval/runs",
    authed(({ request }) => {
      const stage = new URL(request.url).searchParams.get("stage");
      return json<ResponseOf<"/api/eval/runs", "get">>(db.runs.filter((r) => !stage || r.stage === stage));
    }),
  ),
  http.get(
    "/api/eval/runs/:id",
    authed(({ params }) => {
      const r = db.runs.find((x) => x.id === params.id);
      return r ? json<ResponseOf<"/api/eval/runs/{id}", "get">>(r) : notFound("Run");
    }),
  ),
  http.get(
    "/api/profile",
    authed(() =>
      json<ResponseOf<"/api/profile", "get">>({
        active: db.profiles.find((p) => p.active) ?? null,
        profiles: db.profiles,
        reindex_job: db.jobs.find((j) => j.kind === "reindex" && j.file_sha === null && j.status !== "done") ?? null,
      }),
    ),
  ),
  http.put(
    "/api/profile",
    authed(async ({ request }) => {
      const { name } = (await request.json()) as { name: string };
      const target = db.profiles.find((p) => p.name === name);
      if (!target) return notFound("Profile");
      for (const p of db.profiles) p.active = p === target;
      const now = new Date().toISOString();
      const job: Job = { id: nextId(), kind: "reindex", status: "queued", progress: 0, detail: "queued", error: null, file_sha: null, created_at: now, updated_at: now };
      startJob(job);
      return json<ResponseOf<"/api/profile", "put">>({ active: target, profiles: db.profiles, reindex_job: job });
    }),
  ),

  // ---------------------------------------------------------- traces
  http.get(
    "/api/traces",
    authed(({ request }) => {
      const limit = Number(new URL(request.url).searchParams.get("limit") ?? 50);
      return json<ResponseOf<"/api/traces", "get">>(
        db.traces.slice(0, limit).map((t) => ({ id: t.id!, created_at: t.created_at!, outcome: t.outcome ?? null, total_ms: t.total_ms ?? null, ttft_ms: t.ttft_ms ?? null, cost_usd: t.cost_usd })),
      );
    }),
  ),
  http.get(
    "/api/traces/:id",
    authed(({ params }) => {
      const t = db.traces.find((x) => x.id === params.id);
      return t ? json<ResponseOf<"/api/traces/{id}", "get">>(t) : notFound("Trace");
    }),
  ),
];

export { SHA };
