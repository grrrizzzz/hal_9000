/** The mocks' in-memory state. Reset between Vitest tests; lives for the page in the browser. */
import type { JudgeLabelPending, MessageOutPending } from "@/api/pending-contract";
import type {
  ConversationDetail,
  DocumentSummary,
  EvalRun,
  GoldItem,
  IndexProfile,
  Job,
  Trace,
} from "@/api/types";
import { seedDocuments, seedGold, seedJudgeLabels, seedProfiles, seedRuns, seedTraces } from "./data";

export interface MockDb {
  signedIn: boolean;
  failedSignIns: number;
  documents: DocumentSummary[];
  jobs: Job[];
  conversations: (Omit<ConversationDetail, "messages"> & { messages: MessageOutPending[] })[];
  gold: GoldItem[];
  judgeLabels: JudgeLabelPending[];
  runs: EvalRun[];
  profiles: IndexProfile[];
  traces: Trace[];
  /** One row per message (a later POST for the same message updates it). */
  feedback: { id: number; message_id: number; rating: number | null; comment: string | null; add_to_gold: boolean; gold_item_id: number | null }[];
  nextId: number;
  /** Uploaded files by sha: name, relative path, collection. */
  uploads: Map<string, { name: string; relativePath: string; collection: string; bytes: number }>;
  /** Global degraded switches (the Health endpoint reports them). */
  health: { retrievalOnly: boolean };
  jobListeners: Set<(job: Job) => void>;
  timers: ReturnType<typeof setInterval>[];
  /** Job tick interval in ms (Vitest shortens it). */
  jobTickMs: number;
  /** Delay between streamed tokens in ms. */
  tokenDelayMs: number;
}

function fresh(): MockDb {
  return {
    signedIn: false,
    failedSignIns: 0,
    documents: seedDocuments(),
    jobs: [],
    conversations: [],
    gold: seedGold(),
    judgeLabels: seedJudgeLabels(),
    runs: seedRuns(),
    profiles: seedProfiles(),
    traces: seedTraces(),
    feedback: [],
    nextId: 1000,
    uploads: new Map(),
    health: { retrievalOnly: false },
    jobListeners: new Set(),
    timers: [],
    jobTickMs: 350,
    tokenDelayMs: 35,
  };
}

export let db: MockDb = fresh();

export function resetMockDb(overrides: Partial<MockDb> = {}): MockDb {
  for (const t of db.timers) clearInterval(t);
  db = Object.assign(fresh(), overrides);
  return db;
}

export const nextId = () => ++db.nextId;
