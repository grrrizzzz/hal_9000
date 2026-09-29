"""``data/knowledge.db`` schema and forward-only migrations (plan §12).

Each migration is a script applied once, atomically, and recorded in
``schema_version``. Migrations are never edited or removed after they merge:
a change is a new, higher-numbered entry. Opening a database whose version is
newer than this code knows is refused rather than guessed at.

The per-profile search tables (``fts_<profile_id>``, ``vec_<profile_id>``) are
not migrations: they are created when a profile is built (see
:mod:`hal9000.knowledge.storage.db`).
"""

from __future__ import annotations

__all__ = ["MIGRATIONS", "SCHEMA_VERSION", "TABLES"]

_V1 = """
CREATE TABLE files (
    sha256 TEXT PRIMARY KEY CHECK (length(sha256) = 64),
    original_name TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    collection TEXT NOT NULL DEFAULT '',
    mime_type TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK (bytes >= 0),
    version_of TEXT REFERENCES files(sha256) ON DELETE SET NULL,
    uploaded_at TEXT NOT NULL
);
CREATE INDEX ix_files_path ON files(relative_path);
CREATE INDEX ix_files_collection ON files(collection);

CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    file_sha TEXT NOT NULL REFERENCES files(sha256) ON DELETE CASCADE,
    page_start INTEGER NOT NULL CHECK (page_start >= 1),
    page_end INTEGER NOT NULL,
    title TEXT NOT NULL,
    doc_type TEXT NOT NULL DEFAULT 'other',
    vendor TEXT,
    doc_date TEXT,
    meta_source TEXT NOT NULL DEFAULT 'auto' CHECK (meta_source IN ('auto', 'user')),
    CHECK (page_end >= page_start)
);
CREATE INDEX ix_documents_file ON documents(file_sha, page_start);

CREATE TABLE pages (
    file_sha TEXT NOT NULL REFERENCES files(sha256) ON DELETE CASCADE,
    page INTEGER NOT NULL CHECK (page >= 1),
    parser TEXT NOT NULL,
    text TEXT NOT NULL,
    has_text_layer INTEGER NOT NULL CHECK (has_text_layer IN (0, 1)),
    offset_map TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (file_sha, parser, page)
);

CREATE TABLE profiles (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    parser TEXT NOT NULL,
    chunker TEXT NOT NULL
        CHECK (chunker IN ('fixed', 'recursive', 'structure', 'parent_child')),
    size INTEGER NOT NULL CHECK (size > 0),
    overlap INTEGER NOT NULL DEFAULT 0 CHECK (overlap >= 0),
    parent_size INTEGER,
    header INTEGER NOT NULL CHECK (header IN (0, 1)),
    embed_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    status TEXT NOT NULL DEFAULT 'building' CHECK (status IN ('building', 'ready', 'failed')),
    active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL
);
-- Exactly one profile serves the chat (plan §6.2): at most one row may be active.
CREATE UNIQUE INDEX ux_profiles_one_active ON profiles(active) WHERE active = 1;

CREATE TABLE abstain_thresholds (
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    reranker TEXT NOT NULL DEFAULT '',
    threshold REAL NOT NULL,
    auc REAL,
    eval_run_id TEXT,
    calibrated_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, reranker)
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    header TEXT NOT NULL DEFAULT '',
    tokens INTEGER NOT NULL,
    parent_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    CHECK (char_end >= char_start AND page_end >= page_start)
);
CREATE INDEX ix_chunks_profile_doc ON chunks(profile_id, document_id);
CREATE INDEX ix_chunks_parent ON chunks(parent_id);

CREATE TABLE embed_cache (
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    text_sha TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model, dimensions, text_sha)
);

CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('ingest', 'reindex', 'regression')),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'done', 'failed')),
    progress REAL NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    detail TEXT NOT NULL DEFAULT '',
    error TEXT,
    file_sha TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_jobs_status ON jobs(status, created_at);

CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_conversations_session ON conversations(session_id, updated_at);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]',
    trace_id TEXT,
    outcome TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX ix_messages_conversation ON messages(conversation_id, id);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment TEXT,
    add_to_gold INTEGER NOT NULL DEFAULT 0 CHECK (add_to_gold IN (0, 1)),
    created_at TEXT NOT NULL
);

-- One row per question (plan §9.1); kept 180 days. The scalar columns are what
-- Grafana's SQLite panels read; ``body`` is the full Trace as JSON.
CREATE TABLE traces (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'question',
    outcome TEXT,
    profile TEXT,
    total_ms REAL,
    ttft_ms REAL,
    cost_usd REAL NOT NULL DEFAULT 0,
    body TEXT NOT NULL
);
CREATE INDEX ix_traces_created ON traces(created_at);

CREATE TABLE gold_items (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    reference_answer TEXT,
    category TEXT NOT NULL CHECK (category IN
        ('factoid', 'money', 'date', 'list', 'cross_document', 'unanswerable')),
    difficulty TEXT NOT NULL DEFAULT 'medium',
    paraphrased INTEGER NOT NULL DEFAULT 0 CHECK (paraphrased IN (0, 1)),
    evidence TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'rejected')),
    split TEXT CHECK (split IN ('dev', 'test')),
    reviewer_note TEXT,
    origin TEXT NOT NULL DEFAULT 'generated' CHECK (origin IN ('generated', 'feedback')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX ix_gold_status ON gold_items(status, split);

CREATE TABLE judge_labels (
    id INTEGER PRIMARY KEY,
    gold_item_id INTEGER NOT NULL REFERENCES gold_items(id) ON DELETE CASCADE,
    eval_run_id TEXT,
    answer TEXT NOT NULL,
    human_labels TEXT,
    judge_labels TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE eval_runs (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    split TEXT NOT NULL DEFAULT 'dev' CHECK (split IN ('dev', 'test')),
    config TEXT NOT NULL DEFAULT '{}',
    metrics TEXT,
    host TEXT NOT NULL DEFAULT '',
    cost_usd REAL NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    results_path TEXT
);

CREATE TABLE regression_runs (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    profile_id INTEGER REFERENCES profiles(id) ON DELETE SET NULL,
    trigger TEXT NOT NULL CHECK (trigger IN ('ingest', 'manual')),
    recall_2k REAL,
    recall_low REAL,
    recall_high REAL,
    n_questions INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX ix_regression_at ON regression_runs(at);
"""

# KWP3: the persistent segmentation cache (LLM labels keyed by file SHA-256 and
# SEGMENT_VERSION, so a retried or rebuilt file never pays for labels twice), and
# the parameters a job needs to be retried (relative path, collection, --no-llm,
# the profile a reindex builds).
_V2 = """
CREATE TABLE segment_cache (
    key TEXT PRIMARY KEY,
    file_sha TEXT NOT NULL,
    documents TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX ix_segment_cache_file ON segment_cache(file_sha);

ALTER TABLE jobs ADD COLUMN params TEXT NOT NULL DEFAULT '{}';
"""

# KWP3 (decision D1): a file superseded by a newer version at the same relative
# path records which one replaced it. Its files, documents and pages rows stay (so
# re-uploading it restores it); only its chunks leave the indexes.
_V3 = """
ALTER TABLE files ADD COLUMN superseded_by TEXT;
CREATE INDEX ix_files_current ON files(relative_path, superseded_by);
"""

# KWP6 (approved contract additions): an answer's verify result on its message;
# feedback is one row per message whose rating may be left out when the answer
# is only added to the question set, linked to the draft gold item it created;
# the citations of a judge-calibration answer.
_V4 = """
ALTER TABLE messages ADD COLUMN grounding TEXT
    CHECK (grounding IS NULL OR grounding IN ('verified', 'failed', 'skipped'));
ALTER TABLE messages ADD COLUMN issues TEXT NOT NULL DEFAULT '[]';

CREATE TABLE feedback_v4 (
    id INTEGER PRIMARY KEY,
    message_id INTEGER NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating IN (-1, 1)),
    comment TEXT,
    add_to_gold INTEGER NOT NULL DEFAULT 0 CHECK (add_to_gold IN (0, 1)),
    gold_item_id INTEGER REFERENCES gold_items(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (rating IS NOT NULL OR add_to_gold = 1)
);
INSERT INTO feedback_v4 (id, message_id, rating, comment, add_to_gold, created_at, updated_at)
    SELECT id, message_id, rating, comment, add_to_gold, created_at, created_at FROM feedback
    WHERE id IN (SELECT MAX(id) FROM feedback GROUP BY message_id);
DROP TABLE feedback;
ALTER TABLE feedback_v4 RENAME TO feedback;

ALTER TABLE judge_labels ADD COLUMN citations TEXT NOT NULL DEFAULT '[]';
"""

#: (version, script). Append only.
MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _V1), (2, _V2), (3, _V3), (4, _V4))
SCHEMA_VERSION = MIGRATIONS[-1][0]

#: Every plan §12 table created by the migrations (the per-profile tables are separate,
#: and so is ``segment_cache``, an implementation detail of segmentation).
TABLES: tuple[str, ...] = (
    "files",
    "documents",
    "pages",
    "profiles",
    "abstain_thresholds",
    "chunks",
    "embed_cache",
    "jobs",
    "conversations",
    "messages",
    "feedback",
    "traces",
    "gold_items",
    "judge_labels",
    "eval_runs",
    "regression_runs",
)
