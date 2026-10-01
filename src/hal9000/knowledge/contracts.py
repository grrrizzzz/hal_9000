"""Pydantic contracts shared by every knowledge-base package (handoff KWP0).

These are the shapes that cross module boundaries and the API: ingestion
(``FileRecord`` → ``Page`` → ``DocumentMeta`` → ``Chunk``), retrieval
(``RetrievedChunk``, ``RetrievalResult``), answering (``Citation``,
``Answer``), observability (``Trace``, ``StageTiming``), background work
(``Job``) and evaluation (``EvidenceQuote``, ``GoldItem``, ``EvalRun``,
``EvalMetrics``). The API's request and response models are built from them,
so the exported OpenAPI schema is the contract with the UI.

Changing this module needs the orchestrator's approval (rule K4). Every model
round-trips through JSON (a test checks it).
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    field_validator,
    model_validator,
)

from hal9000.knowledge.obs.timing import Stage

__all__ = [
    "Answer",
    "ChatDoneEvent",
    "ChatEvent",
    "ChatEventEnvelope",
    "ChatRequest",
    "ChatResetEvent",
    "ChatSourcesEvent",
    "ChatStatusEvent",
    "ChatTokenEvent",
    "ChatVerifyEvent",
    "Chunk",
    "ChunkSpan",
    "ChunkerName",
    "Citation",
    "ConversationDetail",
    "ConversationSummary",
    "DocType",
    "DocumentDetail",
    "DocumentMeta",
    "DocumentPatch",
    "DocumentSummary",
    "ErrorResponse",
    "EvalMetrics",
    "EvalQuestionResult",
    "EvalResultsPage",
    "EvalRun",
    "EvidenceQuote",
    "FeedbackRequest",
    "FeedbackResponse",
    "FileRecord",
    "GoldCategory",
    "GoldItem",
    "GoldPatch",
    "Health",
    "IndexProfile",
    "Job",
    "JudgeLabel",
    "JudgeLabelUpdate",
    "JudgeReasons",
    "MessageOut",
    "MetricCI",
    "Outcome",
    "Page",
    "PageView",
    "ProfileActivate",
    "ProfileState",
    "RetrievalResult",
    "RetrievedChunk",
    "RubricLabels",
    "SearchFilters",
    "SessionRequest",
    "SessionResponse",
    "StageTiming",
    "Trace",
    "TraceHit",
    "TraceSummary",
    "UploadResponse",
    "UploadResult",
]

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PageNumber = Annotated[int, Field(ge=1)]

#: Sub-document types the segmenter may assign (plan §5.1 step 6).
DocType = Literal[
    "inspection_report",
    "estimate",
    "invoice",
    "pest_report",
    "completion_notice",
    "disclosure_form",
    "manual",
    "warranty",
    "permit",
    "other",
]
ChunkerName = Literal["fixed", "recursive", "structure", "parent_child"]
#: How a question ended (plan §9.2 ``kb_questions_total{outcome}``). ``cancelled``: the
#: client went away before the answer finished (KWP8 records it; not an error).
Outcome = Literal["answered", "abstained", "retrieval_only", "error", "cancelled"]
Grounding = Literal["verified", "failed", "skipped"]
GoldCategory = Literal["factoid", "money", "date", "list", "cross_document", "unanswerable"]


def _now() -> datetime:
    return datetime.now(UTC)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ ingestion
class FileRecord(_Model):
    """An uploaded original, stored at ``data/knowledge/files/<sha256>`` (plan §12 ``files``)."""

    sha256: Sha256
    original_name: str
    relative_path: str
    collection: str = ""
    mime_type: str
    bytes: int = Field(ge=0)
    version_of: Sha256 | None = None
    uploaded_at: datetime = Field(default_factory=_now)


class Page(_Model):
    """One page (or pseudo-page) of cleaned text (plan §12 ``pages``).

    ``offset_map`` holds ``(clean_offset, source_offset)`` pairs at each point
    where cleaning shifted the text, so a span in cleaned text maps back.
    """

    file_sha: Sha256
    page: PageNumber
    parser: str
    text: str
    has_text_layer: bool
    offset_map: list[tuple[int, int]] = Field(default_factory=list)


class DocumentMeta(_Model):
    """A logical sub-document inside a file (plan §5.1 step 6, §12 ``documents``)."""

    id: int | None = None
    file_sha: Sha256
    page_start: PageNumber
    page_end: PageNumber
    title: str
    doc_type: DocType = "other"
    vendor: str | None = None
    doc_date: date | None = None
    meta_source: Literal["auto", "user"] = "auto"

    @model_validator(mode="after")
    def _pages_ordered(self) -> DocumentMeta:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class IndexProfile(_Model):
    """One full indexing combination (plan §6.2)."""

    id: int | None = None
    name: str
    parser: str
    chunker: ChunkerName
    size: int = Field(gt=0)
    overlap: int = Field(default=0, ge=0)
    parent_size: int | None = Field(default=None, gt=0)
    header: bool = True
    embed_model: str
    dimensions: int = Field(gt=0)
    active: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> IndexProfile:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        if (self.chunker == "parent_child") != (self.parent_size is not None):
            raise ValueError("parent_size is required for parent_child and only for it")
        return self


class Chunk(_Model):
    """A retrievable unit (plan §6.1). ``char_start``/``char_end`` index cleaned document text."""

    id: int | None = None
    profile_id: int
    document_id: int
    page_start: PageNumber
    page_end: PageNumber
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    header: str = ""
    tokens: int = Field(ge=0)
    parent_id: int | None = None

    @model_validator(mode="after")
    def _spans_ordered(self) -> Chunk:
        if self.char_end < self.char_start or self.page_end < self.page_start:
            raise ValueError("chunk spans must not be reversed")
        return self


# ------------------------------------------------------------------ retrieval
class RetrievedChunk(_Model):
    """A chunk as retrieval ranked it. Scores are ``None`` where that retriever didn't see it."""

    chunk_id: int
    document_id: int
    text: str
    header: str = ""
    page_start: PageNumber
    page_end: PageNumber
    tokens: int = Field(ge=0)
    rank: int = Field(ge=1)
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    document: DocumentMeta | None = None
    parent_id: int | None = None
    # additive (KWP4): the document's metadata flattened for citations, and budget accounting
    file_sha: str = ""
    title: str = ""
    doc_type: DocType = "other"
    vendor: str | None = None
    doc_date: date | None = None
    collection: str = ""
    #: what this chunk costs in the context budget: header + text tokens when headers are
    #: sent, else ``tokens``
    prompt_tokens: int = Field(default=0, ge=0)
    #: the fusion score before any reranking (RRF or weighted); never the abstain score
    fused_score: float | None = None


class RetrievalResult(_Model):
    """Everything retrieval decided for one question (plan §7.1)."""

    query: str
    standalone_query: str
    profile: str
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    candidates: list[RetrievedChunk] = Field(default_factory=list)
    selected: list[RetrievedChunk] = Field(default_factory=list)
    context_tokens: int = Field(default=0, ge=0)
    #: dense cosine or reranker score of the top chunk -- never the RRF score (plan §7.1)
    abstain_score: float | None = None
    threshold: float | None = None
    abstained: bool = False
    #: set when retrieval ran in a degraded mode (plan §7.4): the query embedding failed,
    #: so the result is BM25-only and there was no retrieval abstain check
    degraded: Literal["embedding_unavailable"] | None = None


# ------------------------------------------------------------------ answering
class Citation(_Model):
    """A ``[S#]`` source in an answer; opens the viewer at ``page``."""

    marker: Annotated[str, StringConstraints(pattern=r"^S[1-9][0-9]*$")]
    chunk_id: int
    document_id: int
    file_sha: Sha256
    page: PageNumber
    title: str
    doc_type: DocType = "other"
    doc_date: date | None = None
    vendor: str | None = None
    snippet: str = ""


class Answer(_Model):
    """A finished answer, as sent in the chat stream's ``done`` event (plan §11)."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    outcome: Outcome
    grounding: Grounding = "skipped"
    issues: list[str] = Field(default_factory=list)
    trace_id: str
    message_id: int | None = None
    conversation_id: int | None = None
    cost_usd: float = Field(default=0.0, ge=0)


# -------------------------------------------------------------- observability
class StageTiming(_Model):
    """One timed stage. ``started_ms`` is relative to the trace start."""

    stage: Stage
    started_ms: float = Field(ge=0)
    duration_ms: float = Field(ge=0)
    depth: int = Field(default=0, ge=0)
    parent: Stage | None = None
    error: str | None = None


class TraceHit(_Model):
    """A retrieved chunk as recorded in a trace: IDs, scores and ranks, no text."""

    chunk_id: int
    rank: int = Field(ge=1)
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    selected: bool = False


class Trace(_Model):
    """One question's trace row (plan §9.1)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=_now)
    kind: Literal["question", "ingest"] = "question"
    question: str = ""
    standalone_query: str | None = None
    profile: str | None = None
    stages: list[StageTiming] = Field(default_factory=list)
    hits: list[TraceHit] = Field(default_factory=list)
    context_tokens: int = Field(default=0, ge=0)
    #: model -> {"prompt": n, "cached": n, "completion": n}
    tokens_by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
    cost_usd: float = Field(default=0.0, ge=0)
    outcome: Outcome | None = None
    grounding: Grounding | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None

    _t0: float = PrivateAttr(default=0.0)


class Job(_Model):
    """A background job (plan §5.1; ``jobs`` table)."""

    id: int | None = None
    kind: Literal["ingest", "reindex", "regression"]
    status: Literal["queued", "running", "done", "failed"] = "queued"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    detail: str = ""
    error: str | None = None
    file_sha: Sha256 | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ----------------------------------------------------------------- evaluation
class EvidenceQuote(_Model):
    """A verbatim quote from one page of cleaned text (plan §8.1)."""

    quote: str = Field(min_length=1)
    page: PageNumber
    file_sha: Sha256 | None = None

    @field_validator("quote")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("an evidence quote must contain text")
        return value


class GoldItem(_Model):
    """One question-set item (plan §8.1; ``gold_items``)."""

    id: int | None = None
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    category: GoldCategory
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    paraphrased: bool = False
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    status: Literal["draft", "approved", "rejected"] = "draft"
    split: Literal["dev", "test"] | None = None
    reviewer_note: str | None = None
    origin: Literal["generated", "feedback"] = "generated"

    @model_validator(mode="after")
    def _evidence_matches_category(self) -> GoldItem:
        if self.category == "unanswerable" and self.evidence:
            raise ValueError("an unanswerable item carries no evidence")
        if self.category != "unanswerable" and not self.evidence:
            raise ValueError("an answerable item needs at least one evidence quote")
        return self


class MetricCI(_Model):
    """A point estimate with its bootstrap 95% interval (plan §8.2)."""

    value: float
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> MetricCI:
        if not self.low <= self.value <= self.high:
            raise ValueError("expected low <= value <= high")
        return self


class EvalMetrics(_Model):
    """Aggregate retrieval and answer metrics for one run (plan §8.2, §8.4.1).

    Budgeted metrics are keyed by the token budget (1000, 2000, 4000) or by *k*;
    a missing key means N/A (e.g. a chunk size larger than the budget).
    """

    n_questions: int = Field(ge=0)
    budgeted_recall: dict[int, MetricCI] = Field(default_factory=dict)
    full_support_hit: dict[int, MetricCI] = Field(default_factory=dict)
    mrr: MetricCI | None = None
    context_precision: dict[int, MetricCI] = Field(default_factory=dict)
    abstain_auc: MetricCI | None = None
    correct: MetricCI | None = None
    faithful: MetricCI | None = None
    abstention_correct: MetricCI | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    cost_per_question_usd: float | None = None
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)


class EvalRun(_Model):
    """One sweep stage's run (``eval_runs``; files under ``data/knowledge/eval/runs/<id>/``)."""

    id: str
    stage: str
    split: Literal["dev", "test"] = "dev"
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: EvalMetrics | None = None
    host: str = ""
    cost_usd: float = Field(default=0.0, ge=0)
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    results_path: str | None = None


# ================================================================ API shapes
# Request and response bodies of the plan §11 API. They are built from the
# domain models above; the exported OpenAPI (web/src/api/openapi.json) is
# generated from them, so this section is the UI's contract too.


class SessionRequest(_Model):
    passcode: str = Field(min_length=1, max_length=256)


class SessionResponse(_Model):
    ok: bool
    expires_at: datetime | None = None


class UploadResult(_Model):
    """What happened to one uploaded file (plan §5.1 steps 1–3)."""

    relative_path: str
    status: Literal["added", "duplicate", "skipped", "failed"]
    sha256: Sha256 | None = None
    collection: str = ""
    job_id: int | None = None
    reason: str | None = None


class UploadResponse(_Model):
    results: list[UploadResult]


class DocumentSummary(_Model):
    """A Library row (plan §10): a sub-document with its file and index state."""

    document: DocumentMeta
    original_name: str
    relative_path: str
    collection: str = ""
    pages: int = Field(ge=0)
    pages_without_text: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    status: Literal["queued", "indexing", "indexed", "failed"] = "indexed"


class DocumentDetail(DocumentSummary):
    file: FileRecord
    siblings: list[DocumentMeta] = Field(default_factory=list)


class DocumentPatch(_Model):
    """Inline metadata edits; a user edit always wins over segmentation (plan §5.1)."""

    title: str | None = None
    doc_type: DocType | None = None
    vendor: str | None = None
    doc_date: date | None = None
    page_start: PageNumber | None = None
    page_end: PageNumber | None = None


class EvalQuestionResult(_Model):
    """One question's result in an eval run: ``GET /api/eval/runs/{id}/results`` items.

    ``recall`` is keyed by the token budget as a string (``"2000"``); a missing
    key means N/A (the chunk size exceeds the budget, or the item is
    unanswerable).
    """

    gold_item_id: int
    question: str
    category: GoldCategory
    recall: dict[str, float]
    #: every evidence quote >= 80% covered within the top k
    full_support: bool
    #: rank of the first chunk covering any evidence; None if none did
    first_hit_rank: int | None
    abstain_score: float | None
    abstained: bool


class EvalResultsPage(_Model):
    """A page of an eval run's per-question results."""

    items: list[EvalQuestionResult]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)


class ChunkSpan(_Model):
    chunk_id: int
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class PageView(_Model):
    """Cleaned page (or pseudo-page) text and the active profile's chunk spans on it."""

    file_sha: Sha256
    page: PageNumber
    page_count: int = Field(ge=1)
    text: str
    has_text_layer: bool
    spans: list[ChunkSpan] = Field(default_factory=list)


class SearchFilters(_Model):
    doc_type: DocType | None = None
    collection: str | None = None
    date_from: date | None = None
    date_to: date | None = None


class ChatRequest(_Model):
    conversation_id: int | None = None
    message: str = Field(min_length=1, max_length=4000)
    filters: SearchFilters | None = None


class ChatStatusEvent(_Model):
    """SSE ``status``: the pipeline stage now running."""

    event: Literal["status"] = "status"
    stage: Stage
    detail: str = ""


class ChatSourcesEvent(_Model):
    """SSE ``sources``: the numbered sources the answer may cite."""

    event: Literal["sources"] = "sources"
    sources: list[Citation]
    retrieval_only: bool = False
    banner: str | None = None


class ChatTokenEvent(_Model):
    event: Literal["token"] = "token"
    text: str


class ChatResetEvent(_Model):
    """SSE ``reset``: clear the partial answer; a regeneration follows (plan §7.1 verify)."""

    event: Literal["reset"] = "reset"
    reason: str


class ChatVerifyEvent(_Model):
    event: Literal["verify"] = "verify"
    grounding: Grounding
    issues: list[str] = Field(default_factory=list)


class ChatDoneEvent(_Model):
    event: Literal["done"] = "done"
    answer: Answer


ChatEvent = Annotated[
    ChatStatusEvent
    | ChatSourcesEvent
    | ChatTokenEvent
    | ChatResetEvent
    | ChatVerifyEvent
    | ChatDoneEvent,
    Field(discriminator="event"),
]


class ChatEventEnvelope(_Model):
    """Documents the ``POST /api/chat`` stream: each SSE ``data:`` line is one ``event``."""

    event: ChatEvent


class MessageOut(_Model):
    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str | None = None
    outcome: Outcome | None = None
    #: the verify result of an assistant answer (plan §7.1); None for user messages
    grounding: Grounding | None = None
    issues: list[str] = Field(default_factory=list)
    created_at: datetime


class ConversationSummary(_Model):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut] = Field(default_factory=list)


class FeedbackRequest(_Model):
    """Feedback on one answer. One row per message: a later POST updates it.

    ``rating`` may be left out only to add the answer to the question set.
    """

    message_id: int
    rating: Literal[-1, 1] | None = None
    comment: str | None = None
    add_to_gold: bool = False

    @model_validator(mode="after")
    def _rating_or_gold(self) -> FeedbackRequest:
        if self.rating is None and not self.add_to_gold:
            raise ValueError("a rating is required unless add_to_gold is set")
        return self


class FeedbackResponse(_Model):
    id: int
    gold_item_id: int | None = None


class GoldPatch(_Model):
    """A review decision or edit on the Evaluation page (plan §8.1, gate G-K1)."""

    question: str | None = None
    reference_answer: str | None = None
    category: GoldCategory | None = None
    evidence: list[EvidenceQuote] | None = None
    status: Literal["draft", "approved", "rejected"] | None = None
    reviewer_note: str | None = None


class RubricLabels(_Model):
    """The judge rubric (plan §8.4.1), as labelled by Mike or by the judge."""

    correctness: Literal["correct", "partial", "wrong"]
    faithful: bool
    citations_correct: bool
    abstention_correct: bool | None = None


class JudgeReasons(_Model):
    """The judge's one-sentence reason for each rubric label (shown once Mike has labelled)."""

    correctness: str = ""
    faithful: str = ""
    citations_correct: str = ""
    abstention_correct: str = ""


class JudgeLabel(_Model):
    id: int
    gold_item_id: int
    eval_run_id: str | None = None
    question: str
    answer: str
    #: the sources the answer cited (the viewer opens them)
    citations: list[Citation] = Field(default_factory=list)
    human: RubricLabels | None = None
    judge: RubricLabels | None = None
    #: why the judge chose each label (None for labels judged before reasons were recorded)
    judge_reasons: JudgeReasons | None = None


class JudgeLabelUpdate(_Model):
    human: RubricLabels


class ProfileState(_Model):
    active: IndexProfile | None = None
    profiles: list[IndexProfile] = Field(default_factory=list)
    reindex_job: Job | None = None


class ProfileActivate(_Model):
    name: str = Field(min_length=1)


class TraceSummary(_Model):
    """A Traces-page row (the in-app list, behind the passcode).

    Grafana reads the ``traces`` table's scalar columns, never this model, so
    the question text here doesn't reach it.
    """

    id: str
    created_at: datetime
    question: str | None = None
    outcome: Outcome | None = None
    total_ms: float | None = None
    ttft_ms: float | None = None
    cost_usd: float = 0.0


class Health(_Model):
    status: Literal["ok", "degraded"]
    version: str
    active_profile: str | None = None
    month_spend_usd: float | None = None
    month_cap_usd: float | None = None
    retrieval_only: bool = False


class ErrorResponse(_Model):
    detail: str
