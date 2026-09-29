"""FastAPI app factory with every plan §11 route as a ``501`` stub (KWP0).

Operation IDs are the handler names (``list_documents``), so the generated
TypeScript client has stable, readable names.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, NoReturn

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.routing import APIRoute

from hal9000 import __version__
from hal9000.knowledge.contracts import (
    ChatEventEnvelope,
    ChatRequest,
    ConversationDetail,
    ConversationSummary,
    DocType,
    DocumentDetail,
    DocumentMeta,
    DocumentPatch,
    DocumentSummary,
    ErrorResponse,
    EvalRun,
    FeedbackRequest,
    FeedbackResponse,
    GoldItem,
    GoldPatch,
    Health,
    Job,
    JudgeLabel,
    JudgeLabelUpdate,
    PageView,
    ProfileActivate,
    ProfileState,
    RetrievalResult,
    SessionRequest,
    SessionResponse,
    Trace,
    TraceSummary,
    UploadResponse,
)

__all__ = ["ROUTES", "create_app"]

#: Every plan §11 route as (method, path). A test checks the app serves exactly these.
ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/api/session"),
    ("DELETE", "/api/session"),
    ("POST", "/api/uploads"),
    ("GET", "/api/jobs"),
    ("GET", "/api/jobs/stream"),
    ("GET", "/api/documents"),
    ("GET", "/api/documents/{id}"),
    ("PATCH", "/api/documents/{id}"),
    ("DELETE", "/api/documents/{id}"),
    ("POST", "/api/documents/{id}/reindex"),
    ("GET", "/api/files/{sha}"),
    ("GET", "/api/files/{sha}/pages/{n}"),
    ("POST", "/api/chat"),
    ("GET", "/api/conversations"),
    ("GET", "/api/conversations/{id}"),
    ("DELETE", "/api/conversations"),
    ("DELETE", "/api/conversations/{id}"),
    ("POST", "/api/feedback"),
    ("GET", "/api/search"),
    ("GET", "/api/eval/gold"),
    ("GET", "/api/eval/gold/{id}"),
    ("PATCH", "/api/eval/gold/{id}"),
    ("GET", "/api/eval/judge-labels"),
    ("GET", "/api/eval/judge-labels/{id}"),
    ("PUT", "/api/eval/judge-labels/{id}"),
    ("GET", "/api/eval/runs"),
    ("GET", "/api/eval/runs/{id}"),
    ("GET", "/api/profile"),
    ("PUT", "/api/profile"),
    ("GET", "/api/traces"),
    ("GET", "/api/traces/{id}"),
    ("GET", "/health"),
    ("GET", "/metrics"),
)

_ERRORS = {
    401: {"model": ErrorResponse, "description": "Not signed in"},
    501: {"model": ErrorResponse, "description": "Not implemented yet (KWP6)"},
}
_SSE = {"content": {"text/event-stream": {"schema": {"type": "string"}}}}
Sha = Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]


def _stub() -> NoReturn:
    raise HTTPException(status_code=501, detail="not implemented yet (KWP6)")


def _operation_id(route: APIRoute) -> str:
    return route.name


def _api_router() -> APIRouter:
    r = APIRouter(prefix="/api", responses=_ERRORS)

    # -- session (plan §14) ------------------------------------------------------
    @r.post("/session", response_model=SessionResponse, tags=["session"])
    def create_session(body: SessionRequest) -> SessionResponse:
        """Exchange the household passcode for a signed session cookie."""
        _stub()

    @r.delete("/session", status_code=204, tags=["session"])
    def delete_session() -> None:
        """Sign out."""
        _stub()

    # -- uploads and jobs (plan §5) ---------------------------------------------
    @r.post("/uploads", response_model=UploadResponse, tags=["library"])
    def create_uploads(
        files: Annotated[list[UploadFile], File(description="the files")],
        relative_path: Annotated[
            list[str], Form(description="each file's path within its chosen folder, in order")
        ],
    ) -> UploadResponse:
        """Upload many files; folder paths become collections. Returns job IDs."""
        _stub()

    @r.get("/jobs", response_model=list[Job], tags=["library"])
    def list_jobs() -> list[Job]:
        _stub()

    @r.get(
        "/jobs/stream",
        tags=["library"],
        responses={200: {**_SSE, "description": "SSE: one `job` event (a Job) per change"}},
    )
    def stream_jobs() -> None:
        """Live job progress (server-sent events)."""
        _stub()

    # -- library -----------------------------------------------------------------
    @r.get("/documents", response_model=list[DocumentSummary], tags=["library"])
    def list_documents(
        collection: str | None = None, doc_type: DocType | None = None
    ) -> list[DocumentSummary]:
        _stub()

    @r.get("/documents/{id}", response_model=DocumentDetail, tags=["library"])
    def get_document(id: int) -> DocumentDetail:
        _stub()

    @r.patch("/documents/{id}", response_model=DocumentMeta, tags=["library"])
    def update_document(id: int, body: DocumentPatch) -> DocumentMeta:
        """Edit metadata; user edits always win over segmentation."""
        _stub()

    @r.delete("/documents/{id}", status_code=204, tags=["library"])
    def delete_document(id: int) -> None:
        _stub()

    @r.post("/documents/{id}/reindex", response_model=Job, tags=["library"])
    def reindex_document(id: int) -> Job:
        _stub()

    @r.get(
        "/files/{sha}",
        tags=["library"],
        responses={
            200: {
                "description": "The original. PDFs inline with range support; anything else "
                "as a sandboxed attachment.",
                "content": {
                    "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            206: {"description": "Partial content (HTTP range request)"},
        },
    )
    def get_file(sha: Sha) -> None:
        _stub()

    @r.get("/files/{sha}/pages/{n}", response_model=PageView, tags=["library"])
    def get_page(sha: Sha, n: Annotated[int, Path(ge=1)]) -> PageView:
        """Cleaned page (or pseudo-page) text and the chunk spans on it."""
        _stub()

    # -- chat (plan §7) ----------------------------------------------------------
    @r.post(
        "/chat",
        tags=["chat"],
        responses={
            200: {
                "model": ChatEventEnvelope,
                "description": "SSE over the POST response. Events: status, sources, token, "
                "reset, verify, done; each `data:` line is the matching ChatEventEnvelope.event.",
            }
        },
    )
    def chat(body: ChatRequest) -> None:
        """Ask a question. Read the stream with fetch + ReadableStream, not EventSource."""
        _stub()

    @r.get("/conversations", response_model=list[ConversationSummary], tags=["chat"])
    def list_conversations() -> list[ConversationSummary]:
        _stub()

    @r.get("/conversations/{id}", response_model=ConversationDetail, tags=["chat"])
    def get_conversation(id: int) -> ConversationDetail:
        _stub()

    @r.delete("/conversations", status_code=204, tags=["chat"])
    def delete_conversations() -> None:
        """Delete this session's conversation history."""
        _stub()

    @r.delete("/conversations/{id}", status_code=204, tags=["chat"])
    def delete_conversation(id: int) -> None:
        _stub()

    @r.post("/feedback", response_model=FeedbackResponse, tags=["chat"])
    def create_feedback(body: FeedbackRequest) -> FeedbackResponse:
        """Thumbs up/down, a comment, and optionally "Add to question set"."""
        _stub()

    @r.get("/search", response_model=RetrievalResult, tags=["chat"])
    def search(
        q: Annotated[str, Query(min_length=1)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        doc_type: DocType | None = None,
        collection: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> RetrievalResult:
        """Retrieval only, with scores (debugging and retrieval-only mode)."""
        _stub()

    # -- evaluation (plan §8) ----------------------------------------------------
    @r.get("/eval/gold", response_model=list[GoldItem], tags=["eval"])
    def list_gold(
        status: str | None = None, split: str | None = None, category: str | None = None
    ) -> list[GoldItem]:
        _stub()

    @r.get("/eval/gold/{id}", response_model=GoldItem, tags=["eval"])
    def get_gold(id: int) -> GoldItem:
        _stub()

    @r.patch("/eval/gold/{id}", response_model=GoldItem, tags=["eval"])
    def update_gold(id: int, body: GoldPatch) -> GoldItem:
        """Approve, edit or reject a question-set item."""
        _stub()

    @r.get("/eval/judge-labels", response_model=list[JudgeLabel], tags=["eval"])
    def list_judge_labels(run: str | None = None) -> list[JudgeLabel]:
        _stub()

    @r.get("/eval/judge-labels/{id}", response_model=JudgeLabel, tags=["eval"])
    def get_judge_label(id: int) -> JudgeLabel:
        _stub()

    @r.put("/eval/judge-labels/{id}", response_model=JudgeLabel, tags=["eval"])
    def put_judge_label(id: int, body: JudgeLabelUpdate) -> JudgeLabel:
        """Mike's rubric labels for one generated answer (gate G-K1b)."""
        _stub()

    @r.get("/eval/runs", response_model=list[EvalRun], tags=["eval"])
    def list_eval_runs(stage: str | None = None) -> list[EvalRun]:
        _stub()

    @r.get("/eval/runs/{id}", response_model=EvalRun, tags=["eval"])
    def get_eval_run(id: str) -> EvalRun:
        _stub()

    # -- profile and traces --------------------------------------------------------
    @r.get("/profile", response_model=ProfileState, tags=["profile"])
    def get_profile() -> ProfileState:
        _stub()

    @r.put("/profile", response_model=ProfileState, tags=["profile"])
    def put_profile(body: ProfileActivate) -> ProfileState:
        """Switch the active profile; re-indexes in the background."""
        _stub()

    @r.get("/traces", response_model=list[TraceSummary], tags=["traces"])
    def list_traces(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[TraceSummary]:
        _stub()

    @r.get("/traces/{id}", response_model=Trace, tags=["traces"])
    def get_trace(id: str) -> Trace:
        """The Query Inspector's data for one question."""
        _stub()

    return r


def _ops_router() -> APIRouter:
    r = APIRouter(responses={501: _ERRORS[501]})

    @r.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        """Unauthenticated (LAN only); never exposes question text."""
        _stub()

    @r.get(
        "/metrics",
        tags=["ops"],
        responses={
            200: {
                "description": "Prometheus exposition format",
                "content": {"text/plain": {"schema": {"type": "string"}}},
            }
        },
    )
    def metrics() -> None:
        _stub()

    return r


def create_app() -> FastAPI:
    """Build the app. KWP6 adds auth, the pipeline and static UI serving."""
    app = FastAPI(
        title="HAL 9000 house knowledge",
        version=__version__,
        description="The contract between the knowledge-base backend and its web UI "
        "(docs/KNOWLEDGE_PLAN.md §11).",
        generate_unique_id_function=_operation_id,
    )
    app.include_router(_api_router())
    app.include_router(_ops_router())
    _document_chat_stream(app)
    return app


def _document_chat_stream(app: FastAPI) -> None:
    """Describe ``POST /api/chat`` as an event stream of :class:`ChatEventEnvelope` only.

    FastAPI documents a response ``model`` as ``application/json``; the chat
    answers only ``text/event-stream``, so the generated schema is rewritten once.
    """
    build = app.openapi

    def openapi() -> dict:
        if app.openapi_schema is None:
            schema = build()
            ok = schema["paths"]["/api/chat"]["post"]["responses"]["200"]
            ok["content"] = {
                "text/event-stream": {"schema": {"$ref": "#/components/schemas/ChatEventEnvelope"}}
            }
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
