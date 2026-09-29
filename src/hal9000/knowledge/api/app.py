"""The knowledge-base HTTP API (plan §11) and the web UI's static files.

``create_app(services)`` serves every plan §11 route (and the approved
additions: ``GET /api/eval/runs/{id}/results``, ``POST /api/jobs/{id}/retry``)
over :class:`~hal9000.knowledge.api.services.Services`. ``create_app()``
without services builds the same routes for the OpenAPI export; they answer
``503`` until services are attached.

- **Auth** (plan §14): every ``/api`` route except ``POST /api/session``
  needs the signed session cookie (``401`` otherwise). ``/health`` and
  ``/metrics`` are unauthenticated (LAN only) and carry no question text.
- **Streams**: ``POST /api/chat`` and ``GET /api/jobs/stream`` are server-sent
  events (sse-starlette). Each question runs on its own thread inside a copy
  of the request's context (the trace and usage sink are context variables);
  when the client goes away the thread is told to cancel, which closes the
  upstream stream and records the estimated usage (outcome ``cancelled``).
- **Files**: PDFs inline with HTTP ``Range``; anything else only as a
  sandboxed download (``attachment``, ``nosniff``, ``CSP: sandbox``).
- **The UI**: ``web/dist`` is served at ``/`` with a history-API fallback to
  ``index.html`` for every other non-``/api`` GET.

Operation IDs are the handler names (``list_documents``), so the generated
TypeScript client has stable, readable names.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sqlite3
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi import (
    Path as PathParam,
)
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

from hal9000 import __version__
from hal9000.knowledge.auth import COOKIE, Session
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
    EvalResultsPage,
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
    SearchFilters,
    SessionRequest,
    SessionResponse,
    Trace,
    TraceSummary,
    UploadResponse,
    UploadResult,
)

__all__ = ["ROUTES", "WEB_DIST", "create_app"]

log = logging.getLogger(__name__)

#: ``web/dist`` in this checkout (``npm run build``).
WEB_DIST = Path(__file__).resolve().parents[4] / "web" / "dist"

#: Every served API route as (method, path). A test checks the app serves exactly these.
ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/api/session"),
    ("DELETE", "/api/session"),
    ("POST", "/api/uploads"),
    ("GET", "/api/jobs"),
    ("GET", "/api/jobs/stream"),
    ("POST", "/api/jobs/{id}/retry"),
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
    ("GET", "/api/eval/runs/{id}/results"),
    ("GET", "/api/profile"),
    ("PUT", "/api/profile"),
    ("GET", "/api/traces"),
    ("GET", "/api/traces/{id}"),
    ("GET", "/health"),
    ("GET", "/metrics"),
)

_ERRORS = {
    401: {"model": ErrorResponse, "description": "Not signed in"},
}
_SSE = {"content": {"text/event-stream": {"schema": {"type": "string"}}}}
Sha = Annotated[str, PathParam(pattern=r"^[0-9a-f]{64}$")]
_END = object()


# ------------------------------------------------------------ dependencies
def services_of(request: Request):
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="the knowledge service is not configured")
    return services


def require_session(request: Request) -> Session:
    """The signed-in session, or ``401``."""
    services = services_of(request)
    session = services.auth.session(request.cookies.get(COOKIE))
    if session is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return session


def database(request: Request) -> Iterator[sqlite3.Connection]:
    conn = services_of(request).connect()
    try:
        yield conn
    finally:
        conn.close()


Services = Annotated[Any, Depends(services_of)]
Db = Annotated[sqlite3.Connection, Depends(database)]
Signed = Annotated[Session, Depends(require_session)]


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} not found")


def _operation_id(route: APIRoute) -> str:
    return route.name


def _put(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, item: object) -> None:
    try:
        loop.call_soon_threadsafe(queue.put_nowait, item)
    except RuntimeError:  # the loop closed: nobody is listening any more
        pass


# ------------------------------------------------------------------ routers
def _session_router() -> APIRouter:
    r = APIRouter(prefix="/api", responses=_ERRORS)

    @r.post(
        "/session",
        response_model=SessionResponse,
        tags=["session"],
        responses={429: {"model": ErrorResponse, "description": "Too many failed attempts"}},
    )
    def create_session(
        body: SessionRequest, request: Request, response: Response, services: Services
    ) -> SessionResponse:
        """Exchange the household passcode for a signed session cookie (30 days)."""
        auth = services.auth
        client = request.client.host if request.client else "unknown"
        if auth.sign_ins.blocked(client):
            raise HTTPException(
                status_code=429,
                detail="Too many attempts. Wait a minute and try again.",
                headers={"Retry-After": str(auth.sign_ins.retry_after(client))},
            )
        if not auth.check_passcode(body.passcode):
            auth.sign_ins.hit(client)
            raise HTTPException(status_code=401, detail="That passcode isn't right.")
        token, _session, expires = auth.issue()
        response.set_cookie(
            COOKIE,
            token,
            max_age=auth.max_age_s,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )
        return SessionResponse(ok=True, expires_at=expires)

    return r


def _api_router() -> APIRouter:
    r = APIRouter(prefix="/api", responses=_ERRORS, dependencies=[Depends(require_session)])

    @r.delete("/session", status_code=204, tags=["session"])
    def delete_session(response: Response) -> None:
        """Sign out."""
        response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict")

    # -- uploads and jobs (plan §5) ---------------------------------------------
    @r.post("/uploads", response_model=UploadResponse, tags=["library"])
    def create_uploads(
        files: Annotated[list[UploadFile], File(description="the files")],
        relative_path: Annotated[
            list[str], Form(description="each file's path within its chosen folder, in order")
        ],
        services: Services,
        conn: Db,
    ) -> UploadResponse:
        """Upload many files; folder paths become collections. Returns job IDs."""
        if services.ingest is None:
            raise HTTPException(status_code=503, detail="ingestion is not available")
        if len(files) != len(relative_path):
            raise HTTPException(
                status_code=422, detail="send one relative_path for every file, in order"
            )
        from hal9000.knowledge.api.library import display_collection

        results: list[UploadResult] = []
        spool = services.settings.data_dir / "knowledge" / "tmp"
        spool.mkdir(parents=True, exist_ok=True)
        for upload, rel in zip(files, relative_path):
            name = rel or upload.filename or "upload"
            with tempfile.NamedTemporaryFile(dir=spool, delete=False) as tmp:
                while chunk := upload.file.read(1 << 20):
                    tmp.write(chunk)
                path = Path(tmp.name)
            try:
                result = services.ingest.submit(conn, path, name)
            finally:
                path.unlink(missing_ok=True)
            results.append(
                result.model_copy(update={"collection": display_collection(result.collection)})
            )
        services.notify_worker()
        return UploadResponse(results=results)

    @r.get("/jobs", response_model=list[Job], tags=["library"])
    def list_jobs(conn: Db) -> list[Job]:
        from hal9000.knowledge.ingest.jobs import list_jobs as jobs

        return jobs(conn, limit=200)

    @r.get(
        "/jobs/stream",
        tags=["library"],
        responses={200: {**_SSE, "description": "SSE: one `job` event (a Job) per change"}},
    )
    async def stream_jobs(services: Services) -> Response:
        """Live job progress (server-sent events): the recent jobs, then every change."""
        from sse_starlette import EventSourceResponse
        from starlette.concurrency import run_in_threadpool

        from hal9000.knowledge.ingest.jobs import list_jobs as jobs

        bus = services.bus
        inbox, unsubscribe = bus.asyncio_queue() if bus is not None else (asyncio.Queue(), None)

        def snapshot() -> list[Job]:
            conn = services.connect()
            try:
                return list(reversed(jobs(conn, limit=200)))
            finally:
                conn.close()

        recent = await run_in_threadpool(snapshot)

        async def events() -> AsyncIterator[dict]:
            try:
                for job in recent:
                    yield {"event": "job", "data": job.model_dump_json()}
                while True:
                    job = await inbox.get()
                    yield {"event": "job", "data": job.model_dump_json()}
            finally:
                if unsubscribe is not None:
                    unsubscribe()

        return EventSourceResponse(events(), ping=15)

    @r.post(
        "/jobs/{id}/retry",
        response_model=Job,
        tags=["library"],
        responses={
            409: {"model": ErrorResponse, "description": "Only a failed job can be retried"}
        },
    )
    def retry_job(id: int, services: Services, conn: Db) -> Job:
        """Queue a failed job again with its original parameters."""
        from hal9000.knowledge.ingest.jobs import JobNotFound, JobNotRetryable
        from hal9000.knowledge.ingest.jobs import retry_job as retry

        try:
            return retry(conn, id, bus=services.bus, worker=services.worker)
        except JobNotFound:
            raise _not_found("Job") from None
        except JobNotRetryable as error:
            raise HTTPException(status_code=409, detail=str(error)) from None

    # -- library -----------------------------------------------------------------
    @r.get("/documents", response_model=list[DocumentSummary], tags=["library"])
    def list_documents(
        conn: Db, collection: str | None = None, doc_type: DocType | None = None
    ) -> list[DocumentSummary]:
        from hal9000.knowledge.api.library import list_documents as documents

        try:
            return documents(conn, collection=collection, doc_type=doc_type)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    @r.get("/documents/{id}", response_model=DocumentDetail, tags=["library"])
    def get_document(id: int, conn: Db) -> DocumentDetail:
        from hal9000.knowledge.api.library import document_detail

        detail = document_detail(conn, id)
        if detail is None:
            raise _not_found("Document")
        return detail

    @r.patch("/documents/{id}", response_model=DocumentMeta, tags=["library"])
    def update_document(id: int, body: DocumentPatch, services: Services, conn: Db) -> DocumentMeta:
        """Edit metadata; user edits always win over segmentation (the file is re-indexed)."""
        from hal9000.knowledge.api.library import EditError, patch_document

        try:
            doc = patch_document(conn, id, body)
        except EditError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        if doc is None:
            raise _not_found("Document")
        _reindex_file(services, conn, doc.file_sha)
        return doc

    @r.delete("/documents/{id}", status_code=204, tags=["library"])
    def delete_document(id: int, services: Services, conn: Db) -> None:
        from hal9000.knowledge.api.library import delete_document as delete

        if not delete(conn, id, services.settings.files_dir):
            raise _not_found("Document")

    @r.post("/documents/{id}/reindex", response_model=Job, tags=["library"])
    def reindex_document(id: int, services: Services, conn: Db) -> Job:
        """Re-run ingestion for the document's file (embeddings and labels are cached)."""
        row = conn.execute("SELECT file_sha FROM documents WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise _not_found("Document")
        return _reindex_file(services, conn, row[0])

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
    def get_file(sha: Sha, services: Services, conn: Db) -> Response:
        from hal9000.knowledge.api.library import file_record

        record = file_record(conn, sha)
        stored = services.settings.files_dir / sha
        if record is None or not stored.is_file():
            raise _not_found("File")
        headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"}
        if record.mime_type == "application/pdf":
            return FileResponse(
                stored,
                media_type="application/pdf",
                headers=headers,
                filename=_safe_name(record.original_name),
                content_disposition_type="inline",
            )
        headers["Content-Security-Policy"] = "sandbox"
        return FileResponse(
            stored,
            media_type="application/octet-stream",
            headers=headers,
            filename=_safe_name(record.original_name),
            content_disposition_type="attachment",
        )

    @r.get("/files/{sha}/pages/{n}", response_model=PageView, tags=["library"])
    def get_page(sha: Sha, n: Annotated[int, PathParam(ge=1)], conn: Db) -> PageView:
        """Cleaned page (or pseudo-page) text and the chunk spans on it."""
        from hal9000.knowledge.api.library import page_view

        view = page_view(conn, sha, n)
        if view is None:
            raise _not_found("Page")
        return view

    # -- chat (plan §7) ----------------------------------------------------------
    @r.post(
        "/chat",
        tags=["chat"],
        responses={
            200: {
                "model": ChatEventEnvelope,
                "description": "SSE over the POST response. Events: status, sources, token, "
                "reset, verify, done; each `data:` line is the matching ChatEventEnvelope.event.",
            },
            409: {"model": ErrorResponse, "description": "No index profile is active yet"},
            429: {"model": ErrorResponse, "description": "More than 20 questions a minute"},
        },
    )
    async def chat(body: ChatRequest, session: Signed, services: Services) -> Response:
        """Ask a question. Read the stream with fetch + ReadableStream, not EventSource."""
        return await _chat(body, session, services)

    @r.get("/conversations", response_model=list[ConversationSummary], tags=["chat"])
    def list_conversations(session: Signed, conn: Db) -> list[ConversationSummary]:
        from hal9000.knowledge.answer.store import list_conversations as conversations

        return conversations(conn, session.id)

    @r.get("/conversations/{id}", response_model=ConversationDetail, tags=["chat"])
    def get_conversation(id: int, session: Signed, conn: Db) -> ConversationDetail:
        from hal9000.knowledge.answer.store import get_conversation as conversation

        found = conversation(conn, session.id, id)
        if found is None:
            raise _not_found("Conversation")
        return found

    @r.delete("/conversations", status_code=204, tags=["chat"])
    def delete_conversations(session: Signed, conn: Db) -> None:
        """Delete this session's conversation history."""
        from hal9000.knowledge.answer.store import delete_conversations as delete

        delete(conn, session.id)

    @r.delete("/conversations/{id}", status_code=204, tags=["chat"])
    def delete_conversation(id: int, session: Signed, conn: Db) -> None:
        from hal9000.knowledge.answer.store import delete_conversation as delete

        if not delete(conn, session.id, id):
            raise _not_found("Conversation")

    @r.post("/feedback", response_model=FeedbackResponse, tags=["chat"])
    def create_feedback(body: FeedbackRequest, session: Signed, conn: Db) -> FeedbackResponse:
        """Thumbs up/down, a comment, and optionally "Add to question set".

        One row per message: a later post for the same message updates it.
        """
        from hal9000.knowledge.answer.store import message_session, save_feedback

        if message_session(conn, body.message_id) != session.id:
            raise _not_found("Message")
        return save_feedback(conn, body)

    @r.get("/search", response_model=RetrievalResult, tags=["chat"])
    def search(
        q: Annotated[str, Query(min_length=1)],
        services: Services,
        conn: Db,
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        doc_type: DocType | None = None,
        collection: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> RetrievalResult:
        """Retrieval only, with scores (debugging and retrieval-only mode)."""
        from hal9000.cost.metering import BudgetExceeded
        from hal9000.knowledge.answer.pipeline import NoActiveProfile

        filters = SearchFilters(
            doc_type=doc_type, collection=collection, date_from=date_from, date_to=date_to
        )
        pipeline = services.answerer
        try:
            try:
                pipeline.deps.check_budget(q)
                result = pipeline.retrieve(conn, q, filters=filters)
            except BudgetExceeded:
                result = pipeline.retrieve(conn, q, filters=filters, bm25_only=True)
        except NoActiveProfile as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        return result.model_copy(update={"candidates": result.candidates[:k]})

    # -- evaluation (plan §8) ----------------------------------------------------
    @r.get("/eval/gold", response_model=list[GoldItem], tags=["eval"])
    def list_gold(
        conn: Db, status: str | None = None, split: str | None = None, category: str | None = None
    ) -> list[GoldItem]:
        from hal9000.knowledge.eval.store import list_gold as gold

        return list(reversed(gold(conn, status=status, split=split, category=category)))

    @r.get("/eval/gold/{id}", response_model=GoldItem, tags=["eval"])
    def get_gold(id: int, conn: Db) -> GoldItem:
        from hal9000.knowledge.eval.store import get_gold as gold

        item = gold(conn, id)
        if item is None:
            raise _not_found("Question")
        return item

    @r.patch("/eval/gold/{id}", response_model=GoldItem, tags=["eval"])
    def update_gold(id: int, body: GoldPatch, conn: Db) -> GoldItem:
        """Approve, edit or reject a question-set item."""
        from hal9000.knowledge.api.evaluation import patch_gold

        try:
            item = patch_gold(conn, id, body)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        if item is None:
            raise _not_found("Question")
        return item

    @r.get("/eval/judge-labels", response_model=list[JudgeLabel], tags=["eval"])
    def list_judge_labels(conn: Db, run: str | None = None) -> list[JudgeLabel]:
        from hal9000.knowledge.eval.store import list_judge_labels as labels

        return labels(conn, eval_run_id=run)

    @r.get("/eval/judge-labels/{id}", response_model=JudgeLabel, tags=["eval"])
    def get_judge_label(id: int, conn: Db) -> JudgeLabel:
        from hal9000.knowledge.api.evaluation import judge_label

        label = judge_label(conn, id)
        if label is None:
            raise _not_found("Label")
        return label

    @r.put("/eval/judge-labels/{id}", response_model=JudgeLabel, tags=["eval"])
    def put_judge_label(id: int, body: JudgeLabelUpdate, conn: Db) -> JudgeLabel:
        """Mike's rubric labels for one generated answer (gate G-K1b)."""
        from hal9000.knowledge.api.evaluation import judge_label
        from hal9000.knowledge.eval.store import set_human_labels

        try:
            set_human_labels(conn, id, body.human)
        except KeyError:
            raise _not_found("Label") from None
        label = judge_label(conn, id)
        assert label is not None
        return label

    @r.get("/eval/runs", response_model=list[EvalRun], tags=["eval"])
    def list_eval_runs(conn: Db, stage: str | None = None) -> list[EvalRun]:
        from hal9000.knowledge.eval.runs import list_runs

        return list_runs(conn, stage=stage)

    @r.get("/eval/runs/{id}", response_model=EvalRun, tags=["eval"])
    def get_eval_run(id: str, conn: Db) -> EvalRun:
        from hal9000.knowledge.eval.runs import get_run

        run = get_run(conn, id)
        if run is None:
            raise _not_found("Run")
        return run

    @r.get("/eval/runs/{id}/results", response_model=EvalResultsPage, tags=["eval"])
    def get_eval_run_results(
        id: str,
        conn: Db,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 25,
    ) -> EvalResultsPage:
        """One run's per-question results, a page at a time."""
        from hal9000.knowledge.eval.runs import results_page

        try:
            return results_page(conn, id, offset=offset, limit=limit)
        except KeyError:
            raise _not_found("Run") from None

    # -- profile and traces --------------------------------------------------------
    @r.get("/profile", response_model=ProfileState, tags=["profile"])
    def get_profile(conn: Db) -> ProfileState:
        from hal9000.knowledge.api.evaluation import profile_state

        return profile_state(conn)

    @r.put(
        "/profile",
        response_model=ProfileState,
        tags=["profile"],
        responses={409: {"model": ErrorResponse, "description": "The profile isn't built yet"}},
    )
    def put_profile(body: ProfileActivate, conn: Db) -> ProfileState:
        """Make an existing, built profile the one that serves the chat."""
        from hal9000.knowledge.api.evaluation import profile_state
        from hal9000.knowledge.index.profiles import (
            ProfileError,
            ProfileNotReady,
            activate_profile,
        )

        try:
            activate_profile(conn, body.name)
        except ProfileNotReady as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        except ProfileError:
            raise _not_found("Profile") from None
        return profile_state(conn)

    @r.get("/traces", response_model=list[TraceSummary], tags=["traces"])
    def list_traces(
        conn: Db, limit: Annotated[int, Query(ge=1, le=500)] = 50
    ) -> list[TraceSummary]:
        from hal9000.knowledge.api.evaluation import trace_summaries

        return trace_summaries(conn, limit)

    @r.get("/traces/{id}", response_model=Trace, tags=["traces"])
    def get_trace(id: str, conn: Db) -> Trace:
        """The Query Inspector's data for one question."""
        from hal9000.knowledge.obs.traces import load_trace

        trace = load_trace(conn, id)
        if trace is None:
            raise _not_found("Trace")
        return trace

    return r


def _safe_name(name: str) -> str:
    base = PurePosixPath(name).name or "original"
    return "".join(ch for ch in base if ch.isprintable() and ch not in '"\\;') or "original"


def _reindex_file(services, conn: sqlite3.Connection, sha: str) -> Job:
    from hal9000.knowledge.ingest.jobs import create_job

    if services.ingest is None:
        raise HTTPException(status_code=503, detail="ingestion is not available")
    record = conn.execute(
        "SELECT relative_path, collection FROM files WHERE sha256 = ?", (sha,)
    ).fetchone()
    if record is None:
        raise _not_found("File")
    active = conn.execute(
        "SELECT id FROM jobs WHERE file_sha = ? AND status IN ('queued', 'running') LIMIT 1",
        (sha,),
    ).fetchone()
    if active is not None:
        from hal9000.knowledge.ingest.jobs import get_job

        job = get_job(conn, active[0])
        assert job is not None
        return job
    job = create_job(
        conn,
        "ingest",
        file_sha=sha,
        params={
            "relative_path": record[0],
            "collection": record[1],
            "use_llm": bool(services.ingest.use_llm),
        },
        detail="queued (re-index)",
        bus=services.bus,
    )
    services.notify_worker()
    return job


# ------------------------------------------------------------------- chat
async def _chat(body: ChatRequest, session: Session, services) -> Response:
    from sse_starlette import EventSourceResponse
    from starlette.concurrency import run_in_threadpool

    from hal9000.knowledge.answer import store
    from hal9000.knowledge.answer.generate import Cancelled
    from hal9000.knowledge.answer.pipeline import NoActiveProfile
    from hal9000.knowledge.answer.retrieval import Turn
    from hal9000.knowledge.contracts import Answer

    question = body.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Ask a question first.")
    limiter = services.chat_limiter
    if not limiter.allow(session.id):
        raise HTTPException(
            status_code=429,
            detail="That's a lot of questions at once. Wait a moment and ask again.",
            headers={"Retry-After": str(limiter.retry_after(session.id))},
        )

    def prepare() -> tuple[int, list[Turn]]:
        conn = services.connect()
        try:
            services.answerer._profile(conn)  # NoActiveProfile before anything streams
            turns = services.knowledge_file().retrieval.history_turns
            if body.conversation_id is not None:
                if store.get_conversation(conn, session.id, body.conversation_id) is None:
                    raise _not_found("Conversation")
                conversation = body.conversation_id
                past = store.history(conn, conversation, turns)
            else:
                conversation = store.create_conversation(conn, session.id, question)
                past = []
            store.add_message(conn, conversation, "user", question)
            return conversation, [Turn(role, text) for role, text in past]  # type: ignore[arg-type]
        finally:
            conn.close()

    try:
        conversation, history = await run_in_threadpool(prepare)
    except NoActiveProfile as error:
        raise HTTPException(status_code=409, detail=str(error)) from None

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def emit(event: Any) -> None:
        _put(loop, queue, event)

    def answer_question() -> None:
        conn = services.connect()
        try:

            def on_answer(answer: Answer) -> Answer:
                message = store.add_message(
                    conn,
                    conversation,
                    "assistant",
                    answer.text,
                    citations=answer.citations,
                    trace_id=answer.trace_id,
                    outcome=answer.outcome,
                    grounding=answer.grounding,
                    issues=answer.issues,
                )
                if answer.outcome == "answered":
                    services.write_health(conn)
                return answer.model_copy(
                    update={"message_id": message.id, "conversation_id": conversation}
                )

            services.answerer.ask(
                conn,
                question,
                emit=emit,
                history=history,
                filters=body.filters,
                cancel=cancel,
                on_answer=on_answer,
            )
        except Cancelled:
            log.info("a question was cancelled: the client went away")
        except Exception:
            log.exception("answering failed")
        finally:
            conn.close()
            _put(loop, queue, _END)

    context = contextvars.copy_context()  # the trace and usage sink live in context variables
    threading.Thread(
        target=context.run, args=(answer_question,), name="kb-question", daemon=True
    ).start()

    async def events() -> AsyncIterator[dict]:
        try:
            while True:
                event = await queue.get()
                if event is _END:
                    break
                yield {"event": event.event, "data": event.model_dump_json()}
        finally:
            cancel.set()  # the client went away (or the stream ended: then it's a no-op)

    return EventSourceResponse(events(), ping=15, headers={"Cache-Control": "no-cache"})


# -------------------------------------------------------------------- ops
def _ops_router(db_path: Path | None) -> APIRouter:
    from hal9000.knowledge.obs.metrics import metrics_router

    r = APIRouter()

    @r.get("/health", response_model=Health, tags=["ops"])
    def health(request: Request) -> Health:
        """Unauthenticated (LAN only); never exposes question text.

        Also writes ``data/knowledge_health.json`` for HAL's watchdog.
        """
        services = services_of(request)
        from hal9000.knowledge.index.profiles import active_profile

        conn = services.connect()
        try:
            profile = active_profile(conn)
            body = services.write_health(conn)
        finally:
            conn.close()
        retrieval_only = services.retrieval_only()
        return Health(
            status="ok" if profile is not None and not retrieval_only else "degraded",
            version=__version__,
            active_profile=profile.name if profile else None,
            month_spend_usd=body["month_spend_usd"],
            month_cap_usd=services.settings.monthly_cap_usd,
            retrieval_only=retrieval_only,
        )

    r.include_router(metrics_router(db_path))
    return r


def _static(app: FastAPI, dist: Path) -> None:
    """``web/dist`` at ``/``: real files, else ``index.html`` (the UI's history routes)."""
    root = dist.resolve()
    index = root / "index.html"
    reserved = ("api", "health", "metrics")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        first = path.split("/", 1)[0]
        if first in reserved:
            raise HTTPException(status_code=404, detail="Not Found")
        target = (root / path).resolve()
        if path and target.is_file() and target.is_relative_to(root):
            cache = "public, max-age=31536000, immutable" if first == "assets" else "no-cache"
            return FileResponse(target, headers={"Cache-Control": cache})
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def create_app(services: Any | None = None, *, web_dist: Path | None | bool = None) -> FastAPI:
    """Build the app. ``web_dist``: the UI build to serve (default ``web/dist`` if built)."""
    app = FastAPI(
        title="HAL 9000 house knowledge",
        version=__version__,
        description="The contract between the knowledge-base backend and its web UI "
        "(docs/KNOWLEDGE_PLAN.md §11).",
        generate_unique_id_function=_operation_id,
    )
    app.state.services = services
    app.include_router(_session_router())
    app.include_router(_api_router())
    app.include_router(_ops_router(services.settings.db_path if services else None))
    _document_chat_stream(app)
    dist = WEB_DIST if web_dist is None else web_dist
    if services is not None and isinstance(dist, Path) and (dist / "index.html").is_file():
        _static(app, dist)
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
