"""One question, end to end (plan §7): retrieve → abstain → generate → verify → done.

:meth:`AnswerPipeline.ask` answers one question and reports progress through
``emit`` as the chat stream's events (``status``, ``sources``, ``token``,
``reset``, ``verify``, ``done``: the contracts' ``Chat*Event`` models), in
this order:

1. ``status`` for each stage as it starts; ``sources`` (every numbered source,
   with the degraded-mode banner) before any ``token``.
2. ``token`` deltas of the streamed answer. A reply of exactly
   ``NOT_IN_SOURCES`` is held back and becomes the abstention text.
3. **verify** (plan §7.1). On failure: ``reset`` (the client drops the partial
   answer), one regeneration listing the problems, verify again. If that fails
   too the answer keeps ``grounding="failed"`` (the UI's "figures not
   verified" badge) and ``kb_grounding_failures_total{kind}`` counts it.
4. ``verify`` with the result, then ``done`` with the :class:`Answer`
   (citations, trace ID, outcome, cost), after the trace row is written, so
   the Query Inspector can open it at once.

**Degraded modes** (plan §7.4):

- monthly cap reached (before or during the question): retrieval-only, BM25
  over the index, no chat call, ``sources`` with ``retrieval_only`` and a banner;
- query embedding failed: BM25 only, with a banner;
- chat call failed or the 90 s wall clock passed: the partial answer is reset
  and the sources are sent again as retrieval-only with an error banner
  (outcome ``error``);
- ``answer.max_output_tokens`` caps the output.

Every question runs inside KWP8's ``observe()``: stage timings, hits,
context size, tokens and cost (from the usage sink), outcome and grounding.
Run it on one thread: the trace lives in a context variable (the API runs
each question on its own thread inside ``contextvars.copy_context()``).
Client disconnects arrive as ``cancel``: generation stops, the estimated
usage is recorded, and the question ends as ``cancelled``.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from hal9000.cost.metering import BudgetExceeded
from hal9000.knowledge.answer.generate import (
    DeadlineExceeded,
    call_within,
    metered_generate,
    stream_answer,
)
from hal9000.knowledge.answer.prompt import (
    NOT_IN_SOURCES,
    NumberedSource,
    build_prompt,
    number_sources,
    read_policy,
    regeneration_prompt,
)
from hal9000.knowledge.answer.retrieval import NoActiveProfile as _RetrieveNoActiveProfile
from hal9000.knowledge.answer.retrieval import (
    RetrievalConfig,
    Retriever,
    Turn,
    embedding_unavailable,
    search_documents,
)
from hal9000.knowledge.answer.verify import Verification, verify_answer
from hal9000.knowledge.config import AGENT, KnowledgeFile, KnowledgeSettings, load_knowledge_file
from hal9000.knowledge.contracts import (
    Answer,
    ChatDoneEvent,
    ChatResetEvent,
    ChatSourcesEvent,
    ChatStatusEvent,
    ChatTokenEvent,
    ChatVerifyEvent,
    IndexProfile,
    RetrievalResult,
    SearchFilters,
    Trace,
)
from hal9000.knowledge.obs import metrics
from hal9000.knowledge.obs.timing import Stage, stage
from hal9000.knowledge.obs.traces import describe_error, hits_from_retrieval, observe

__all__ = [
    "ABSTAIN_NOTHING",
    "ABSTAIN_TEXT",
    "BANNER_CAP",
    "BANNER_CHAT_ERROR",
    "BANNER_EMBEDDING",
    "BANNER_FAILED",
    "BANNER_TIMEOUT",
    "AnswerDeps",
    "AnswerPipeline",
    "Emit",
    "NoActiveProfile",
]

log = logging.getLogger(__name__)

Emit = Callable[[Any], None]

ABSTAIN_TEXT = (
    "I couldn't find that in the house documents. The closest passages are listed as sources."
)
ABSTAIN_NOTHING = (
    "I couldn't find anything about that in the house documents. A document that covers it "
    "may not have been added yet."
)
BANNER_CAP = (
    "This month's answer budget is used up, so here are the closest passages instead. "
    "Search still works."
)
BANNER_EMBEDDING = (
    "Semantic search is unavailable right now, so these results come from keyword search "
    "only and may be less complete."
)
BANNER_CHAT_ERROR = (
    "The answer service didn't respond, so here are the closest passages instead. "
    "Try again in a moment."
)
BANNER_TIMEOUT = "The answer took too long, so here are the closest passages instead."
BANNER_FAILED = "Something went wrong while answering. Try again in a moment."


def _is_refusal(text: str) -> bool:
    return text.strip().startswith(NOT_IN_SOURCES)


#: No index profile serves the chat yet (plan §6.2, gate G-K2): KWP4's error, so the
#: API and the CLI catch one class whichever side notices first.
NoActiveProfile = _RetrieveNoActiveProfile


@dataclass
class AnswerDeps:
    """What the pipeline calls out to; tests and ``--fake`` swap these.

    Args:
        stream_fn: like :func:`hal9000.llm.stream_response`.
        generate_fn: like :func:`hal9000.llm.generate_response` (condensing).
        meter: meters every paid call (``None``: fake mode, nothing is paid).
        embed_fn: the embedding callable for query vectors (``None``: by model).
        retriever_cls: KWP4's ``Retriever`` (one is built per question).
        query_cache: the query-vector cache (``None``: KWP4's process-wide one).
        budget_check: raises ``BudgetExceeded`` when the question may not pay for
            anything (default: the meter's cap check for the agent).
        agent: ``knowledge``, or ``knowledge-eval`` for the evaluation harness.
    """

    stream_fn: Callable[..., Any] | None = None
    generate_fn: Callable[..., str] | None = None
    meter: Any | None = None
    embed_fn: Callable[..., Any] | None = None
    retriever_cls: type = Retriever
    query_cache: Any | None = None
    budget_check: Callable[[str], None] | None = None
    agent: str = AGENT
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.stream_fn is None or self.generate_fn is None:
            from hal9000 import llm

            self.stream_fn = self.stream_fn or llm.stream_response
            self.generate_fn = self.generate_fn or llm.generate_response

    def check_budget(self, question: str) -> None:
        if self.budget_check is not None:
            self.budget_check(question)
        elif self.meter is not None:
            self.meter.precheck(self.agent)


@dataclass
class _State:
    """What one question has decided so far (for the done event and the error paths)."""

    sources: list[NumberedSource] = field(default_factory=list)
    emitted: bool = False  # any token went out since the last reset
    outcome: str = "answered"
    grounding: str = "skipped"
    issues: list[str] = field(default_factory=list)
    text: str = ""
    banner: str | None = None
    verified: bool = False  # a verify event went out


class _Fallback(Exception):  # noqa: N818 - control flow, not an error
    """Stop generating and show the sources as retrieval-only (with this banner)."""

    def __init__(self, outcome: str, banner: str, cause: BaseException | None = None) -> None:
        super().__init__(banner)
        self.outcome = outcome
        self.banner = banner
        self.cause = cause


class AnswerPipeline:
    """Answers questions over the active (or a given) index profile.

    ``knowledge_file`` pins ``config/knowledge.yaml`` (tests); by default it is
    read per question, like the policy. ``profile`` pins the index profile
    (an evaluation run's) instead of the active one. ``record_traces=False``
    keeps questions out of the ``traces`` table (evaluation answers).
    """

    def __init__(
        self,
        settings: KnowledgeSettings,
        deps: AnswerDeps | None = None,
        *,
        knowledge_file: KnowledgeFile | None = None,
        tier: str | None = None,
        profile: IndexProfile | None = None,
        record_traces: bool = True,
    ) -> None:
        self.settings = settings
        self.deps = deps or AnswerDeps()
        self.knowledge_file = knowledge_file
        self.tier = tier or settings.chat_tier
        self.profile = profile
        self.record_traces = record_traces

    # ------------------------------------------------------------ helpers
    def _kf(self) -> KnowledgeFile:
        return self.knowledge_file or load_knowledge_file(self.settings.yaml_path)

    def _profile(self, conn: sqlite3.Connection) -> IndexProfile:
        from hal9000.knowledge.index.profiles import active_profile

        profile = self.profile or active_profile(conn)
        if profile is None:
            raise NoActiveProfile(
                "No index profile is active yet: ingest documents with an active_profile in "
                "config/knowledge.yaml, or activate one with `hal9000 kb profile activate`."
            )
        return profile

    def _embeddings(
        self, conn: sqlite3.Connection, profile: IndexProfile, deadline: float | None = None
    ):
        from hal9000.knowledge.index.embedding import EmbeddingService, default_embed_fn

        embed_fn = self.deps.embed_fn
        if deadline is not None:
            inner = embed_fn or default_embed_fn(profile.embed_model)
            clock = self.deps.clock

            def embed_fn(*args, **kw):  # the query embedding counts against the deadline
                return call_within(deadline, clock, inner, *args, **kw)

        return EmbeddingService(
            conn,
            embed_fn=embed_fn,
            meter=self.deps.meter,
            agent=self.deps.agent,
            max_attempts=2,
            backoff_s=0.25,
        )

    def _retriever(
        self,
        conn: sqlite3.Connection,
        profile: IndexProfile,
        config,
        deadline: float | None = None,
    ):
        condense = metered_generate(
            self.deps.generate_fn,  # type: ignore[arg-type]
            meter=self.deps.meter,
            agent=self.deps.agent,
            purpose="condense",
            default_tier=self.settings.condense_tier,
            deadline=deadline,
            clock=self.deps.clock,
        )
        return self.deps.retriever_cls(
            conn,
            self._embeddings(conn, profile, deadline),
            generate=condense,
            config=config,
            profile=profile,
            tier=self.settings.condense_tier,
            query_cache=self.deps.query_cache,
        )

    def search(
        self,
        conn: sqlite3.Connection,
        query: str,
        *,
        k: int = 10,
        filters: SearchFilters | None = None,
        bm25_only: bool = False,
    ) -> RetrievalResult:
        """Retrieval alone (``GET /api/search``), with KWP4's :func:`search_documents`."""
        profile = self._profile(conn)
        config = RetrievalConfig.from_settings(self.knowledge_file or self.settings)
        if bm25_only:
            config = replace(config, mode="bm25")
        return search_documents(
            conn,
            self._embeddings(conn, profile),
            query,
            k=k,
            filters=filters,
            config=config,
            profile=profile,
        )

    # ---------------------------------------------------------------- ask
    def ask(
        self,
        conn: sqlite3.Connection,
        question: str,
        *,
        emit: Emit,
        history: Sequence[Turn] = (),
        filters: SearchFilters | None = None,
        cancel: threading.Event | None = None,
        on_answer: Callable[[Answer], Answer] | None = None,
        on_sources: Callable[[list[NumberedSource]], None] | None = None,
    ) -> Answer:
        """Answer ``question``, emitting the chat events; returns the final :class:`Answer`.

        ``on_answer`` runs before ``done`` is emitted (the API stores the
        message there and fills its IDs); ``on_sources`` receives the
        numbered sources with their full text (the evaluation harness).
        Raises :class:`NoActiveProfile` before anything is emitted, and
        :class:`Cancelled` when ``cancel`` was set.
        """
        profile = self._profile(conn)
        kf = self._kf()
        deadline = self.deps.clock() + kf.answer.wall_clock_seconds
        state = _State()
        trace = Trace(question=question, profile=profile.name)
        downstream = emit

        def emit(event) -> None:  # noqa: F811 - tracks what went out
            if event.event == "verify":
                state.verified = True
            downstream(event)

        try:
            with observe(conn if self.record_traces else None, trace=trace):
                self._answer(
                    conn, question, profile, kf, emit, history, filters, cancel, deadline, state
                )
                trace.outcome = state.outcome  # type: ignore[assignment]
                trace.grounding = state.grounding  # type: ignore[assignment]
        except Exception:  # recorded on the trace as an error; the client still gets done
            log.exception("question %s failed", trace.id)
            state.outcome, state.grounding, state.issues, state.text = "error", "skipped", [], ""
            if state.emitted:
                emit(ChatResetEvent(reason=BANNER_FAILED))
            self._sources(emit, state, retrieval_only=True, banner=BANNER_FAILED)
        if not state.verified:  # every stream carries a verify event before done
            emit(ChatVerifyEvent(grounding=state.grounding, issues=state.issues))  # type: ignore[arg-type]
        if on_sources is not None:
            on_sources(list(state.sources))
        citations = self._citations(state)
        answer = Answer(
            text=state.text,
            citations=citations,
            outcome=state.outcome,  # type: ignore[arg-type]
            grounding=state.grounding,  # type: ignore[arg-type]
            issues=state.issues,
            trace_id=trace.id,
            cost_usd=round(trace.cost_usd, 6),
        )
        if on_answer is not None:
            answer = on_answer(answer)
        emit(ChatDoneEvent(answer=answer))
        return answer

    def _citations(self, state: _State):
        from hal9000.knowledge.answer.verify import cited_markers

        all_sources = [source.citation() for source in state.sources]
        if state.outcome != "answered":
            return all_sources  # the closest passages, shown instead of an answer
        cited = set(cited_markers(state.text))
        return [c for c in all_sources if c.marker in cited]

    def _answer(
        self,
        conn: sqlite3.Connection,
        question: str,
        profile: IndexProfile,
        kf: KnowledgeFile,
        emit: Emit,
        history: Sequence[Turn],
        filters: SearchFilters | None,
        cancel: threading.Event | None,
        deadline: float,
        state: _State,
    ) -> None:
        from hal9000.knowledge.obs.timing import current_trace

        trace = current_trace()
        assert trace is not None
        config = RetrievalConfig.from_settings(self.knowledge_file or self.settings)
        capped = False
        try:
            self.deps.check_budget(question)
        except BudgetExceeded:
            capped = True

        result = None
        if not capped:
            emit(ChatStatusEvent(stage=Stage.CONDENSE if history else Stage.EMBED_QUERY))
            try:
                result = self._retriever(conn, profile, config, deadline).retrieve(
                    question, history, filters, config
                )
            except BudgetExceeded:
                capped = True
            except DeadlineExceeded as error:  # condensing or embedding ran out the clock
                trace.error = describe_error(error)
                state.outcome = "error"
                self._sources(emit, state, retrieval_only=True, banner=BANNER_TIMEOUT)
                return
        if capped:
            emit(ChatStatusEvent(stage=Stage.BM25, detail="retrieval only"))
            bm25 = replace(config, mode="bm25")
            result = self._retriever(conn, profile, bm25).retrieve(question, (), filters, bm25)
        assert result is not None
        emit(ChatStatusEvent(stage=Stage.SELECT))

        trace.standalone_query = result.standalone_query
        trace.hits = hits_from_retrieval(result)
        trace.context_tokens = result.context_tokens
        state.sources = number_sources(conn, result.selected)
        if embedding_unavailable(result):
            state.banner = BANNER_EMBEDDING

        if capped:
            self._sources(emit, state, retrieval_only=True, banner=BANNER_CAP)
            state.outcome = "retrieval_only"
            return
        if self.deps.clock() >= deadline:
            # the retriever falls back quietly when condensing or embedding time out
            # (the question as asked, BM25 only); the question's time is still up
            if state.banner:
                self._sources(emit, state, banner=state.banner)
            trace.error = describe_error(DeadlineExceeded("the question's time limit passed"))
            state.outcome = "error"
            self._sources(emit, state, retrieval_only=True, banner=BANNER_TIMEOUT)
            return
        if result.abstained or not state.sources:
            self._sources(emit, state, banner=state.banner)
            self._abstain(emit, state)
            return

        self._sources(emit, state, banner=state.banner)
        try:
            self._generate_and_verify(conn, question, result, kf, emit, cancel, deadline, state)
        except _Fallback as fallback:
            if state.emitted:
                emit(ChatResetEvent(reason=fallback.banner))
                state.emitted = False
            state.text = ""
            state.grounding = "skipped"
            state.issues = []
            state.outcome = fallback.outcome
            if fallback.cause is not None and fallback.outcome == "error":
                trace.error = describe_error(fallback.cause)
            self._sources(emit, state, retrieval_only=True, banner=fallback.banner)

    # ------------------------------------------------------------ events
    @staticmethod
    def _sources(
        emit: Emit, state: _State, *, retrieval_only: bool = False, banner: str | None = None
    ) -> None:
        emit(
            ChatSourcesEvent(
                sources=[source.citation() for source in state.sources],
                retrieval_only=retrieval_only,
                banner=banner,
            )
        )

    @staticmethod
    def _abstain(emit: Emit, state: _State) -> None:
        state.text = ABSTAIN_TEXT if state.sources else ABSTAIN_NOTHING
        state.outcome = "abstained"
        state.grounding = "skipped"
        emit(ChatTokenEvent(text=state.text))
        emit(ChatVerifyEvent(grounding="skipped"))

    # -------------------------------------------------------- generation
    def _generate_once(
        self,
        prompt,
        kf: KnowledgeFile,
        emit: Emit,
        cancel: threading.Event | None,
        deadline: float,
        state: _State,
    ) -> str:
        """One streamed generation; returns its text (``NOT_IN_SOURCES`` is never emitted)."""
        held: list[str] = []

        def on_delta(delta: str) -> None:
            if not state.emitted:
                held.append(delta)
                sofar = "".join(held).lstrip()
                if NOT_IN_SOURCES.startswith(sofar) or sofar.startswith(NOT_IN_SOURCES):
                    return  # might be the refusal sentinel: hold it back
                emit(ChatTokenEvent(text="".join(held)))
                held.clear()
                state.emitted = True
                return
            emit(ChatTokenEvent(text=delta))
            state.emitted = True

        emit(ChatStatusEvent(stage=Stage.GENERATE))
        try:
            generation = stream_answer(
                prompt,
                tier=self.tier,
                stream_fn=self.deps.stream_fn,  # type: ignore[arg-type]
                on_delta=on_delta,
                meter=self.deps.meter,
                agent=self.deps.agent,
                cancel=cancel,
                deadline=deadline,
                max_tokens=kf.answer.max_output_tokens,
                clock=self.deps.clock,
            )
        except BudgetExceeded as error:
            raise _Fallback("retrieval_only", BANNER_CAP, error) from error
        except DeadlineExceeded as error:
            raise _Fallback("error", BANNER_TIMEOUT, error) from error
        except Exception as error:
            log.warning("the chat call failed: %s", describe_error(error))
            raise _Fallback("error", BANNER_CHAT_ERROR, error) from error
        text = generation.text.strip()
        if held and not _is_refusal(text):  # a short reply that only looked like the sentinel
            emit(ChatTokenEvent(text="".join(held)))
            state.emitted = True
        return text

    def _generate_and_verify(
        self,
        conn: sqlite3.Connection,
        question: str,
        result: RetrievalResult,
        kf: KnowledgeFile,
        emit: Emit,
        cancel: threading.Event | None,
        deadline: float,
        state: _State,
    ) -> None:
        policy = read_policy(self.settings.policy_path)
        prompt = build_prompt(
            question, state.sources, policy=policy, standalone=result.standalone_query
        )
        facts = [source.facts() for source in state.sources]

        text = self._generate_once(prompt, kf, emit, cancel, deadline, state)
        if _is_refusal(text):
            self._abstain(emit, state)
            return
        check = self._verify(text, facts, emit)
        if not check.ok:
            emit(ChatResetEvent(reason=check.issues[0].message))
            state.emitted = False
            again = regeneration_prompt(prompt, text, check.messages)
            text = self._generate_once(again, kf, emit, cancel, deadline, state)
            if _is_refusal(text):
                self._abstain(emit, state)
                return
            check = self._verify(text, facts, emit)
            if not check.ok:
                for kind in sorted(check.kinds):
                    metrics.record_grounding_failure(kind)
                log.warning("answer failed verification twice: %s", "; ".join(check.messages))
        state.text = text
        state.outcome = "answered"
        state.grounding = check.grounding
        state.issues = check.messages
        emit(ChatVerifyEvent(grounding=check.grounding, issues=check.messages))

    @staticmethod
    def _verify(text: str, facts, emit: Emit) -> Verification:
        emit(ChatStatusEvent(stage=Stage.VERIFY))
        with stage(Stage.VERIFY):
            return verify_answer(text, facts)
