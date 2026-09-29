"""Every plan §9.2 metric, the helpers the pipeline calls, and the ``/metrics`` router.

Each metric is defined exactly once, here, in the knowledge base's own
registry (:data:`~hal9000.knowledge.obs.timing.REGISTRY`), except
``kb_stage_seconds``, which lives beside the stage timer in ``timing.py``, and
the five gauges that are read from ``knowledge.db`` at scrape time by
:class:`~hal9000.knowledge.obs.collector.KnowledgeDbCollector`
(``kb_regression_recall``, ``kb_eval_metric``, ``kb_documents``, ``kb_chunks``,
``kb_db_bytes``).

Label values are closed sets wherever the plan gives one, and every helper
validates them, so a caller can't create an unbounded series or leak free text
(a question, a file name) into Prometheus (plan §14). The helpers never take
question text.

KWP6 mounts :func:`metrics_router` at the application root, outside ``/api``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector

from hal9000.knowledge.contracts import Outcome
from hal9000.knowledge.obs.timing import (
    REGISTRY,
    STAGE_SECONDS,
    Stage,
    current_trace,
    recorded_by_trace,
)

if TYPE_CHECKING:  # pragma: no cover - fastapi is imported only when the router is built
    from fastapi import APIRouter

__all__ = [
    "ANSWER_SECONDS",
    "CANCELLED",
    "CHUNKS_NAME",
    "CONTEXT_TOKENS",
    "COST_USD_TOTAL",
    "DB_BYTES_NAME",
    "DOCUMENTS_NAME",
    "ERRORS_TOTAL",
    "ERROR_STAGES",
    "EVAL_METRIC_NAME",
    "FEEDBACK_RATINGS",
    "FEEDBACK_TOTAL",
    "GROUNDING_FAILURES_TOTAL",
    "GROUNDING_KINDS",
    "INGEST_FILES_TOTAL",
    "INGEST_PAGES_TOTAL",
    "INGEST_STATUSES",
    "METRICS",
    "MONTH_CAP_USD",
    "MONTH_SPEND_USD",
    "OUTCOMES",
    "PURPOSES",
    "QUESTIONS_TOTAL",
    "REGRESSION_RECALL_NAME",
    "RETRIEVAL_SECONDS",
    "RETRIEVAL_STAGES",
    "STAGE_SECONDS",
    "TOKENS_TOTAL",
    "TOP_SCORE",
    "TTFT_SECONDS",
    "bind_month_budget",
    "configured_month_cap",
    "count_trace_error",
    "install_db_collector",
    "metrics_router",
    "record_error",
    "record_feedback",
    "record_grounding_failure",
    "record_ingest_file",
    "record_ingest_pages",
    "record_question",
    "record_usage",
    "render_metrics",
    "set_month_budget",
    "use_configured_month_budget",
]

# ------------------------------------------------------------ closed label sets
#: ``kb_questions_total{outcome}``: the contracts' ``Outcome`` plus ``cancelled``
#: (the client went away, e.g. a closed stream: not an error, so not in the error rate).
CANCELLED = "cancelled"
OUTCOMES: tuple[str, ...] = tuple(dict.fromkeys((*get_args(Outcome), CANCELLED)))
#: ``kb_tokens_total`` / ``kb_cost_usd_total`` ``{purpose}``.
PURPOSES: tuple[str, ...] = ("embed", "condense", "answer", "judge", "segment")
#: ``kb_grounding_failures_total{kind}``.
GROUNDING_KINDS: tuple[str, ...] = ("numeral", "citation")
#: ``kb_feedback_total{rating}``; the ``feedback`` table stores 1 and -1.
FEEDBACK_RATINGS: tuple[str, ...] = ("up", "down")
#: ``kb_ingest_files_total{status}`` (the contracts' ``UploadResult.status``).
INGEST_STATUSES: tuple[str, ...] = ("added", "duplicate", "skipped", "failed")
#: ``kb_errors_total{stage}``: a stage name, or ``request`` for an error outside any stage.
ERROR_STAGES: tuple[str, ...] = (*(s.value for s in Stage), "request")
#: plan §9.2: local retrieval is vector + BM25 + fuse + rerank + select.
RETRIEVAL_STAGES: frozenset[str] = frozenset({"vector", "bm25", "fuse", "rerank", "select"})

# ------------------------------------------------------------------ metrics
# The ``*_created`` companion series only double the series count on the Pi.
disable_created_metrics()

RETRIEVAL_SECONDS = Histogram(
    "kb_retrieval_seconds",
    "Total local retrieval time per question (vector + bm25 + fuse + rerank + select).",
    registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 2.5, 5, 10),
)
TTFT_SECONDS = Histogram(
    "kb_ttft_seconds",
    "Time from the question arriving to the first answer token.",
    registry=REGISTRY,
    buckets=(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 30, 60),
)
ANSWER_SECONDS = Histogram(
    "kb_answer_seconds",
    "End-to-end time per question, until the answer is complete.",
    registry=REGISTRY,
    buckets=(0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120),
)
QUESTIONS_TOTAL = Counter(
    "kb_questions",
    "Questions answered, by outcome.",
    ["outcome"],
    registry=REGISTRY,
)
ERRORS_TOTAL = Counter(
    "kb_errors",
    "Errors by stage (or 'request') and kind (an exception type name).",
    ["stage", "kind"],
    registry=REGISTRY,
)
TOKENS_TOTAL = Counter(
    "kb_tokens",
    "Tokens sent and received (prompt + completion), by model and purpose.",
    ["model", "purpose"],
    registry=REGISTRY,
)
COST_USD_TOTAL = Counter(
    "kb_cost_usd",
    "Spend in US dollars, by model and purpose.",
    ["model", "purpose"],
    registry=REGISTRY,
)
MONTH_SPEND_USD = Gauge(
    "kb_month_spend_usd",
    "Month-to-date spend of agent 'knowledge' (excludes 'knowledge-eval').",
    registry=REGISTRY,
)
MONTH_CAP_USD = Gauge(
    "kb_month_cap_usd",
    "The monthly cap KB_MONTHLY_CAP_USD; above it the chat is retrieval-only.",
    registry=REGISTRY,
)
TOP_SCORE = Histogram(
    "kb_top_score",
    "The abstain score per question: top dense cosine, or cross-encoder score when reranking.",
    registry=REGISTRY,
    buckets=(
        -10,
        -5,
        -2,
        -1,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
        2,
        5,
        10,
    ),  # fmt: skip
)
CONTEXT_TOKENS = Histogram(
    "kb_context_tokens",
    "Tokens of retrieved context sent with each question.",
    registry=REGISTRY,
    buckets=(250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000),
)
GROUNDING_FAILURES_TOTAL = Counter(
    "kb_grounding_failures",
    "Answers that failed verification after the one regeneration, by kind.",
    ["kind"],
    registry=REGISTRY,
)
FEEDBACK_TOTAL = Counter(
    "kb_feedback",
    "Thumbs up or down on answers.",
    ["rating"],
    registry=REGISTRY,
)
INGEST_FILES_TOTAL = Counter(
    "kb_ingest_files",
    "Uploaded files by result.",
    ["status"],
    registry=REGISTRY,
)
INGEST_PAGES_TOTAL = Counter(
    "kb_ingest_pages",
    "Ingested pages, by whether they had a text layer.",
    ["text_layer"],
    registry=REGISTRY,
)

#: Read from knowledge.db at scrape time (see ``collector.py``).
REGRESSION_RECALL_NAME = "kb_regression_recall"
EVAL_METRIC_NAME = "kb_eval_metric"
DOCUMENTS_NAME = "kb_documents"
CHUNKS_NAME = "kb_chunks"
DB_BYTES_NAME = "kb_db_bytes"

#: Plan §9.2 as exposed: metric name -> (type, label names). The tests and the
#: dashboard static check read this table.
METRICS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "kb_stage_seconds": ("histogram", ("stage",)),
    "kb_retrieval_seconds": ("histogram", ()),
    "kb_ttft_seconds": ("histogram", ()),
    "kb_answer_seconds": ("histogram", ()),
    "kb_questions_total": ("counter", ("outcome",)),
    "kb_errors_total": ("counter", ("stage", "kind")),
    "kb_tokens_total": ("counter", ("model", "purpose")),
    "kb_cost_usd_total": ("counter", ("model", "purpose")),
    "kb_month_spend_usd": ("gauge", ()),
    "kb_month_cap_usd": ("gauge", ()),
    "kb_top_score": ("histogram", ()),
    "kb_context_tokens": ("histogram", ()),
    "kb_grounding_failures_total": ("counter", ("kind",)),
    "kb_feedback_total": ("counter", ("rating",)),
    "kb_ingest_files_total": ("counter", ("status",)),
    "kb_ingest_pages_total": ("counter", ("text_layer",)),
    DOCUMENTS_NAME: ("gauge", ("profile",)),
    CHUNKS_NAME: ("gauge", ("profile",)),
    DB_BYTES_NAME: ("gauge", ("profile",)),
    REGRESSION_RECALL_NAME: ("gauge", ()),
    EVAL_METRIC_NAME: ("gauge", ("profile", "metric")),
}


def _start_closed_sets_at_zero() -> None:
    """Closed label sets start at zero, so rate() and ratios work before the first event."""
    for value in OUTCOMES:
        QUESTIONS_TOTAL.labels(outcome=value)
    for value in GROUNDING_KINDS:
        GROUNDING_FAILURES_TOTAL.labels(kind=value)
    for value in FEEDBACK_RATINGS:
        FEEDBACK_TOTAL.labels(rating=value)
    for value in INGEST_STATUSES:
        INGEST_FILES_TOTAL.labels(status=value)
    for value in ("yes", "no"):
        INGEST_PAGES_TOTAL.labels(text_layer=value)


_start_closed_sets_at_zero()


# ------------------------------------------------------------------ helpers
def _closed(value: str, allowed: Iterable[str], what: str) -> str:
    allowed = tuple(allowed)
    if value not in allowed:
        raise ValueError(f"{what} must be one of {', '.join(allowed)}; got {value!r}")
    return value


_KIND_RE = re.compile(r"[^A-Za-z0-9_.]")
_MODEL_RE = re.compile(r"[^A-Za-z0-9_.:/@-]")


def _kind(kind: str | BaseException | type[BaseException]) -> str:
    """An error kind label: an exception type name, never free text."""
    if isinstance(kind, BaseException):
        kind = type(kind).__name__
    elif isinstance(kind, type):
        kind = kind.__name__
    cleaned = _KIND_RE.sub("", str(kind))[:64]
    return cleaned or "unknown"


def _model(model: str) -> str:
    cleaned = _MODEL_RE.sub("", str(model or ""))[:100]
    return cleaned or "unknown"


def _nonneg(value: float | None, what: str) -> float | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{what} must be >= 0, got {value!r}")
    return float(value)


def record_question(
    outcome: str,
    *,
    ttft_s: float | None = None,
    answer_s: float | None = None,
    retrieval_s: float | None = None,
    top_score: float | None = None,
    context_tokens: int | None = None,
) -> None:
    """Count one finished question and observe its per-question histograms.

    Tokens and cost are *not* taken here: they are counted per call by
    :func:`record_usage`, so a question's spend is never counted twice.
    Omitted values are not observed (e.g. no ``ttft_s`` for an abstention).
    """
    _closed(outcome, OUTCOMES, "outcome")
    ttft_s = _nonneg(ttft_s, "ttft_s")
    answer_s = _nonneg(answer_s, "answer_s")
    retrieval_s = _nonneg(retrieval_s, "retrieval_s")
    if context_tokens is not None and context_tokens < 0:
        raise ValueError(f"context_tokens must be >= 0, got {context_tokens!r}")
    QUESTIONS_TOTAL.labels(outcome=outcome).inc()
    if ttft_s is not None:
        TTFT_SECONDS.observe(ttft_s)
    if answer_s is not None:
        ANSWER_SECONDS.observe(answer_s)
    if retrieval_s is not None:
        RETRIEVAL_SECONDS.observe(retrieval_s)
    if top_score is not None:
        TOP_SCORE.observe(float(top_score))
    if context_tokens is not None:
        CONTEXT_TOKENS.observe(context_tokens)


def record_usage(usage: Mapping, purpose: str, *, cost_usd: float | None = None) -> float:
    """Count one call's tokens and cost; returns the cost.

    ``usage`` is an :class:`hal9000.llm.Usage` (``model``, ``prompt``,
    ``cached``, ``completion``). Without ``cost_usd`` the cost is priced with
    the same function the ledger uses, so Prometheus and the ledger agree.
    When a trace is bound, its ``tokens_by_model`` and ``cost_usd`` grow too.
    """
    _closed(purpose, PURPOSES, "purpose")
    if cost_usd is None:
        from hal9000.cost.metering import usage_cost

        cost_usd = usage_cost(dict(usage))
    cost_usd = _nonneg(cost_usd, "cost_usd") or 0.0
    model = _model(usage.get("model", ""))
    prompt = int(usage.get("prompt", 0) or 0)
    cached = int(usage.get("cached", 0) or 0)
    completion = int(usage.get("completion", 0) or 0)
    if prompt < 0 or completion < 0 or cached < 0:
        raise ValueError("token counts must be >= 0")
    TOKENS_TOTAL.labels(model=model, purpose=purpose).inc(prompt + completion)
    COST_USD_TOTAL.labels(model=model, purpose=purpose).inc(cost_usd)
    trace = current_trace()
    if trace is not None:
        counts = trace.tokens_by_model.setdefault(model, {})
        for key, value in (("prompt", prompt), ("cached", cached), ("completion", completion)):
            counts[key] = counts.get(key, 0) + value
        trace.cost_usd = round(trace.cost_usd + cost_usd, 10)
    return cost_usd


def count_trace_error(stage: str | Stage, kind: str | BaseException | type[BaseException]) -> None:
    """Count an error unconditionally (``traces.finish_trace`` counts a trace's errors with it)."""
    stage_value = stage.value if isinstance(stage, Stage) else str(stage)
    _closed(stage_value, ERROR_STAGES, "stage")
    ERRORS_TOTAL.labels(stage=stage_value, kind=_kind(kind)).inc()


def record_error(stage: str | Stage, kind: str | BaseException | type[BaseException]) -> None:
    """Count an error. ``stage`` is a stage name or ``request``; ``kind`` is a type name.

    Inside an observed request, the very exception object that already left a
    ``stage()`` block of that name is not counted here: the trace counts it
    when it finishes, so a caller that catches it outside the stage and also
    reports it (the retriever's BM25-only fallback) doesn't count it twice.
    Any other report (another exception of the same type, a type or a name)
    is counted.
    """
    stage_value = stage.value if isinstance(stage, Stage) else str(stage)
    _closed(stage_value, ERROR_STAGES, "stage")
    if isinstance(kind, BaseException) and recorded_by_trace(kind, stage_value):
        return
    ERRORS_TOTAL.labels(stage=stage_value, kind=_kind(kind)).inc()


def record_grounding_failure(kind: str) -> None:
    """Count an answer whose ``numeral`` or ``citation`` check failed (plan §7.1 verify)."""
    GROUNDING_FAILURES_TOTAL.labels(kind=_closed(kind, GROUNDING_KINDS, "kind")).inc()


def record_feedback(rating: int | str) -> None:
    """Count a thumbs up (``1`` / ``"up"``) or down (``-1`` / ``"down"``)."""
    mapped = {1: "up", -1: "down"}.get(rating, rating) if isinstance(rating, int) else rating
    FEEDBACK_TOTAL.labels(rating=_closed(str(mapped), FEEDBACK_RATINGS, "rating")).inc()


def record_ingest_file(status: str) -> None:
    """Count one uploaded file: ``added``, ``duplicate``, ``skipped`` or ``failed``."""
    INGEST_FILES_TOTAL.labels(status=_closed(status, INGEST_STATUSES, "status")).inc()


def record_ingest_pages(*, with_text_layer: int = 0, without_text_layer: int = 0) -> None:
    """Count ingested pages by whether they had a text layer."""
    if with_text_layer < 0 or without_text_layer < 0:
        raise ValueError("page counts must be >= 0")
    INGEST_PAGES_TOTAL.labels(text_layer="yes").inc(with_text_layer)
    INGEST_PAGES_TOTAL.labels(text_layer="no").inc(without_text_layer)


def configured_month_cap() -> float:
    """``KB_MONTHLY_CAP_USD`` as configured now (``knowledge.config.settings()``)."""
    from hal9000.knowledge.config import settings

    return float(settings().monthly_cap_usd)


def _safe(fn: Callable[[], float]) -> Callable[[], float]:
    """A scrape-time reader that reports ``NaN`` instead of breaking the scrape."""

    def read() -> float:
        try:
            return float(fn())
        except Exception:
            return float("nan")

    return read


def use_configured_month_budget() -> None:
    """The default: nothing spent, and the cap as configured (read at scrape time).

    This is what a process that binds no meter reports (``kb serve --fake``,
    where nothing is paid), so the spend-vs-cap panel and alert never divide
    by a zero cap. The real service replaces it with :func:`bind_month_budget`.
    """
    MONTH_SPEND_USD.set_function(lambda: 0.0)
    MONTH_CAP_USD.set_function(_safe(configured_month_cap))


use_configured_month_budget()


def set_month_budget(spend_usd: float, cap_usd: float) -> None:
    """Set month-to-date spend and the cap (e.g. after each metered call).

    Replaces anything bound with :func:`bind_month_budget`.
    """
    spend = _nonneg(spend_usd, "spend_usd")
    cap = _nonneg(cap_usd, "cap_usd")
    # set_function, not set(): a function bound earlier would otherwise keep winning
    MONTH_SPEND_USD.set_function(lambda: spend)
    MONTH_CAP_USD.set_function(lambda: cap)


def bind_month_budget(spend: Callable[[], float], cap: float | Callable[[], float]) -> None:
    """Read spend (and the cap) at scrape time, e.g. ``lambda: meter.month_spend("knowledge")``.

    Reading at scrape time means spend recorded by another process (a CLI
    ingest) shows up too. A failing callable reports ``NaN`` rather than
    breaking the scrape.
    """
    MONTH_SPEND_USD.set_function(_safe(spend))
    if callable(cap):
        MONTH_CAP_USD.set_function(_safe(cap))
    else:
        MONTH_CAP_USD.set_function(lambda: float(cap))


# ----------------------------------------------------------- the endpoint
#: Default process metrics (process_*, python_*), kept out of the kb registry
#: so tests reading REGISTRY see only kb_* series.
_PROCESS_REGISTRY = CollectorRegistry(auto_describe=True)
ProcessCollector(registry=_PROCESS_REGISTRY)
PlatformCollector(registry=_PROCESS_REGISTRY)
GCCollector(registry=_PROCESS_REGISTRY)

_db_collector = None


def install_db_collector(db_path: Path | str | None):
    """Register (or replace) the scrape-time reader of ``knowledge.db``.

    ``None`` removes it. Returns the installed collector.
    """
    global _db_collector
    from hal9000.knowledge.obs.collector import KnowledgeDbCollector

    if _db_collector is not None:
        REGISTRY.unregister(_db_collector)
        _db_collector = None
    if db_path is not None:
        _db_collector = KnowledgeDbCollector(db_path)
        REGISTRY.register(_db_collector)
    return _db_collector


def render_metrics(*, process: bool = True) -> bytes:
    """The Prometheus text exposition of the kb registry (plus process metrics)."""
    body = generate_latest(REGISTRY)
    if process:
        body += generate_latest(_PROCESS_REGISTRY)
    return body


def metrics_router(db_path: Path | str | None = None) -> APIRouter:
    """``GET /metrics`` (unauthenticated, LAN only, no question text: plan §14).

    With ``db_path``, the knowledge.db collector is installed too. FastAPI is
    imported here, not at module import, so the CLI doesn't pay for it.
    """
    from fastapi import APIRouter, Response

    if db_path is not None:
        install_db_collector(db_path)
    router = APIRouter()

    @router.get(
        "/metrics",
        tags=["ops"],
        response_class=Response,
        responses={
            200: {
                "description": "Prometheus exposition format",
                "content": {"text/plain": {"schema": {"type": "string"}}},
            }
        },
    )
    def metrics():  # no annotation: it would be resolved in module globals
        return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)

    return router
