"""Optional OpenTelemetry export of traces (plan §9.1 output 3).

Off unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (the ``otel`` dependency
group must then be installed). Each finished
:class:`~hal9000.knowledge.contracts.Trace` becomes one root span
(``kb.question`` or ``kb.ingest``) with a child span per recorded stage, named
exactly like the stage (``vector``, ``generate``, ...); nested stages nest.

Spans are built after the fact from the trace's stage timings, so the stage
timer stays one code path and OpenTelemetry is never imported unless enabled.
Spans carry IDs, outcome, profile, sizes and cost, never question text.

``OTEL_EXPORTER_OTLP_PROTOCOL=grpc`` selects the gRPC exporter; anything else
uses OTLP over HTTP (``http/protobuf``, the specification's default).
``OTEL_SERVICE_NAME`` overrides the service name ``hal-kb``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from hal9000.knowledge.contracts import StageTiming, Trace

__all__ = ["ENDPOINT_ENV", "configure", "enabled", "export_trace", "reset"]

log = logging.getLogger(__name__)

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
_SCOPE = "hal9000.knowledge"

_lock = threading.Lock()
_configured = False
_provider: Any = None


def _from_environment() -> Any:
    endpoint = os.environ.get(ENDPOINT_ENV, "").strip()
    if not endpoint:
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip() == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        log.warning(
            "%s is set but OpenTelemetry is not installed (uv sync --group otel); "
            "spans are not exported.",
            ENDPOINT_ENV,
        )
        return None
    service = os.environ.get("OTEL_SERVICE_NAME", "").strip() or "hal-kb"
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


def configure(provider: Any = None) -> bool:
    """Use ``provider`` (a ``TracerProvider``), or build one from the environment.

    Returns whether export is enabled. Called lazily by :func:`export_trace`;
    tests pass a provider with an in-memory exporter.
    """
    global _configured, _provider
    with _lock:
        _provider = provider if provider is not None else _from_environment()
        _configured = True
        return _provider is not None


def reset() -> None:
    """Forget the provider (it is not shut down); the next export re-reads the environment."""
    global _configured, _provider
    with _lock:
        _configured = False
        _provider = None


def enabled() -> bool:
    if not _configured:
        configure()
    return _provider is not None


def _ns(trace: Trace) -> int:
    return int(trace.created_at.timestamp() * 1_000_000_000)


def _attributes(trace: Trace) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "kb.trace_id": trace.id,
        "kb.kind": trace.kind,
        "kb.context_tokens": trace.context_tokens,
        "kb.cost_usd": trace.cost_usd,
        "kb.hits": len(trace.hits),
    }
    for key in ("outcome", "profile", "grounding", "ttft_ms", "total_ms", "error"):
        value = getattr(trace, key)
        if value is not None:
            attrs[f"kb.{key}"] = value
    return attrs


def _ordered(stages: list[StageTiming]) -> list[StageTiming]:
    # Parents start no later than their children and are one level shallower.
    return sorted(stages, key=lambda t: (t.started_ms, t.depth, -t.duration_ms))


def export_trace(trace: Trace) -> bool:
    """Export ``trace`` as spans if enabled. Never raises; returns whether it exported."""
    if not enabled():
        return False
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import Status, StatusCode

        tracer = _provider.get_tracer(_SCOPE)
        start = _ns(trace)
        stage_end = max(
            (int((t.started_ms + t.duration_ms) * 1e6) for t in trace.stages), default=0
        )
        total = int(trace.total_ms * 1e6) if trace.total_ms is not None else stage_end
        root = tracer.start_span(
            f"kb.{trace.kind}",
            start_time=start,
            attributes=_attributes(trace),
            context=otel_trace.set_span_in_context(otel_trace.INVALID_SPAN),
        )
        if trace.error:
            root.set_status(Status(StatusCode.ERROR, trace.error))
        # (span, depth, end_ns) of the stages still able to hold children
        open_spans: list[tuple[Any, int, int]] = []
        for timing in _ordered(trace.stages):
            begin = start + int(timing.started_ms * 1e6)
            end = begin + int(timing.duration_ms * 1e6)
            while open_spans and (open_spans[-1][1] >= timing.depth or open_spans[-1][2] < begin):
                open_spans.pop()
            parent = open_spans[-1][0] if open_spans else root
            span = tracer.start_span(
                timing.stage.value,
                start_time=begin,
                context=otel_trace.set_span_in_context(parent),
                attributes={"kb.stage": timing.stage.value, "kb.depth": timing.depth},
            )
            if timing.error:
                span.set_status(Status(StatusCode.ERROR, timing.error))
            span.end(end_time=end)
            open_spans.append((span, timing.depth, end))
        root.end(end_time=start + max(total, stage_end))
        return True
    except Exception:  # pragma: no cover - export must never break a request
        log.exception("OpenTelemetry export failed")
        return False
