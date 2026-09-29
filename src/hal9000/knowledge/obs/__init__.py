"""Knowledge-base observability (plan §9).

- ``timing``: the stage enum and the ``stage()`` timer (KWP0).
- ``metrics``: every plan §9.2 metric, the recording helpers and the ``/metrics`` router.
- ``collector``: gauges read from ``knowledge.db`` at scrape time.
- ``traces``: ``observe()`` / ``finish_trace()``, trace rows and their 180-day retention.
- ``otel``: optional OpenTelemetry export, on when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

Nothing is imported here, so ``import hal9000.knowledge.obs.timing`` stays light.
"""
