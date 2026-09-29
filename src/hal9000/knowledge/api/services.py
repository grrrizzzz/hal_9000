"""Everything the API's handlers use, built once per process (``kb serve``, tests).

:func:`build_services` wires the real service: the metered answer pipeline,
the ingestion pipeline and its one job worker (``pipeline.worker(db_path,
locked=True)``, so a ``kb ingest`` only queues jobs while the service runs),
the post-ingest regression hook, the passcode and session signer, the
month's spend on ``/metrics``. With ``fake=True`` (``kb serve --fake``)
nothing is paid: the fake chat model, the fake embedder and heuristic
segmentation.

It also writes ``data/knowledge_health.json`` (plan §16): the last successful
question, the last finished ingestion and the month's spend, for HAL's
watchdog.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hal9000.knowledge.answer.pipeline import AnswerPipeline
from hal9000.knowledge.auth import Auth, RateLimiter, SessionStore, load_secret
from hal9000.knowledge.config import AGENT, KnowledgeFile, KnowledgeSettings, load_knowledge_file

__all__ = ["HEALTH_FILE", "Services", "build_services"]

log = logging.getLogger(__name__)

HEALTH_FILE = "knowledge_health.json"


@dataclass
class Services:
    settings: KnowledgeSettings
    auth: Auth
    answerer: AnswerPipeline
    ingest: Any | None = None  # IngestPipeline
    worker: Any | None = None  # JobWorker
    meter: Any | None = None  # Meter
    bus: Any | None = None  # ProgressBus
    fake: bool = False
    chat_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(20, 60.0))
    sessions: SessionStore | None = None

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = SessionStore(self.settings.db_path)

    def connect(self) -> sqlite3.Connection:
        from hal9000.knowledge.storage import connect

        return connect(self.settings.db_path)

    def knowledge_file(self) -> KnowledgeFile:
        return load_knowledge_file(self.settings.yaml_path)

    def max_upload_bytes(self) -> int:
        """``ingest.max_upload_bytes``: the ingest pipeline's limit, else the yaml's."""
        if self.ingest is not None and getattr(self.ingest, "knowledge_file", None) is not None:
            return int(self.ingest.max_bytes)
        return self.knowledge_file().ingest.max_upload_bytes

    def month_spend(self) -> float | None:
        if self.meter is None:
            return 0.0 if self.fake else None
        try:
            return float(self.meter.month_spend(AGENT))
        except Exception:  # pragma: no cover - the ledger is unreadable
            log.exception("could not read the month's spend")
            return None

    def retrieval_only(self) -> bool:
        spend = self.month_spend()
        return spend is not None and spend >= self.settings.monthly_cap_usd

    def notify_worker(self) -> None:
        if self.worker is not None:
            self.worker.notify()

    def write_health(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """``data/knowledge_health.json`` (plan §16); returns what was written."""
        question = conn.execute(
            "SELECT MAX(created_at) FROM traces WHERE kind = 'question' AND outcome = 'answered'"
        ).fetchone()[0]
        ingest = conn.execute(
            "SELECT MAX(updated_at) FROM jobs WHERE kind = 'ingest' AND status = 'done'"
        ).fetchone()[0]
        body = {
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "last_successful_question_at": question,
            "last_ingestion_at": ingest,
            "month_spend_usd": self.month_spend(),
            "month_cap_usd": self.settings.monthly_cap_usd,
        }
        target = self.settings.data_dir / HEALTH_FILE
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(body, indent=2) + "\n")
            tmp.replace(target)
        except OSError:  # pragma: no cover - a read-only data dir must not break /health
            log.exception("could not write %s", target)
        return body

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop(timeout=10)
        if self.ingest is not None:
            self.ingest.close()


def _meter(settings: KnowledgeSettings):
    from hal9000 import config as hal_config
    from hal9000.cost.metering import Meter
    from hal9000.knowledge.config import monthly_caps
    from hal9000.storage.db import Database

    hal = hal_config.settings()
    return Meter(
        Database(Path(hal.data_dir) / "hal9000.db"),
        f"kb-serve-{uuid.uuid4().hex[:8]}",
        soft_cap=math.inf,
        hard_cap=math.inf,
        tz=hal.tz,
        monthly_caps=monthly_caps(settings),
    )


def build_services(
    settings: KnowledgeSettings,
    *,
    fake: bool = False,
    meter: Any | None = None,
    start_worker: bool = True,
    answer_deps: Any | None = None,
    ingest: Any | None = None,
    chat_limit: int | None = None,
) -> Services:
    """Build (and, unless told not to, start) the service's parts."""
    from hal9000.knowledge.answer.pipeline import AnswerDeps
    from hal9000.knowledge.eval.regression import ingest_hook
    from hal9000.knowledge.ingest.jobs import BUS
    from hal9000.knowledge.ingest.pipeline import IngestPipeline, no_regression
    from hal9000.knowledge.obs import metrics, traces
    from hal9000.knowledge.storage import open_db

    open_db(settings.db_path).close()  # migrated before anything else opens it
    traces.configure_retention(settings.yaml_path)  # trace_retention_days
    kf = load_knowledge_file(settings.yaml_path)
    auth = Auth(
        settings.require_passcode(), load_secret(settings.data_dir / "knowledge" / "session.key")
    )
    if fake:
        from hal9000.knowledge.answer.fake import fake_deps
        from hal9000.knowledge.index.fake import FakeEmbedder, enable_fake_mode

        enable_fake_mode()
        # nothing is paid: spend 0 against this service's cap (never a 0/0 on the dashboard)
        metrics.set_month_budget(0.0, settings.monthly_cap_usd)
        deps = answer_deps or fake_deps()
        if ingest is None:
            ingest = IngestPipeline(
                settings.files_dir,
                embed_fn=FakeEmbedder(),
                use_llm=False,
                bus=BUS,
                knowledge_file=kf,
            )
    else:
        from hal9000 import llm

        meter = meter if meter is not None else _meter(settings)
        deps = answer_deps or AnswerDeps(meter=meter)
        if ingest is None:
            ingest = IngestPipeline(
                settings.files_dir,
                meter=meter,
                generate=meter.wrap(llm.generate_response, AGENT),
                bus=BUS,
                knowledge_file=kf,
            )
        metrics.bind_month_budget(lambda: meter.month_spend(AGENT), settings.monthly_cap_usd)
    if ingest is not None and ingest.regression is no_regression:
        # the post-ingest regression (plan §9.2), metered as knowledge-eval
        ingest.regression = ingest_hook(settings, meter=meter)
    worker = ingest.worker(settings.db_path, locked=True) if ingest is not None else None
    if worker is not None and start_worker:
        worker.start()
    return Services(
        settings=settings,
        auth=auth,
        answerer=AnswerPipeline(settings, deps),
        ingest=ingest,
        worker=worker,
        meter=meter,
        bus=BUS,
        fake=fake,
        chat_limiter=RateLimiter(chat_limit or kf.answer.questions_per_minute, 60.0),
    )
