"""Everything one evening run shares: the date, the stores, the clients, the flags.

Tools receive it as ``_run`` (hidden from the model). The triage sub-agent shares
the same object for the run store and the date, but never gets delivery or
record-writing tools.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from hal9000.config import HalFeatures, Settings
from hal9000.solar.clients.homeassistant import HAProtocol
from hal9000.solar.clients.openmeteo import WeatherProtocol
from hal9000.solar.domain.sitecfg import Site, Thresholds
from hal9000.solar.domain.tariff import Tariff
from hal9000.storage.db import Database
from hal9000.storage.run_store import RunStore

__all__ = ["RunContext", "RunFlags"]


@dataclass
class RunFlags:
    delivered: bool = False
    renderer: str | None = None
    report_text: str | None = None
    guard_failures: int = 0  # numeric-guard rejections
    completeness_bounces: int = 0
    length_bounces: int = 0
    record_written: bool = False
    telegram_sent: int = 0
    triage_ran: bool = False
    verdict: dict | None = None
    alerts_raised: list[str] = field(default_factory=list)
    alerts_cleared: list[str] = field(default_factory=list)
    soft_cap_hit: bool = False
    hard_cap_hit: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    production_missing_reason: str | None = None
    weather_missing_reason: str | None = None
    savings_absent_reason: str | None = None


@dataclass
class RunContext:
    day: date
    now: datetime
    run_id: str
    settings: Settings
    site: Site
    thresholds: Thresholds
    tariff: Tariff
    db: Database
    ha: HAProtocol | None
    weather: WeatherProtocol | None
    delivery: Any  # coordinator.delivery.Delivery
    features: HalFeatures
    store: RunStore = field(default_factory=RunStore)
    flags: RunFlags = field(default_factory=RunFlags)
    trace: list[tuple[str, str]] = field(default_factory=list)
    triage_runner: Callable[[str], dict] | None = None
    report_policy: str = ""

    def note(self, step: str, outcome: str) -> None:
        self.trace.append((step, outcome))

    def degrade(self, reason: str) -> None:
        if reason not in self.flags.degraded_reasons:
            self.flags.degraded_reasons.append(reason)
