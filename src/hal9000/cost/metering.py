"""Cost control at the one place every LLM call passes through.

:class:`Meter` wraps ``generate_response``. Because the solar agent, the report
narrative and the triage sub-agent all share the wrapped callable, every call is
metered -- a capability would only see the parent loop's own calls.

- **Soft cap** (per run): sets ``flags.soft_cap_hit``; the loop injects a nudge
  to deliver now.
- **Hard cap** (per run): the next call raises :class:`BudgetExceeded`; HAL
  renders the deterministic report.
- **Monthly cap**: checked by :func:`monthly_precheck` before the run starts.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from hal9000 import config
from hal9000.core.prompt import Prompt
from hal9000.storage.db import Database

__all__ = ["BudgetExceeded", "Meter", "MonthlyStatus", "monthly_precheck", "usage_cost"]


class BudgetExceeded(RuntimeError):
    """The per-run hard cap has been reached. HAL falls back to the deterministic report."""

    error_type = "BudgetExceeded"
    recoverable = False
    guidance = "The cost cap for this run has been reached."


def usage_cost(usage: dict) -> float:
    rate_in, rate_cached, rate_out = config.rates_for(usage.get("model", ""))
    prompt = int(usage.get("prompt", 0))
    cached = min(int(usage.get("cached", 0)), prompt)
    completion = int(usage.get("completion", 0))
    return ((prompt - cached) * rate_in + cached * rate_cached + completion * rate_out) / 1e6


class Meter:
    def __init__(
        self,
        db: Database,
        run_id: str,
        soft_cap: float,
        hard_cap: float,
        flags=None,
    ):
        self.db = db
        self.run_id = run_id
        self.soft_cap = soft_cap
        self.hard_cap = hard_cap
        self.flags = flags
        self.spent = 0.0
        self.calls = 0

    def record(self, usage: dict, agent: str) -> float:
        usd = usage_cost(usage)
        self.spent += usd
        self.calls += 1
        self.db.add_ledger(self.run_id, agent, usage, usd)
        if self.flags is not None and self.spent >= self.soft_cap:
            self.flags.soft_cap_hit = True
        return usd

    def wrap(self, generate_response: Callable, agent: str) -> Callable:
        accepts_usage = "on_usage" in inspect.signature(generate_response).parameters

        def metered(prompt: Prompt, tier: str = "powerful") -> str:
            if self.spent >= self.hard_cap:
                if self.flags is not None:
                    self.flags.hard_cap_hit = True
                raise BudgetExceeded(
                    f"Run cost ${self.spent:.2f} reached the ${self.hard_cap:.2f} hard cap."
                )
            if accepts_usage:
                return generate_response(
                    prompt, tier, on_usage=lambda usage: self.record(usage, agent)
                )
            return generate_response(prompt, tier)

        return metered


@dataclass(frozen=True)
class MonthlyStatus:
    spent: float
    cap: float
    skip_agent: bool
    triage_enabled: bool


def monthly_precheck(db: Database, today: date, cap: float) -> MonthlyStatus:
    spent = db.month_cost(today.year, today.month)
    return MonthlyStatus(
        spent=spent,
        cap=cap,
        skip_agent=spent >= cap,
        triage_enabled=spent < 0.8 * cap,
    )
