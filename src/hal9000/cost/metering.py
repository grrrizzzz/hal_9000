"""Cost control at the one place every LLM call passes through.

:class:`Meter` wraps ``generate_response``. Because the solar agent, the report
narrative and the triage sub-agent all share the wrapped callable, every call is
metered -- a capability would only see the parent loop's own calls.

- **Soft cap** (per run): sets ``flags.soft_cap_hit``; the loop injects a nudge
  to deliver now.
- **Hard cap** (per run): the next call raises :class:`BudgetExceeded`; HAL
  renders the deterministic report.
- **Monthly cap**: checked by :func:`monthly_precheck` before the run starts.

The knowledge base (plan §13) adds:

- :meth:`Meter.stream` and :meth:`Meter.embed`, the metered forms of
  :func:`hal9000.llm.stream_response` and :func:`hal9000.llm.embed`. Each checks
  the caps *before* calling out and writes a ledger row after.
- **Per-agent monthly caps** (``monthly_caps``): an agent listed there is
  refused with :class:`MonthlyCapExceeded` once *its own* month-to-date spend
  reaches its cap. ``knowledge`` and ``knowledge-eval`` are capped separately,
  so evaluation spend never pushes the chat into retrieval-only mode. The
  solar agents are not listed, so their behaviour is unchanged, and
  :func:`monthly_precheck` (the solar cap) leaves the knowledge agents' spend
  out, so the knowledge base can't switch the evening report off either.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from hal9000 import config
from hal9000.core.prompt import Prompt
from hal9000.storage.db import Database

__all__ = [
    "KNOWLEDGE_AGENTS",
    "BudgetExceeded",
    "Meter",
    "MonthlyCapExceeded",
    "MonthlyStatus",
    "monthly_precheck",
    "usage_cost",
]

#: Ledger agents with caps of their own (plan §13); left out of the solar cap.
KNOWLEDGE_AGENTS: tuple[str, ...] = ("knowledge", "knowledge-eval")


class BudgetExceeded(RuntimeError):
    """The per-run hard cap has been reached. HAL falls back to the deterministic report."""

    error_type = "BudgetExceeded"
    recoverable = False
    guidance = "The cost cap for this run has been reached."


class MonthlyCapExceeded(BudgetExceeded):
    """An agent's own monthly cap has been reached (the knowledge chat goes retrieval-only)."""

    error_type = "MonthlyCapExceeded"
    guidance = "The monthly cost cap for this agent has been reached."


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
        tz=None,
        monthly_caps: Mapping[str, float] | None = None,
    ):
        self.monthly_caps: dict[str, float] = dict(monthly_caps or {})
        self.tz = tz  # ledger rows are stamped in local time so months match the bill
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
        at = datetime.now(self.tz).isoformat(timespec="seconds") if self.tz else None
        self.db.add_ledger(self.run_id, agent, usage, usd, at=at)
        if self.flags is not None and self.spent >= self.soft_cap:
            self.flags.soft_cap_hit = True
        return usd

    def wrap(self, generate_response: Callable, agent: str) -> Callable:
        accepts_usage = "on_usage" in inspect.signature(generate_response).parameters

        def metered(prompt: Prompt, tier: str = "powerful") -> str:
            self._check_run_cap()
            self._check_monthly_cap(agent)
            if accepts_usage:
                return generate_response(
                    prompt, tier, on_usage=lambda usage: self.record(usage, agent)
                )
            return generate_response(prompt, tier)

        return metered

    # ------------------------------------------------------------- caps
    def _check_run_cap(self) -> None:
        if self.spent >= self.hard_cap:
            if self.flags is not None:
                self.flags.hard_cap_hit = True
            raise BudgetExceeded(
                f"Run cost ${self.spent:.2f} reached the ${self.hard_cap:.2f} hard cap."
            )

    def month_spend(self, agent: str) -> float:
        """``agent``'s own month-to-date spend, in the month the ledger stamps rows with."""
        now = datetime.now(self.tz) if self.tz else datetime.now(UTC)
        return self.db.month_cost_for(now.year, now.month, agents=[agent])

    def _check_monthly_cap(self, agent: str) -> None:
        cap = self.monthly_caps.get(agent)
        if cap is None:
            return
        spent = self.month_spend(agent)
        if spent >= cap:
            raise MonthlyCapExceeded(
                f"{agent} has spent ${spent:.2f} this month, reaching its ${cap:.2f} cap."
            )

    def precheck(self, agent: str) -> None:
        """Raise :class:`BudgetExceeded` if a call for ``agent`` must not go out now."""
        self._check_run_cap()
        self._check_monthly_cap(agent)

    # ------------------------------------------------- knowledge base calls
    def stream(
        self,
        stream_response: Callable[..., Iterator[Any]],
        prompt: Prompt,
        tier: str = "fast",
        *,
        agent: str,
        **kw: Any,
    ) -> Iterator[Any]:
        """Metered :func:`hal9000.llm.stream_response`.

        The caps are checked here, when ``stream`` is called, so an over-cap
        request raises before any connection is made -- not on the first
        ``next()``. The ledger row is written when the final ``Usage`` arrives.
        """
        self.precheck(agent)
        return stream_response(prompt, tier, on_usage=lambda usage: self.record(usage, agent), **kw)

    def embed(
        self,
        embed: Callable[..., Any],
        texts: Sequence[str],
        model: str,
        dimensions: int | None = None,
        *,
        agent: str,
        **kw: Any,
    ) -> Any:
        """Metered :func:`hal9000.llm.embed`: caps first, then one ledger row per request."""
        self.precheck(agent)
        return embed(
            texts, model, dimensions, on_usage=lambda usage: self.record(usage, agent), **kw
        )


@dataclass(frozen=True)
class MonthlyStatus:
    spent: float
    cap: float
    skip_agent: bool
    triage_enabled: bool


def monthly_precheck(db: Database, today: date, cap: float) -> MonthlyStatus:
    """The solar run's monthly cap. Knowledge agents have caps of their own and don't count."""
    spent = db.month_cost_for(today.year, today.month, exclude=KNOWLEDGE_AGENTS)
    return MonthlyStatus(
        spent=spent,
        cap=cap,
        skip_agent=spent >= cap,
        triage_enabled=spent < 0.8 * cap,
    )
