"""Money for evaluation commands (rule K9, plan §13).

Every paid evaluation command prints an estimate first, refuses to start
when the estimate exceeds its ``--max-cost`` or what is left of this month's
``KB_EVAL_CAP_USD``, asks for confirmation unless ``--yes``, and runs its
LLM calls through a :class:`~hal9000.cost.metering.Meter` whose hard cap is
``--max-cost`` and whose monthly cap for ``knowledge-eval`` is
``KB_EVAL_CAP_USD``: the run stops at either cap and keeps partial results.
"""

from __future__ import annotations

import math
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO

from hal9000 import config as hal_config
from hal9000.cost.metering import Meter
from hal9000.knowledge.config import EVAL_AGENT, KnowledgeSettings, monthly_caps
from hal9000.knowledge.eval.types import Generate

if TYPE_CHECKING:  # pragma: no cover
    from hal9000.knowledge.eval.checkpoint import Guard

__all__ = ["EvalSpend", "confirm", "open_spend", "suggest_max_cost"]


def suggest_max_cost(estimate_usd: float) -> float:
    """A ``--max-cost`` with headroom over an estimate: +25%, rounded up to the next $0.50."""
    return max(0.5, math.ceil(estimate_usd * 1.25 * 2 - 1e-9) / 2)


@dataclass
class EvalSpend:
    meter: Meter
    #: the command's --max-cost; None for commands without one (the eval cap bounds them)
    max_cost: float | None
    eval_cap: float

    @property
    def month_spent(self) -> float:
        return self.meter.month_spend(EVAL_AGENT)

    @property
    def remaining(self) -> float:
        return max(0.0, self.eval_cap - self.month_spent)

    def refusal(self, estimate_usd: float) -> str | None:
        """Why a run estimated at ``estimate_usd`` must not start, or None."""
        if self.max_cost is not None and estimate_usd > self.max_cost:
            return (
                f"the estimate ${estimate_usd:.2f} exceeds --max-cost ${self.max_cost:.2f}; "
                f"to run it, pass --max-cost {suggest_max_cost(estimate_usd):.2f}"
            )
        if estimate_usd > self.remaining:
            return (
                f"the estimate ${estimate_usd:.2f} exceeds what is left of KB_EVAL_CAP_USD "
                f"this month (${self.remaining:.2f} of ${self.eval_cap:.2f})"
            )
        return None

    def generate(self, generate_response: Callable | None = None) -> Generate:
        """``generate_response`` metered as ``knowledge-eval`` (needs an LLM key)."""
        if generate_response is None:
            hal_config.require_llm_key()
            from hal9000.llm import generate_response
        return self.meter.wrap(generate_response, EVAL_AGENT)

    def guard_records(self, guard: Guard) -> None:
        """Write every ledger row inside ``guard.shield()``, then ``guard.hold()``.

        A signal can't cut a ledger row in half, and one arriving after the row
        is written waits until the reply it paid for is saved.
        """
        record = self.meter.record

        def guarded(usage: dict, agent: str) -> float:
            with guard.shield():
                usd = record(usage, agent)
                guard.hold()
            return usd

        self.meter.record = guarded  # type: ignore[method-assign]

    def close(self) -> None:
        self.meter.db.close()


def open_spend(settings: KnowledgeSettings, *, max_cost: float | None, label: str) -> EvalSpend:
    """A meter for one evaluation command, writing to the shared ledger in ``hal9000.db``.

    The run's hard cap is ``max_cost``, or ``KB_EVAL_CAP_USD`` when None.
    """
    from hal9000.storage.db import Database

    hal = hal_config.settings()
    database = Database(hal.data_dir / "hal9000.db")
    meter = Meter(
        database,
        f"kb-eval-{label}-{uuid.uuid4().hex[:8]}",
        soft_cap=settings.eval_cap_usd if max_cost is None else max_cost,
        hard_cap=settings.eval_cap_usd if max_cost is None else max_cost,
        tz=hal.tz,
        monthly_caps=monthly_caps(settings),
    )
    return EvalSpend(meter, max_cost, settings.eval_cap_usd)


def confirm(
    question: str, *, yes: bool, stdin: TextIO | None = None, out: TextIO | None = None
) -> bool:
    """``--yes``, or an interactive "y"; a non-interactive stdin without ``--yes`` is a no."""
    if yes:
        return True
    stdin = stdin or sys.stdin
    out = out or sys.stdout
    if not stdin.isatty():
        print("Not confirmed: pass --yes to run without a prompt.", file=out)
        return False
    print(f"{question} [y/N] ", end="", file=out, flush=True)
    return stdin.readline().strip().lower() in ("y", "yes")
