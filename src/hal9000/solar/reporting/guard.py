"""The numeric guard and the completeness check for anything the household hears.

``find_unbacked_numerals`` returns every number in a text that does not appear,
exactly, among the figures tools stored for this run. Matching is strict: a
rounded or derived number fails, because a guard with fuzzy matching is a guard
with a hole in it.

Clock times ("8 pm", "4:00") and dates ("September 22, 2026") are not
quantities: they are skipped by the extractor and never back anything. Beyond
the stored figures, only 0 and 1 are always allowed ("one of the best days").
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from hal9000.solar.domain.numerals import extract_numerals
from hal9000.storage.run_store import RunStore

__all__ = ["allowed_values", "completeness_problems", "find_unbacked_numerals"]

_ALWAYS_ALLOWED = {Decimal(0), Decimal(1)}


def allowed_values(store: RunStore, day: date) -> set[Decimal]:
    values = set(store.numeric_values())
    values |= _ALWAYS_ALLOWED
    return values


def find_unbacked_numerals(text: str, store: RunStore, day: date) -> list[str]:
    allowed = allowed_values(store, day)
    return [n.text for n in extract_numerals(text) if n.value not in allowed]


def completeness_problems(store: RunStore, flags) -> list[str]:
    """What a report must contain, or explain the absence of."""
    problems = []
    if not store.has("production_kwh") and not flags.production_missing_reason:
        problems.append("production has not been fetched (call get_home_energy)")
    band = store.data("band")
    if band is None:
        problems.append("performance has not been assessed (call assess_performance)")
    if not store.has("savings_usd") and not flags.savings_absent_reason:
        problems.append("savings have not been computed (call compute_savings)")
    return problems
