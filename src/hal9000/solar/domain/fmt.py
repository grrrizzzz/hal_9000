"""Display formatting: every figure the household may hear is formatted here, once.

The numeric guard compares the narrative against these exact strings, so a
figure has one spoken form: kWh to one decimal, dollars to cents, percentages as
whole numbers.
"""

from __future__ import annotations

__all__ = ["dollars", "kwh", "percent", "plain"]


def kwh(value: float) -> str:
    return f"{value:.1f}"


def dollars(value: float) -> str:
    return f"${value:,.2f}"


def percent(value: float) -> str:
    """``value`` is a fraction (0.74 -> "74%")."""
    return f"{round(value * 100):d}%"


def plain(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"
