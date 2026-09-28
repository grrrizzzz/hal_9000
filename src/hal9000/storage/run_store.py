"""The run store: every figure a run may speak, and the data behind it.

Tools write here; the numeric guard reads here. A figure is stored once, already
formatted the way it will be spoken ("8.2", "$1.37", "74%"), so the narrative can
only pass the guard by quoting a tool's own output.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from hal9000.solar.domain.numerals import extract_numerals

__all__ = ["RunStore"]


class RunStore:
    def __init__(self) -> None:
        self._figures: dict[str, str] = {}
        self._data: dict[str, Any] = {}

    # figures -----------------------------------------------------------
    def put(self, key: str, display: str) -> str:
        self._figures[key] = display
        return display

    def figure(self, key: str) -> str | None:
        return self._figures.get(key)

    def figures(self, prefix: str = "") -> dict[str, str]:
        return {k: v for k, v in self._figures.items() if k.startswith(prefix)}

    def has(self, key: str) -> bool:
        return key in self._figures

    def numeric_values(self) -> set[tuple[str, Decimal]]:
        """Every ``(unit_class, value)`` in any stored figure.

        Unit classes: ``money`` ("$1.37"), ``percent`` ("74%"), ``number``
        (kWh and every other plain figure).
        """
        values: set[tuple[str, Decimal]] = set()
        for display in self._figures.values():
            for numeral in extract_numerals(display):
                if numeral.unit in ("money", "percent"):
                    values.add((numeral.unit, numeral.value))
                else:
                    values.add(("number", numeral.value))
        return values

    # data --------------------------------------------------------------
    def put_data(self, key: str, value: Any) -> None:
        self._data[key] = value

    def data(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def snapshot(self) -> dict:
        return {"figures": dict(self._figures), "data_keys": sorted(self._data)}
