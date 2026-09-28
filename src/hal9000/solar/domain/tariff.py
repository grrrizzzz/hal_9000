"""Tariff engine: the marginal $/kWh for any local hour, from effective-dated components.

Marginal rate for a TOU period on a date::

    (udc + wildfire + pcia + ipca) * (1 + franchise_fee)
      + state_regulatory_fee + cea_generation[season] + clean_impact_plus

Every component is looked up by effective date, so historical days use the
rates in force then. A date with no configured component raises
:class:`TariffNotConfigured`; the engine never guesses and never returns 0 for
a missing value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from hal9000.solar.domain.sitecfg import load_yaml
from hal9000.solar.domain.tou_calendar import PERIODS, TouCalendar

__all__ = ["DayRates", "Tariff", "TariffNotConfigured", "load_tariff"]


class TariffNotConfigured(Exception):
    """A rate needed for this date is not configured. Recoverable: report without savings."""

    error_type = "TariffNotConfigured"
    recoverable = True
    guidance = (
        "Deliver the report without the savings figure and say that rates are not "
        "configured for this date. Do not estimate savings."
    )


@dataclass(frozen=True)
class DayRates:
    """Marginal $/kWh per TOU period for one date, plus provenance."""

    day: date
    season: str
    rates: dict[str, float]
    schedule_version: str
    provisional: bool


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _in_force(entries: list[dict], day: date, name: str) -> dict:
    """The last entry whose ``from`` is on or before ``day``."""
    chosen = None
    for entry in sorted(entries or [], key=lambda e: _as_date(e["from"])):
        if _as_date(entry["from"]) <= day:
            chosen = entry
    if chosen is None:
        raise TariffNotConfigured(
            f"No {name} rate is configured for {day.isoformat()} "
            f"(earliest configured date is later). Add it to config/tariff.yaml."
        )
    return chosen


def _required(mapping: dict, key: str, name: str, day: date) -> float:
    value = mapping.get(key) if mapping else None
    if value is None:
        raise TariffNotConfigured(
            f"The {name} rate for {key} is not configured for {day.isoformat()}."
        )
    return float(value)


class Tariff:
    """Marginal-rate lookup built from the ``tariff.yaml`` structure."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.calendar = TouCalendar(cfg)
        self.components = cfg.get("components", {})
        self.adders = cfg.get("per_kwh_adders", {})
        self.franchise_fee = float(cfg.get("franchise_fee", 0.0))
        confirmed = cfg.get("confirmed_through")
        self.confirmed_through = _as_date(confirmed) if confirmed else None

    def day_rates(self, day: date) -> DayRates:
        season = self.calendar.season(day)
        udc = _in_force(self.components.get("udc"), day, "SDG&E delivery (UDC)")
        pcia = _in_force(self.components.get("pcia"), day, "PCIA")
        gen = _in_force(self.components.get("cea_generation"), day, "CEA generation")
        wildfire = _required(self.adders, "wildfire_fund", "wildfire fund", day)
        ipca = _required(self.adders, "ipca", "IPCA", day)
        state_fee = _required(self.adders, "state_regulatory_fee", "state regulatory fee", day)
        cip = float(self.adders.get("clean_impact_plus", 0.0) or 0.0)
        pcia_value = _required(pcia, "value", "PCIA", day)

        rates = {}
        for period in PERIODS:
            delivery = _required(udc, period, "UDC", day)
            generation = _required(gen.get(season, {}), period, f"CEA {season} generation", day)
            sdge_part = (delivery + wildfire + pcia_value + ipca) * (1 + self.franchise_fee)
            rates[period] = round(sdge_part + state_fee + generation + cip, 5)

        provisional = bool(
            self.confirmed_through and day > self.confirmed_through and season == "winter"
        )
        version = f"udc@{udc['from']}|pcia@{pcia['from']}|cea@{gen['from']}|{season}" + (
            "|provisional" if provisional else ""
        )
        return DayRates(
            day=day, season=season, rates=rates, schedule_version=version, provisional=provisional
        )

    def rate(self, day: date, hour: int) -> float:
        return self.day_rates(day).rates[self.calendar.period(day, hour)]


def load_tariff(config_dir: Path | None = None) -> Tariff:
    return Tariff(load_yaml("tariff.yaml", config_dir))
