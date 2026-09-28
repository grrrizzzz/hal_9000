"""Expected production from the day's actual weather.

Two models, same interface (hourly plane-of-array insolation and air
temperature in, hourly kWh out):

- **Bootstrap** -- physics with a fixed performance ratio and degradation
  factor. Used until a fitted model exists, and after any database loss.
- **Fitted** -- ridge regression on this site's own history,
  ``kWh = a*G + b*G*(Tcell - 25)``, no intercept. It learns the array's real
  output per unit of irradiance, so ageing reads as the new normal instead of a
  permanent fault.

Never nameplate: a nameplate comparison would read 11 years of degradation as a
fault every day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from hal9000.solar.domain.sitecfg import Site

__all__ = [
    "ExpectationModel",
    "bootstrap_model",
    "cell_temperature",
    "fit_model",
    "MIN_FIT_DAYS",
]

MIN_FIT_DAYS = 90
TEMP_COEFF = -0.004  # per degC, typical crystalline silicon


def cell_temperature(air_c: float, poa_kwh_m2: float) -> float:
    """Rough cell temperature: ~25 degC above ambient at 1 kW/m2."""
    return air_c + 25.0 * poa_kwh_m2


@dataclass
class ExpectationModel:
    kind: str  # "bootstrap" | "fitted"
    a: float  # kWh per (kWh/m2)
    b: float  # temperature term
    n_days: int = 0
    calibrated_on: str | None = None
    notes: dict = field(default_factory=dict)

    def predict_hourly(self, poa_kwh_m2: list[float], temp_c: list[float]) -> list[float]:
        out = []
        for g, t in zip(poa_kwh_m2, temp_c):
            g = max(g, 0.0)
            tc = cell_temperature(t if t is not None else 20.0, g)
            out.append(max(self.a * g + self.b * g * (tc - 25.0), 0.0))
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "a": self.a,
            "b": self.b,
            "n_days": self.n_days,
            "calibrated_on": self.calibrated_on,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExpectationModel:
        return cls(
            kind=data["kind"],
            a=float(data["a"]),
            b=float(data["b"]),
            n_days=int(data.get("n_days", 0)),
            calibrated_on=data.get("calibrated_on"),
            notes=data.get("notes", {}),
        )


def bootstrap_model(site: Site) -> ExpectationModel:
    a = site.dc_kw * site.bootstrap_performance_ratio * site.degradation_factor
    return ExpectationModel(kind="bootstrap", a=a, b=a * TEMP_COEFF)


def fit_model(
    days: list[tuple[date, list[float], list[float], list[float]]],
    site: Site,
    ridge_lambda: float = 1e-3,
    min_days: int = MIN_FIT_DAYS,
    today: date | None = None,
) -> ExpectationModel | None:
    """Fit on ``(day, poa_kwh_m2[24], temp_c[24], actual_kwh[24])`` tuples.

    Hours with little light, and hours far below the bootstrap expectation
    (outages, sensor gaps), are excluded so faults don't teach the model to
    expect less. Returns ``None`` when fewer than ``min_days`` usable days exist.
    """
    boot = bootstrap_model(site)
    rows, targets, used_days = [], [], set()
    for day, poa, temp, actual in days:
        expected = boot.predict_hourly(poa, temp)
        if sum(actual) <= 0.05 * max(sum(expected), 1e-9):
            continue  # an outage day, not information about the array
        for g, t, y, e in zip(poa, temp, actual, expected):
            if g < 0.05 or y is None or e <= 0:
                continue
            if y < 0.3 * e or y > 2.0 * e:
                continue
            tc = cell_temperature(t if t is not None else 20.0, g)
            rows.append([g, g * (tc - 25.0)])
            targets.append(y)
            used_days.add(day)
    if len(used_days) < min_days:
        return None
    x = np.asarray(rows)
    y = np.asarray(targets)
    xtx = x.T @ x + ridge_lambda * len(y) * np.eye(2)
    coef = np.linalg.solve(xtx, x.T @ y)
    residual = y - x @ coef
    return ExpectationModel(
        kind="fitted",
        a=float(coef[0]),
        b=float(coef[1]),
        n_days=len(used_days),
        calibrated_on=(today or max(used_days)).isoformat(),
        notes={"rmse_kwh_per_hour": float(np.sqrt(np.mean(residual**2))), "n_hours": len(y)},
    )
