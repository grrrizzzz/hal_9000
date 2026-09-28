"""Performance bands and the hourly anomaly profile."""

from __future__ import annotations

from dataclasses import dataclass

from hal9000.solar.domain.sitecfg import Thresholds

__all__ = ["AnomalyProfile", "band_for", "classify_profile"]

NORMAL = "normal"
LOW = "low"
SEVERE = "severe"
ZERO = "zero"


def band_for(actual_kwh: float, expected_kwh: float, thresholds: Thresholds) -> tuple[str, float]:
    """``(band, ratio)``. Bands: normal, low, severe, zero (see thresholds.yaml)."""
    if expected_kwh < thresholds.min_expected_kwh:
        return ("unassessed", 0.0)
    ratio = actual_kwh / expected_kwh
    if ratio <= thresholds.zero_max:
        return (ZERO, ratio)
    if ratio < thresholds.low_min:
        return (SEVERE, ratio)
    if ratio < thresholds.normal_min:
        return (LOW, ratio)
    return (NORMAL, ratio)


@dataclass(frozen=True)
class AnomalyProfile:
    kind: str  # normal | flat_zero | late_start | early_cutoff | midday_dropout | uniform_shortfall
    description: str
    worst_hour: int | None
    hourly_ratio: list[float | None]


def _productive(actual: float, expected: float) -> bool:
    return expected > 0.02 and actual >= 0.15 * expected


def classify_profile(actual: list[float], expected: list[float]) -> AnomalyProfile:
    """Where in the day actual production departs from expected."""
    ratios: list[float | None] = [(a / e) if e > 0.02 else None for a, e in zip(actual, expected)]
    daylight = [h for h, e in enumerate(expected) if e > 0.02]
    if not daylight:
        return AnomalyProfile("normal", "No meaningful sunlight expected.", None, ratios)
    total_a = sum(actual[h] for h in daylight)
    total_e = sum(expected[h] for h in daylight)
    if total_e > 0 and total_a <= 0.05 * total_e:
        return AnomalyProfile(
            "flat_zero",
            "Production stayed at or near zero through daylight hours.",
            None,
            ratios,
        )
    productive = [h for h in daylight if _productive(actual[h], expected[h])]
    worst = min(daylight, key=lambda h: ratios[h] if ratios[h] is not None else 9)
    strong = [h for h in daylight if expected[h] >= 0.25 * max(expected)]
    if productive:
        if productive[0] - daylight[0] >= 2 and any(
            h in strong for h in range(daylight[0], productive[0])
        ):
            return AnomalyProfile(
                "late_start",
                f"Production started late, around {productive[0]}:00.",
                productive[0],
                ratios,
            )
        if daylight[-1] - productive[-1] >= 2 and any(
            h in strong for h in range(productive[-1] + 1, daylight[-1] + 1)
        ):
            return AnomalyProfile(
                "early_cutoff",
                f"Production stopped early, after {productive[-1]}:00.",
                productive[-1] + 1,
                ratios,
            )
    strong_ratios = [ratios[h] for h in strong if ratios[h] is not None]
    for h in strong[1:-1]:
        r, before, after = ratios[h], ratios[h - 1], ratios[h + 1]
        if r is not None and before is not None and after is not None:
            if r < 0.4 and before >= 0.7 and after >= 0.7:
                return AnomalyProfile(
                    "midday_dropout",
                    f"A sharp dropout around {h}:00 with normal output either side.",
                    h,
                    ratios,
                )
    if strong_ratios and max(strong_ratios) < 0.85:
        mean = sum(strong_ratios) / len(strong_ratios)
        return AnomalyProfile(
            "uniform_shortfall",
            f"Output was low all day, about {round(mean * 100)}% of expected in the main hours.",
            worst,
            ratios,
        )
    return AnomalyProfile("normal", "No distinctive shape in the hourly profile.", worst, ratios)
