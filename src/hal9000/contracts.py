"""Typed contracts shared across HAL: the triage verdict, the nightly record, run results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Evidence",
    "NightlyRecord",
    "RecommendedAlert",
    "RunResult",
    "Verdict",
    "VERDICT_JSON_SCHEMA",
]

VerdictKind = Literal[
    "inverter_suspected",
    "weather_explained",
    "shading",
    "degradation_trend",
    "sensor_fault",
    "inconclusive",
]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: str = Field(min_length=3)
    source_tool: str = Field(min_length=3)
    supports: str = Field(min_length=3)


class RecommendedAlert(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Literal["info", "warn", "critical"]
    text: str = Field(min_length=3)


class Verdict(BaseModel):
    """What the triage sub-agent concludes. ``figures`` names run-store keys to speak."""

    model_config = ConfigDict(extra="forbid")
    verdict: VerdictKind
    confidence: Literal["high", "medium", "low"]
    evidence: list[Evidence] = Field(min_length=1)
    recommended_alert: RecommendedAlert
    figure_keys: list[str] = Field(default_factory=list)


#: The JSON Schema offered to the model for ``submit_verdict`` (arrays carry ``items``).
VERDICT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": list(VerdictKind.__args__),
            "description": "Your conclusion about the cause.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {
            "type": "array",
            "description": "At least one observation, each citing the tool it came from.",
            "items": {
                "type": "object",
                "properties": {
                    "observation": {"type": "string"},
                    "source_tool": {"type": "string"},
                    "supports": {"type": "string"},
                },
                "required": ["observation", "source_tool", "supports"],
            },
        },
        "recommended_alert": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["info", "warn", "critical"]},
                "text": {"type": "string"},
            },
            "required": ["severity", "text"],
        },
        "figure_keys": {
            "type": "array",
            "description": (
                "Keys of figures (from tool results) the parent may speak. Only keys that "
                "tools actually returned are accepted."
            ),
            "items": {"type": "string"},
        },
    },
    "required": ["verdict", "confidence", "evidence", "recommended_alert"],
}


class NightlyRecord(BaseModel):
    """One evening's summary. Append-only; ``kind`` is provisional (20:00) or final (next day)."""

    model_config = ConfigDict(extra="forbid")
    date: str
    kind: Literal["provisional", "final"]
    production_kwh: float | None
    expected_kwh: float | None
    ratio: float | None
    band: str
    model_kind: str | None = None
    calibration_date: str | None = None
    savings_usd: float | None = None
    savings_by_period: dict[str, float] | None = None
    savings_absent_reason: str | None = None
    tariff_schedule_version: str | None = None
    import_kwh: float | None = None
    export_kwh: float | None = None
    household_kwh: float | None = None
    net_position: dict[str, float] | None = None
    insolation_kwh_m2: float | None = None
    anomaly: str | None = None
    verdict: str | None = None
    alerts_raised: list[str] = Field(default_factory=list)
    alerts_cleared: list[str] = Field(default_factory=list)
    renderer: str | None = None
    degraded: bool = False
    sensor_fault: bool = False
    transition_day: bool = False
    missing_solar_hours: int = 0
    production_missing: bool = False


class RunResult(BaseModel):
    run_id: str
    date: str
    status: Literal["ok", "degraded", "failed"]
    report_delivered: bool
    record_written: bool
    announced: bool
    renderer: str | None
    report_text: str | None
    band: str | None = None
    cost_usd: float = 0.0
    iterations: int = 0
    degraded_reasons: list[str] = Field(default_factory=list)
    transcript_path: str | None = None
    trace: list[tuple[str, str]] = Field(default_factory=list)
