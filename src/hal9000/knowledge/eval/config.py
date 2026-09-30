"""``config/knowledge_eval.yaml``: scoring, gold generation, judge and sweep settings."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hal9000 import config as hal_config
from hal9000.knowledge.contracts import GoldCategory
from hal9000.knowledge.eval.scoring import ScoringConfig

__all__ = [
    "EvalConfig",
    "GoldConfig",
    "JudgeConfig",
    "load_eval_config",
]

CATEGORIES: tuple[GoldCategory, ...] = (
    "factoid",
    "money",
    "date",
    "list",
    "cross_document",
    "unanswerable",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScoringSection(_Model):
    budgets: list[int] = Field(default_factory=lambda: [1000, 2000, 4000])
    hit_ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    min_similarity: float = Field(default=90.0, ge=0, le=100)
    full_support: float = Field(default=0.8, gt=0, le=1)
    min_span_words: int = Field(default=3, ge=1)
    word_cover: float = Field(default=0.5, gt=0, le=1)
    min_fuzzy_chars: int = Field(default=12, ge=1)
    n_boot: int = Field(default=1000, ge=0)
    seed: int = 20260928
    min_paired: int = Field(default=10, ge=1)

    def to_config(self) -> ScoringConfig:
        return ScoringConfig(
            budgets=tuple(self.budgets),
            hit_ks=tuple(self.hit_ks),
            min_similarity=self.min_similarity,
            full_support=self.full_support,
            min_span_words=self.min_span_words,
            word_cover=self.word_cover,
            min_fuzzy_chars=self.min_fuzzy_chars,
            n_boot=self.n_boot,
            seed=self.seed,
            min_paired=self.min_paired,
        )


def _default_completion() -> dict[str, int]:
    return {"fast": 1000, "powerful": 3200}


#: gold.estimate keys of the first estimate (no reasoning tokens), accepted and ignored
RETIRED_ESTIMATE_KEYS = ("output_tokens_per_item", "repair_allowance", "check_output_tokens")


class GoldEstimate(_Model):
    """Allowances for ``kb gold generate``'s estimate when the ledger has too few calls."""

    @model_validator(mode="before")
    @classmethod
    def _retired(cls, data: Any) -> Any:
        if isinstance(data, dict):
            found = [key for key in RETIRED_ESTIMATE_KEYS if key in data]
            if found:
                warnings.warn(
                    f"knowledge_eval.yaml gold.estimate: {', '.join(found)} "
                    f"{'is' if len(found) == 1 else 'are'} no longer used and ignored (the "
                    "estimate now allows completion_tokens_per_call incl. reasoning)",
                    UserWarning,
                    stacklevel=2,
                )
                data = {k: v for k, v in data.items() if k not in RETIRED_ESTIMATE_KEYS}
        return data

    #: completion tokens per call *including reasoning* (billed as output), by tier
    completion_tokens_per_call: dict[Literal["fast", "powerful"], int] = Field(
        default_factory=_default_completion
    )
    seconds_per_call: float = Field(default=60.0, gt=0)
    #: extra calls reserved for repairs, as a share of generation calls
    repair_reserve: float = Field(default=0.10, ge=0)
    #: the ledger's mean is used once it has at least this many calls to the model
    min_ledger_calls: int = Field(default=5, ge=1)
    #: passage tokens an unanswerable check shows at most (the eval retriever's largest
    #: scored budget, ``scoring.budgets``)
    check_passage_tokens: int = Field(default=4000, ge=1)

    def completion(self, tier: str) -> int:
        return int(self.completion_tokens_per_call.get(tier, 3200))  # type: ignore[call-overload]


def _default_quotas() -> dict[str, int]:
    return {
        "factoid": 34,
        "money": 18,
        "date": 12,
        "list": 16,
        "cross_document": 16,
        "unanswerable": 24,
    }


class GoldConfig(_Model):
    tier: Literal["fast", "powerful"] = "powerful"
    quotas: dict[GoldCategory, int] = Field(default_factory=_default_quotas)
    paraphrase_quota: float = Field(default=0.30, ge=0, le=1)
    paraphrase_jaccard: float = Field(default=0.30, gt=0, le=1)
    dev_fraction: float = Field(default=0.60, gt=0, lt=1)
    overgenerate: float = Field(default=1.5, ge=1)
    window_pages: int = Field(default=3, ge=1)
    window_max_chars: int = Field(default=12000, ge=500)
    min_window_chars: int = Field(default=200, ge=0)
    max_link_sets: int = Field(default=12, ge=0)
    max_link_docs: int = Field(default=4, ge=2)
    unanswerable_check_k: int = Field(default=30, ge=1)
    dedup_similarity: float = Field(default=90.0, ge=0, le=100)
    min_quote_words: int = Field(default=3, ge=1)
    min_quote_chars: int = Field(default=12, ge=1)
    estimate: GoldEstimate = Field(default_factory=GoldEstimate)

    @model_validator(mode="after")
    def _quotas(self) -> GoldConfig:
        for category in CATEGORIES:
            self.quotas.setdefault(category, 0)
        if any(v < 0 for v in self.quotas.values()):
            raise ValueError("quotas must be >= 0")
        return self


def _default_reasoning() -> dict[str, int]:
    return {"fast": 600, "powerful": 1500}


class JudgeEstimate(_Model):
    answer_prompt_tokens: int = 3000
    answer_output_tokens: int = 350
    judge_output_tokens: int = 150
    #: reasoning tokens per call by tier, billed as output (reasoning models)
    reasoning_tokens: dict[Literal["fast", "powerful"], int] = Field(
        default_factory=_default_reasoning
    )
    #: the metered cost per answer / judgement from judge-prep, when recorded, times this
    metered_margin: float = Field(default=1.25, ge=1)

    def reasoning(self, tier: str) -> int:
        return int(self.reasoning_tokens.get(tier, 0))  # type: ignore[call-overload]


class JudgeConfig(_Model):
    n_items: int = Field(default=20, ge=1)
    agreement_gate: float = Field(default=0.85, gt=0, le=1)
    estimate: JudgeEstimate = Field(default_factory=JudgeEstimate)


class RegressionConfig(_Model):
    budget: int = Field(default=2000, gt=0)


class EvalConfig(_Model):
    scoring: ScoringSection = Field(default_factory=ScoringSection)
    gold: GoldConfig = Field(default_factory=GoldConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    #: stage grids for the sweep runner (plan §8.4); validated by part B
    sweep: dict[str, Any] = Field(default_factory=dict)


def load_eval_config(path: Path | str | None = None) -> EvalConfig:
    """Parse ``config/knowledge_eval.yaml``; a missing file gives the defaults."""
    if path is None:
        from hal9000.knowledge.config import settings

        path = settings().config_dir / "knowledge_eval.yaml"
    target = Path(path)
    if not target.exists():
        return EvalConfig()
    raw: Any = yaml.safe_load(target.read_text()) or {}
    if not isinstance(raw, dict):
        raise hal_config.ConfigError(f"{target} must hold a mapping, not {type(raw).__name__}.")
    return EvalConfig.model_validate(raw)
