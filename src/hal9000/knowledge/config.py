"""Knowledge-base settings: ``KB_*`` environment variables and ``config/knowledge.yaml``.

Secrets and machine-specific values come from the environment (usually
``.env``); tunable behaviour (the active profile, retrieval sizes) lives in
``config/knowledge.yaml``. Everything is read at call time so tests can change
it with ``monkeypatch``.

Model *names* and *prices* are not here. Chat calls ask for a tier (``fast`` or
``powerful``), mapped to a model by :func:`hal9000.config.model_for`; prices,
including the embedding prices, live in :data:`hal9000.config.MODEL_RATES`
(rule K8, plan §13: one source of truth).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hal9000 import config as hal_config

__all__ = [
    "AGENT",
    "EVAL_AGENT",
    "ConfigMissing",
    "KnowledgeFile",
    "KnowledgeSettings",
    "load_knowledge_file",
    "monthly_caps",
    "settings",
]

#: Ledger agent names (rule K9). Evaluation spend never counts toward ``AGENT``'s cap.
AGENT = "knowledge"
EVAL_AGENT = "knowledge-eval"

DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
DEFAULT_PORT = 8910
TIERS = ("fast", "powerful")


class ConfigMissing(hal_config.ConfigError):
    """A required knowledge setting is not configured (rule K8: never guess)."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise hal_config.ConfigError(f"{name}={raw!r} is not a number.") from error


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise hal_config.ConfigError(f"{name}={raw!r} is not an integer.") from error


def _tier(name: str, default: str) -> str:
    value = _env(name, default).lower()
    if value not in TIERS:
        raise hal_config.ConfigError(f"{name}={value!r} must be one of: {', '.join(TIERS)}.")
    return value


@dataclass(frozen=True)
class KnowledgeSettings:
    """The ``KB_*`` settings from plan §2, §13, §14 and handoff KWP0."""

    embed_model: str
    chat_tier: str
    condense_tier: str
    judge_tier: str
    monthly_cap_usd: float
    eval_cap_usd: float
    passcode: str
    port: int
    data_dir: Path
    house_pdf: Path
    config_dir: Path

    @property
    def db_path(self) -> Path:
        """``data/knowledge.db`` (plan §12)."""
        return self.data_dir / "knowledge.db"

    @property
    def files_dir(self) -> Path:
        """Originals, stored by hash: ``data/knowledge/files/<sha256>``."""
        return self.data_dir / "knowledge" / "files"

    @property
    def eval_dir(self) -> Path:
        """The question set and sweep runs; never in git (plan §8.1)."""
        return self.data_dir / "knowledge" / "eval"

    @property
    def policy_path(self) -> Path:
        return self.config_dir / "knowledge_policy.txt"

    @property
    def yaml_path(self) -> Path:
        return self.config_dir / "knowledge.yaml"

    def chat_model(self) -> str:
        return hal_config.model_for(self.chat_tier)

    def require_passcode(self) -> str:
        if not self.passcode:
            raise ConfigMissing("KB_PASSCODE is not set. Add a household passcode to .env.")
        return self.passcode


def _yaml_chat_tier(config_dir: Path) -> str | None:
    """``answer.chat_tier`` of ``config/knowledge.yaml`` (set at gate G-K2), if any.

    ``KB_CHAT_TIER`` in the environment overrides it. A missing or unreadable
    file gives None here; :func:`load_knowledge_file` reports its errors.
    """
    path = config_dir / "knowledge.yaml"
    try:
        raw = yaml.safe_load(path.read_text()) if path.exists() else None
    except (OSError, yaml.YAMLError):
        return None
    answer = raw.get("answer") if isinstance(raw, dict) else None
    tier = answer.get("chat_tier") if isinstance(answer, dict) else None
    return str(tier) if tier else None


def settings() -> KnowledgeSettings:
    """Read :class:`KnowledgeSettings` from the environment now.

    The chat tier is ``KB_CHAT_TIER``, else ``answer.chat_tier`` in
    ``config/knowledge.yaml``, else ``fast``.
    """
    hal = hal_config.settings()
    data_dir = Path(_env("KB_DATA_DIR", str(hal.data_dir)))
    return KnowledgeSettings(
        embed_model=_env("KB_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        chat_tier=_tier("KB_CHAT_TIER", _yaml_chat_tier(hal.config_dir) or "fast"),
        condense_tier=_tier("KB_CONDENSE_TIER", "fast"),
        judge_tier=_tier("KB_JUDGE_TIER", "powerful"),
        monthly_cap_usd=_env_float("KB_MONTHLY_CAP_USD", 10.0),
        eval_cap_usd=_env_float("KB_EVAL_CAP_USD", 10.0),
        passcode=_env("KB_PASSCODE"),
        port=_env_int("KB_PORT", DEFAULT_PORT),
        data_dir=data_dir,
        house_pdf=Path(
            _env("KB_HOUSE_PDF", str(data_dir / "knowledge" / "inbox" / "Home_Inspection.pdf"))
        ),
        config_dir=hal.config_dir,
    )


def monthly_caps(current: KnowledgeSettings | None = None) -> dict[str, float]:
    """Per-agent monthly caps for :class:`hal9000.cost.metering.Meter` (plan §13)."""
    s = current or settings()
    return {AGENT: s.monthly_cap_usd, EVAL_AGENT: s.eval_cap_usd}


# ------------------------------------------------------------ knowledge.yaml
class RetrievalConfig(BaseModel):
    """Retrieval knobs (plan §7.1). Defaults are the plan's starting values; G-K2 sets them."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["dense", "bm25", "rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    weighted_alpha: float | None = None
    top_k_vector: int = 30
    top_k_bm25: int = 30
    context_budget_tokens: int = 2000
    history_turns: int = 6
    #: query transformation (plan §8.3 T10), chosen by the setup evaluation
    transform: Literal["none", "multi", "hyde"] = "none"


class AnswerConfig(BaseModel):
    """Answer limits (plan §7.4). ``max_output_tokens`` null means the tier default."""

    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int | None = None
    #: the chat tier chosen at G-K2 (plan §8.3 T11); ``KB_CHAT_TIER`` overrides it
    chat_tier: Literal["fast", "powerful"] | None = None
    wall_clock_seconds: float = 90.0
    questions_per_minute: int = 20


class IngestConfig(BaseModel):
    """Upload and parse limits (plan §5)."""

    model_config = ConfigDict(extra="forbid")

    max_upload_bytes: int = 100 * 1024 * 1024
    parse_timeout_seconds: float = 300.0
    header_repeat_fraction: float = 0.30
    pseudo_page_chars: int = 3000


class KnowledgeFile(BaseModel):
    """``config/knowledge.yaml``. Unknown keys are rejected so a typo can't pass silently."""

    model_config = ConfigDict(extra="forbid")

    active_profile: str | None = None
    reranker: str | None = None
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    trace_retention_days: int = 180

    def require_active_profile(self) -> str:
        if not self.active_profile:
            raise ConfigMissing(
                "No active index profile. Set active_profile in config/knowledge.yaml "
                "(chosen at gate G-K2) or activate one with `hal9000 kb profile activate`."
            )
        return self.active_profile


def load_knowledge_file(path: Path | str | None = None) -> KnowledgeFile:
    """Parse ``config/knowledge.yaml``; a missing file gives the defaults."""
    target = Path(path) if path is not None else settings().yaml_path
    if not target.exists():
        return KnowledgeFile()
    raw: Any = yaml.safe_load(target.read_text()) or {}
    if not isinstance(raw, dict):
        raise hal_config.ConfigError(f"{target} must hold a mapping, not {type(raw).__name__}.")
    return KnowledgeFile.model_validate(raw)
