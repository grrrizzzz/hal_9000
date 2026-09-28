"""Configuration: provider and model tiers, feature flags, and site settings.

Three ideas live here.

**Model tiers.** Agents ask for a *tier* -- ``"powerful"`` or ``"fast"`` -- and
this module maps it to a LiteLLM ``provider/model`` string. The solar agent runs
on the fast tier, the triage sub-agent on the powerful one.

**Feature flags.** :class:`HalFeatures` holds the few runtime switches HAL has:
shadow mode (dry run), whether triage may run, and verbose loop logging.

**Settings.** Everything site-specific that is secret or machine-specific comes
from environment variables (usually ``.env``): the Home Assistant URL and token,
the Emporia entity IDs, the Telegram chat, the solar-channel correction. Anything
derived from the environment is a *function*, read at call time, so tests can
change it with ``monkeypatch``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

__all__ = [
    "ConfigError",
    "HalFeatures",
    "MODELS",
    "MODEL_RATES",
    "Settings",
    "model_for",
    "resolve_provider",
    "settings",
]


class ConfigError(RuntimeError):
    """Raised when something the run needs is not configured."""


#: provider -> tier -> LiteLLM model string. Override either tier with
#: ``AGENT_MODEL_FAST`` / ``AGENT_MODEL_POWERFUL`` -- no code change needed.
MODELS: dict[str, dict[str, str]] = {
    "openai": {
        "powerful": "openai/gpt-5",
        "fast": "openai/gpt-5-mini",
    },
    "anthropic": {
        "powerful": "anthropic/claude-sonnet-5",
        "fast": "anthropic/claude-haiku-4-5",
    },
}

#: model (without provider prefix) -> USD per million tokens:
#: (input, cached input, output). Verify against the provider's price page;
#: unknown models fall back to :data:`DEFAULT_RATES`, which is deliberately
#: pessimistic so cost caps still bite.
MODEL_RATES: dict[str, tuple[float, float, float]] = {
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-5-mini": (0.25, 0.025, 2.0),
    "gpt-5-nano": (0.05, 0.005, 0.4),
    "claude-haiku-4-5": (1.0, 0.1, 5.0),
    "claude-sonnet-5": (2.0, 0.2, 10.0),
}
DEFAULT_RATES: tuple[float, float, float] = (5.0, 0.5, 25.0)

TIER_OVERRIDES: dict[str, str] = {
    "powerful": "AGENT_MODEL_POWERFUL",
    "fast": "AGENT_MODEL_FAST",
}

_NO_PROVIDER_MESSAGE = (
    "No LLM provider configured. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY) in .env, "
    "or run with --no-llm for the deterministic report."
)

#: Output ceiling per call. Reasoning tokens count toward it, so the powerful
#: tier (medium effort, long triage prompts) gets far more room.
DEFAULT_MAX_TOKENS = {"fast": 8192, "powerful": 16384}
DEFAULT_MAX_ITERATIONS = 12
DEFAULT_MAX_DURATION_SECONDS = 600


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def max_tokens(tier: str = "fast") -> int:
    """Output token ceiling per LLM call (reasoning tokens count toward it).

    ``AGENT_MAX_TOKENS`` overrides both tiers.
    """
    return _env_int("AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS.get(tier, DEFAULT_MAX_TOKENS["fast"]))


def max_iterations() -> int:
    return _env_int("AGENT_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)


def max_duration_seconds() -> int:
    return _env_int("AGENT_MAX_DURATION_SECONDS", DEFAULT_MAX_DURATION_SECONDS)


def resolve_provider() -> str:
    """``LLM_PROVIDER``, else the first provider whose key is present."""
    explicit = _env("LLM_PROVIDER").lower()
    if explicit:
        if explicit not in MODELS:
            raise ConfigError(
                f"LLM_PROVIDER={explicit!r} is not supported. "
                f"Choose one of: {', '.join(sorted(MODELS))}."
            )
        return explicit
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise ConfigError(_NO_PROVIDER_MESSAGE)


def require_llm_key() -> None:
    """Raise :class:`ConfigError` unless the resolved provider has a key."""
    provider = resolve_provider()
    key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[provider]
    if not _env(key):
        raise ConfigError(f"{key} is not set. Add it to .env, or run with --no-llm.")


def model_for(tier: str = "powerful") -> str:
    """LiteLLM model string for ``tier``; the override variables win outright."""
    if tier not in TIER_OVERRIDES:
        raise ConfigError(
            f"Unknown model tier {tier!r}. Choose one of: {', '.join(sorted(TIER_OVERRIDES))}."
        )
    override = _env(TIER_OVERRIDES[tier])
    if override:
        return override
    return MODELS[resolve_provider()][tier]


def rates_for(model: str) -> tuple[float, float, float]:
    """USD per million tokens (input, cached input, output) for ``model``."""
    name = model.split("/", 1)[-1]
    return MODEL_RATES.get(name, DEFAULT_RATES)


def reasoning_effort(tier: str) -> str:
    """Reasoning effort requested from reasoning models, per tier."""
    default = "low" if tier == "fast" else "medium"
    return _env(f"AGENT_REASONING_{tier.upper()}", default)


@dataclass
class HalFeatures:
    """Runtime switches. ``AGENT_ENABLE_<NAME>=1`` turns each on (triage defaults on)."""

    shadow_mode: bool = False  # dry run: no Sonos; Telegram to the test chat / prefixed
    triage_enabled: bool = True  # the solar agent may call the triage sub-agent
    verbose_logging: bool = False  # one structured log line per loop event

    @classmethod
    def env_var(cls, flag: str) -> str:
        return f"AGENT_ENABLE_{flag.upper()}"

    @classmethod
    def from_env(cls) -> HalFeatures:
        values = {}
        for f in fields(cls):
            raw = os.environ.get(cls.env_var(f.name))
            values[f.name] = f.default if raw is None else _env_flag(cls.env_var(f.name))
        return cls(**values)

    def enabled(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name)]


@dataclass(frozen=True)
class Settings:
    """Site and integration settings, read from the environment."""

    ha_base_url: str
    ha_token: str
    entity_solar: str
    entity_grid_import: str
    entity_grid_export: str
    entity_speaker: str
    tts_entity: str
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_test_chat_id: str
    solar_scale: float
    solar_fixed_at: datetime | None
    timezone: str
    data_dir: Path
    config_dir: Path
    api_token: str
    run_soft_cap_usd: float
    run_hard_cap_usd: float
    monthly_cap_usd: float
    deadman_ping_url: str

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def telegram_configured(self) -> bool:
        token = self.telegram_bot_token
        return bool(token and token != "placeholder" and self.telegram_chat_id)

    @property
    def ha_configured(self) -> bool:
        return bool(self.ha_base_url and self.ha_token)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_fixed_at(raw: str) -> datetime | None:
    """``SOLAR_FIXED_AT`` must be an ISO timestamp with a UTC offset.

    A naive time can't be compared with the timezone-aware statistics, and a
    typo must not silently switch the scale correction off.
    """
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise ConfigError(
            f"SOLAR_FIXED_AT={raw!r} is not an ISO timestamp "
            "(expected e.g. 2026-09-27T18:35:00-07:00)"
        ) from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise ConfigError(
            f"SOLAR_FIXED_AT={raw!r} has no UTC offset (expected e.g. 2026-09-27T18:35:00-07:00)"
        )
    return value


def settings() -> Settings:
    """Read :class:`Settings` from the environment now."""
    fixed_at = _parse_fixed_at(_env("SOLAR_FIXED_AT"))
    root = _repo_root()
    return Settings(
        ha_base_url=_env("HA_BASE_URL").rstrip("/"),
        ha_token=_env("HA_TOKEN"),
        entity_solar=_env("HA_ENTITY_SOLAR", "sensor.electron_inspector_energy_today_13"),
        entity_grid_import=_env("HA_ENTITY_GRID_IMPORT", "sensor.mainstogrid_energy_today"),
        entity_grid_export=_env("HA_ENTITY_GRID_EXPORT", "sensor.mainsfromgrid_energy_today"),
        entity_speaker=_env("HA_ENTITY_SPEAKER", "media_player.unnamed_room_4"),
        tts_entity=_env("HA_TTS_ENTITY", "tts.home_assistant_cloud"),
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
        telegram_test_chat_id=_env("TELEGRAM_TEST_CHAT_ID"),
        solar_scale=_env_float("SOLAR_SCALE", 1.0),
        solar_fixed_at=fixed_at,
        timezone=_env("HAL_TIMEZONE", "America/Los_Angeles"),
        data_dir=Path(_env("HAL_DATA_DIR", str(root / "data"))),
        config_dir=Path(_env("HAL_CONFIG_DIR", str(root / "config"))),
        api_token=_env("HAL_API_TOKEN"),
        run_soft_cap_usd=_env_float("HAL_RUN_SOFT_CAP_USD", 0.25),
        run_hard_cap_usd=_env_float("HAL_RUN_HARD_CAP_USD", 1.00),
        monthly_cap_usd=_env_float("HAL_MONTHLY_CAP_USD", 10.0),
        deadman_ping_url=_env("DEADMAN_PING_URL"),
    )


def timezone() -> str:
    """Timezone for the optional TimeAwareCapability (``AGENT_TIMEZONE``)."""
    return _env("AGENT_TIMEZONE", _env("HAL_TIMEZONE", "America/Los_Angeles"))
