"""Configuration: provider resolution, model tiers, and feature flags.

Two ideas live here.

**Model tiers (MATE "model efficiency").** Rather than hard-coding one model
string, the agent asks for a *tier* -- ``"powerful"`` or ``"fast"`` -- and this
module maps that onto a concrete LiteLLM ``provider/model`` string. Cheap work
(structured extraction) runs on the fast tier; the work that needs judgement
(composing a greeting) runs on the powerful tier.

**Feature flags.** The default run is lean on purpose: five tools, zero
capabilities. Every advanced pattern in this template ships complete and tested
but switched off, so turning one on costs one environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

__all__ = ["ConfigError", "Features", "MODELS", "model_for", "resolve_provider"]


class ConfigError(RuntimeError):
    """Raised when the template cannot work out how to reach an LLM."""


#: provider -> tier -> LiteLLM model string. These ids are current; do not
#: "correct" them to dated variants such as ``claude-opus-5-20260401``.
MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        "powerful": "anthropic/claude-opus-5",
        "fast": "anthropic/claude-haiku-4-5",
    },
    "openai": {
        "powerful": "openai/gpt-4o",
        "fast": "openai/gpt-4o-mini",
    },
}

#: tier -> environment variable that overrides the mapping above.
TIER_OVERRIDES: dict[str, str] = {
    "powerful": "AGENT_MODEL_POWERFUL",
    "fast": "AGENT_MODEL_FAST",
}

_NO_PROVIDER_MESSAGE = (
    "No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
    "(or force one with LLM_PROVIDER=anthropic|openai). "
    "Copy .env.example to .env and fill in a key -- see .env.example for the "
    "full list of supported variables."
)


def _env_int(name: str, default: int) -> int:
    """Read an integer setting, falling back to ``default`` if unset or junk."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    """Interpret an environment variable as a boolean. Unset means False."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# House rule for this module: anything derived from the environment is a
# *function*, so it is read at call time and a test can change it with
# monkeypatch. Anything constant is a constant.
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_DURATION_SECONDS = 180


def max_tokens() -> int:
    """Output token ceiling per LLM call. Override with ``AGENT_MAX_TOKENS``."""
    return _env_int("AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS)


def max_iterations() -> int:
    """Agent loop iteration ceiling. Override with ``AGENT_MAX_ITERATIONS``."""
    return _env_int("AGENT_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)


def max_duration_seconds() -> int:
    """Agent loop wall-clock ceiling. Override with ``AGENT_MAX_DURATION_SECONDS``."""
    return _env_int("AGENT_MAX_DURATION_SECONDS", DEFAULT_MAX_DURATION_SECONDS)


def resolve_provider() -> str:
    """Decide which provider to talk to.

    Order of precedence:

    1. an explicit ``LLM_PROVIDER`` environment variable,
    2. ``ANTHROPIC_API_KEY`` being present,
    3. ``OPENAI_API_KEY`` being present,
    4. otherwise raise :class:`ConfigError`.
    """
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in MODELS:
            raise ConfigError(
                f"LLM_PROVIDER={explicit!r} is not supported. "
                f"Choose one of: {', '.join(sorted(MODELS))}."
            )
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise ConfigError(_NO_PROVIDER_MESSAGE)


def model_for(tier: str = "powerful") -> str:
    """Return the LiteLLM model string for ``tier`` (``"powerful"`` or ``"fast"``).

    ``AGENT_MODEL_POWERFUL`` / ``AGENT_MODEL_FAST`` win outright, so you can point
    a tier at any LiteLLM-supported model without touching this file.
    """
    if tier not in TIER_OVERRIDES:
        raise ConfigError(
            f"Unknown model tier {tier!r}. Choose one of: {', '.join(sorted(TIER_OVERRIDES))}."
        )
    override = os.environ.get(TIER_OVERRIDES[tier], "").strip()
    if override:
        return override
    return MODELS[resolve_provider()][tier]


def guestbook_path() -> str:
    """Where the guestbook JSON file lives. Override with ``AGENT_GUESTBOOK_PATH``."""
    return os.environ.get("AGENT_GUESTBOOK_PATH", "").strip() or "guestbook.json"


def timezone() -> str:
    """Timezone for ``TimeAwareCapability``. Override with ``AGENT_TIMEZONE``."""
    return os.environ.get("AGENT_TIMEZONE", "").strip() or "America/Chicago"


@dataclass
class Features:
    """Which optional patterns are switched on. Every flag defaults to off.

    ``AGENT_ENABLE_TIME_AWARE=1`` enables :attr:`time_aware`, and so on for each
    field: the environment variable is ``AGENT_ENABLE_`` plus the field name in
    upper case.
    """

    time_aware: bool = False
    plan_first: bool = False
    progress_tracking: bool = False
    verbose_logging: bool = False
    delegation: bool = False  # adds the translator sub-agent + call_agent tools
    staged_review: bool = False  # routes writes through StagedActionEnvironment
    dynamic_experts: bool = False  # create_and_consult_expert

    @classmethod
    def env_var(cls, flag: str) -> str:
        """The environment variable that controls ``flag``."""
        return f"AGENT_ENABLE_{flag.upper()}"

    @classmethod
    def from_env(cls) -> Features:
        """Build a :class:`Features` from ``AGENT_ENABLE_*``. Unset means all off."""
        return cls(**{f.name: _env_flag(cls.env_var(f.name)) for f in fields(cls)})

    @classmethod
    def all_on(cls) -> Features:
        """Every flag enabled. Handy for tests and for the guided tour in the docs."""
        return cls(**{f.name: True for f in fields(cls)})

    def enabled(self) -> list[str]:
        """Names of the flags that are on, in declaration order."""
        return [f.name for f in fields(self) if getattr(self, f.name)]
