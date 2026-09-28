"""Provider resolution, model tiers, feature flags and settings."""

from __future__ import annotations

from datetime import datetime

import pytest

from hal9000 import config
from hal9000.config import ConfigError, HalFeatures


def test_explicit_provider_wins_over_present_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert config.resolve_provider() == "openai"


def test_openai_key_selects_openai(openai_env):
    assert config.resolve_provider() == "openai"


def test_anthropic_key_selects_anthropic(anthropic_env):
    assert config.resolve_provider() == "anthropic"


def test_openai_wins_when_both_keys_are_present(anthropic_env, openai_env):
    assert config.resolve_provider() == "openai"


def test_no_provider_raises_with_a_fix():
    with pytest.raises(ConfigError) as excinfo:
        config.resolve_provider()
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert "--no-llm" in str(excinfo.value)


def test_unknown_explicit_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "llamahouse")
    with pytest.raises(ConfigError, match="llamahouse"):
        config.resolve_provider()


def test_require_llm_key_needs_the_providers_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        config.require_llm_key()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    config.require_llm_key()


@pytest.mark.parametrize(
    ("provider", "tier", "expected"),
    [
        ("openai", "powerful", "openai/gpt-5"),
        ("openai", "fast", "openai/gpt-5-mini"),
        ("anthropic", "fast", "anthropic/claude-haiku-4-5"),
    ],
)
def test_model_for_maps_tier_onto_provider_model(monkeypatch, provider, tier, expected):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    assert config.model_for(tier) == expected


def test_tiers_are_distinct_for_every_provider():
    for provider, tiers in config.MODELS.items():
        assert tiers["powerful"] != tiers["fast"], provider


@pytest.mark.parametrize(
    ("tier", "var"), [("powerful", "AGENT_MODEL_POWERFUL"), ("fast", "AGENT_MODEL_FAST")]
)
def test_env_override_beats_the_table(monkeypatch, tier, var):
    monkeypatch.setenv(var, "openai/gpt-5.6-luna")
    assert config.model_for(tier) == "openai/gpt-5.6-luna"


def test_unknown_tier_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ConfigError, match="tier"):
        config.model_for("galaxy-brain")


def test_rates_known_and_pessimistic_default():
    assert config.rates_for("openai/gpt-5-mini") == config.MODEL_RATES["gpt-5-mini"]
    assert config.rates_for("openai/unknown-model") == config.DEFAULT_RATES
    assert config.DEFAULT_RATES[0] >= max(r[0] for r in config.MODEL_RATES.values())


def test_limits_have_defaults_and_overrides(monkeypatch):
    assert config.max_tokens() == 4096
    assert config.max_iterations() == 12
    monkeypatch.setenv("AGENT_MAX_TOKENS", "2000")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "junk")
    assert config.max_tokens() == 2000
    assert config.max_iterations() == 12


def test_reasoning_effort_per_tier(monkeypatch):
    assert config.reasoning_effort("fast") == "low"
    assert config.reasoning_effort("powerful") == "medium"
    monkeypatch.setenv("AGENT_REASONING_FAST", "minimal")
    assert config.reasoning_effort("fast") == "minimal"


def test_features_defaults_and_env(monkeypatch):
    features = HalFeatures.from_env()
    assert features.shadow_mode is False and features.triage_enabled is True
    monkeypatch.setenv("AGENT_ENABLE_SHADOW_MODE", "1")
    monkeypatch.setenv("AGENT_ENABLE_TRIAGE_ENABLED", "0")
    features = HalFeatures.from_env()
    assert features.shadow_mode is True and features.triage_enabled is False
    assert features.enabled() == ["shadow_mode"]
    assert HalFeatures.env_var("verbose_logging") == "AGENT_ENABLE_VERBOSE_LOGGING"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("HA_BASE_URL", "http://ha.local:8123/")
    monkeypatch.setenv("HA_TOKEN", "tok")
    monkeypatch.setenv("SOLAR_SCALE", "2.0")
    monkeypatch.setenv("SOLAR_FIXED_AT", "2026-09-27T18:35:00-07:00")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100")
    monkeypatch.setenv("HA_ENTITY_SPEAKER", "media_player.kitchen")
    s = config.settings()
    assert s.ha_base_url == "http://ha.local:8123"
    assert s.ha_configured
    assert s.solar_scale == 2.0
    assert s.solar_fixed_at == datetime.fromisoformat("2026-09-27T18:35:00-07:00")
    assert not s.telegram_configured  # placeholder token
    assert s.entity_speaker == "media_player.kitchen"
    assert s.tz.key == "America/Los_Angeles"


def test_settings_defaults():
    s = config.settings()
    assert not s.ha_configured
    assert s.solar_scale == 1.0 and s.solar_fixed_at is None
    assert s.entity_solar.startswith("sensor.")
    assert s.config_dir.name == "config"
