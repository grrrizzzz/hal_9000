"""Provider resolution, model tiers, and feature flags."""

from __future__ import annotations

import pytest

from hello_agent import config
from hello_agent.config import ConfigError, Features

FLAG_NAMES = [
    "time_aware",
    "plan_first",
    "progress_tracking",
    "verbose_logging",
    "delegation",
    "staged_review",
    "dynamic_experts",
]


# --- provider resolution: the four cases ------------------------------------


def test_explicit_provider_wins_over_present_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert config.resolve_provider() == "openai"


def test_anthropic_key_selects_anthropic(anthropic_env):
    assert config.resolve_provider() == "anthropic"


def test_openai_key_selects_openai(openai_env):
    assert config.resolve_provider() == "openai"


def test_anthropic_wins_when_both_keys_are_present(anthropic_env, openai_env):
    assert config.resolve_provider() == "anthropic"


def test_no_provider_raises_naming_both_variables_and_the_example_file():
    with pytest.raises(ConfigError) as excinfo:
        config.resolve_provider()
    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert ".env.example" in message


def test_unknown_explicit_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "llamahouse")
    with pytest.raises(ConfigError, match="llamahouse"):
        config.resolve_provider()


# --- model tiers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "tier", "expected"),
    [
        ("anthropic", "powerful", "anthropic/claude-opus-5"),
        ("anthropic", "fast", "anthropic/claude-haiku-4-5"),
        ("openai", "powerful", "openai/gpt-4o"),
        ("openai", "fast", "openai/gpt-4o-mini"),
    ],
)
def test_model_for_maps_tier_onto_provider_model(monkeypatch, provider, tier, expected):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    assert config.model_for(tier) == expected


def test_powerful_is_the_default_tier(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert config.model_for() == config.model_for("powerful")


def test_tiers_are_distinct_for_every_provider():
    for provider, tiers in config.MODELS.items():
        assert tiers["powerful"] != tiers["fast"], provider


@pytest.mark.parametrize(
    ("tier", "var"),
    [("powerful", "AGENT_MODEL_POWERFUL"), ("fast", "AGENT_MODEL_FAST")],
)
def test_env_override_beats_the_table_and_needs_no_key(monkeypatch, tier, var):
    monkeypatch.setenv(var, "ollama/llama3")
    assert config.model_for(tier) == "ollama/llama3"


def test_unknown_tier_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ConfigError, match="tier"):
        config.model_for("galaxy-brain")


def test_model_for_without_a_provider_raises(monkeypatch):
    with pytest.raises(ConfigError):
        config.model_for("fast")


# --- feature flags ----------------------------------------------------------


def test_all_seven_flags_exist_and_default_to_false():
    features = Features()
    assert [f for f in vars(features)] == FLAG_NAMES
    assert all(getattr(features, name) is False for name in FLAG_NAMES)


def test_from_env_with_nothing_set_is_all_false():
    assert Features.from_env() == Features()
    assert Features.from_env().enabled() == []


@pytest.mark.parametrize("flag", FLAG_NAMES)
def test_each_flag_reads_its_own_env_var(monkeypatch, flag):
    monkeypatch.setenv(Features.env_var(flag), "1")
    features = Features.from_env()
    assert getattr(features, flag) is True
    assert features.enabled() == [flag]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("AGENT_ENABLE_DELEGATION", value)
    assert Features.from_env().delegation is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  "])
def test_falsy_spellings(monkeypatch, value):
    monkeypatch.setenv("AGENT_ENABLE_DELEGATION", value)
    assert Features.from_env().delegation is False


def test_all_on_enables_everything():
    assert Features.all_on().enabled() == FLAG_NAMES


def test_env_var_naming_convention():
    assert Features.env_var("time_aware") == "AGENT_ENABLE_TIME_AWARE"


# --- misc knobs -------------------------------------------------------------


def test_numeric_defaults():
    assert config.max_tokens() == 1024
    assert config.max_iterations() == 10
    assert config.max_duration_seconds() == 180


@pytest.mark.parametrize(
    ("var", "getter"),
    [
        ("AGENT_MAX_TOKENS", config.max_tokens),
        ("AGENT_MAX_ITERATIONS", config.max_iterations),
        ("AGENT_MAX_DURATION_SECONDS", config.max_duration_seconds),
    ],
)
def test_numeric_overrides_and_junk_fallback(monkeypatch, var, getter):
    default = getter()
    monkeypatch.setenv(var, "7")
    assert getter() == 7
    monkeypatch.setenv(var, "not-a-number")
    assert getter() == default


def test_guestbook_path_default_and_override(monkeypatch):
    assert config.guestbook_path() == "guestbook.json"
    monkeypatch.setenv("AGENT_GUESTBOOK_PATH", "/tmp/book.json")
    assert config.guestbook_path() == "/tmp/book.json"


def test_timezone_default_and_override(monkeypatch):
    assert config.timezone() == "America/Chicago"
    monkeypatch.setenv("AGENT_TIMEZONE", "Europe/Lisbon")
    assert config.timezone() == "Europe/Lisbon"
