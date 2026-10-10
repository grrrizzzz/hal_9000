"""``KB_*`` settings, ``config/knowledge.yaml`` and the pricing config (handoff KWP0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hal9000 import config as hal_config
from hal9000.knowledge import config as kc

REPO = Path(__file__).resolve().parents[2]


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path))
    s = kc.settings()
    assert s.embed_model == "openai/text-embedding-3-small"
    assert (s.chat_tier, s.condense_tier, s.judge_tier) == ("fast", "fast", "powerful")
    assert (s.monthly_cap_usd, s.eval_cap_usd) == (10.0, 10.0)
    assert s.port == 8910 and s.passcode == ""
    assert s.db_path == tmp_path / "knowledge.db"
    assert s.files_dir == tmp_path / "knowledge" / "files"
    assert s.eval_dir == tmp_path / "knowledge" / "eval"
    assert s.house_pdf == tmp_path / "knowledge" / "inbox" / "Home_Inspection.pdf"
    assert s.policy_path.name == "knowledge_policy.txt" and s.policy_path.exists()
    assert s.yaml_path.name == "knowledge.yaml"


def test_overrides(monkeypatch, tmp_path):
    for name, value in {
        "KB_EMBED_MODEL": "openai/text-embedding-3-large",
        "KB_CHAT_TIER": "POWERFUL",
        "KB_CONDENSE_TIER": "fast",
        "KB_JUDGE_TIER": "fast",
        "KB_MONTHLY_CAP_USD": "3.5",
        "KB_EVAL_CAP_USD": "1",
        "KB_PASSCODE": "open sesame",
        "KB_PORT": "9000",
        "KB_DATA_DIR": str(tmp_path / "scratch"),
        "KB_HOUSE_PDF": str(tmp_path / "bundle.pdf"),
    }.items():
        monkeypatch.setenv(name, value)
    s = kc.settings()
    assert s.embed_model.endswith("-large") and s.chat_tier == "powerful"
    assert s.judge_tier == "fast" and s.monthly_cap_usd == 3.5 and s.eval_cap_usd == 1.0
    assert s.require_passcode() == "open sesame" and s.port == 9000
    assert s.db_path == tmp_path / "scratch" / "knowledge.db"
    assert s.house_pdf == tmp_path / "bundle.pdf"


@pytest.mark.parametrize(
    ("name", "value"),
    [("KB_CHAT_TIER", "medium"), ("KB_MONTHLY_CAP_USD", "ten"), ("KB_PORT", "80a")],
)
def test_bad_values_raise_config_errors(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(hal_config.ConfigError, match=name):
        kc.settings()


def test_missing_passcode_raises_config_missing():
    with pytest.raises(kc.ConfigMissing, match="KB_PASSCODE"):
        kc.settings().require_passcode()
    assert issubclass(kc.ConfigMissing, hal_config.ConfigError)  # the CLI exits 2 on it


def test_chat_model_comes_from_the_tier_config(openai_env, monkeypatch):
    assert kc.settings().chat_model() == "openai/gpt-5-mini"
    monkeypatch.setenv("KB_CHAT_TIER", "powerful")
    assert kc.settings().chat_model() == "openai/gpt-5"


def test_embedding_prices_live_in_the_one_pricing_table():
    assert hal_config.MODEL_RATES["text-embedding-3-small"] == (0.02, 0.02, 0.0)
    assert hal_config.MODEL_RATES["text-embedding-3-large"] == (0.13, 0.13, 0.0)
    assert hal_config.rates_for(kc.DEFAULT_EMBED_MODEL) == (0.02, 0.02, 0.0)
    assert not hasattr(kc, "MODEL_RATES")  # no second copy


def test_committed_knowledge_yaml_parses():
    file = kc.load_knowledge_file(REPO / "config" / "knowledge.yaml")
    # the plan's baseline, the default for the first ingest until G-K2 replaces it
    assert file.active_profile == "pdfium.structure-768.hdr.te3s-1536"
    assert file.retrieval.rrf_k == 60 and file.retrieval.top_k_vector == 30
    assert file.retrieval.context_budget_tokens == 2000
    assert file.answer.wall_clock_seconds == 90 and file.answer.questions_per_minute == 20
    assert file.ingest.max_upload_bytes == 100 * 1024 * 1024
    assert file.trace_retention_days == 180
    assert file.require_active_profile() == file.active_profile
    with pytest.raises(kc.ConfigMissing, match="active_profile"):
        kc.KnowledgeFile().require_active_profile()


def test_knowledge_yaml_defaults_typos_and_shape(tmp_path, monkeypatch):
    assert kc.load_knowledge_file(tmp_path / "missing.yaml") == kc.KnowledgeFile()
    typo = tmp_path / "typo.yaml"
    typo.write_text("retrieval:\n  rrf_kk: 60\n")
    with pytest.raises(ValueError, match="rrf_kk"):
        kc.load_knowledge_file(typo)
    listy = tmp_path / "list.yaml"
    listy.write_text("- a\n")
    with pytest.raises(hal_config.ConfigError, match="mapping"):
        kc.load_knowledge_file(listy)
    good = tmp_path / "knowledge.yaml"
    good.write_text("active_profile: pdfium.fixed-512.hdr.te3s-1536\n")
    monkeypatch.setenv("HAL_CONFIG_DIR", str(tmp_path))
    assert kc.load_knowledge_file().require_active_profile().startswith("pdfium")


def test_env_example_documents_every_kb_setting():
    text = (REPO / ".env.example").read_text()
    for name in (
        "KB_EMBED_MODEL",
        "KB_CHAT_TIER",
        "KB_CONDENSE_TIER",
        "KB_JUDGE_TIER",
        "KB_MONTHLY_CAP_USD",
        "KB_EVAL_CAP_USD",
        "KB_PASSCODE",
        "KB_PORT",
        "KB_DATA_DIR",
        "KB_HOUSE_PDF",
        "KB_DENYLIST",
    ):
        assert name in text, name


def test_the_answer_policy_keeps_citation_discipline():
    """S5 (2026-10): half the answers failed the citation check, mostly uncited closing
    sentences and markers for merely related sources. The policy rules that prevent it."""
    from pathlib import Path

    policy = (Path(__file__).parents[2] / "config" / "knowledge_policy.txt").read_text()
    assert "End every sentence that states something from the sources with the marker" in policy
    assert "Cite only a source that itself states what the sentence says" in policy
    assert "no closing remark or summary that would need a citation" in policy
    assert "don't guess which document might have it" in policy
    assert "reply with exactly NOT_IN_SOURCES" in policy
