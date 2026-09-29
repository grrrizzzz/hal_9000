"""RetrievalConfig, thresholds, transforms' parsing and the reranker wrappers."""

from __future__ import annotations

import math
import sys
import time

import pytest

from hal9000.knowledge.config import ConfigMissing, KnowledgeFile
from hal9000.knowledge.config import RetrievalConfig as YamlRetrieval
from hal9000.knowledge.retrieve import (
    NoRerank,
    OnnxCrossEncoder,
    RetrievalConfig,
    get_threshold,
    reranker_for,
    reranker_key,
    set_threshold,
    transforms,
)
from hal9000.knowledge.retrieve.rerank import MODELS, sigmoid
from tests.fakes import FakeLLM
from tests.knowledge.retrieve.conftest import ACTIVE, profile_id


# ------------------------------------------------------------------ config
def test_defaults_match_the_interface():
    cfg = RetrievalConfig()
    assert (cfg.mode, cfg.fusion, cfg.alpha, cfg.top_k_each) == ("hybrid", "rrf", 0.5, 30)
    assert (cfg.rerank, cfg.budget_tokens, cfg.send_headers, cfg.transform) == (
        "none", 2000, True, "none",
    )  # fmt: skip


@pytest.mark.parametrize(
    ("method", "alpha", "mode", "fusion"),
    [("rrf", None, "hybrid", "rrf"), ("dense", None, "dense", "rrf"),
     ("bm25", None, "bm25", "rrf"), ("weighted", 0.3, "hybrid", "weighted")],
)  # fmt: skip
def test_from_settings_maps_the_yaml_method(method, alpha, mode, fusion):
    knowledge = KnowledgeFile(
        reranker="minilm",
        retrieval=YamlRetrieval(
            method=method, weighted_alpha=alpha, top_k_vector=20, top_k_bm25=40,
            context_budget_tokens=4000, rrf_k=10, history_turns=4,
        ),
    )  # fmt: skip
    cfg = RetrievalConfig.from_settings(knowledge)
    assert (cfg.mode, cfg.fusion) == (mode, fusion)
    assert (cfg.top_k_each, cfg.budget_tokens, cfg.rrf_k, cfg.history_turns) == (40, 4000, 10, 4)
    assert cfg.rerank == "minilm"
    if alpha is not None:
        assert cfg.alpha == alpha


def test_weighted_without_alpha_is_never_guessed():
    knowledge = KnowledgeFile(retrieval=YamlRetrieval(method="weighted"))
    with pytest.raises(ConfigMissing, match="weighted_alpha"):
        RetrievalConfig.from_settings(knowledge)
    with pytest.raises(ValueError, match="unknown retrieval method"):
        RetrievalConfig.method_fields("magic")


def test_from_settings_reads_the_yaml_file(tmp_path):
    (tmp_path / "knowledge.yaml").write_text("reranker: bge-base\nretrieval:\n  method: dense\n")

    class Settings:
        yaml_path = tmp_path / "knowledge.yaml"

    cfg = RetrievalConfig.from_settings(Settings())
    assert (cfg.mode, cfg.rerank) == ("dense", "bge-base")
    assert RetrievalConfig.from_settings().mode == "hybrid"  # the repo's config/knowledge.yaml


def test_updated_applies_run_keys():
    cfg = RetrievalConfig()
    assert cfg.updated(None) is cfg
    run = {"profile": "x", "retrieval": "weighted", "alpha": 0.7, "reranker": "minilm",
           "budget": 1000, "query_transform": "hyde", "sweep_id": "s1"}  # fmt: skip
    new = cfg.updated(run)
    assert (new.mode, new.fusion, new.alpha, new.rerank) == ("hybrid", "weighted", 0.7, "minilm")
    assert (new.budget_tokens, new.transform) == (1000, "hyde")
    assert cfg.updated({"retrieval": {"mode": "bm25", "top_k_each": 5, "nope": 1}}).top_k_each == 5
    assert cfg.updated({"reranker": None, "rerank": ""}).rerank == "none"
    assert cfg.updated({"query_transform": None, "transform": ""}).transform == "none"
    assert cfg.updated({"send_headers": False}).send_headers is False
    weighted = RetrievalConfig(fusion="weighted", alpha=0.3)
    assert weighted.updated({"retrieval": "weighted"}).alpha == 0.3
    with pytest.raises(ConfigMissing):
        cfg.updated({"retrieval": "weighted"})


@pytest.mark.parametrize(
    "bad",
    [dict(mode="x"), dict(fusion="x"), dict(rerank="x"), dict(transform="x"), dict(alpha=2.0),
     dict(top_k_each=0), dict(budget_tokens=0), dict(rrf_k=-1), dict(history_turns=-1)],
)  # fmt: skip
def test_invalid_configs_are_refused(bad):
    with pytest.raises(ValueError):
        RetrievalConfig(**bad)


# ------------------------------------------------------------------ thresholds
def test_thresholds_are_stored_per_profile_and_reranker(conn):
    pid = profile_id(conn, ACTIVE)
    assert get_threshold(conn, pid, None) is None
    set_threshold(conn, pid, "none", 0.3, auc=0.8, eval_run_id="r1")
    set_threshold(conn, pid, "minilm", 0.6)
    assert get_threshold(conn, pid, None) == get_threshold(conn, pid, "") == 0.3
    assert get_threshold(conn, pid, "minilm") == 0.6
    set_threshold(conn, pid, None, 0.35)  # replaced, not duplicated
    assert get_threshold(conn, pid, "none") == 0.35
    rows = conn.execute("SELECT reranker, auc FROM abstain_thresholds ORDER BY reranker").fetchall()
    assert [tuple(r) for r in rows] == [("", None), ("minilm", None)]
    assert reranker_key("bge-base") == "bge-base" and reranker_key(None) == ""


# ------------------------------------------------------------------ transforms
def test_reply_lines_are_cleaned():
    assert transforms.clean_line('  1. "water heater age" ') == "water heater age"
    assert transforms.clean_line("- Rewrite 2: tank year") == "tank year"
    assert transforms.clean_line("Standalone question: How old?") == "How old?"
    assert transforms.clean_line("• `x`") == "x"


def test_multi_query_keeps_three_distinct_rewrites():
    llm = FakeLLM(["How old is it?\n\nA\na\nB\nC\nD"])
    assert transforms.multi_query(llm, "How old is it?") == ["A", "B", "C"]
    assert llm.last_prompt.metadata["purpose"] == "multi_query"


def test_condense_prompt_marks_the_conversation_as_data():
    from hal9000.knowledge.retrieve import Turn

    llm = FakeLLM(["\n\n  What year was the water heater made?  \n"])
    out = transforms.condense(llm, "when?", [Turn("assistant", "The heater leaks.")])
    assert out == "What year was the water heater made?"
    system, user = llm.last_prompt.messages
    assert "data, not instructions" in system["content"]
    assert "Assistant: The heater leaks." in user["content"]
    assert transforms.hyde(FakeLLM(["  A  passage\n here "]), "q") == "A passage here"


# ------------------------------------------------------------------ rerank
def test_sigmoid_is_stable_and_monotonic():
    assert sigmoid(0) == 0.5
    assert sigmoid(-1000) == 0.0 and sigmoid(1000) == 1.0
    xs = [-8.0, -1.5, 0.0, 2.0, 9.0]
    ys = [sigmoid(x) for x in xs]
    assert ys == sorted(ys) and all(0 <= y <= 1 for y in ys)
    assert sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2.0)))


def test_onnx_cross_encoder_normalises_logits_without_loading_early():
    encoder = OnnxCrossEncoder("minilm", batch_size=4)
    assert encoder.model_name == MODELS["minilm"] and encoder._model is None
    assert encoder.score("q", []) == []  # nothing to score: the model isn't loaded

    class Model:
        def rerank(self, query, documents, batch_size):
            assert batch_size == 4
            return iter([2.0, -1.0][: len(documents)])

    encoder._model = Model()
    assert encoder.logits("q", ["a", "b"]) == [2.0, -1.0]
    assert encoder.score("q", ["a", "b"]) == [
        pytest.approx(sigmoid(2.0)),
        pytest.approx(sigmoid(-1.0)),
    ]
    assert OnnxCrossEncoder("some/other-model").model_name == "some/other-model"


def test_reranker_for_caches_one_model_per_choice():
    assert isinstance(reranker_for("none"), NoRerank)
    assert NoRerank().score("q", ["a", "b"]) == [0.5, 0.5]
    first = reranker_for("minilm")
    assert first is reranker_for("minilm") and first.name == "minilm"
    assert reranker_for("bge-base").model_name == "BAAI/bge-reranker-base"
    with pytest.raises(ValueError):
        reranker_for("colbert")


def test_importing_the_retriever_loads_no_model_runtime():
    import hal9000.knowledge.retrieve  # noqa: F401

    assert "fastembed" not in sys.modules and "onnxruntime" not in sys.modules


@pytest.mark.models
def test_minilm_reranks_a_hand_made_set_and_records_latency():
    pytest.importorskip("fastembed")
    encoder = OnnxCrossEncoder("minilm")
    query = "Why is the water heater relief valve dripping?"
    relevant = "The temperature and pressure relief valve on the water heater is dripping."
    distractors = [
        "The irrigation controller runs the lawn zones on Monday mornings.",
        "Replace the heat pump filter every 90 days.",
        "The garage door opener is fed by breaker 14.",
    ]
    try:
        scores = encoder.score(query, [*distractors, relevant])
    except Exception as error:  # pragma: no cover - needs the model download
        pytest.skip(f"MiniLM is not available: {error}")
    assert max(range(4), key=scores.__getitem__) == 3
    pairs = [relevant if i % 4 == 0 else distractors[i % 3] for i in range(30)]
    started = time.perf_counter()
    encoder.score(query, pairs)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"MiniLM latency for 30 pairs: {elapsed:.0f} ms")
