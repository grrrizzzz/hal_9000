"""Plan §8.6 on constructed runs: the recall winner, ties to the simpler option, the latency
filter (Pi when measured, else the sweep host, flagged), the chat tier, and ``choose``'s yaml."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hal9000.knowledge.config import load_knowledge_file
from hal9000.knowledge.contracts import EvalMetrics, EvalRun, MetricCI
from hal9000.knowledge.eval.decide import (
    Ranker,
    find_run,
    identity,
    recommend,
    render_recommendation,
    render_targets,
    simplicity,
    target_scores,
    write_choice_yaml,
)
from hal9000.knowledge.eval.scoring import ScoringConfig, bootstrap_ci
from hal9000.knowledge.retrieve.config import RetrievalConfig

REPO = Path(__file__).resolve().parents[3]
SCORING = ScoringConfig(n_boot=300, min_paired=5)
N = 30


def base(values: list[float], shift: float = 0.0) -> list[float]:
    return [min(1.0, max(0.0, v + shift)) for v in values]


#: per-question recall with spread, so the bootstrap intervals are not degenerate
SPREAD = [0.5 + 0.4 * ((i * 7) % 11) / 10 for i in range(N)]
#: a small difference whose sign alternates: a tie
NOISE = [0.02 if i % 2 else -0.015 for i in range(N)]


def make_run(
    tmp: Path,
    run_id: str,
    recalls: list[float] | None = None,
    *,
    stage: str = "S3",
    host: str = "mac",
    p95: float | None = 50.0,
    correct: list[float] | None = None,
    **config,
) -> EvalRun:
    folder = tmp / run_id
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(N):
        record = {"gold_item_id": i + 1, "category": "factoid", "question": f"q{i}"}
        if recalls is not None:
            record["recall"] = {"2000": recalls[i]}
        if correct is not None:
            record["correct"] = correct[i]
        records.append(record)
    (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    metrics = EvalMetrics(
        n_questions=N,
        budgeted_recall={2000: bootstrap_ci(recalls, n_boot=200)} if recalls else {},
        correct=bootstrap_ci(correct, n_boot=200) if correct else None,
        latency_ms_p95=p95,
    )
    values = {
        "sweep_id": "sweep-1",
        "profile": "pdfium.structure-512.hdr.te3s-1536",
        "parser": "pdfium",
        "embed_model": "openai/text-embedding-3-small",
        "dimensions": 1536,
        "retrieval": "rrf",
        "reranker": "none",
        "query_transform": "none",
        "name": run_id,
    }
    values.update(config)
    return EvalRun(
        id=run_id,
        stage=stage,
        config=values,
        metrics=metrics,
        host=host,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        results_path=str(folder / "results.jsonl"),
    )


def test_a_config_that_wins_on_every_question_is_chosen_despite_a_reranker(tmp_path):
    plain = make_run(tmp_path, "plain", SPREAD)
    reranked = make_run(tmp_path, "reranked", base(SPREAD, 0.08), reranker="bge-base")
    rec = recommend([plain, reranked], scoring=SCORING)
    assert rec.chosen.run_id == "reranked"
    assert rec.alternatives[0].run_id == "plain" and rec.alternatives[0].versus.verdict == "a"
    text = render_recommendation(rec)
    assert "reranked" in text and "worse" in text


def test_a_tie_goes_to_the_simpler_option(tmp_path):
    simple = make_run(tmp_path, "simple", SPREAD)
    fancy = make_run(tmp_path, "fancy", [s + n for s, n in zip(SPREAD, NOISE)], reranker="minilm")
    assert fancy.metrics.budgeted_recall[2000].value > simple.metrics.budgeted_recall[2000].value
    rec = recommend([fancy, simple], scoring=SCORING)
    assert rec.chosen.run_id == "simple"
    reason = next(r for r in rec.reasons if "ties by paired bootstrap" in r)
    assert (
        reason.startswith("simple ") and "the highest budgeted recall @ 2k on dev, fancy" in reason
    )
    assert not any(r.startswith("highest") for r in rec.reasons)
    assert rec.alternatives[0].versus.verdict == "tie"


@pytest.mark.parametrize(
    ("simpler", "other"),
    [
        ({"reranker": "none"}, {"reranker": "minilm"}),
        ({"reranker": "minilm"}, {"reranker": "bge-base"}),
        ({"parser": "pdfium"}, {"parser": "docling"}),
        ({"dimensions": 256, "embed_model": "openai/text-embedding-3-large"}, {"dimensions": 1536}),
        ({"query_transform": "none"}, {"query_transform": "hyde"}),
        ({"chat_tier": "fast"}, {"chat_tier": "powerful"}),
        ({"k": 3}, {"k": 8}),
    ],
)
def test_the_simplicity_order(tmp_path, simpler, other):
    common = {
        "parser": "pdfium",
        "dimensions": 1536,
        "embed_model": "openai/text-embedding-3-small",
    }
    assert simplicity({**common, **simpler}) < simplicity({**common, **other})
    a = make_run(tmp_path, "a", SPREAD, **simpler)
    b = make_run(tmp_path, "b", [s + n for s, n in zip(SPREAD, NOISE)], **other)
    assert Ranker(SCORING).rank([b, a])[0].run.id == "a"


def test_the_reranker_outranks_the_parser_in_the_tie_order():
    assert simplicity({"reranker": "none", "parser": "docling"}) < simplicity(
        {"reranker": "minilm", "parser": "pdfium"}
    )


def test_pi_numbers_filter_when_they_exist(tmp_path):
    best = make_run(tmp_path, "best", base(SPREAD, 0.1), profile="p-best")
    good = make_run(tmp_path, "good", SPREAD, profile="p-good")
    unmeasured = make_run(tmp_path, "unmeasured", base(SPREAD, 0.2), profile="p-other")
    pi_best = make_run(tmp_path, "s6-best", SPREAD, stage="S6", host="pi", p95=420.0,
                       profile="p-best")  # fmt: skip
    pi_good = make_run(tmp_path, "s6-good", SPREAD, stage="S6", host="pi", p95=120.0,
                       profile="p-good")  # fmt: skip
    rec = recommend([best, good, unmeasured, pi_best, pi_good], scoring=SCORING)
    assert rec.pi_measured and rec.chosen.run_id == "good"
    assert rec.chosen.latency_host == "pi" and rec.chosen.p95_ms == 120.0
    assert {c.run_id for c in rec.excluded} == {"best", "unmeasured"}
    assert "Pi p95" in rec.reasons[0]
    assert "(pi" not in render_recommendation(rec).split("Over the")[0]  # no flag on Pi numbers


def test_without_pi_numbers_the_host_numbers_are_used_and_flagged(tmp_path):
    slow = make_run(tmp_path, "slow", base(SPREAD, 0.1), p95=500.0, profile="p-slow")
    fast = make_run(tmp_path, "fast", SPREAD, p95=40.0, profile="p-fast")
    rec = recommend([slow, fast], scoring=SCORING, p95_limit_ms=300)
    assert not rec.pi_measured and rec.chosen.run_id == "fast"
    assert rec.chosen.latency_host == "mac" and "flagged" in rec.reasons[0]
    assert "(mac, flagged)" in render_recommendation(rec)
    none_fit = recommend([slow], scoring=SCORING, p95_limit_ms=100)
    assert none_fit.chosen is None and "no configuration meets" in none_fit.reasons[-1]


def test_k_variants_and_unfinished_runs_are_not_candidates(tmp_path):
    base_run = make_run(tmp_path, "baseline", SPREAD, stage="S4")
    k3 = make_run(tmp_path, "k3", base(SPREAD, 0.1), stage="S4", k=3)
    unfinished = make_run(tmp_path, "unfinished", base(SPREAD, 0.2), profile="p2")
    unfinished = unfinished.model_copy(update={"finished_at": None})
    rec = recommend([base_run, k3, unfinished], scoring=SCORING)
    assert rec.chosen.run_id == "baseline" and not rec.alternatives
    assert recommend([], scoring=SCORING).chosen is None


def test_the_latest_stage_stands_for_a_configuration(tmp_path):
    s3 = make_run(tmp_path, "s3", SPREAD, stage="S3")
    s4 = make_run(tmp_path, "s4", SPREAD, stage="S4")
    rec = recommend([s3, s4], scoring=SCORING)
    assert rec.chosen.run_id == "s4" and not rec.alternatives


def test_the_chat_tier_is_fast_unless_powerful_is_significantly_better(tmp_path):
    winner = make_run(tmp_path, "winner", SPREAD, stage="S4", abstain_threshold=0.42)
    fast = make_run(tmp_path, "s5-fast", stage="S5", correct=SPREAD, chat_tier="fast", p95=None)
    tied = make_run(tmp_path, "s5-pow", stage="S5", chat_tier="powerful", p95=None,
                    correct=[s + n for s, n in zip(SPREAD, NOISE)])  # fmt: skip
    rec = recommend([winner, fast, tied], scoring=SCORING)
    assert rec.chat_tier == "fast" and "tie" in rec.chat_tier_reason
    assert rec.abstain_threshold == 0.42
    better = make_run(tmp_path, "s5-pow2", stage="S5", chat_tier="powerful", p95=None,
                      correct=base(SPREAD, 0.1))  # fmt: skip
    rec = recommend([winner, fast, better], scoring=SCORING)
    assert rec.chat_tier == "powerful" and "significantly" in rec.chat_tier_reason
    only = recommend([winner], scoring=SCORING)
    assert only.chat_tier is None and "no S5" in only.chat_tier_reason
    thresholds = {("pdfium.structure-512.hdr.te3s-1536", ""): 0.5}
    assert recommend([winner], scoring=SCORING, thresholds=thresholds).abstain_threshold == 0.5


def test_render_names_the_choice_and_a_transform_is_a_normal_choice(tmp_path):
    hyde = make_run(tmp_path, "hyde", SPREAD, query_transform="hyde")
    rec = recommend([hyde], scoring=SCORING)
    assert rec.chosen.unsupported == ()  # retrieval.transform is a knowledge.yaml key
    text = render_recommendation(rec, choice={"config": "hyde", "run_id": "hyde", "at": "today"})
    assert "Chosen at G-K2:** hyde" in text and "note:" not in text
    assert "not calibrated" in text
    assert "No recommendation yet" in render_recommendation(recommend([], scoring=SCORING))


def test_top_distinct_and_find_run(tmp_path):
    a = make_run(tmp_path, "a", base(SPREAD, 0.2), profile="p1")
    b = make_run(tmp_path, "b", base(SPREAD, 0.1), profile="p1", retrieval="dense")
    c = make_run(tmp_path, "c", SPREAD, profile="p2")
    top = Ranker(SCORING).top([c, b, a], 2, distinct=lambda cfg: cfg["profile"])
    assert [r.id for r in top] == ["a", "c"]
    assert identity(a.config) != identity(b.config)
    s4 = make_run(tmp_path, "s4", SPREAD, stage="S4", name="a")
    assert find_run([a, s4], "a").id == "a"  # a run ID wins
    assert find_run([s4, c], "a").id == "s4"
    assert find_run([c], "missing") is None


def test_write_choice_yaml_keeps_the_comments_and_validates(tmp_path):
    target = tmp_path / "knowledge.yaml"
    target.write_text((REPO / "config" / "knowledge.yaml").read_text())
    write_choice_yaml(
        target,
        profile="pdfium.structure-512.hdr.te3s-1536",
        reranker="minilm",
        method="weighted",
        alpha=0.3,
        budget=2000,
        comment=["chosen by a test"],
        transform="multi",
        chat_tier="powerful",
    )
    text = target.read_text()
    parsed = load_knowledge_file(target)
    assert parsed.retrieval.transform == "multi" and parsed.answer.chat_tier == "powerful"
    assert RetrievalConfig.from_file(parsed).transform == "multi"
    assert parsed.active_profile == "pdfium.structure-512.hdr.te3s-1536"
    assert parsed.reranker == "minilm"
    assert (parsed.retrieval.method, parsed.retrieval.weighted_alpha) == ("weighted", 0.3)
    assert parsed.retrieval.context_budget_tokens == 2000
    assert "# G-K2: chosen by a test" in text and "rrf_k: 60" in text
    assert "# dense | bm25 | rrf | weighted" in text  # the line comments survive
    write_choice_yaml(target, profile="p.x", reranker="none", method="rrf", alpha=None,
                      budget=4000, comment=["again"])  # fmt: skip
    text = target.read_text()
    assert text.count("# G-K2:") == 1 and "reranker: null" in text
    assert load_knowledge_file(target).retrieval.weighted_alpha is None
    empty = tmp_path / "new.yaml"
    write_choice_yaml(empty, profile="p.y", reranker="none", method="dense", alpha=None,
                      budget=1000, comment=["new"])  # fmt: skip
    assert load_knowledge_file(empty).retrieval.method == "dense"
    with pytest.raises(ValueError):
        write_choice_yaml(empty, profile="p.y", reranker="none", method="nope", alpha=None,
                          budget=1000, comment=[])  # fmt: skip
    assert load_knowledge_file(empty).retrieval.method == "dense"  # unchanged on failure


def test_only_the_last_retrieval_stage_is_a_candidate(tmp_path):
    """Earlier stages' configurations lost on the way; the finalists carry S5 and S6."""
    earlier = make_run(tmp_path, "s2-best", base(SPREAD, 0.2), stage="S2", profile="p-early")
    finalist = make_run(tmp_path, "s4-final", SPREAD, stage="S4")
    rec = recommend([earlier, finalist], scoring=SCORING)
    assert rec.chosen.run_id == "s4-final" and not rec.alternatives


def test_exact_ties_break_by_name_not_by_run_id(tmp_path):
    x = make_run(tmp_path, "zz-run", SPREAD, name="a-config", profile="p1")
    y = make_run(tmp_path, "aa-run", SPREAD, name="b-config", profile="p2")
    assert Ranker(SCORING).rank([y, x])[0].run.id == "zz-run"


def test_the_chat_tier_comes_from_knowledge_yaml_unless_kb_chat_tier_is_set(tmp_path, monkeypatch):
    from hal9000.knowledge.config import settings

    (tmp_path / "knowledge.yaml").write_text("answer:\n  chat_tier: powerful\n")
    monkeypatch.setenv("HAL_CONFIG_DIR", str(tmp_path))
    assert settings().chat_tier == "powerful"
    monkeypatch.setenv("KB_CHAT_TIER", "fast")
    assert settings().chat_tier == "fast"
    monkeypatch.delenv("KB_CHAT_TIER")
    (tmp_path / "knowledge.yaml").write_text("answer:\n  chat_tier: null\n")
    assert settings().chat_tier == "fast"
    (tmp_path / "knowledge.yaml").write_text("answer: [unclosed\n")
    assert settings().chat_tier == "fast"  # the yaml's own loader reports the error
    (tmp_path / "knowledge.yaml").write_text("answer:\n  chat_tier: medium\n")
    with pytest.raises(Exception, match="must be one of"):
        settings()


def test_the_retrieval_transform_is_read_from_knowledge_yaml(tmp_path):
    from hal9000.knowledge.config import KnowledgeFile

    kf = KnowledgeFile.model_validate({"retrieval": {"transform": "hyde"}})
    assert RetrievalConfig.from_file(kf).transform == "hyde"
    assert RetrievalConfig.from_file(KnowledgeFile()).transform == "none"
    with pytest.raises(ValueError):
        KnowledgeFile.model_validate({"retrieval": {"transform": "rewrite"}})


def _test_run(tmp_path, *, recall, mrr, correct, faithful, declined, forced=False):
    ci = lambda v, lo, hi: MetricCI(value=v, low=lo, high=hi)  # noqa: E731
    metrics = EvalMetrics(
        n_questions=48,
        budgeted_recall={2000: ci(*recall)},
        mrr=ci(*mrr),
        correct=ci(*correct) if correct else None,
        faithful=ci(*faithful),
    )
    config = {"name": "chosen", "declined": ci(*declined).model_dump()}
    if forced:
        config["forced_rerun"] = True
    return EvalRun(id="t1", stage="test", split="test", config=config, metrics=metrics,
                   finished_at=datetime.now(UTC))  # fmt: skip


def test_the_plan_s1_pass_rule_on_test_split_scores(tmp_path):
    run = _test_run(
        tmp_path,
        recall=(0.86, 0.80, 0.91),  # upper bound reaches 0.90, point within 5 points: pass
        mrr=(0.69, 0.60, 0.80),  # point 6 points short: fail
        correct=None,  # not measured
        faithful=(0.92, 0.90, 0.94),  # point within 5 points, upper bound short: fail
        declined=(0.80, 0.70, 0.90),  # exactly on the target: pass
    )
    scores = {s.key: s for s in target_scores(run)}
    assert [s.key for s in target_scores(run)] == ["recall", "mrr", "correct", "faithful",
                                                   "declined"]  # fmt: skip
    assert scores["recall"].passes is True
    assert scores["mrr"].passes is False
    assert scores["correct"].passes is None
    assert scores["faithful"].passes is False
    assert scores["declined"].passes is True
    assert "PASS" in scores["recall"].describe() and "not measured" in scores["correct"].describe()
    text = render_targets([run])
    assert "| Evidence recall within a 2,000-token budget | >= 0.90 |" in text
    assert text.count("| PASS |") == 2 and text.count("| FAIL |") == 2
    assert "not measured" in text and render_targets([]) == ""
    forced = _test_run(tmp_path, recall=(0.9, 0.8, 1.0), mrr=(0.8, 0.7, 0.9), correct=None,
                       faithful=(1, 1, 1), declined=(1, 1, 1), forced=True)  # fmt: skip
    assert "a forced re-run" in render_targets([forced])


def test_s6_rows_match_finalists_by_configuration(tmp_path):
    """A transform finalist and two finalists on one profile each find their own Pi row."""
    multi = make_run(tmp_path, "multi", SPREAD, stage="S4", query_transform="multi")
    rrf = make_run(tmp_path, "rrf", SPREAD, stage="S4")
    dense = make_run(tmp_path, "dense", SPREAD, stage="S4", retrieval="dense")
    pi = [
        make_run(tmp_path, f"s6-{run.id}", SPREAD, stage="S6", host="pi", p95=p95,
                 **{k: v for k, v in run.config.items() if k != "name"})
        for run, p95 in ((multi, 90.0), (rrf, 80.0), (dense, 350.0))
    ]  # fmt: skip
    rec = recommend([multi, rrf, dense, *pi], scoring=SCORING)
    assert rec.pi_measured
    measured = {rec.chosen.run_id: rec.chosen.p95_ms}
    measured.update({a.run_id: a.p95_ms for a in rec.alternatives})
    assert measured == {"rrf": 80.0, "multi": 90.0}  # each its own row, both eligible
    assert [(e.run_id, e.p95_ms) for e in rec.excluded] == [("dense", 350.0)]  # over 300 ms
