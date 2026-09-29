"""The plan §8.6 decision rule, and the tie-aware ranking the sweep selects with.

**Ranking** (:class:`Ranker`). Configurations are ordered by budgeted recall
@ 2k on the dev split, but a lead only counts when a paired bootstrap on the
per-question differences says so (plan §8.2). Repeatedly: the configuration
with the highest point estimate leads; every remaining configuration the
paired bootstrap calls a tie with it (or can't compare: too few shared
questions) is a contender; the **simplest** contender is placed next
(:func:`simplicity`), then the higher point estimate. The sweep takes its
top *n* between stages from this order, and the decision rule its winner.

**Simplicity** (plan §8.6, in priority order): no reranker over a reranker
(MiniLM over bge-base), ``pypdfium2`` over Docling, a smaller embedding over
a larger one (fewer dimensions, then the cheaper model), no query
transformation over one, the fast chat tier over the powerful one, and, for
the T9 measurements, a smaller *k*. Nothing else (size, strategy, method) is
ranked by simplicity: the plan gives no order for it, so the point estimate
decides.

**Decision** (:func:`recommend`). Candidates are the finished dev runs of
the sweep's last retrieval stage (normally the S4 baselines: the finalists,
which were calibrated, answered in S5 and measured in S6; earlier stages'
configurations lost to them on the way). The S4 *k* variants are T9
measurements, not candidates: ``config/knowledge.yaml`` has no *k*. Local retrieval p95
must be within ``p95_local_retrieval_ms`` (300 ms): measured on the Pi (S6
runs, matched by configuration) when any Pi numbers exist, else the Mac
numbers of every configuration, flagged as such. The winner of the ranking is
recommended; the next ones are the alternatives, each with its comparison to
the winner. The chat tier comes from S5: the fast tier unless the powerful
one is significantly more correct on the same configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from hal9000 import config as hal_config
from hal9000.knowledge.contracts import EvalRun, MetricCI
from hal9000.knowledge.eval.runs import load_results
from hal9000.knowledge.eval.scoring import Comparison, ScoringConfig, paired_bootstrap

__all__ = [
    "DECISION_BUDGET",
    "RETRIEVAL_STAGES",
    "Candidate",
    "Ranked",
    "Ranker",
    "Recommendation",
    "identity",
    "recommend",
    "render_recommendation",
    "render_targets",
    "simplicity",
    "target_scores",
]

#: plan §8.6: the highest budgeted recall @ 2k on dev
DECISION_BUDGET = 2000
RETRIEVAL_STAGES = ("S1a", "S1b", "S2", "S3", "S4")

_RERANK_ORDER = {"none": 0, "": 0, "minilm": 1, "bge-base": 2}
_PARSER_ORDER = {"pdfium": 0, "docling": 1}
_TIER_ORDER = {"fast": 0, "powerful": 1}
#: identity of a configuration for matching runs across stages (S5, S6 against S1-S4)
IDENTITY_KEYS = ("profile", "retrieval", "alpha", "reranker", "query_transform")


def _embed_rate(model: str) -> float:
    return hal_config.rates_for(model)[0] if model else 0.0


def simplicity(config: Mapping) -> tuple:
    """Sort key: lower is simpler (plan §8.6 order, then *k*)."""
    reranker = str(config.get("reranker") or "none")
    transform = str(config.get("query_transform") or "none")
    k = config.get("k")
    return (
        _RERANK_ORDER.get(reranker, 3),
        _PARSER_ORDER.get(str(config.get("parser") or "pdfium"), 2),
        int(config.get("dimensions") or 0),
        _embed_rate(str(config.get("embed_model") or "")),
        0 if transform == "none" else 1,
        _TIER_ORDER.get(str(config.get("chat_tier") or "fast"), 2),
        k if isinstance(k, int) else 10**6,
    )


def identity(config: Mapping) -> tuple:
    """What makes two runs the same chat configuration (ignores stage, k, tier, repeats)."""
    return tuple(
        (config.get(key) if config.get(key) is not None else "none")
        if key in ("reranker", "query_transform")
        else config.get(key)
        for key in IDENTITY_KEYS
    )


def _label(run: EvalRun) -> str:
    """Exact ties are broken by configuration name (stable across stages), then run ID."""
    return f"{run.config.get('name') or ''}\x00{run.id}"


def _recall(run: EvalRun, budget: int) -> MetricCI | None:
    return run.metrics.budgeted_recall.get(budget) if run.metrics else None


@dataclass(frozen=True)
class Ranked:
    """One run in ranked order, with the runs it was tied with when it was placed."""

    run: EvalRun
    value: float | None
    #: IDs of runs the paired bootstrap could not separate from the leader at this step
    tied: tuple[str, ...] = ()
    #: the leader at this step when this run was placed by simplicity, else None
    over: str | None = None


class Ranker:
    """Tie-aware ranking by one per-question metric (budgeted recall @ 2k by default)."""

    def __init__(
        self,
        scoring: ScoringConfig | None = None,
        *,
        budget: int = DECISION_BUDGET,
        simplicity_key: Callable[[Mapping], tuple] = simplicity,
    ) -> None:
        self.scoring = scoring or ScoringConfig()
        self.budget = budget
        self.simplicity_key = simplicity_key
        self._values: dict[str, dict[int, float]] = {}

    # ------------------------------------------------------------ values
    def values(self, run: EvalRun, metric: str = "recall") -> dict[int, float]:
        """Per-question values of ``run`` from its results file (answerable questions)."""
        key = f"{run.id}:{metric}"
        if key not in self._values:
            out: dict[int, float] = {}
            for record in load_results(run):
                if metric == "recall":
                    value = (record.get("recall") or {}).get(str(self.budget))
                else:
                    value = record.get(metric)
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, int | float):
                    out[int(record["gold_item_id"])] = float(value)
            self._values[key] = out
        return self._values[key]

    def point(self, run: EvalRun, metric: str = "recall") -> float | None:
        if metric == "recall":
            ci = _recall(run, self.budget)
            return ci.value if ci else None
        values = self.values(run, metric)
        return sum(values.values()) / len(values) if values else None

    def compare(self, a: EvalRun, b: EvalRun, metric: str = "recall") -> Comparison:
        """Paired bootstrap of A against B on the questions both scored."""
        cfg = self.scoring
        return paired_bootstrap(
            self.values(a, metric),
            self.values(b, metric),
            n_boot=cfg.n_boot,
            seed=cfg.seed,
            min_paired=cfg.min_paired,
        )

    # ------------------------------------------------------------ ranking
    def rank(self, runs: Iterable[EvalRun], metric: str = "recall") -> list[Ranked]:
        """Every run, best first; runs without the metric come last (by ID)."""
        scored: list[tuple[EvalRun, float]] = []
        unscored: list[EvalRun] = []
        for run in runs:
            value = self.point(run, metric)
            (unscored.append(run) if value is None else scored.append((run, value)))
        order: list[Ranked] = []
        remaining = sorted(scored, key=lambda p: _label(p[0]))
        while remaining:
            leader, _ = min(
                remaining,
                key=lambda p: (-p[1], self.simplicity_key(p[0].config), _label(p[0])),
            )
            contenders = [
                (run, value)
                for run, value in remaining
                if run is leader
                or self.compare(leader, run, metric).verdict in ("tie", "insufficient")
            ]
            pick, value = min(
                contenders,
                key=lambda p: (self.simplicity_key(p[0].config), -p[1], _label(p[0])),
            )
            tied = tuple(r.id for r, _ in contenders if r is not leader)
            order.append(Ranked(pick, value, tied, over=leader.id if pick is not leader else None))
            remaining = [p for p in remaining if p[0] is not pick]
        order.extend(Ranked(run, None) for run in sorted(unscored, key=lambda r: r.id))
        return order

    def top(
        self,
        runs: Iterable[EvalRun],
        n: int,
        *,
        distinct: Callable[[Mapping], object] | None = None,
        metric: str = "recall",
    ) -> list[EvalRun]:
        """The first ``n`` ranked runs (only the best run per ``distinct`` key, if given)."""
        chosen: list[EvalRun] = []
        seen: set = set()
        for item in self.rank(runs, metric):
            if item.value is None:
                continue
            if distinct is not None:
                key = distinct(item.run.config)
                if key in seen:
                    continue
                seen.add(key)
            chosen.append(item.run)
            if len(chosen) >= n:
                break
        return chosen


# ------------------------------------------------------------------ decision
@dataclass(frozen=True)
class Candidate:
    run_id: str
    name: str
    recall: MetricCI | None
    p95_ms: float | None
    #: where p95 was measured: "pi", or the host class of the sweep run (flagged)
    latency_host: str | None
    within_latency: bool
    #: the paired comparison with the recommendation (None for the recommendation itself)
    versus: Comparison | None = None
    unsupported: tuple[str, ...] = ()


@dataclass
class Recommendation:
    """The plan §8.6 outcome for one sweep."""

    chosen: Candidate | None
    alternatives: list[Candidate] = field(default_factory=list)
    excluded: list[Candidate] = field(default_factory=list)
    pi_measured: bool = False
    p95_limit_ms: float = 300.0
    reasons: list[str] = field(default_factory=list)
    chat_tier: str | None = None
    chat_tier_reason: str = ""
    abstain_threshold: float | None = None
    config: dict = field(default_factory=dict)


def _unsupported(config: Mapping) -> tuple[str, ...]:
    """Settings of a configuration the chat can't be configured with (none at present:
    ``retrieval.transform`` and ``answer.chat_tier`` are ``knowledge.yaml`` keys, and
    *k* variants are never candidates)."""
    return ()


def _chat_tier(ranker: Ranker, winner: EvalRun, s5: Sequence[EvalRun]) -> tuple[str | None, str]:
    """fast unless powerful is significantly more correct on the winner's configuration."""
    same = [r for r in s5 if identity(r.config) == identity(winner.config) and r.metrics]
    by_tier: dict[str, EvalRun] = {}
    for run in sorted(same, key=lambda r: (int(r.config.get("repeat") or 0), r.id)):
        by_tier.setdefault(str(run.config.get("chat_tier")), run)
    if not by_tier:
        return None, "no S5 answers for this configuration: keep KB_CHAT_TIER as it is"
    if "fast" in by_tier and "powerful" in by_tier:
        comparison = ranker.compare(by_tier["powerful"], by_tier["fast"], metric="correct")
        fast, powerful = (ranker.point(by_tier[t], "correct") for t in ("fast", "powerful"))
        numbers = f"correct {fast:.3f} (fast) vs {powerful:.3f} (powerful)"
        if comparison.verdict == "a":
            return "powerful", f"{numbers}; the powerful tier is significantly more correct"
        return "fast", f"{numbers}; {comparison.verdict} by paired bootstrap, so the fast tier"
    only = next(iter(by_tier))
    return only, f"only the {only} tier was answered in S5"


def recommend(
    runs: Sequence[EvalRun],
    *,
    scoring: ScoringConfig | None = None,
    p95_limit_ms: float = 300.0,
    budget: int = DECISION_BUDGET,
    alternatives: int = 3,
    thresholds: Mapping[tuple[str, str], float] | None = None,
) -> Recommendation:
    """Apply plan §8.6 to a sweep's runs (and any S6 runs among them)."""
    ranker = Ranker(scoring, budget=budget)
    finished = [r for r in runs if r.finished_at is not None and r.metrics is not None]
    from hal9000.knowledge.obs.collector import stage_order

    retrieval = [
        r
        for r in finished
        if r.split == "dev" and r.stage in RETRIEVAL_STAGES and r.config.get("k") is None
    ]
    last = max((r.stage for r in retrieval), key=stage_order, default=None)
    candidates = [r for r in retrieval if r.stage == last]
    pi_runs = [r for r in finished if r.stage == "S6" and r.host == "pi"]
    pi_p95: dict[tuple, float] = {}
    for run in pi_runs:
        if run.metrics.latency_ms_p95 is not None:
            pi_p95[identity(run.config)] = run.metrics.latency_ms_p95
    rec = Recommendation(chosen=None, pi_measured=bool(pi_p95), p95_limit_ms=p95_limit_ms)
    if not candidates:
        rec.reasons.append("no finished retrieval runs (S1a-S4) to choose from")
        return rec

    def latency(run: EvalRun) -> tuple[float | None, str | None]:
        if rec.pi_measured:
            p95 = pi_p95.get(identity(run.config))
            return p95, "pi" if p95 is not None else None
        return run.metrics.latency_ms_p95, run.host or None

    # one run per configuration (a stage may hold the same one twice)
    latest: dict[tuple, EvalRun] = {}
    for run in sorted(candidates, key=lambda r: r.id):
        latest[identity(run.config)] = run
    pool = list(latest.values())

    def candidate(run: EvalRun, versus: Comparison | None = None) -> Candidate:
        p95, host = latency(run)
        return Candidate(
            run_id=run.id,
            name=str(run.config.get("name") or run.id),
            recall=_recall(run, budget),
            p95_ms=p95,
            latency_host=host,
            within_latency=p95 is not None and p95 <= p95_limit_ms,
            versus=versus,
            unsupported=_unsupported(run.config),
        )

    eligible = [r for r in pool if candidate(r).within_latency]
    rec.excluded = [candidate(r) for r in pool if r not in eligible]
    if rec.pi_measured:
        rec.reasons.append(
            f"latency filter: Pi p95 local retrieval <= {p95_limit_ms:.0f} ms (S6); "
            "configurations not measured on the Pi are not eligible"
        )
    else:
        rec.reasons.append(
            f"latency filter: no Pi numbers yet (S6), so the sweep host's p95 local retrieval "
            f"<= {p95_limit_ms:.0f} ms is used instead -- flagged: re-check on the Pi"
        )
    if not eligible:
        rec.reasons.append("no configuration meets the latency limit")
        return rec
    ranked = ranker.rank(eligible)
    first = ranked[0]
    winner = first.run
    rec.chosen = candidate(winner)
    rec.config = dict(winner.config)
    value = f"{first.value:.3f}" if first.value is not None else "N/A"
    rec.reasons.append(
        f"highest budgeted recall @ {budget // 1000}k on dev: {rec.chosen.name} ({value})"
    )
    if first.over is not None:
        rec.reasons.append(
            f"it ties (paired bootstrap) with the point-estimate leader {first.over} and is the "
            "simpler option (plan §8.6 order)"
        )
    elif first.tied:
        rec.reasons.append(
            f"{len(first.tied)} configuration(s) tie with it by paired bootstrap; it is also "
            "the simplest of them"
        )
    for item in ranked[1 : 1 + alternatives]:
        rec.alternatives.append(candidate(item.run, ranker.compare(winner, item.run)))
    s5 = [r for r in finished if r.stage == "S5"]
    rec.chat_tier, rec.chat_tier_reason = _chat_tier(ranker, winner, s5)
    reranker = str(winner.config.get("reranker") or "none")
    key = (str(winner.config.get("profile")), "" if reranker == "none" else reranker)
    if thresholds and key in thresholds:
        rec.abstain_threshold = thresholds[key]
    elif isinstance(winner.config.get("abstain_threshold"), int | float):
        rec.abstain_threshold = float(winner.config["abstain_threshold"])
    return rec


def _ci(ci: MetricCI | None) -> str:
    return "N/A" if ci is None else f"{ci.value:.3f} [{ci.low:.3f}-{ci.high:.3f}]"


def _ms(c: Candidate) -> str:
    if c.p95_ms is None:
        return "not measured"
    flag = "" if c.latency_host == "pi" else f" ({c.latency_host or 'host'}, flagged)"
    return f"{c.p95_ms:.1f}{flag}"


def _versus(c: Candidate) -> str:
    v = c.versus
    if v is None:
        return ""
    word = {"a": "worse", "b": "better", "tie": "tie", "insufficient": "too few shared"}[v.verdict]
    return f"{word} (diff {-v.mean_diff:+.3f} [{-v.high:+.3f}, {-v.low:+.3f}], n={v.n})"


def render_recommendation(rec: Recommendation, *, choice: Mapping | None = None) -> str:
    """Markdown for the report's "Chosen configuration" section (names and numbers only)."""
    lines = ["### Recommendation (plan §8.6)", ""]
    if rec.chosen is None:
        lines.append("No recommendation yet: " + "; ".join(rec.reasons) + ".")
        lines.append("")
    else:
        c = rec.chosen
        lines += [
            f"**{c.name}** (run `{c.run_id}`): recall @ 2k {_ci(c.recall)}, "
            f"p95 local retrieval {_ms(c)} ms.",
            "",
        ]
        lines += [f"- {reason}" for reason in rec.reasons]
        lines.append(f"- chat tier: {rec.chat_tier or 'unchanged'} ({rec.chat_tier_reason})")
        threshold = (
            f"{rec.abstain_threshold:.4f}"
            if rec.abstain_threshold is not None
            else "not calibrated (the generation-side NOT_IN_SOURCES reply decides)"
        )
        lines.append(f"- abstain threshold: {threshold}")
        for note in c.unsupported:
            lines.append(f"- note: {note}")
        lines.append("")
        if rec.alternatives:
            lines += [
                "| alternative | recall@2k | p95 ms | vs recommendation |",
                "|---|---|---|---|",
            ]
            lines += [
                f"| {a.name} | {_ci(a.recall)} | {_ms(a)} | {_versus(a)} |"
                for a in rec.alternatives
            ]
            lines.append("")
    if rec.excluded:
        lines.append(
            f"Over the {rec.p95_limit_ms:.0f} ms latency limit or not measured: "
            + ", ".join(f"{e.name} ({_ms(e)})" for e in rec.excluded[:10])
            + (f" and {len(rec.excluded) - 10} more" if len(rec.excluded) > 10 else "")
            + "."
        )
        lines.append("")
    if choice:
        lines.append(
            f"**Chosen at G-K2:** {choice.get('config')} (run `{choice.get('run_id')}`), "
            f"recorded {choice.get('at')}."
        )
        lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------- choose
_MARK = "# G-K2:"


def _scalar(value: object) -> str:
    """A YAML scalar for ``value`` (JSON strings are valid YAML)."""
    import json
    import re

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_.\-/]+", text) and text not in ("null", "true", "false"):
        return text
    return json.dumps(text)


def _set_line(lines: list[str], key: str, value: object, parent: str | None = None) -> None:
    """Replace ``key:`` (top level, or inside ``parent:``), keeping its comment; add if missing."""
    import re

    pattern = re.compile(rf"^(\s*){re.escape(key)}:(\s*)([^#]*?)(\s*#.*)?$")
    start, end, indent = 0, len(lines), ""
    if parent is not None:
        head = next(
            (
                i
                for i, line in enumerate(lines)
                if re.match(rf"^{re.escape(parent)}:\s*(#.*)?$", line)
            ),
            None,
        )
        if head is None:
            lines.append(f"{parent}:")
            head = len(lines) - 1
        start = head + 1
        end = next(
            (i for i in range(start, len(lines)) if lines[i] and not lines[i][0].isspace()
             and not lines[i].startswith("#")),
            len(lines),
        )  # fmt: skip
        indent = "  "
    for i in range(start, end):
        match = pattern.match(lines[i])
        if match and (match.group(1) == "") == (parent is None):
            gap = match.group(2) or " "
            lines[i] = f"{match.group(1)}{key}:{gap}{_scalar(value)}{match.group(4) or ''}"
            return
    lines.insert(end if parent is not None else len(lines), f"{indent}{key}: {_scalar(value)}")


def write_choice_yaml(
    path, *, profile: str, reranker: str, method: str, alpha: float | None, budget: int,
    comment: Sequence[str], transform: str = "none", chat_tier: str | None = None,
) -> None:  # fmt: skip
    """Write the chosen configuration into ``config/knowledge.yaml``, keeping its comments.

    Sets ``active_profile``, ``reranker``, the ``retrieval:`` block's ``method``,
    ``weighted_alpha``, ``context_budget_tokens`` and ``transform``, and
    ``answer.chat_tier`` (``KB_CHAT_TIER`` in ``.env`` still overrides it);
    replaces the ``# G-K2:`` comment lines. The abstain threshold lives in
    ``abstain_thresholds`` in the database. The result is validated before it
    replaces the file.
    """
    from pathlib import Path

    from hal9000.knowledge.config import load_knowledge_file

    target = Path(path)
    original = target.read_text() if target.exists() else ""
    lines = [line for line in original.splitlines() if not line.startswith(_MARK)]
    _set_line(lines, "active_profile", profile)
    _set_line(lines, "reranker", None if reranker in ("", "none") else reranker)
    _set_line(lines, "method", method, parent="retrieval")
    _set_line(lines, "weighted_alpha", alpha if method == "weighted" else None, parent="retrieval")
    _set_line(lines, "context_budget_tokens", budget, parent="retrieval")
    _set_line(lines, "transform", transform or "none", parent="retrieval")
    if chat_tier is not None:
        _set_line(lines, "chat_tier", chat_tier, parent="answer")
    at = next((i for i, line in enumerate(lines) if line.startswith("active_profile:")), 0)
    lines[at:at] = [f"{_MARK} {text}" for text in comment]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    try:
        load_knowledge_file(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(target)


def find_run(runs: Sequence[EvalRun], name: str) -> EvalRun | None:
    """The run with ID ``name``, else the latest-stage finished run whose config is named so."""
    from hal9000.knowledge.obs.collector import stage_order

    for run in runs:
        if run.id == name:
            return run
    named = [r for r in runs if r.config.get("name") == name and r.finished_at is not None]
    if not named:
        return None
    return max(named, key=lambda r: (stage_order(r.stage), r.id))


# ------------------------------------------------------------ plan §1 targets
#: plan §1 quality targets on the test split: (label, target)
TARGETS: tuple[tuple[str, str, float], ...] = (
    ("recall", "Evidence recall within a 2,000-token budget", 0.90),
    ("mrr", "MRR of the first evidence-bearing chunk", 0.75),
    ("correct", "Correct (judge)", 0.85),
    ("faithful", "Faithful to the cited sources (judge)", 0.95),
    ("declined", "Unanswerable questions correctly declined", 0.80),
)
#: plan §1: the point estimate must be within this of the target
TARGET_SLACK = 0.05


@dataclass(frozen=True)
class TargetScore:
    key: str
    label: str
    target: float
    ci: MetricCI | None

    @property
    def passes(self) -> bool | None:
        """Plan §1: the CI's upper bound reaches the target and the point estimate is
        within 5 points of it; None when not measured."""
        if self.ci is None:
            return None
        return self.ci.high >= self.target and self.ci.value >= self.target - TARGET_SLACK

    def describe(self) -> str:
        verdict = {True: "PASS", False: "FAIL", None: "not measured"}[self.passes]
        return f"{_ci(self.ci)} against >= {self.target:.2f}: {verdict}"


def target_scores(run: EvalRun, *, budget: int = DECISION_BUDGET) -> list[TargetScore]:
    """A test-split run's scores against the plan §1 quality targets."""
    m = run.metrics
    declined = run.config.get("declined")
    values = {
        "recall": m.budgeted_recall.get(budget) if m else None,
        "mrr": m.mrr if m else None,
        "correct": m.correct if m else None,
        "faithful": m.faithful if m else None,
        "declined": MetricCI.model_validate(declined) if isinstance(declined, dict) else None,
    }
    return [TargetScore(key, label, target, values[key]) for key, label, target in TARGETS]


def render_targets(runs: Sequence[EvalRun]) -> str:
    """Markdown: each test-split run's scores with CIs against the plan §1 targets."""
    tests = [r for r in runs if r.split == "test" and r.metrics is not None]
    if not tests:
        return ""
    lines = [
        "Plan §1 targets (a target passes when the 95% interval's upper bound reaches it and "
        "the point estimate is within 5 points of it):",
        "",
    ]
    for run in tests:
        forced = " (a forced re-run)" if run.config.get("forced_rerun") else ""
        lines += [
            f"**{run.config.get('name', run.id)}** (run `{run.id}`{forced}, "
            f"{run.metrics.n_questions} test questions):",
            "",
            "| metric | target | test split [95% CI] | result |",
            "|---|---|---|---|",
        ]
        for score in target_scores(run):
            verdict = {True: "PASS", False: "FAIL", None: "not measured"}[score.passes]
            lines.append(f"| {score.label} | >= {score.target:.2f} | {_ci(score.ci)} | {verdict} |")
        lines.append("")
    return "\n".join(lines)
