"""The staged sweep S1a-S5 (plan §8.4), resumable, estimated and capped (rule K9).

``hal9000 kb eval sweep --stages S1a,S1b,S2,S3,S4`` and, after judge
calibration, ``--stages S5 --resume SWEEP``. One *sweep* (``sweep-...`` ID)
is a set of ``eval_runs`` rows, one per configuration, each with
``config.sweep_id``; their files are under ``data/knowledge/eval/runs/``.
Notes (unavailable arms, calibrations) go to ``eval/sweeps/<id>.json``.

Stages (grids in ``config/knowledge_eval.yaml`` ``sweep:``):

- **S1a** size x strategy (T1 x T3): fixed, recursive and structure at every
  size, parent-child children under one parent size; pdfium, 15% overlap for
  fixed only, header on, one embedding; dense and RRF hybrid.
- **S1b** overlap x header x parser (T2 x T4 x T5) on the top 5 chunkings of
  S1a, each with its better method. Docling only if importable; otherwise the
  arm is skipped and "docling unavailable" recorded.
- **S2** embedding models and dimensions (T6) on the top 4 profiles of S1,
  with the best method of S1.
- **S3** retrieval methods x rerankers (T7 x T8) on the top 3 profiles of S2
  (a reranker only if its model loads, else recorded unavailable), then query
  transforms (T10, paid) on the top 2 configurations.
- **S4** *k* (T9) on the top 3 configurations of S3, plus each one's
  baseline (no *k*: ``config/knowledge.yaml`` has none), which carries the
  abstain calibration (T12, plan §8.5); the top 3 baselines are the finalists,
  and their thresholds are written to ``abstain_thresholds``.
- **S5** end to end on dev: the finalists x chat tier, answers judged, the
  top configuration repeated ``repeat_top`` more time(s). Refused unless the
  judge agrees with Mike's labels at >= 85% (``--force-judge`` overrides it,
  logged and recorded on the runs).

Between stages the top *n* come from :class:`~.decide.Ranker`: budgeted
recall @ 2k on dev with the paired bootstrap, ties to the simpler option.

**Budgets.** Retrieval runs select up to the largest scored budget and are
scored at every budget (1k, 2k, 4k), so T9's budgets are columns of every
run. **Latency.** Every retrieval run records local retrieval latency (every
retrieval stage except ``condense`` and ``embed_query``, from the stage timer)
on this host; the host class is stored with the run.

**Money.** Before each stage (and S3's paid second half) the estimate is
printed: embedding tokens x price for the texts not yet in ``embed_cache``
(profile builds and dev questions), LLM calls x estimated tokens x price for
T10, and answers plus judging for S5. The stage needs ``--yes`` or a "y". The
meter's hard cap is ``--max-cost`` and ``KB_EVAL_CAP_USD`` caps the month;
at either the sweep stops, keeps every finished run and can be resumed (a
configuration whose run is finished is skipped). Everything is metered as
``knowledge-eval``. Evaluation profiles are built but never activated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TextIO

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hal9000.cost.metering import BudgetExceeded
from hal9000.knowledge.config import EVAL_AGENT, KnowledgeSettings
from hal9000.knowledge.contracts import EvalMetrics, EvalRun, GoldItem, IndexProfile, MetricCI
from hal9000.knowledge.eval.config import EvalConfig
from hal9000.knowledge.eval.decide import DECISION_BUDGET, Ranker
from hal9000.knowledge.eval.gold import estimate_cost
from hal9000.knowledge.eval.runs import get_run, list_runs, new_run_id, question_record, record_run
from hal9000.knowledge.eval.scoring import (
    QuestionScore,
    ScoringItem,
    aggregate,
    bootstrap_ci,
    roc_auc,
    score_question,
)
from hal9000.knowledge.eval.store import list_gold
from hal9000.knowledge.eval.types import AnswerOutput, RetrievalOutput, Source

__all__ = [
    "STAGES",
    "Calibration",
    "SweepConfig",
    "SweepDeps",
    "SweepOutcome",
    "SweepRunner",
    "SweepSettings",
    "calibrate_threshold",
    "config_answerer",
    "load_sweep_settings",
    "local_latency_ms",
    "manifest_path",
    "profile_texts",
    "read_manifest",
]

log = logging.getLogger(__name__)

STAGES = ("S1a", "S1b", "S2", "S3", "S4", "S5")
METHODS = ("dense", "bm25", "rrf", "weighted")
TRANSFORMS = ("none", "multi", "hyde")
RERANKER_NAMES = ("none", "minilm", "bge-base")
#: stages excluded from local retrieval latency (plan §1: the embedding call is not local)
NOT_LOCAL = frozenset({"condense", "embed_query"})


# ==================================================================== settings
class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingChoice(_M):
    model: str
    dimensions: int = Field(gt=0)


class ParentChildGrid(_M):
    child_sizes: list[int]
    parent_size: int = Field(gt=0)


def _methods(value: list[str]) -> list[str]:
    bad = [v for v in value if v not in METHODS]
    if bad:
        raise ValueError(f"unknown retrieval method(s) {bad}: use {', '.join(METHODS)}")
    return value


class S1aGrid(_M):
    sizes: list[int]
    chunkers: list[Literal["fixed", "recursive", "structure"]]
    fixed_overlap: float = Field(default=0.15, ge=0, lt=1)
    parent_child: ParentChildGrid | None = None
    parser: str = "pdfium"
    header: bool = True
    embedding: EmbeddingChoice
    retrieval: list[str] = Field(default_factory=lambda: ["dense", "rrf"])

    _check = field_validator("retrieval")(_methods)


class S1bGrid(_M):
    top: int = Field(default=5, ge=1)
    overlap: list[float] = Field(default_factory=lambda: [0.0, 0.15])
    header: list[bool] = Field(default_factory=lambda: [False, True])
    parser: list[str] = Field(default_factory=lambda: ["pdfium", "docling"])


class S2Grid(_M):
    top: int = Field(default=4, ge=1)
    embeddings: list[EmbeddingChoice]


class S3Grid(_M):
    top: int = Field(default=3, ge=1)
    retrieval: list[str] = Field(default_factory=lambda: list(METHODS))
    weighted_alpha: list[float] = Field(default_factory=lambda: [0.3, 0.5, 0.7])
    rerankers: list[str] = Field(default_factory=lambda: list(RERANKER_NAMES))
    query_transform_top: int = Field(default=2, ge=0)
    query_transforms: list[str] = Field(default_factory=lambda: ["multi", "hyde"])

    _check = field_validator("retrieval")(_methods)

    @field_validator("rerankers")
    @classmethod
    def _rerankers(cls, value: list[str]) -> list[str]:
        bad = [v for v in value if v not in RERANKER_NAMES]
        if bad:
            raise ValueError(f"unknown reranker(s) {bad}: use {', '.join(RERANKER_NAMES)}")
        return value

    @field_validator("query_transforms")
    @classmethod
    def _transforms(cls, value: list[str]) -> list[str]:
        bad = [v for v in value if v not in TRANSFORMS]
        if bad:
            raise ValueError(f"unknown query transform(s) {bad}: use {', '.join(TRANSFORMS)}")
        return value


class S4Grid(_M):
    top: int = Field(default=3, ge=1)
    ks: list[int] = Field(default_factory=lambda: [3, 5, 8, 10, 15])
    abstain_decline_min: float = Field(default=0.80, ge=0, le=1)
    abstain_refuse_max: float = Field(default=0.05, ge=0, le=1)
    abstain_auc_min: float = Field(default=0.75, ge=0, le=1)


class S5Grid(_M):
    top: int = Field(default=3, ge=1)
    chat_tiers: list[Literal["fast", "powerful"]] = Field(
        default_factory=lambda: ["fast", "powerful"]
    )
    repeat_top: int = Field(default=1, ge=0)


class S6Grid(_M):
    top: int = Field(default=3, ge=1)
    replay: int = Field(default=200, ge=1)
    end_to_end: int = Field(default=30, ge=0)
    p95_local_retrieval_ms: float = Field(default=300.0, gt=0)


class SweepEstimate(_M):
    transform_prompt_tokens: int = 250
    multi_output_tokens: int = 80
    hyde_output_tokens: int = 200


class SweepSettings(_M):
    decision_budget: int = Field(default=DECISION_BUDGET, gt=0)
    S1a: S1aGrid
    S1b: S1bGrid = Field(default_factory=S1bGrid)
    S2: S2Grid
    S3: S3Grid = Field(default_factory=S3Grid)
    S4: S4Grid = Field(default_factory=S4Grid)
    S5: S5Grid = Field(default_factory=S5Grid)
    S6: S6Grid = Field(default_factory=S6Grid)
    estimate: SweepEstimate = Field(default_factory=SweepEstimate)


def load_sweep_settings(config: EvalConfig) -> SweepSettings:
    """The ``sweep:`` section of ``knowledge_eval.yaml``, validated."""
    from hal9000 import config as hal_config

    if not config.sweep:
        raise hal_config.ConfigError(
            "config/knowledge_eval.yaml has no sweep: section (the stage grids, plan §8.4)"
        )
    return SweepSettings.model_validate(config.sweep)


# ==================================================================== configs
@dataclass(frozen=True)
class SweepConfig:
    """One configuration a run evaluates: an index profile plus retrieval (and answer) settings."""

    parser: str = "pdfium"
    chunker: str = "structure"
    size: int = 512
    #: overlap as a fraction of ``size`` (the profile stores tokens)
    overlap: float = 0.0
    parent_size: int | None = None
    header: bool = True
    embed_model: str = "openai/text-embedding-3-small"
    dimensions: int = 1536
    retrieval: str = "rrf"
    alpha: float | None = None
    reranker: str = "none"
    query_transform: str = "none"
    k: int | None = None
    chat_tier: str | None = None
    repeat: int = 0

    def __post_init__(self) -> None:
        if self.retrieval not in METHODS:
            raise ValueError(f"unknown retrieval method {self.retrieval!r}")
        if (self.retrieval == "weighted") != (self.alpha is not None):
            raise ValueError("alpha is set for weighted retrieval, and only for it")

    # ---------------------------------------------------------- the profile
    @property
    def overlap_percent(self) -> int:
        return round(self.overlap * 100)

    def profile(self) -> IndexProfile:
        from hal9000.knowledge.chunking import make_profile, overlap_tokens

        return make_profile(
            parser=self.parser,
            chunker=self.chunker,
            size=self.size,
            overlap=overlap_tokens(self.size, self.overlap_percent),
            parent_size=self.parent_size,
            header=self.header,
            embed_model=self.embed_model,
            dimensions=self.dimensions,
        )

    @property
    def profile_name(self) -> str:
        return self.profile().name

    @property
    def name(self) -> str:
        """``<profile>+<method>[+rerank][+transform][+k8][+tier][+rep1]``: readable, unique."""
        method = self.retrieval if self.alpha is None else f"weighted{self.alpha:g}"
        parts = [self.profile_name, method]
        if self.reranker != "none":
            parts.append(self.reranker)
        if self.query_transform != "none":
            parts.append(self.query_transform)
        if self.k is not None:
            parts.append(f"k{self.k}")
        if self.chat_tier:
            parts.append(self.chat_tier)
        if self.repeat:
            parts.append(f"rep{self.repeat}")
        return "+".join(parts)

    def run_config(self, *, stage: str, sweep_id: str | None) -> dict[str, Any]:
        """The ``eval_runs.config`` (the well-known keys of :mod:`.runs`), None values dropped."""
        from hal9000.knowledge.chunking import overlap_tokens

        values: dict[str, Any] = {
            "sweep_id": sweep_id,
            "stage": stage,
            "name": self.name,
            "profile": self.profile_name,
            "overlap_tokens": overlap_tokens(self.size, self.overlap_percent),
            **{f.name: getattr(self, f.name) for f in fields(self)},
        }
        if not self.repeat:
            values.pop("repeat")
        return {key: value for key, value in values.items() if value is not None}

    @classmethod
    def from_run_config(cls, config: Mapping[str, Any]) -> SweepConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in names})

    @classmethod
    def from_profile(cls, profile: IndexProfile, **retrieval: Any) -> SweepConfig:
        overlap = round(profile.overlap / profile.size, 2) if profile.overlap else 0.0
        base = cls(
            parser=profile.parser,
            chunker=profile.chunker,
            size=profile.size,
            overlap=overlap,
            parent_size=profile.parent_size,
            header=profile.header,
            embed_model=profile.embed_model,
            dimensions=profile.dimensions,
        )
        return replace(base, **retrieval)

    def retrieval_overrides(self) -> dict[str, Any]:
        """What the retrieval binding reads from a run's config."""
        values = {
            "profile": self.profile_name,
            "retrieval": self.retrieval,
            "alpha": self.alpha,
            "reranker": self.reranker,
            "query_transform": self.query_transform,
        }
        return {k: v for k, v in values.items() if v is not None}


def _shape(config: Mapping) -> tuple:
    """The chunking of a configuration (S1a -> S1b: the top chunkings)."""
    return (config.get("chunker"), config.get("size"), config.get("parent_size"))


def _profile_sans_embedding(config: Mapping) -> tuple:
    keys = ("parser", "chunker", "size", "overlap", "parent_size", "header")
    return tuple(config.get(k) for k in keys)


# ==================================================================== helpers
def local_latency_ms(trace: Any) -> float:
    """Local retrieval time from a trace: its top-level stages except condense and embed_query."""
    return float(
        sum(s.duration_ms for s in trace.stages if s.depth == 0 and str(s.stage) not in NOT_LOCAL)
    )


@dataclass(frozen=True)
class Calibration:
    """An abstain threshold chosen on dev (plan §8.5), or why there is none."""

    threshold: float | None
    auc: float | None
    decline: float | None = None
    refuse: float | None = None
    met: bool = False
    reason: str = ""


def calibrate_threshold(
    answerable: Sequence[float],
    unanswerable: Sequence[float],
    *,
    decline_min: float = 0.80,
    refuse_max: float = 0.05,
    auc_min: float = 0.75,
) -> Calibration:
    """The threshold (decline when score < t) that declines the most unanswerables while
    refusing at most ``refuse_max`` of the answerables; None when the score can't separate
    them (AUC < ``auc_min``: the check moves to generation, ``NOT_IN_SOURCES``)."""
    auc = roc_auc(answerable, unanswerable)
    if auc is None:
        return Calibration(None, None, reason="needs abstain scores for answerable and "
                           "unanswerable dev questions")  # fmt: skip
    if auc < auc_min:
        return Calibration(
            None,
            auc,
            reason=f"AUC {auc:.3f} < {auc_min:.2f}: the abstain score can't separate "
            "answerable from unanswerable questions, so the check moves to the generation "
            "step (NOT_IN_SOURCES)",
        )
    pos = np.asarray(answerable, dtype=float)
    neg = np.asarray(unanswerable, dtype=float)
    points = np.unique(np.concatenate([pos, neg]))
    candidates = [float(points[0]) - 1e-6]
    candidates += [float((a + b) / 2) for a, b in zip(points[:-1], points[1:])]
    candidates.append(float(points[-1]) + 1e-6)
    best: tuple[float, float, float] | None = None  # (decline, -refuse, threshold)
    for t in candidates:
        refuse = float(np.mean(pos < t))
        if refuse > refuse_max + 1e-12:
            continue
        decline = float(np.mean(neg < t))
        key = (decline, -refuse, -t)
        if best is None or key > (best[0], -best[1], -best[2]):
            best = (decline, refuse, t)
    assert best is not None  # the lowest candidate refuses nothing
    decline, refuse, t = best
    met = decline >= decline_min
    reason = f"declines {decline:.0%} of unanswerables, refuses {refuse:.0%} of answerables" + (
        "" if met else f" (target: >= {decline_min:.0%} declined; not met)"
    )
    return Calibration(round(t, 6), auc, decline, refuse, met, reason)


def manifest_path(eval_dir: Path, sweep_id: str) -> Path:
    from hal9000.knowledge.eval.runs import run_dir

    run_dir(eval_dir, sweep_id)  # validates the ID
    return eval_dir / "sweeps" / f"{sweep_id}.json"


def read_manifest(eval_dir: Path, sweep_id: str) -> dict[str, Any]:
    path = manifest_path(eval_dir, sweep_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_manifest(eval_dir: Path, sweep_id: str, manifest: Mapping[str, Any]) -> None:
    path = manifest_path(eval_dir, sweep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def profile_texts(conn: sqlite3.Connection, profile: IndexProfile) -> list[str]:
    """The texts a build of ``profile`` would embed, for the estimate (no embedding, no parse).

    Stored pages of the profile's parser, or the pdfium pages as a stand-in
    when that parser's pages aren't stored yet (Docling parses at build time).
    """
    from hal9000.knowledge.index.library import documents_for, list_files, load_pages
    from hal9000.knowledge.index.profiles import parser_key
    from hal9000.knowledge.index.writer import PreparedDocument, chunk_documents, searchable
    from hal9000.knowledge.ingest.clean import CleanResult

    texts: list[str] = []
    for file in list_files(conn):
        docs = documents_for(conn, file.sha256)
        if not docs:
            continue
        pages = load_pages(conn, file.sha256, parser_key(file, profile.parser)) or load_pages(
            conn, file.sha256, parser_key(file, "pdfium")
        )
        if not pages:
            continue
        cleaned = CleanResult(pages=pages)
        prepared = [
            PreparedDocument(d, cleaned.document(d.page_start, d.page_end), file.collection)
            for d in docs
        ]
        texts += [d.index_text for d in searchable(chunk_documents(prepared, profile))]
    return texts


# ==================================================================== answers
_CORRECT = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def config_answerer(
    conn: sqlite3.Connection,
    settings: KnowledgeSettings,
    config: SweepConfig,
    *,
    meter: Any,
    deps: SweepDeps,
    budget: int = DECISION_BUDGET,
) -> Callable[..., tuple[AnswerOutput, Any]]:
    """``(question, emit=None) -> (AnswerOutput, Answer)`` with *every* setting of ``config``.

    The chat's pipeline (KWP6) over the configuration's profile, with its
    retrieval method, reranker, query transform, *k* and chat tier pinned and a
    context budget of ``budget`` tokens, metered as ``knowledge-eval``, without
    traces.
    """
    from hal9000.knowledge.answer.pipeline import AnswerDeps, AnswerPipeline
    from hal9000.knowledge.config import load_knowledge_file
    from hal9000.knowledge.index.profiles import get_profile
    from hal9000.knowledge.retrieve.config import RetrievalConfig
    from hal9000.knowledge.retrieve.retriever import Retriever

    profile = get_profile(conn, config.profile_name)
    if profile is None:
        raise LookupError(f"profile {config.profile_name} is not built")
    knowledge = load_knowledge_file(settings.yaml_path)
    pinned = replace(
        RetrievalConfig.from_file(knowledge).updated(config.retrieval_overrides()),
        budget_tokens=budget,
    )
    reranker = deps.reranker(config.reranker) if config.reranker != "none" else None
    k = config.k

    class PinnedRetriever(Retriever):
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            if reranker is not None:
                self.reranker = reranker

        def retrieve(self, question, history=(), filters=None, config=None):  # noqa: ANN001
            result = super().retrieve(question, history, filters, pinned)
            if k is not None and len(result.selected) > k:
                kept = result.selected[:k]
                result = result.model_copy(
                    update={"selected": kept, "context_tokens": sum(c.prompt_tokens for c in kept)}
                )
            return result

    from hal9000 import llm

    answer_deps = AnswerDeps(
        stream_fn=deps.stream_response or llm.stream_response,
        generate_fn=deps.generate_response or llm.generate_response,
        meter=meter,
        agent=EVAL_AGENT,
        embed_fn=deps.embed_fn,
        retriever_cls=PinnedRetriever,
    )
    pipeline = AnswerPipeline(
        settings,
        answer_deps,
        knowledge_file=knowledge,
        tier=config.chat_tier or settings.chat_tier,
        profile=profile,
        record_traces=False,
    )

    def answer(question: str, emit: Callable[[Any], None] | None = None):
        from hal9000.knowledge.answer.prompt import NumberedSource

        kept: list[NumberedSource] = []
        result = pipeline.ask(
            conn, question, emit=emit or (lambda event: None), on_sources=kept.extend
        )
        output = AnswerOutput(
            text=result.text,
            sources=tuple(
                Source(marker=s.marker, text=s.text, title=s.document.title) for s in kept
            ),
            abstained=result.outcome == "abstained",
            cost_usd=result.cost_usd,
            citations=tuple(result.citations),
        )
        return output, result

    return answer


# ==================================================================== runner
@dataclass
class SweepDeps:
    """What the runner calls out to; tests inject fakes, the CLI leaves the defaults."""

    #: embedding callable (default: by model -- the fake embedder for ``fake/*``)
    embed_fn: Callable[..., Any] | None = None
    #: ``generate_response(prompt, tier, *, on_usage)``: T10 rewrites, the judge, condensing
    generate_response: Callable[..., str] | None = None
    #: ``stream_response`` for S5 answers
    stream_response: Callable[..., Any] | None = None
    #: ``name -> Reranker`` (default: the ONNX cross-encoders, probed once)
    reranker_factory: Callable[[str], Any] | None = None
    #: whether the Docling parser can be imported
    docling_available: Callable[[], bool] | None = None
    #: parse in the child process with the time limit (tests parse in-process)
    isolate_parse: bool = True
    _rerankers: dict[str, Any] = field(default_factory=dict, repr=False)

    def reranker(self, name: str) -> Any:
        """A loaded reranker; raises when its model can't load."""
        if name not in self._rerankers:
            if self.reranker_factory is not None:
                model = self.reranker_factory(name)
            else:
                from hal9000.knowledge.retrieve.rerank import reranker_for

                model = reranker_for(name)
            model.score("probe", ["probe"])  # loads the model now, or raises
            self._rerankers[name] = model
        return self._rerankers[name]

    def has_docling(self) -> bool:
        if self.docling_available is not None:
            return self.docling_available()
        from hal9000.knowledge.ingest.parsers.docling import docling_available

        return docling_available()


class SweepStopped(Exception):  # noqa: N818 - control flow
    """A spending cap was reached: finished runs are kept, the sweep can be resumed."""


class SweepRefused(Exception):  # noqa: N818 - control flow
    """A stage was not started (estimate over a cap, not confirmed, nothing to run from)."""


@dataclass
class SweepOutcome:
    sweep_id: str
    status: Literal["done", "stopped", "refused"]
    message: str = ""
    runs: int = 0
    spent: float = 0.0


@dataclass
class _Estimate:
    usd: float = 0.0
    lines: list[str] = field(default_factory=list)

    def add(self, usd: float, line: str) -> None:
        self.usd += usd
        self.lines.append(line)


class SweepRunner:
    """Runs sweep stages against ``knowledge.db`` (see the module notes)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: KnowledgeSettings,
        *,
        eval_config: EvalConfig,
        meter: Any,
        max_cost: float,
        eval_cap: float | None = None,
        deps: SweepDeps | None = None,
        confirm: Callable[[str], bool] = lambda question: True,
        out: TextIO | None = None,
    ) -> None:
        import sys

        self.conn = conn
        self.settings = settings
        self.eval_config = eval_config
        self.grid = load_sweep_settings(eval_config)
        self.scoring = eval_config.scoring.to_config()
        self.meter = meter
        self.max_cost = max_cost
        self.eval_cap = settings.eval_cap_usd if eval_cap is None else eval_cap
        self.deps = deps or SweepDeps()
        self.confirm = confirm
        self.out = out or sys.stdout
        self.ranker = Ranker(self.scoring, budget=self.grid.decision_budget)
        self.sweep_id = ""
        self.manifest: dict[str, Any] = {}
        self._pipeline = None
        self._generate: Callable[..., str] | None = None
        self._items: list[GoldItem] | None = None
        self._gate: dict[str, Any] = {}

    # ------------------------------------------------------------ plumbing
    def say(self, message: str) -> None:
        print(message, file=self.out, flush=True)

    @property
    def eval_dir(self) -> Path:
        return self.settings.eval_dir

    def note(self, stage: str, message: str) -> None:
        notes = self.manifest.setdefault("notes", {}).setdefault(stage, [])
        if message not in notes:
            notes.append(message)
            write_manifest(self.eval_dir, self.sweep_id, self.manifest)
        self.say(f"  note: {message}")

    def dev_items(self) -> list[GoldItem]:
        if self._items is None:
            self._items = list_gold(self.conn, status="approved", split="dev")
        return self._items

    def generate(self) -> Callable[..., str]:
        """``(prompt, tier) -> str`` metered as ``knowledge-eval`` (needs a key unless injected)."""
        if self._generate is None:
            fn = self.deps.generate_response
            if fn is None:
                from hal9000 import config as hal_config

                hal_config.require_llm_key()
                from hal9000.llm import generate_response as fn
            self._generate = self.meter.wrap(fn, EVAL_AGENT)
        return self._generate

    def pipeline(self):
        """KWP3's ingestion pipeline, for profile builds (metered as ``knowledge-eval``)."""
        if self._pipeline is None:
            from hal9000.knowledge.config import load_knowledge_file
            from hal9000.knowledge.ingest.pipeline import IngestPipeline

            self._pipeline = IngestPipeline(
                self.settings.files_dir,
                embed_fn=self.deps.embed_fn,
                meter=self.meter,
                agent=EVAL_AGENT,
                use_llm=False,
                bus=None,
                isolate_parse=self.deps.isolate_parse,
                knowledge_file=load_knowledge_file(self.settings.yaml_path),
            )
        return self._pipeline

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    def run_id(self, stage: str, config: SweepConfig) -> str:
        digest = hashlib.sha1(config.name.encode()).hexdigest()[:10]
        return f"{self.sweep_id}.{stage.lower()}.{digest}"

    def stage_runs(self, *stages: str, finished: bool = True) -> list[EvalRun]:
        runs = [
            r
            for r in list_runs(self.conn)
            if r.config.get("sweep_id") == self.sweep_id and r.stage in stages
        ]
        return [r for r in runs if r.finished_at is not None] if finished else runs

    def _remaining(self) -> float:
        run_left = max(0.0, self.max_cost - self.meter.spent)
        month_left = max(0.0, self.eval_cap - self.meter.month_spend(EVAL_AGENT))
        return min(run_left, month_left)

    # ------------------------------------------------------------ entry
    def run(
        self,
        stages: Sequence[str],
        *,
        sweep_id: str | None = None,
        force_judge: bool = False,
    ) -> SweepOutcome:
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}: choose from {', '.join(STAGES)}")
        ordered = [s for s in STAGES if s in stages]
        if "S5" in ordered and sweep_id is None:
            raise ValueError("S5 continues a sweep: pass --resume SWEEP (the ID S1a-S4 printed)")
        if sweep_id is not None:
            if not any(r.config.get("sweep_id") == sweep_id for r in list_runs(self.conn)):
                if not read_manifest(self.eval_dir, sweep_id):
                    raise ValueError(f"no sweep {sweep_id!r} to resume")
        self.sweep_id = sweep_id or new_run_id("sweep")
        self.manifest = read_manifest(self.eval_dir, self.sweep_id) or {
            "sweep_id": self.sweep_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        write_manifest(self.eval_dir, self.sweep_id, self.manifest)
        if not self.dev_items():
            return SweepOutcome(
                self.sweep_id,
                "refused",
                "no approved dev questions: review the question set first (gate G-K1)",
            )
        self.say(
            f"Sweep {self.sweep_id}: stages {', '.join(ordered)} on {len(self.dev_items())} "
            f"approved dev questions. Cap ${self.max_cost:.2f} for this run "
            f"(KB_EVAL_CAP_USD ${self.eval_cap:.2f}/month), metered as {EVAL_AGENT}."
        )
        handlers = {
            "S1a": self._s1a,
            "S1b": self._s1b,
            "S2": self._s2,
            "S3": self._s3,
            "S4": self._s4,
            "S5": lambda: self._s5(force_judge),
        }
        try:
            for stage in ordered:
                handlers[stage]()
        except SweepStopped as stop:
            return self._outcome("stopped", f"STOPPED at the cost cap: {stop}. Finished runs are "
                                 f"kept; resume with --resume {self.sweep_id}.")  # fmt: skip
        except SweepRefused as refused:
            return self._outcome("refused", str(refused))
        finally:
            self.close()
        return self._outcome("done", "")

    def _outcome(self, status, message) -> SweepOutcome:
        return SweepOutcome(
            self.sweep_id,
            status,
            message,
            runs=len(self.stage_runs(*STAGES)),
            spent=self.meter.spent,
        )

    # ------------------------------------------------------------ batches
    def _batch(self, stage: str, configs: Sequence[SweepConfig], label: str = "") -> None:
        """Estimate, confirm and run the configurations not yet finished."""
        configs = list(dict.fromkeys(configs))
        todo = []
        for config in configs:
            existing = get_run(self.conn, self.run_id(stage, config))
            if existing is not None and existing.finished_at is not None:
                continue
            todo.append(config)
        title = f"{stage}{label}"
        self.say(
            f"\n{title}: {len(configs)} configuration(s), {len(configs) - len(todo)} already done."
        )
        if not todo:
            return
        estimate = self._estimate(stage, todo)
        self.say(f"Estimate for {title}: ${estimate.usd:.4f}")
        for line in estimate.lines:
            self.say(f"  - {line}")
        left = self._remaining()
        if estimate.usd > left:
            raise SweepRefused(
                f"{title} not started: the estimate ${estimate.usd:.4f} exceeds what is left of "
                f"the caps (${left:.4f}: --max-cost ${self.max_cost:.2f}, KB_EVAL_CAP_USD "
                f"${self.eval_cap:.2f}/month). Finished runs are kept; resume with "
                f"--resume {self.sweep_id}."
            )
        if not self.confirm(f"Run {title} ({len(todo)} configurations)?"):
            raise SweepRefused(f"{title} not confirmed. Resume with --resume {self.sweep_id}.")
        for number, config in enumerate(todo, start=1):
            self._execute(stage, config, f"[{number}/{len(todo)}]")

    def _execute(self, stage: str, config: SweepConfig, counter: str) -> None:
        from hal9000.knowledge.index.profiles import ProfileError
        from hal9000.knowledge.ingest.parsers import ParseError

        run_id = self.run_id(stage, config)
        stale = get_run(self.conn, run_id)
        if stale is not None:  # an unfinished row from an interrupted run
            with self.conn:
                self.conn.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))
        try:
            profile = self._ensure_profile(config.profile())
        except BudgetExceeded as error:
            raise SweepStopped(str(error)) from None
        except (ProfileError, ParseError) as error:
            self.note(stage, f"{config.profile_name} could not be built ({error}); skipped")
            return
        started = time.perf_counter()
        try:
            if stage == "S5":
                run = self._answer_run(stage, config, run_id)
            else:
                run = self._retrieval_run(stage, config, run_id, profile)
        except BudgetExceeded as error:
            raise SweepStopped(str(error)) from None
        m = run.metrics
        budget = self.grid.decision_budget
        summary = (
            f"correct {m.correct.value:.3f}"
            if stage == "S5" and m and m.correct
            else f"recall@{budget} {m.budgeted_recall[budget].value:.3f}"
            if m and budget in m.budgeted_recall
            else f"recall@{budget} N/A"
        )
        p95 = f", p95 {m.latency_ms_p95:.1f} ms" if m and m.latency_ms_p95 is not None else ""
        self.say(
            f"  {counter} {config.name}: {summary}{p95}, ${run.cost_usd:.4f} "
            f"({time.perf_counter() - started:.1f} s)"
        )

    # ------------------------------------------------------------ profiles
    def _ensure_profile(self, profile: IndexProfile) -> IndexProfile:
        from hal9000.knowledge.index.profiles import get_profile, profile_status

        if profile_status(self.conn, profile.name) == "ready":
            found = get_profile(self.conn, profile.name)
            assert found is not None
            return found
        self.say(f"  building {profile.name} (not activated)")
        report = self.pipeline().build(self.conn, profile)
        if report.embedded:
            self.say(f"    embedded {report.embedded} new text(s), {report.cache_hits} cached")
        return report.profile

    def _estimate(self, stage: str, todo: Sequence[SweepConfig]) -> _Estimate:
        from hal9000.knowledge.index.embedding import EmbeddingService, is_fake_model
        from hal9000.knowledge.index.profiles import profile_status

        estimate = _Estimate()
        service = EmbeddingService(self.conn)
        texts: dict[tuple[str, int], list[str]] = {}
        builds = 0
        for profile in {c.profile_name: c.profile() for c in todo}.values():
            if profile_status(self.conn, profile.name) == "ready":
                continue
            builds += 1
            if is_fake_model(profile.embed_model):
                continue
            texts.setdefault((profile.embed_model, profile.dimensions), []).extend(
                profile_texts(self.conn, profile)
            )
        questions = [item.question for item in self.dev_items()]
        for model, dims in {(c.embed_model, c.dimensions) for c in todo}:
            texts.setdefault((model, dims), []).extend(questions)
        for (model, dims), batch in sorted(texts.items()):
            if not batch or is_fake_model(model):
                continue
            tokens, usd = service.estimate(batch, model, dims)
            estimate.add(usd, f"embeddings: ~{tokens:,} uncached tokens with {model}-{dims}")
        if builds:
            estimate.lines.append(f"{builds} profile(s) to build from stored pages (not activated)")
        n = len(self.dev_items())
        cfg = self.grid.estimate
        for transform in ("multi", "hyde"):
            count = sum(c.query_transform == transform for c in todo)
            if count:
                output = cfg.multi_output_tokens if transform == "multi" else cfg.hyde_output_tokens
                cost = estimate_cost(
                    count * n,
                    count * n * cfg.transform_prompt_tokens,
                    count * n * output,
                    self.settings.condense_tier,
                )
                estimate.add(cost.usd, f"T10 {transform}: {cost.describe()}")
        if stage == "S5":
            from hal9000.knowledge.eval.judge import estimate_judge_prep

            for config in todo:
                cost = estimate_judge_prep(
                    n,
                    self.eval_config.judge,
                    config.chat_tier or self.settings.chat_tier,
                    self.settings.judge_tier,
                )
                estimate.add(cost.usd, f"{config.name}: {n} answers + judgements, "
                             f"{cost.describe()}")  # fmt: skip
        return estimate

    # ------------------------------------------------------------ retrieval runs
    def _binding(self, config: SweepConfig):
        from hal9000.knowledge.retrieve.eval_binding import binding

        return binding(
            self.conn,
            self.settings,
            meter=self.meter,
            run=SimpleNamespace(config=config.retrieval_overrides()),
            generate=self.generate() if config.query_transform != "none" else None,
            embed_fn=self.deps.embed_fn,
            reranker=self.deps.reranker(config.reranker) if config.reranker != "none" else None,
        )

    def retrieve_timed(self, retrieve: Callable[[str], RetrievalOutput], question: str,
                       k: int | None = None) -> RetrievalOutput:  # fmt: skip
        """One retrieval with its local latency (the stage timer), cut to ``k`` items."""
        from hal9000.knowledge.obs.timing import start_trace

        with start_trace(question="") as trace:
            output = retrieve(question)
        selected = output.selected[:k] if k is not None else output.selected
        return replace(output, selected=tuple(selected), latency_ms=local_latency_ms(trace))

    def _retrieval_run(
        self, stage: str, config: SweepConfig, run_id: str, profile: IndexProfile
    ) -> EvalRun:
        bound = self._binding(config)
        spent = self.meter.spent
        items = self.dev_items()
        if config.query_transform == "none":  # warm caches and imports (untimed, free)
            bound.retrieve(items[0].question)
        scores: list[QuestionScore] = []
        records: list[dict[str, Any]] = []
        for item in items:
            output = self.retrieve_timed(bound.retrieve, item.question, config.k)
            score = score_question(
                ScoringItem.from_gold(item), output, chunk_size=bound.chunk_size,
                config=self.scoring,
            )  # fmt: skip
            scores.append(score)
            records.append(
                question_record(
                    score, question=item.question, selected=[s.chunk_id for s in output.selected]
                )
            )
        cost = self.meter.spent - spent
        metrics = aggregate(scores, config=self.scoring)
        metrics = metrics.model_copy(
            update={"cost_per_question_usd": self._query_cost(config, items, cost)}
        )
        run_config = config.run_config(stage=stage, sweep_id=self.sweep_id)
        if stage == "S4" and config.k is None:
            calibration = self._calibration(scores)
            run_config.update(
                {
                    "abstain_threshold": calibration.threshold,
                    "abstain_auc": calibration.auc,
                    "abstain_decline": calibration.decline,
                    "abstain_refuse": calibration.refuse,
                    "abstain_note": calibration.reason,
                }
            )
            run_config = {k: v for k, v in run_config.items() if v is not None}
        return record_run(
            self.conn,
            eval_dir=self.eval_dir,
            stage=stage,
            config=run_config,
            metrics=metrics,
            records=records,
            cost_usd=cost,
            run_id=run_id,
        )

    def _query_cost(self, config: SweepConfig, items: Sequence[GoldItem], spent: float) -> float:
        """Production cost per question: one query embedding (+ the transform call, measured)."""
        from hal9000.cost.metering import usage_cost
        from hal9000.knowledge.index.embedding import is_fake_model
        from hal9000.knowledge.tokens import count_tokens

        embed = 0.0
        if not is_fake_model(config.embed_model) and items:
            tokens = np.mean([count_tokens(i.question) for i in items])
            embed = usage_cost({"model": config.embed_model, "prompt": int(round(tokens))})
        return round(embed + (spent / len(items) if items else 0.0), 8)

    def _calibration(self, scores: Sequence[QuestionScore]) -> Calibration:
        grid = self.grid.S4
        pos = [s.abstain_score for s in scores if s.answerable and s.abstain_score is not None]
        neg = [s.abstain_score for s in scores if not s.answerable and s.abstain_score is not None]
        return calibrate_threshold(
            pos,
            neg,
            decline_min=grid.abstain_decline_min,
            refuse_max=grid.abstain_refuse_max,
            auc_min=grid.abstain_auc_min,
        )

    # ------------------------------------------------------------ stages
    def _require(self, runs: Sequence[EvalRun], previous: str) -> None:
        if not runs:
            raise SweepRefused(
                f"no finished {previous} runs in sweep {self.sweep_id}: run {previous} first"
            )

    def _s1a(self) -> None:
        g = self.grid.S1a
        configs: list[SweepConfig] = []
        base = SweepConfig(
            parser=g.parser,
            header=g.header,
            embed_model=g.embedding.model,
            dimensions=g.embedding.dimensions,
        )
        for chunker in g.chunkers:
            for size in g.sizes:
                overlap = g.fixed_overlap if chunker == "fixed" else 0.0
                for method in g.retrieval:
                    configs.append(
                        replace(base, chunker=chunker, size=size, overlap=overlap, retrieval=method)
                    )
        if g.parent_child is not None:
            for size in g.parent_child.child_sizes:
                for method in g.retrieval:
                    configs.append(
                        replace(
                            base,
                            chunker="parent_child",
                            size=size,
                            parent_size=g.parent_child.parent_size,
                            retrieval=method,
                        )
                    )
        self._batch("S1a", configs)

    def _s1b(self) -> None:
        g = self.grid.S1b
        previous = self.stage_runs("S1a")
        self._require(previous, "S1a")
        top = self.ranker.top(previous, g.top, distinct=_shape)
        parsers = list(g.parser)
        if "docling" in parsers and not self.deps.has_docling():
            parsers.remove("docling")
            self.note("S1b", "docling unavailable (the knowledge-docling group is not "
                      "installed): the Docling arm of T5 was skipped")  # fmt: skip
        configs = []
        for run in top:
            base = SweepConfig.from_run_config(run.config)
            for overlap in g.overlap:
                for header in g.header:
                    for parser in parsers:
                        configs.append(replace(base, overlap=overlap, header=header, parser=parser))
        self.say(f"\nS1b on the top {len(top)} of S1a: " + ", ".join(r.config["name"] for r in top))
        self._batch("S1b", configs)

    def _s2(self) -> None:
        g = self.grid.S2
        previous = self.stage_runs("S1a", "S1b")
        self._require(previous, "S1a")
        ranked = self.ranker.top(previous, 1)
        if not ranked:
            raise SweepRefused("no S1 run has a recall at the decision budget to rank by")
        method = SweepConfig.from_run_config(ranked[0].config)
        top = self.ranker.top(previous, g.top, distinct=_profile_sans_embedding)
        self.say(
            f"\nS2 with the best S1 method ({method.retrieval}"
            f"{'' if method.alpha is None else f', alpha {method.alpha:g}'}) on the top "
            f"{len(top)} of S1: " + ", ".join(r.config["profile"] for r in top)
        )
        configs = []
        for run in top:
            base = replace(
                SweepConfig.from_run_config(run.config),
                retrieval=method.retrieval,
                alpha=method.alpha,
            )
            for emb in g.embeddings:
                configs.append(replace(base, embed_model=emb.model, dimensions=emb.dimensions))
        self._batch("S2", configs)

    def _s3(self) -> None:
        g = self.grid.S3
        previous = self.stage_runs("S2")
        self._require(previous, "S2")
        top = self.ranker.top(previous, g.top, distinct=lambda c: c.get("profile"))
        rerankers = []
        for name in g.rerankers:
            if name == "none":
                rerankers.append(name)
                continue
            try:
                self.deps.reranker(name)
                rerankers.append(name)
            except Exception as error:  # the model or fastembed is missing
                self.note("S3", f"reranker {name} unavailable ({type(error).__name__}: "
                          f"{error}); its arm of T8 was skipped")  # fmt: skip
        methods: list[tuple[str, float | None]] = []
        for method in g.retrieval:
            if method == "weighted":
                methods += [("weighted", alpha) for alpha in g.weighted_alpha]
            else:
                methods.append((method, None))
        self.say(
            f"\nS3 on the top {len(top)} of S2: " + ", ".join(r.config["profile"] for r in top)
        )
        configs = []
        for run in top:
            base = SweepConfig.from_run_config(run.config)
            for method, alpha in methods:
                for reranker in rerankers:
                    configs.append(replace(base, retrieval=method, alpha=alpha, reranker=reranker))
        self._batch("S3", configs, " (T7 x T8)")
        transforms = [t for t in g.query_transforms if t != "none"]
        if g.query_transform_top and transforms:
            phase_a = [r for r in self.stage_runs("S3") if
                       (r.config.get("query_transform") or "none") == "none"]  # fmt: skip
            best = self.ranker.top(phase_a, g.query_transform_top)
            configs = [
                replace(SweepConfig.from_run_config(run.config), query_transform=t)
                for run in best
                for t in transforms
            ]
            self._batch("S3", configs, " (T10 query transforms, paid)")

    def _s4(self) -> None:
        g = self.grid.S4
        previous = self.stage_runs("S3")
        self._require(previous, "S3")
        top = self.ranker.top(previous, g.top)
        self.say(f"\nS4 on the top {len(top)} of S3: " + ", ".join(r.config["name"] for r in top))
        configs = []
        for run in top:
            base = SweepConfig.from_run_config(run.config)
            configs.append(base)
            configs += [replace(base, k=k) for k in g.ks]
        self._batch("S4", configs)
        self._store_thresholds()

    def finalists(self) -> list[EvalRun]:
        """The top S4 baselines (no *k*): what S5 answers with."""
        baselines = [r for r in self.stage_runs("S4") if r.config.get("k") is None]
        return self.ranker.top(baselines, self.grid.S5.top)

    def _store_thresholds(self) -> None:
        from hal9000.knowledge.index.profiles import get_profile
        from hal9000.knowledge.retrieve.thresholds import set_threshold

        stored = self.manifest.setdefault("thresholds", {})
        for run in self.finalists():
            profile = get_profile(self.conn, run.config["profile"])
            reranker = run.config.get("reranker", "none")
            threshold = run.config.get("abstain_threshold")
            key = f"{run.config['profile']}|{reranker}"
            if profile is None:
                continue
            if threshold is None:
                self.note("S4", f"no abstain threshold for {key}: {run.config.get('abstain_note')}")
                continue
            set_threshold(
                self.conn,
                profile.id,
                reranker,
                float(threshold),
                auc=run.config.get("abstain_auc"),
                eval_run_id=run.id,
            )
            stored[key] = {"threshold": threshold, "run": run.id,
                           "note": run.config.get("abstain_note")}  # fmt: skip
            self.say(f"  abstain threshold {threshold:.4f} stored for {key} "
                     f"({run.config.get('abstain_note')})")  # fmt: skip
        write_manifest(self.eval_dir, self.sweep_id, self.manifest)

    # ------------------------------------------------------------ S5
    def _judge_gate(self, force: bool) -> dict[str, Any]:
        """Judge agreement with Mike's labels on this sweep's calibration answers
        (``judge-prep`` on one of its runs), or on every label when it has none: the
        rubric is pinned, so any calibration measures the same judge."""
        from hal9000.knowledge.eval.judge import agreement
        from hal9000.knowledge.eval.store import list_judge_labels

        gate = self.eval_config.judge.agreement_gate
        labels = list_judge_labels(self.conn)
        mine = {r.id for r in list_runs(self.conn) if r.config.get("sweep_id") == self.sweep_id}
        own = [label for label in labels if label.eval_run_id in mine]
        result = agreement(own or labels)
        record = {"judge_agreement": round(result.rate, 4), "judge_labels": result.labels}
        if result.passes(gate):
            self.say(f"Judge calibration: {result.describe()} (gate {gate:.0%}).")
            return record
        reason = (
            f"judge agreement not recorded at >= {gate:.0%}: {result.describe()}"
            if result.items
            else "no judge calibration labels recorded yet (run `kb eval judge-prep RUN`, then "
            "label the answers on the Judge calibration tab)"
        )
        if not force:
            raise SweepRefused(f"S5 refused: {reason}. Pass --force-judge to run it anyway "
                               "(recorded on every S5 run).")  # fmt: skip
        log.warning("S5 forced without judge calibration (--force-judge): %s", reason)
        self.note("S5", f"--force-judge: {reason}")
        return {**record, "judge_forced": True}

    def _s5(self, force_judge: bool) -> None:
        g = self.grid.S5
        finalists = self.finalists()
        self._require(finalists, "S4")
        self._gate = self._judge_gate(force_judge)
        self.say("\nS5 on the finalists: " + ", ".join(r.config["name"] for r in finalists))
        configs = []
        for rank, run in enumerate(finalists):
            base = replace(SweepConfig.from_run_config(run.config), k=None)
            for tier in g.chat_tiers:
                configs.append(replace(base, chat_tier=tier))
                if rank == 0:
                    configs += [replace(base, chat_tier=tier, repeat=r + 1)
                                for r in range(g.repeat_top)]  # fmt: skip
        self._batch("S5", configs)

    def _answer_run(self, stage: str, config: SweepConfig, run_id: str) -> EvalRun:
        from hal9000.knowledge.eval.judge import JudgeError, judge_answer

        answer = config_answerer(
            self.conn,
            self.settings,
            config,
            meter=self.meter,
            deps=self.deps,
            budget=self.grid.decision_budget,
        )
        generate = self.generate()
        spent = self.meter.spent
        records: list[dict[str, Any]] = []
        answer_cost = 0.0
        for item in self.dev_items():
            self.meter.precheck(EVAL_AGENT)
            started = time.perf_counter()
            output, result = answer(item.question)
            total_ms = (time.perf_counter() - started) * 1000
            if result.outcome in ("retrieval_only", "error"):
                self.meter.precheck(EVAL_AGENT)  # a cap turns answers retrieval-only: stop
            answer_cost += output.cost_usd
            try:
                labels = judge_answer(generate, self.settings.judge_tier, item, output)
            except JudgeError as error:
                log.warning("S5: the judge failed on item %s: %s", item.id, error)
                labels = None
            records.append(
                {
                    "gold_item_id": item.id,
                    "question": item.question,
                    "category": item.category,
                    "answerable": item.category != "unanswerable",
                    "outcome": result.outcome,
                    "abstained": output.abstained,
                    "correctness": labels.correctness if labels else None,
                    "correct": _CORRECT[labels.correctness] if labels else None,
                    "faithful": labels.faithful if labels else None,
                    "citations_correct": labels.citations_correct if labels else None,
                    "abstention_correct": labels.abstention_correct if labels else None,
                    "cost_usd": output.cost_usd,
                    "latency_ms": total_ms,
                    "recall": {},
                    "full_support": {},
                }
            )
        metrics = self._answer_metrics(records, answer_cost)
        run_config = config.run_config(stage=stage, sweep_id=self.sweep_id)
        run_config.update(self._gate)
        return record_run(
            self.conn,
            eval_dir=self.eval_dir,
            stage=stage,
            config=run_config,
            metrics=metrics,
            records=records,
            cost_usd=self.meter.spent - spent,
            run_id=run_id,
        )

    def _answer_metrics(self, records: Sequence[Mapping], answer_cost: float) -> EvalMetrics:
        cfg = self.scoring

        def ci(key: str, salt: int) -> MetricCI | None:
            values = [float(r[key]) for r in records if r.get(key) is not None]
            return bootstrap_ci(values, n_boot=cfg.n_boot, seed=cfg.seed + salt) if values else None

        by_category: dict[str, dict[str, float]] = {}
        for category in sorted({r["category"] for r in records}):
            rows = [r for r in records if r["category"] == category]
            row = {
                "n": float(len(rows)),
                "abstained": float(np.mean([r["abstained"] for r in rows])),
            }
            correct = [r["correct"] for r in rows if r.get("correct") is not None]
            if correct:
                row["correct"] = float(np.mean(correct))
            by_category[category] = row
        return EvalMetrics(
            n_questions=len(records),
            correct=ci("correct", 500),
            faithful=ci("faithful", 501),
            abstention_correct=ci("abstention_correct", 502),
            cost_per_question_usd=round(answer_cost / len(records), 8) if records else None,
            by_category=by_category,
        )
