"""Shared fixtures for the sweep, latency and decision tests (all offline, fake mode).

- ``corpus_root`` (session): the KWP1 fixture corpus ingested with the KWP3
  pipeline in fake mode (active profile :data:`ACTIVE`) plus a synthetic,
  approved question set built from ``corpus.QA_FACTS`` and invented
  unanswerable questions (rule K2).
- ``swept_root`` (session): a copy with stages S1a-S4 run on :data:`GRID`, a
  reduced grid that keeps the whole fake sweep well under two minutes.
- ``env`` / ``swept_env``: a private copy of either, with settings, a real
  :class:`Meter` on a temp ledger, and fake dependencies (embedder, LLM
  provider that reports priced usage, streaming chat, reranker, no Docling).
"""

from __future__ import annotations

import io
import json
import math
import shutil
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hal9000.knowledge.config import KnowledgeFile, KnowledgeSettings
from hal9000.knowledge.contracts import EvidenceQuote, GoldItem, RubricLabels
from hal9000.knowledge.eval.config import load_eval_config
from hal9000.knowledge.eval.store import insert_gold, insert_judge_label, set_human_labels
from hal9000.knowledge.eval.sweep import SweepDeps, SweepRunner
from hal9000.knowledge.index.fake import FAKE_ENV, FakeEmbedder, concepts, enable_fake_mode
from hal9000.knowledge.storage import open_db
from hal9000.llm import Usage
from tests.fakes import tool_call
from tests.knowledge.fixtures import corpus

REPO = Path(__file__).resolve().parents[3]
ACTIVE = "pdfium.structure-256.hdr.fake-64"

LAYOUT: dict[str, corpus.CorpusFile] = {
    "House/inspection.pdf": corpus.INSPECTION,
    "House/Bills/estimate.pdf": corpus.ESTIMATE,
    "House/Bills/invoice.pdf": corpus.INVOICE,
    "Appliances/heat_pump_guide.docx": corpus.DOCX,
    "Appliances/Garage/garage_door.md": corpus.MARKDOWN,
    "Yard/irrigation_notes.html": corpus.HTML,
    "Yard/pool_log.txt": corpus.TEXT,
}

#: invented questions the fixture corpus does not answer
UNANSWERABLE = (
    "What colour is the mailbox painted?",
    "Who installed the solar panels on the roof?",
    "How many ceiling fans are in the bedrooms?",
    "When was the chimney last swept?",
    "What brand is the kitchen refrigerator?",
    "Is there a warranty on the front door lock?",
)

GRID = """
scoring:
  budgets: [1000, 2000, 4000]
  hit_ks: [1, 5]
  n_boot: 200
  min_paired: 3
judge:
  n_items: 3
  agreement_gate: 0.85
sweep:
  decision_budget: 2000
  S1a:
    sizes: [128, 512]
    chunkers: [fixed, structure]
    fixed_overlap: 0.15
    parent_child: {child_sizes: [128], parent_size: 512}
    parser: pdfium
    header: true
    embedding: {model: fake/embed-64, dimensions: 64}
    retrieval: [dense, rrf]
  S1b:
    top: 2
    overlap: [0.0, 0.15]
    header: [false, true]
    parser: [pdfium, docling]
  S2:
    top: 2
    embeddings:
      - {model: fake/embed-64, dimensions: 64}
      - {model: fake/embed-64, dimensions: 32}
  S3:
    top: 2
    retrieval: [dense, rrf, weighted]
    weighted_alpha: [0.5]
    rerankers: [none, minilm]
    query_transform_top: 1
    query_transforms: [multi]
  S4:
    top: 2
    ks: [3]
    abstain_decline_min: 0.5
    abstain_refuse_max: 0.2
    abstain_auc_min: 0.5
  S5:
    top: 2
    chat_tiers: [fast, powerful]
    repeat_top: 1
  S6:
    p95_local_retrieval_ms: 300
"""

#: a grid that only runs a handful of configurations (cap and CLI tests)
TINY_GRID = (
    GRID.replace("sizes: [128, 512]", "sizes: [512]")
    .replace("chunkers: [fixed, structure]", "chunkers: [structure]")
    .replace("    parent_child: {child_sizes: [128], parent_size: 512}\n", "")
)


def gold_items() -> list[GoldItem]:
    """13 answerable fixture facts and 6 unanswerable questions, approved; 4 in test."""
    items = [
        GoldItem(
            question=fact.question,
            reference_answer=fact.answer,
            category=fact.category,
            evidence=[EvidenceQuote(quote=fact.quote, page=fact.page)],
            status="approved",
        )
        for fact in corpus.QA_FACTS
    ]
    items += [
        GoldItem(question=q, category="unanswerable", status="approved") for q in UNANSWERABLE
    ]
    test = {2, 7, 11, 16}
    return [
        item.model_copy(update={"split": "test" if i in test else "dev"})
        for i, item in enumerate(items)
    ]


# ---------------------------------------------------------------- fakes
class FakeProvider:
    """``generate_response``: the judge, T10 rewrites and condensing, with priced usage.

    Each call reports ``usage`` tokens of ``openai/gpt-5-mini`` (about $0.00175),
    so the meter's caps bite. ``log`` receives "FIRST LLM CALL" once.
    """

    def __init__(self, log: io.StringIO | None = None, usage=(3000, 500), judge=None) -> None:
        self.log = log
        self.usage = usage
        self.calls = 0
        self.judge = judge or {}

    def __call__(self, prompt, tier="powerful", *, on_usage=None) -> str:
        if self.calls == 0 and self.log is not None:
            self.log.write("FIRST LLM CALL\n")
        self.calls += 1
        if on_usage is not None:
            on_usage(
                Usage(model="openai/gpt-5-mini", prompt=self.usage[0], cached=0,
                      completion=self.usage[1], reasoning=0)
            )  # fmt: skip
        system = str(prompt.messages[0].get("content", ""))
        purpose = (prompt.metadata or {}).get("purpose")
        if prompt.tools:
            answer = str(prompt.messages[-1]["content"]).rsplit("Assistant's answer:", 1)[-1]
            declined = "couldn't find" in answer.lower() or "not in" in answer.lower()
            return tool_call(
                "record_judgement",
                correctness=self.judge.get("correctness", "correct"),
                faithful=True,
                citations_correct=True,
                abstention_correct=not declined,
            )
        if purpose == "multi_query" or "rewrites" in system:
            question = str(prompt.messages[-1]["content"])
            return f"{question}\nwhat does the report say\nwhere is it noted"
        if purpose == "hyde":
            return "The report notes the item and its details."
        user = str(prompt.messages[-1]["content"])
        return user.rsplit("Last question:", 1)[-1].strip()


class FakeReranker:
    """Scores passages by the share of the query's concepts they contain, in [0, 1]."""

    def __init__(self, name: str = "minilm") -> None:
        self.name = name

    def score(self, query: str, passages) -> list[float]:
        wanted = set(concepts(query)) or {""}
        return [len(wanted & set(concepts(p))) / len(wanted) for p in passages]


def fake_chat():
    """KWP6's extractive fake chat, reporting priced usage so answers cost something."""
    from hal9000.knowledge.answer.fake import FakeChat

    class PricedChat(FakeChat):
        def _stream(self, prompt, text, on_usage):
            for item in super()._stream(prompt, text, None):
                if isinstance(item, str):
                    yield item
            usage = Usage(model="openai/gpt-5-mini", prompt=2000, cached=0, completion=100,
                          reasoning=0)  # fmt: skip
            if on_usage is not None:
                on_usage(usage)
            yield usage

    return PricedChat(delay_s=0)


def deps(log: io.StringIO | None = None, **kw) -> SweepDeps:
    chat = fake_chat()
    values = {
        "embed_fn": FakeEmbedder(),
        "generate_response": FakeProvider(log),
        "stream_response": chat,
        "reranker_factory": FakeReranker,
        "docling_available": lambda: False,
        "isolate_parse": False,
    }
    values.update(kw)
    return SweepDeps(**values)


# ---------------------------------------------------------------- settings
def make_settings(root: Path, grid: str = GRID, **overrides) -> KnowledgeSettings:
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "knowledge.yaml").write_text(
        (REPO / "config" / "knowledge.yaml")
        .read_text()
        .replace("active_profile: null", f"active_profile: {ACTIVE}")
    )
    (config / "knowledge_eval.yaml").write_text(grid)
    shutil.copy(REPO / "config" / "knowledge_policy.txt", config / "knowledge_policy.txt")
    values = dict(
        embed_model="fake/embed-64",
        chat_tier="fast",
        condense_tier="fast",
        judge_tier="powerful",
        monthly_cap_usd=10.0,
        eval_cap_usd=10.0,
        passcode="open-the-pod-bay-doors",
        port=8910,
        data_dir=root / "data",
        house_pdf=root / "none.pdf",
        config_dir=config,
    )
    values.update(overrides)
    return KnowledgeSettings(**values)


def make_meter(root: Path, *, cap: float = 5.0, month_cap: float = 10.0):
    from hal9000.cost.metering import Meter
    from hal9000.storage.db import Database

    db = Database(root / "hal9000.db")
    return Meter(
        db,
        "kb-eval-test",
        soft_cap=cap,
        hard_cap=cap,
        monthly_caps={"knowledge": math.inf, "knowledge-eval": month_cap},
    )


@dataclass
class Env:
    root: Path
    settings: KnowledgeSettings
    conn: sqlite3.Connection
    out: io.StringIO = field(default_factory=io.StringIO)
    meters: list = field(default_factory=list)

    def runner(self, *, cap: float = 5.0, grid: str | None = None, confirm=None, **dep_kw):
        if grid is not None:
            (self.settings.config_dir / "knowledge_eval.yaml").write_text(grid)
        meter = make_meter(self.root, cap=cap)
        self.meters.append(meter)
        cfg = load_eval_config(self.settings.config_dir / "knowledge_eval.yaml")
        return SweepRunner(
            self.conn,
            self.settings,
            eval_config=cfg,
            meter=meter,
            max_cost=cap,
            deps=deps(self.out, **dep_kw),
            confirm=confirm or (lambda question: True),
            out=self.out,
        )

    def ledger(self) -> list[tuple[str, float]]:
        db = sqlite3.connect(self.root / "hal9000.db")
        try:
            return db.execute("SELECT agent, usd FROM ledger").fetchall()
        finally:
            db.close()

    def active(self) -> str | None:
        row = self.conn.execute("SELECT name FROM profiles WHERE active = 1").fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()
        for meter in self.meters:
            meter.db.close()


def _copy(source: Path, target: Path) -> None:
    shutil.copytree(source / "data", target / "data")


def open_env(root: Path, grid: str = GRID) -> Env:
    settings = make_settings(root, grid)
    return Env(root, settings, open_db(settings.db_path))


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch, openai_env) -> None:
    monkeypatch.setenv(FAKE_ENV, "1")


@pytest.fixture(autouse=True)
def fresh_query_cache() -> Iterator[None]:
    from hal9000.knowledge.retrieve.retriever import DEFAULT_QUERY_CACHE

    DEFAULT_QUERY_CACHE.clear()
    yield
    DEFAULT_QUERY_CACHE.clear()


#: session results shared by every module that imports these fixtures (pytest makes one
#: fixture definition per importing module, so a session fixture would run once per module)
_SESSION: dict[str, object] = {}


@pytest.fixture(scope="session")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if "corpus" not in _SESSION:
        _SESSION["corpus"] = _build_corpus(tmp_path_factory)
    return _SESSION["corpus"]  # type: ignore[return-value]


def _build_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from hal9000.knowledge.ingest.pipeline import IngestPipeline

    root = tmp_path_factory.mktemp("sweep-corpus")
    settings = make_settings(root)
    enable_fake_mode(True)
    pipeline = IngestPipeline(
        settings.files_dir,
        knowledge_file=KnowledgeFile(active_profile=ACTIVE),
        embed_fn=FakeEmbedder(),
        use_llm=False,
        bus=None,
        isolate_parse=False,
    )
    conn = open_db(settings.db_path)
    try:
        worker = pipeline.worker(":unused:")
        for relative, item in LAYOUT.items():
            result = pipeline.submit(conn, item.path, relative)
            assert result.status == "added", result
            assert worker.run_job(conn, result.job_id).status == "done"
        insert_gold(conn, gold_items())
    finally:
        conn.close()
        pipeline.close()
        enable_fake_mode(False)
    return root


@pytest.fixture(scope="session")
def swept_root(corpus_root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """S1a-S4 run once on :data:`GRID`; returns (root, sweep ID)."""
    if "swept" not in _SESSION:
        _SESSION["swept"] = _sweep(corpus_root, tmp_path_factory)
    return _SESSION["swept"]  # type: ignore[return-value]


def _sweep(corpus_root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    import os

    from hal9000.knowledge.retrieve.retriever import DEFAULT_QUERY_CACHE

    root = tmp_path_factory.mktemp("swept")
    _copy(corpus_root, root)
    previous = os.environ.get(FAKE_ENV)
    os.environ[FAKE_ENV] = "1"
    key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    env = open_env(root)
    try:
        DEFAULT_QUERY_CACHE.clear()
        outcome = env.runner().run(["S1a", "S1b", "S2", "S3", "S4"])
        assert outcome.status == "done", (outcome, env.out.getvalue())
        (root / "sweep.log").write_text(env.out.getvalue())
    finally:
        env.close()
        DEFAULT_QUERY_CACHE.clear()
        if previous is None:
            os.environ.pop(FAKE_ENV, None)
        else:
            os.environ[FAKE_ENV] = previous
        if key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = key
    return root, outcome.sweep_id


@pytest.fixture
def env(corpus_root: Path, tmp_path: Path) -> Iterator[Env]:
    _copy(corpus_root, tmp_path)
    environment = open_env(tmp_path)
    yield environment
    environment.close()


@pytest.fixture
def swept_env(swept_root: tuple[Path, str], tmp_path: Path) -> Iterator[tuple[Env, str]]:
    root, sweep_id = swept_root
    _copy(root, tmp_path)
    shutil.copy(root / "hal9000.db", tmp_path / "hal9000.db")
    shutil.copy(root / "sweep.log", tmp_path / "sweep.log")
    environment = open_env(tmp_path)
    yield environment, sweep_id
    environment.close()


def label_calibration(conn: sqlite3.Connection, run_id: str, *, agree: bool = True) -> None:
    """Three judge-calibration answers with Mike's labels (agreeing with the judge or not)."""
    from hal9000.knowledge.eval.store import list_gold

    judge = RubricLabels(correctness="correct", faithful=True, citations_correct=True,
                         abstention_correct=True)  # fmt: skip
    human = judge if agree else RubricLabels(correctness="wrong", faithful=False,
                                             citations_correct=False, abstention_correct=False)  # fmt: skip
    for item in list_gold(conn, status="approved", split="dev")[:3]:
        label = insert_judge_label(conn, item.id, "an answer", eval_run_id=run_id, judge=judge)
        set_human_labels(conn, label, human)


def results(run) -> list[dict]:
    return [json.loads(line) for line in Path(run.results_path).read_text().splitlines()]
