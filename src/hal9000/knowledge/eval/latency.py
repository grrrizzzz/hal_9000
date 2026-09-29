"""S6: latency on the Pi (plan §8.4, §15): ``hal9000 kb eval latency --profiles A,B,C``.

For each profile (each must be built):

- **Configuration.** The best finished retrieval run (S1a-S4, no *k*) of that
  profile gives the retrieval method, reranker and so on; without one, the
  ``retrieval:`` block and ``reranker`` of ``config/knowledge.yaml``. Query
  transforms are not replayed (they are LLM calls, not local work).
- **Local retrieval** (plan §1: search + fusion + rerank + select, excluding
  the embedding call): ``--n`` questions replayed from the approved question
  set (cycled), after every distinct question's embedding is cached, so each
  timed retrieval hits the query-vector cache. Its time is the sum of the
  top-level stages except ``condense`` and ``embed_query``, from the stage
  timer. p50, p95 and p99 are recorded.
- **End to end** (``--end-to-end M``, paid): M live answers through the chat
  pipeline, timing the first streamed token (TTFT) and the whole answer.
- **Scratch ingest** (``--ingest-test SHA``, paid): the stored original
  ``data/knowledge/files/<SHA>`` ingested into a temporary data directory
  with an empty embedding cache, into the first profile, timed from submit
  to the finished job.

Everything paid is estimated first and needs ``--yes`` or a "y"; it is
metered as ``knowledge-eval`` and capped by ``KB_EVAL_CAP_USD``. Each profile's
result is an ``S6`` eval run with the host class (``pi`` on the Pi): its
metrics are the source run's dev metrics with the local latency replaced
(so the Pareto chart and the collector see a complete row), and the
measurements (p99, TTFT, end to end, ingest time) are under
``config["measured"]``.
"""

from __future__ import annotations

import sqlite3
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

import numpy as np

from hal9000.cost.metering import BudgetExceeded
from hal9000.knowledge.config import EVAL_AGENT, KnowledgeSettings
from hal9000.knowledge.contracts import EvalMetrics, EvalRun, GoldItem, IndexProfile
from hal9000.knowledge.eval.config import EvalConfig
from hal9000.knowledge.eval.decide import RETRIEVAL_STAGES, Ranker
from hal9000.knowledge.eval.runs import list_runs, record_run
from hal9000.knowledge.eval.store import list_gold
from hal9000.knowledge.eval.sweep import (
    SweepConfig,
    SweepDeps,
    config_answerer,
    load_sweep_settings,
    local_latency_ms,
)

__all__ = ["LatencyOutcome", "LatencyRunner", "percentiles", "source_run"]


def percentiles(values: Sequence[float], qs: Sequence[float] = (50, 95, 99)) -> list[float | None]:
    if not values:
        return [None for _ in qs]
    return [round(float(np.percentile(values, q)), 3) for q in qs]


def source_run(conn: sqlite3.Connection, profile: str, ranker: Ranker) -> EvalRun | None:
    """The best finished dev retrieval run (S1a-S4, no *k*) of ``profile``, if any."""
    runs = [
        r
        for r in list_runs(conn)
        if r.finished_at is not None
        and r.split == "dev"
        and r.stage in RETRIEVAL_STAGES
        and r.config.get("profile") == profile
        and r.config.get("k") is None
    ]
    best = ranker.top(runs, 1)
    return best[0] if best else None


@dataclass
class LatencyOutcome:
    status: str  # done, stopped, refused
    message: str = ""
    runs: list[EvalRun] = field(default_factory=list)


class LatencyRefused(Exception):  # noqa: N818 - control flow
    pass


@dataclass
class _Target:
    profile: IndexProfile
    config: SweepConfig
    source: EvalRun | None
    notes: list[str] = field(default_factory=list)


class LatencyRunner:
    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: KnowledgeSettings,
        *,
        eval_config: EvalConfig,
        meter: Any,
        eval_cap: float | None = None,
        deps: SweepDeps | None = None,
        confirm: Callable[[str], bool] = lambda question: True,
        out: TextIO | None = None,
        segment: bool | None = None,
    ) -> None:
        import sys

        self.conn = conn
        self.settings = settings
        self.eval_config = eval_config
        self.scoring = eval_config.scoring.to_config()
        self.meter = meter
        self.eval_cap = settings.eval_cap_usd if eval_cap is None else eval_cap
        self.deps = deps or SweepDeps()
        self.confirm = confirm
        self.out = out or sys.stdout
        self.segment = segment
        grid = eval_config.sweep and load_sweep_settings(eval_config)
        self.budget = grid.decision_budget if grid else 2000

    def say(self, message: str) -> None:
        print(message, file=self.out, flush=True)

    # ------------------------------------------------------------ planning
    def _targets(self, names: Sequence[str]) -> list[_Target]:
        from hal9000.knowledge.config import load_knowledge_file
        from hal9000.knowledge.index.profiles import get_profile, profile_status

        ranker = Ranker(self.scoring, budget=self.budget)
        knowledge = load_knowledge_file(self.settings.yaml_path)
        targets = []
        for name in names:
            profile = get_profile(self.conn, name)
            if profile is None or profile_status(self.conn, name) != "ready":
                raise LatencyRefused(
                    f"profile {name!r} is not built here; copy the Mac's knowledge.db, or "
                    "build it with `hal9000 kb profile activate`"
                )
            source = source_run(self.conn, name, ranker)
            notes = []
            if source is not None:
                config = SweepConfig.from_run_config(source.config)
            else:
                r = knowledge.retrieval
                config = SweepConfig.from_profile(
                    profile,
                    retrieval=r.method,
                    alpha=r.weighted_alpha if r.method == "weighted" else None,
                    reranker=knowledge.reranker or "none",
                )
                notes.append("no sweep run for this profile: config/knowledge.yaml retrieval")
            if config.query_transform != "none":
                notes.append(f"query transform {config.query_transform} not replayed")
            config = replace(config, query_transform="none", k=None, repeat=0)
            targets.append(_Target(profile, config, source, notes))
        return targets

    def _questions(self, n: int) -> tuple[list[GoldItem], list[GoldItem]]:
        """(the n replayed items, cycled; the distinct items)."""
        items = list_gold(self.conn, status="approved")
        if not items:
            raise LatencyRefused("no approved questions to replay: review the question set first")
        replay = [items[i % len(items)] for i in range(n)]
        return replay, items

    def _estimate(self, targets, distinct, end_to_end, ingest_sha) -> tuple[float, list[str]]:
        from hal9000.knowledge.eval.gold import estimate_cost
        from hal9000.knowledge.index.embedding import EmbeddingService, is_fake_model

        usd, lines = 0.0, []
        service = EmbeddingService(self.conn)
        questions = [i.question for i in distinct]
        for model, dims in sorted({(t.profile.embed_model, t.profile.dimensions) for t in targets}):
            if is_fake_model(model):
                continue
            tokens, cost = service.estimate(questions, model, dims)
            usd += cost
            lines.append(f"query embeddings: ~{tokens:,} uncached tokens with {model}-{dims}")
        if end_to_end:
            e = self.eval_config.judge.estimate
            for target in targets:
                tier = target.config.chat_tier or self.settings.chat_tier
                cost = estimate_cost(
                    end_to_end,
                    end_to_end * e.answer_prompt_tokens,
                    end_to_end * e.answer_output_tokens,
                    tier,
                )
                usd += cost.usd
                lines.append(f"{target.profile.name}: {end_to_end} live answers, {cost.describe()}")
        if ingest_sha:
            cost, line = self._ingest_estimate(targets[0].profile, ingest_sha)
            usd += cost
            lines.append(line)
        return usd, lines

    # ------------------------------------------------------------ ingest
    def _stored(self, sha: str):
        from hal9000.knowledge.index.library import get_file

        record = get_file(self.conn, sha)
        path = self.settings.files_dir / sha
        if record is None or not path.is_file():
            raise LatencyRefused(
                f"no stored file {sha!r} (data/knowledge/files/<sha256>, recorded in the library)"
            )
        return record, path

    def _use_llm(self) -> bool:
        if self.segment is not None:
            return self.segment
        import os

        return bool(os.environ.get("OPENAI_API_KEY", "").strip())

    def _ingest_estimate(self, profile: IndexProfile, sha: str) -> tuple[float, str]:
        """From the stored pages (nothing is parsed or paid before the confirmation)."""
        from hal9000.knowledge.cli.ingest import estimate_cost as ingest_cost
        from hal9000.knowledge.index.embedding import is_fake_model
        from hal9000.knowledge.index.library import load_pages
        from hal9000.knowledge.index.profiles import parser_key
        from hal9000.knowledge.tokens import count_tokens

        record, _path = self._stored(sha)
        pages = load_pages(self.conn, sha, parser_key(record, "pdfium"))
        parsed = SimpleNamespace(pages=pages, page_count=max(len(pages), 1))
        embed, segment, _ = ingest_cost(
            [Path(record.original_name)],
            embed_model=None if is_fake_model(profile.embed_model) else profile.embed_model,
            segment=self._use_llm(),
            parse=lambda path: parsed,
        )
        tokens = sum(count_tokens(p.text) for p in pages)
        return embed + segment, (
            f"scratch ingest of {record.original_name} into {profile.name}: ~{tokens:,} tokens, "
            f"embeddings ${embed:.4f}, segmentation ${segment:.4f} (empty cache)"
        )

    def _ingest(self, profile: IndexProfile, sha: str) -> dict[str, Any]:
        from hal9000.knowledge.config import load_knowledge_file
        from hal9000.knowledge.ingest.pipeline import IngestPipeline
        from hal9000.knowledge.storage import open_db

        record, path = self._stored(sha)
        knowledge = load_knowledge_file(self.settings.yaml_path).model_copy(
            update={"active_profile": profile.name}
        )
        use_llm = self._use_llm()
        generate = None
        if use_llm:
            fn = self.deps.generate_response
            if fn is None:
                from hal9000.llm import generate_response as fn
            generate = self.meter.wrap(fn, EVAL_AGENT)
        with tempfile.TemporaryDirectory(prefix="kb-ingest-test-") as tmp:
            root = Path(tmp)
            scratch = open_db(root / "knowledge.db")
            pipeline = IngestPipeline(
                root / "files",
                embed_fn=self.deps.embed_fn,
                meter=self.meter,
                agent=EVAL_AGENT,
                generate=generate,
                use_llm=use_llm,
                bus=None,
                isolate_parse=self.deps.isolate_parse,
                knowledge_file=knowledge,
            )
            try:
                started = time.perf_counter()
                result = pipeline.submit(scratch, path, record.original_name, use_llm=use_llm)
                if result.job_id is None:
                    raise LatencyRefused(f"scratch ingest not started: {result.reason}")
                job = pipeline.worker(root / "knowledge.db").run_job(scratch, result.job_id)
                seconds = time.perf_counter() - started
                if job.status != "done":
                    if self.meter is not None:
                        self.meter.precheck(EVAL_AGENT)
                    raise LatencyRefused(f"scratch ingest failed: {job.error}")
                pages = scratch.execute("SELECT count(*) FROM pages").fetchone()[0]
                chunks = scratch.execute("SELECT count(*) FROM chunks").fetchone()[0]
            finally:
                pipeline.close()
                scratch.close()
        return {
            "ingest_seconds": round(seconds, 3),
            "ingest_pages": int(pages),
            "ingest_chunks": int(chunks),
            "ingest_file": sha[:12],
            "ingest_llm_segmentation": use_llm,
        }

    # ------------------------------------------------------------ measuring
    def _replay(
        self, target: _Target, replay, distinct
    ) -> tuple[list[float], dict[int, list[float]]]:
        from hal9000.knowledge.obs.timing import start_trace
        from hal9000.knowledge.retrieve.eval_binding import binding

        config = target.config
        bound = binding(
            self.conn,
            self.settings,
            meter=self.meter,
            run=SimpleNamespace(config=config.retrieval_overrides()),
            embed_fn=self.deps.embed_fn,
            reranker=self.deps.reranker(config.reranker) if config.reranker != "none" else None,
        )
        for item in distinct:  # every query vector cached before timing
            bound.retrieve(item.question)
        latencies: list[float] = []
        per_item: dict[int, list[float]] = {}
        for item in replay:
            with start_trace(question="") as trace:
                bound.retrieve(item.question)
            ms = local_latency_ms(trace)
            latencies.append(ms)
            per_item.setdefault(item.id, []).append(ms)
        return latencies, per_item

    def _end_to_end(
        self, target: _Target, items: Sequence[GoldItem]
    ) -> tuple[list[float], list[float], float]:
        answer = config_answerer(
            self.conn, self.settings, target.config, meter=self.meter, deps=self.deps,
            budget=self.budget,
        )  # fmt: skip
        ttfts, totals, cost = [], [], 0.0
        for item in items:
            self.meter.precheck(EVAL_AGENT)
            first: list[float] = []
            started = time.perf_counter()

            def emit(event, first=first, started=started) -> None:
                if getattr(event, "event", None) == "token" and not first:
                    first.append((time.perf_counter() - started) * 1000)

            output, result = answer(item.question, emit)
            totals.append((time.perf_counter() - started) * 1000)
            if first:
                ttfts.append(first[0])
            if result.outcome in ("retrieval_only", "error"):
                self.meter.precheck(EVAL_AGENT)
            cost += output.cost_usd
        return ttfts, totals, cost

    # ------------------------------------------------------------ entry
    def run(
        self,
        profiles: Sequence[str],
        *,
        n: int = 200,
        end_to_end: int = 30,
        ingest_sha: str | None = None,
    ) -> LatencyOutcome:
        from hal9000.knowledge.eval.runs import host_class, new_run_id

        try:
            if not profiles:
                raise LatencyRefused("name at least one profile (--profiles A,B,C)")
            if n < 1 or end_to_end < 0:
                raise LatencyRefused("--n must be >= 1 and --end-to-end >= 0")
            targets = self._targets(profiles)
            replay, distinct = self._questions(n)
            live = [distinct[i % len(distinct)] for i in range(end_to_end)]
            usd, lines = self._estimate(targets, distinct, end_to_end, ingest_sha)
            host = host_class()
            self.say(
                f"S6 latency on this host ({host}): {len(targets)} profile(s), {n} replayed "
                f"retrievals each, {end_to_end} live answers each"
                + (f", scratch ingest of {ingest_sha[:12]}" if ingest_sha else "")
                + "."
            )
            self.say(f"Estimate: ${usd:.4f}")
            for line in lines:
                self.say(f"  - {line}")
            left = max(0.0, self.eval_cap - self.meter.month_spend(EVAL_AGENT))
            left = min(left, max(0.0, self.meter.hard_cap - self.meter.spent))
            if usd > left:
                raise LatencyRefused(
                    f"the estimate ${usd:.4f} exceeds what is left of the caps (${left:.4f})"
                )
            if not self.confirm("Run the latency measurements?"):
                raise LatencyRefused("not confirmed")
        except LatencyRefused as refused:
            return LatencyOutcome("refused", str(refused))

        outcome = LatencyOutcome("done")
        batch = new_run_id("s6")
        try:
            ingest = self._ingest(targets[0].profile, ingest_sha) if ingest_sha else None
            for number, target in enumerate(targets):
                spent = self.meter.spent
                latencies, per_item = self._replay(target, replay, distinct)
                p50, p95, p99 = percentiles(latencies)
                measured: dict[str, Any] = {
                    "replayed": len(latencies),
                    "local_p50_ms": p50,
                    "local_p95_ms": p95,
                    "local_p99_ms": p99,
                }
                e2e_cost = None
                if live:
                    ttfts, totals, e2e_cost = self._end_to_end(target, live)
                    t50, t95, _ = percentiles(ttfts)
                    e50, e95, _ = percentiles(totals)
                    measured.update(
                        {"end_to_end": len(totals), "ttft_p50_ms": t50, "ttft_p95_ms": t95,
                         "e2e_p50_ms": e50, "e2e_p95_ms": e95}
                    )  # fmt: skip
                if ingest and number == 0:
                    measured.update(ingest)
                run = self._record(target, batch, number, measured, per_item, distinct, e2e_cost,
                                   self.meter.spent - spent, host)  # fmt: skip
                outcome.runs.append(run)
                self.say(
                    f"  {target.profile.name}: local p50/p95/p99 {p50}/{p95}/{p99} ms"
                    + (
                        f"; TTFT p95 {measured['ttft_p95_ms']} ms, end to end p95 "
                        f"{measured['e2e_p95_ms']} ms"
                        if live
                        else ""
                    )  # fmt: skip
                    + (
                        f"; scratch ingest {ingest['ingest_seconds']} s"
                        if ingest and number == 0
                        else ""
                    )  # fmt: skip
                )
                for note in target.notes:
                    self.say(f"    note: {note}")
        except BudgetExceeded as error:
            outcome.status = "stopped"
            outcome.message = f"STOPPED at the cost cap: {error} (finished profiles are kept)"
        except LatencyRefused as refused:
            outcome.status = "refused"
            outcome.message = str(refused)
        return outcome

    def _record(self, target, batch, number, measured, per_item, distinct, e2e_cost, cost, host):
        source = target.source
        sweep_id = source.config.get("sweep_id") if source else None
        config = target.config.run_config(stage="S6", sweep_id=sweep_id)
        config.update({"measured": measured, "notes": target.notes})
        if source is not None:
            config["source_run"] = source.id
        if source is not None and source.metrics is not None:
            metrics = source.metrics.model_copy(
                update={
                    "latency_ms_p50": measured["local_p50_ms"],
                    "latency_ms_p95": measured["local_p95_ms"],
                }
            )
        else:
            metrics = EvalMetrics(
                n_questions=len(per_item),
                latency_ms_p50=measured["local_p50_ms"],
                latency_ms_p95=measured["local_p95_ms"],
            )
        if e2e_cost is not None and measured.get("end_to_end"):
            metrics = metrics.model_copy(
                update={"cost_per_question_usd": round(e2e_cost / measured["end_to_end"], 8)}
            )
        by_id = {item.id: item for item in distinct}
        records = [
            {
                "gold_item_id": item_id,
                "question": by_id[item_id].question,
                "category": by_id[item_id].category,
                "answerable": by_id[item_id].category != "unanswerable",
                "latency_ms": statistics.median(values),
                "recall": {},
                "full_support": {},
            }
            for item_id, values in per_item.items()
        ]
        return record_run(
            self.conn,
            eval_dir=self.settings.eval_dir,
            stage="S6",
            config=config,
            metrics=metrics,
            records=records,
            cost_usd=cost,
            run_id=f"{batch}.{number + 1}",
            host=host,
        )
