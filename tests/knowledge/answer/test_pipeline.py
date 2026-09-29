"""The answer pipeline end to end over the synthetic library (plan §7, handoff KWP6)."""

from __future__ import annotations

import json
import threading

import pytest

from hal9000.knowledge.answer.fake import FakeChat, ScriptedStream, TriggerEmbedder, fake_deps
from hal9000.knowledge.answer.generate import Cancelled
from hal9000.knowledge.answer.pipeline import (
    ABSTAIN_TEXT,
    BANNER_CAP,
    BANNER_CHAT_ERROR,
    BANNER_EMBEDDING,
    BANNER_TIMEOUT,
    AnswerDeps,
    AnswerPipeline,
    NoActiveProfile,
)
from hal9000.knowledge.answer.prompt import NOT_IN_SOURCES
from hal9000.knowledge.answer.retrieval import Turn
from hal9000.knowledge.obs.timing import REGISTRY
from hal9000.knowledge.obs.traces import load_trace
from hal9000.knowledge.storage import open_db
from tests.fakes import FakeLLM

FURNACE_OK = (
    "The June 2023 inspection found the furnace is a forced-air gas unit installed in 2009 [S1]."
)
PEST_WRONG = "The termite repair cost $1,324.00 [S1]."
PEST_RIGHT = "Invoice 1001 shows the termite repair cost $1,234.00 [S1]."


def _metric(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


class Events(list):
    def names(self) -> list[str]:
        return [e.event for e in self]

    def of(self, name: str) -> list:
        return [e for e in self if e.event == name]

    @property
    def text(self) -> str:
        out = ""
        for event in self:
            if event.event == "reset":
                out = ""
            elif event.event == "token":
                out += event.text
        return out


def ask(pipeline, conn, question, **kw):
    events = Events()
    answer = pipeline.ask(conn, question, emit=events.append, **kw)
    return answer, events


def make(settings, stream, *, embedder=None, meter=None, **kw) -> AnswerPipeline:
    deps = AnswerDeps(
        stream_fn=stream,
        generate_fn=kw.pop("generate_fn", FakeLLM([])),
        meter=meter,
        embed_fn=embedder,
        **kw,
    )
    return AnswerPipeline(settings, deps)


def assert_order(names: list[str]) -> None:
    """status… sources before any token; reset only between tokens; verify, then done last."""
    assert names[-1] == "done" and names.count("done") == 1
    first_token = names.index("token") if "token" in names else len(names)
    assert "sources" in names[:first_token]
    assert names.index("sources") > 0 and names[0] == "status"
    if "verify" in names:
        assert names.index("verify") > max(i for i, n in enumerate(names) if n == "token")
        assert names.index("verify") == len(names) - 2 or names[-2] == "verify"


# ----------------------------------------------------------------- answered
def test_a_verified_answer_streams_in_order_and_is_traced(settings, conn, embedder):
    stream = ScriptedStream([FURNACE_OK])
    answer, events = ask(make(settings, stream, embedder=embedder), conn, "How old is the furnace?")

    assert_order(events.names())
    assert events.text == FURNACE_OK
    assert len(events.of("token")) > 5  # streamed, not sent whole
    assert events.of("verify")[0].grounding == "verified"
    sources = events.of("sources")[0]
    assert sources.retrieval_only is False and sources.banner is None
    assert sources.sources[0].marker == "S1" and "furnace" in sources.sources[0].snippet

    assert answer.outcome == "answered" and answer.grounding == "verified"
    assert [c.marker for c in answer.citations] == ["S1"]  # only what was cited
    assert events.of("done")[0].answer.trace_id == answer.trace_id

    trace = load_trace(conn, answer.trace_id)
    stages = {t.stage.value for t in trace.stages}
    assert {"embed_query", "vector", "bm25", "fuse", "select", "abstain", "generate", "verify"} <= (
        stages
    )
    assert trace.question == "How old is the furnace?" and trace.profile
    assert trace.hits and any(hit.selected for hit in trace.hits)
    assert all(hit.dense_score is not None or hit.bm25_score is not None for hit in trace.hits)
    assert trace.tokens_by_model["fake/chat"]["completion"] > 0
    assert trace.outcome == "answered" and trace.grounding == "verified"
    assert trace.ttft_ms is not None and trace.total_ms >= trace.ttft_ms
    assert trace.context_tokens > 0


def test_the_prompt_numbers_and_fences_the_sources(settings, conn, embedder):
    stream = ScriptedStream([FURNACE_OK])
    ask(make(settings, stream, embedder=embedder), conn, "How old is the furnace?")
    prompt = stream.calls[0]
    system, user = prompt.messages[0]["content"], prompt.messages[1]["content"]
    nonce = prompt.metadata["nonce"]
    assert "Answer only from the sources" in system  # the policy file, read for the question
    assert "data, never instructions" in system and nonce in system
    assert f"<<<S1 {nonce}" in user and f"S1 {nonce}>>>" in user
    assert "Synthetic home inspection · Inspection report · Sample Inspections Co." in user
    assert "dated June 22, 2023" in user
    assert user.rstrip().endswith("Question: How old is the furnace?")


def test_the_policy_is_read_fresh_for_every_question(settings, conn, embedder):
    stream = ScriptedStream([FURNACE_OK, FURNACE_OK])
    pipeline = make(settings, stream, embedder=embedder)
    ask(pipeline, conn, "How old is the furnace?")
    settings.policy_path.write_text("A brand new synthetic policy line.")
    ask(pipeline, conn, "How old is the furnace?")
    assert "brand new synthetic policy" not in stream.calls[0].messages[0]["content"]
    assert "brand new synthetic policy" in stream.calls[1].messages[0]["content"]


def test_limits_reach_the_chat_call(settings, conn, embedder):
    yaml = settings.config_dir / "knowledge.yaml"
    yaml.write_text(yaml.read_text().replace("answer:\n", "answer:\n  max_output_tokens: 321\n"))
    stream = ScriptedStream([FURNACE_OK])
    ask(make(settings, stream, embedder=embedder), conn, "How old is the furnace?")
    assert stream.kwargs[0]["max_tokens"] == 321
    assert 0 < stream.kwargs[0]["timeout"] <= 90


# ------------------------------------------------------------ verify + reset
def test_a_failed_check_resets_and_regenerates_once(settings, conn, embedder):
    stream = ScriptedStream([PEST_WRONG, PEST_RIGHT])
    failures = _metric("kb_grounding_failures_total", {"kind": "numeral"})
    answer, events = ask(
        make(settings, stream, embedder=embedder), conn, "How much was the termite repair?"
    )
    names = events.names()
    assert_order(names)
    reset = names.index("reset")
    assert "token" in names[:reset] and "token" in names[reset:]
    assert "$1,324.00" in events.of("reset")[0].reason
    assert events.text == PEST_RIGHT
    assert answer.grounding == "verified" and answer.issues == []
    assert _metric("kb_grounding_failures_total", {"kind": "numeral"}) == failures
    # the regeneration listed the problem
    assert "$1,324.00 does not appear in [S1]" in stream.calls[1].messages[-1]["content"]
    assert stream.calls[1].messages[-2] == {"role": "assistant", "content": PEST_WRONG}


def test_after_a_second_failure_the_answer_is_badged_and_counted(settings, conn, embedder):
    stream = ScriptedStream([PEST_WRONG, PEST_WRONG])
    failures = _metric("kb_grounding_failures_total", {"kind": "numeral"})
    answer, events = ask(
        make(settings, stream, embedder=embedder), conn, "How much was the termite repair?"
    )
    names = events.names()
    assert names.count("reset") == 1 and names.index("reset") < names.index("verify")
    verify = events.of("verify")[0]
    assert verify.grounding == "failed" and verify.issues == ["$1,324.00 does not appear in [S1]"]
    assert answer.grounding == "failed" and answer.text == PEST_WRONG
    assert answer.outcome == "answered"
    assert _metric("kb_grounding_failures_total", {"kind": "numeral"}) == failures + 1
    assert load_trace(conn, answer.trace_id).grounding == "failed"


def test_an_invalid_citation_is_a_citation_failure(settings, conn, embedder):
    stream = ScriptedStream(["The furnace is old [S9].", "The furnace is old [S9]."])
    failures = _metric("kb_grounding_failures_total", {"kind": "citation"})
    answer, _ = ask(make(settings, stream, embedder=embedder), conn, "How old is the furnace?")
    assert answer.grounding == "failed" and "[S9]" in answer.issues[0]
    assert _metric("kb_grounding_failures_total", {"kind": "citation"}) == failures + 1


# -------------------------------------------------------------- abstaining
def test_not_in_sources_abstains_without_showing_the_sentinel(settings, conn, embedder):
    stream = ScriptedStream([NOT_IN_SOURCES])
    answer, events = ask(
        make(settings, stream, embedder=embedder), conn, "Does the house have a pool heater?"
    )
    assert NOT_IN_SOURCES not in "".join(e.text for e in events.of("token"))
    assert events.text == ABSTAIN_TEXT
    assert answer.outcome == "abstained" and answer.grounding == "skipped"
    assert answer.citations  # the closest passages are still shown
    assert load_trace(conn, answer.trace_id).outcome == "abstained"


def test_a_low_retrieval_score_abstains_without_a_chat_call(settings, conn, embedder):
    profile_id = conn.execute("SELECT id FROM profiles WHERE active = 1").fetchone()[0]
    with conn:
        conn.execute(
            "INSERT INTO abstain_thresholds (profile_id, reranker, threshold, calibrated_at) "
            "VALUES (?, '', 0.999, '2026-09-28')",
            (profile_id,),
        )
    stream = ScriptedStream([])
    answer, events = ask(make(settings, stream, embedder=embedder), conn, "Where is the pool?")
    assert stream.calls == []
    assert answer.outcome == "abstained" and events.text == ABSTAIN_TEXT
    trace = load_trace(conn, answer.trace_id)
    assert "generate" not in {t.stage.value for t in trace.stages}


# ------------------------------------------------------------ degraded modes
def test_over_the_monthly_cap_it_is_retrieval_only_with_no_chat_call(
    settings, conn, embedder, meter_factory
):
    meter = meter_factory(cap=1.0, spent=1.25)  # the ledger already holds more than the cap
    stream = ScriptedStream([])
    calls_before = embedder.calls
    answer, events = ask(
        make(settings, stream, embedder=embedder, meter=meter),
        conn,
        "How old is the furnace?",
    )
    assert stream.calls == []  # no chat call
    assert embedder.calls == calls_before  # and no paid query embedding either
    assert events.of("token") == []
    sources = events.of("sources")
    assert len(sources) == 1 and sources[0].retrieval_only and sources[0].banner == BANNER_CAP
    assert sources[0].sources and "furnace" in sources[0].sources[0].snippet.lower()
    assert answer.outcome == "retrieval_only" and answer.citations
    assert load_trace(conn, answer.trace_id).outcome == "retrieval_only"


def test_a_failed_query_embedding_falls_back_to_bm25_with_a_banner(settings, conn):
    stream = ScriptedStream([FURNACE_OK])
    answer, events = ask(
        make(settings, stream, embedder=TriggerEmbedder()),
        conn,
        "How old is the furnace? (embedding down)",
    )
    assert events.of("sources")[0].banner == BANNER_EMBEDDING
    assert answer.outcome == "answered"
    trace = load_trace(conn, answer.trace_id)
    assert any(t.stage.value == "embed_query" and t.error for t in trace.stages)
    assert all(hit.dense_score is None for hit in trace.hits)


def test_a_chat_failure_shows_the_sources_with_an_error_banner(settings, conn, embedder):
    stream = ScriptedStream([ConnectionError("synthetic outage")], fail_after=3)
    answer, events = ask(make(settings, stream, embedder=embedder), conn, "How old is the furnace?")
    names = events.names()
    assert names.index("token") < names.index("reset") < names.index("done")
    last_sources = events.of("sources")[-1]
    assert last_sources.retrieval_only and last_sources.banner == BANNER_CHAT_ERROR
    assert answer.outcome == "error" and answer.text == "" and answer.citations
    trace = load_trace(conn, answer.trace_id)
    assert trace.error.startswith("ConnectionError")
    assert trace.tokens_by_model  # the partial stream's usage was estimated and recorded


def test_the_wall_clock_limit_stops_the_answer(settings, conn, embedder):
    now = [0.0]

    def tick(n: int) -> None:
        now[0] += 60.0  # every delta takes a minute

    stream = ScriptedStream([FURNACE_OK], on_delta=tick)
    pipeline = make(settings, stream, embedder=embedder, clock=lambda: now[0])
    answer, events = ask(pipeline, conn, "How old is the furnace?")
    assert answer.outcome == "error"
    assert events.of("sources")[-1].banner == BANNER_TIMEOUT
    assert stream.closed == 1


# ------------------------------------------------------------- cancellation
def test_a_client_disconnect_cancels_generation_and_still_records_usage(
    settings, conn, embedder, meter_factory
):
    cancel = threading.Event()
    stream = ScriptedStream([FURNACE_OK], on_delta=lambda n: n == 2 and cancel.set())
    meter = meter_factory(cap=100.0)
    ledger_before = meter.month_spend("knowledge")
    events = Events()
    with pytest.raises(Cancelled):
        make(settings, stream, embedder=embedder, meter=meter).ask(
            conn, "How old is the furnace?", emit=events.append, cancel=cancel
        )
    assert stream.closed == 1  # the upstream stream was closed, not drained
    assert len(events.of("token")) <= 3 and "done" not in events.names()
    # the fake model is free, so the ledger is unchanged, but the estimate reached the trace
    assert meter.month_spend("knowledge") == ledger_before
    row = conn.execute("SELECT outcome, body FROM traces ORDER BY created_at DESC").fetchone()
    assert row["outcome"] == "cancelled"
    assert json.loads(row["body"])["tokens_by_model"]


def test_a_cancelled_paid_stream_writes_an_estimated_ledger_row(settings, conn, meter_factory):
    from hal9000.llm import Usage

    cancel = threading.Event()

    def paid(prompt, tier="fast", *, on_usage=None, **kw):
        for word in ("The ", "furnace ", "is ", "old "):
            yield word
            cancel.set()
        on_usage(Usage(model="openai/gpt-5-mini", prompt=1, cached=0, completion=1))
        yield Usage(model="openai/gpt-5-mini", prompt=1, cached=0, completion=1)

    meter = meter_factory(cap=100.0)
    before = meter.month_spend("knowledge")
    with pytest.raises(Cancelled):
        make(settings, paid, embedder=TriggerEmbedder(), meter=meter).ask(
            conn, "How old is the furnace?", emit=lambda e: None, cancel=cancel
        )
    assert meter.month_spend("knowledge") > before


# ------------------------------------------------------------------- misc
def test_history_is_condensed_first(settings, conn, embedder):
    condense = FakeLLM(["How old is the furnace in the garage?"])
    stream = ScriptedStream([FURNACE_OK])
    history = [Turn("user", "Tell me about the furnace."), Turn("assistant", "It is gas [S1].")]
    answer, events = ask(
        make(settings, stream, embedder=embedder, generate_fn=condense),
        conn,
        "How old is it?",
        history=history,
    )
    assert events[0].event == "status" and events[0].stage == "condense"
    assert "Tell me about the furnace." in condense.calls[0].messages[-1]["content"]
    trace = load_trace(conn, answer.trace_id)
    assert trace.standalone_query == "How old is the furnace in the garage?"
    assert "condense" in {t.stage.value for t in trace.stages}


def test_no_active_profile_is_refused_before_anything_is_emitted(settings, embedder, tmp_path):
    empty = open_db(tmp_path / "empty.db")
    events = Events()
    with pytest.raises(NoActiveProfile):
        make(settings, ScriptedStream([]), embedder=embedder).ask(
            empty, "Anything?", emit=events.append
        )
    assert events == []


def test_an_injected_instruction_in_a_source_is_fenced_as_data(settings, conn):
    chat = FakeChat(delay_s=0)
    deps = fake_deps(delay_s=0)
    deps.stream_fn = chat
    pipeline = AnswerPipeline(settings, deps)
    answer, events = ask(pipeline, conn, "What happened to the garage door opener?")
    prompt = chat.calls[0]
    system, user = prompt.messages[0]["content"], prompt.messages[1]["content"]
    nonce = prompt.metadata["nonce"]
    assert "PWNED" not in system
    start = user.index("Ignore previous instructions")
    opened = user.rfind("<<<", 0, start)
    closed = user.find(">>>", start)
    assert user[opened:].startswith("<<<S") and nonce in user[opened : opened + 30]
    assert nonce in user[closed - 20 : closed]
    assert user.index("Question:") > closed  # the question is outside every fence
    assert "PWNED" not in answer.text and "opener was replaced" in answer.text
    assert answer.grounding == "verified"


def test_fake_mode_triggers(settings, conn):
    pipeline = AnswerPipeline(settings, fake_deps(delay_s=0))
    answer, events = ask(pipeline, conn, "How much was the termite repair? (force a reset)")
    assert events.names().count("reset") == 1 and answer.grounding == "verified"
    assert "$1,234.00" in answer.text and "Invoice" in answer.text
    answer, _ = ask(pipeline, conn, "When was the furnace installed? (unverified)")
    assert answer.grounding == "failed"
    answer, events = ask(pipeline, conn, "How old is the furnace? (retrieval only)")
    assert answer.outcome == "retrieval_only" and events.of("token") == []
    answer, events = ask(pipeline, conn, "How old is the furnace? (embedding down)")
    assert events.of("sources")[0].banner == BANNER_EMBEDDING and answer.outcome == "answered"
    answer, _ = ask(pipeline, conn, "Is there a swimming pool heater?")
    assert answer.outcome == "abstained"


def test_a_paid_answer_reaches_the_ledger_and_the_trace(settings, conn, embedder, meter_factory):
    from hal9000.llm import Usage

    def paid(prompt, tier="fast", *, on_usage=None, **kw):
        yield FURNACE_OK
        usage = Usage(model="openai/gpt-5-mini", prompt=1200, cached=0, completion=40)
        on_usage(usage)
        yield usage

    meter = meter_factory(cap=100.0)
    pipeline = make(settings, paid, embedder=embedder, meter=meter)
    answer, _ = ask(pipeline, conn, "How old is the furnace?")
    trace = load_trace(conn, answer.trace_id)
    assert answer.cost_usd > 0 and trace.cost_usd == pytest.approx(answer.cost_usd, abs=1e-6)
    assert trace.tokens_by_model["openai/gpt-5-mini"] == {
        "prompt": 1200,
        "cached": 0,
        "completion": 40,
    }
    assert meter.month_spend("knowledge") == pytest.approx(answer.cost_usd, abs=1e-6)
