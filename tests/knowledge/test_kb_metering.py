"""KF3: ``Meter.stream()``/``Meter.embed()`` and per-agent monthly caps (plan §13)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from hal9000 import config, llm
from hal9000.core.prompt import Prompt
from hal9000.cost.metering import (
    BudgetExceeded,
    Meter,
    MonthlyCapExceeded,
    monthly_precheck,
    usage_cost,
)
from hal9000.knowledge.config import AGENT, EVAL_AGENT, monthly_caps
from hal9000.storage.db import Database

CAPS = {AGENT: 10.0, EVAL_AGENT: 10.0}


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "hal.db")


def _meter(db, **kw):
    kw.setdefault("monthly_caps", CAPS)
    return Meter(db, "kb-run", soft_cap=kw.pop("soft", 100.0), hard_cap=kw.pop("hard", 100.0), **kw)


class FakeStream:
    """A ``stream_response`` stand-in that records whether it was called."""

    def __init__(self, usage=None):
        self.calls = 0
        self.usage = usage or llm.Usage(
            model="openai/gpt-5-mini", prompt=2000, cached=1000, completion=300, reasoning=0
        )

    def __call__(self, prompt, tier="fast", *, on_usage=None, **kw):
        self.calls += 1
        self.kw = kw

        def gen():
            yield "An "
            yield "answer."
            on_usage(self.usage)
            yield self.usage

        return gen()


class FakeEmbed:
    def __init__(self, batches=(1000,)):
        self.calls = 0
        self.batches = batches

    def __call__(self, texts, model, dimensions=None, *, on_usage=None, **kw):
        self.calls += 1
        total = 0
        for tokens in self.batches:
            if kw.get("before_request"):
                kw["before_request"]()
            on_usage(llm.Usage(model=model, prompt=tokens, cached=0, completion=0, reasoning=0))
            total += tokens
        return llm.Embeddings(
            vectors=[[0.0]] * len(texts),
            usage=llm.Usage(model=model, prompt=total, cached=0, completion=0, reasoning=0),
            requests=len(self.batches),
        )


def _spend(db, agent, usd):
    db.add_ledger("earlier", agent, {"model": "openai/gpt-5-mini"}, usd)


# -------------------------------------------------------------- over the cap
def test_over_the_knowledge_cap_stream_raises_before_calling_out(db):
    _spend(db, AGENT, 10.0)
    fake = FakeStream()
    with pytest.raises(MonthlyCapExceeded, match="knowledge has spent \\$10.00"):
        _meter(db).stream(fake, Prompt(), agent=AGENT)
    assert fake.calls == 0  # raised at call time, not on the first next()


def test_over_the_knowledge_cap_embed_raises_before_calling_out(db):
    _spend(db, AGENT, 12.5)
    fake = FakeEmbed()
    with pytest.raises(BudgetExceeded):
        _meter(db).embed(fake, ["a"], "openai/text-embedding-3-small", agent=AGENT)
    assert fake.calls == 0


def test_knowledge_eval_spend_does_not_count_toward_the_knowledge_cap(db):
    _spend(db, EVAL_AGENT, 50.0)
    fake = FakeStream()
    items = list(_meter(db).stream(fake, Prompt(), agent=AGENT))
    assert items[:2] == ["An ", "answer."] and fake.calls == 1


def test_the_eval_cap_is_its_own(db):
    _spend(db, AGENT, 50.0)
    meter = _meter(db)
    meter.embed(FakeEmbed(), ["a"], "openai/text-embedding-3-small", agent=EVAL_AGENT)
    _spend(db, EVAL_AGENT, 10.0)
    with pytest.raises(MonthlyCapExceeded, match="knowledge-eval"):
        meter.embed(FakeEmbed(), ["a"], "openai/text-embedding-3-small", agent=EVAL_AGENT)


def test_embed_rechecks_the_cap_before_every_batch(db, monkeypatch):
    """Review (c): a call that crosses the cap stops before its next batch."""
    import litellm

    requests = []

    def embedding(**kw):
        requests.append(len(kw["input"]))
        return SimpleNamespace(
            data=[{"index": i, "embedding": [0.0]} for i in range(len(kw["input"]))],
            usage={"prompt_tokens": 20_000},  # $0.0004 per request at $0.02/M
        )

    monkeypatch.setattr(litellm, "embedding", embedding)
    meter = Meter(db, "r", 100, 100, monthly_caps={AGENT: 0.0003})
    with pytest.raises(MonthlyCapExceeded):
        meter.embed(
            llm.embed,
            ["x"] * 3000,
            "openai/text-embedding-3-small",
            agent=AGENT,
            count_tokens=len,
        )
    assert requests == [2048]  # the second batch never went out
    assert meter.month_spend(AGENT) == pytest.approx(0.0004)  # the first stays billed


def test_last_month_spend_does_not_count(db):
    db.add_ledger("old", AGENT, {"model": "m"}, 99.0, at="2000-01-15T12:00:00")
    list(_meter(db).stream(FakeStream(), Prompt(), agent=AGENT))


def test_stream_and_embed_respect_the_run_hard_cap(db):
    meter = _meter(db, hard=0.0005)
    list(meter.stream(FakeStream(), Prompt(), agent=AGENT))  # spends $0.000875
    with pytest.raises(BudgetExceeded):
        meter.stream(FakeStream(), Prompt(), agent=AGENT)
    with pytest.raises(BudgetExceeded):
        meter.embed(FakeEmbed(), ["a"], "openai/text-embedding-3-small", agent=AGENT)


def test_generate_response_via_wrap_respects_the_knowledge_cap(db):
    """Condensing and segmentation use generate_response through Meter.wrap."""
    _spend(db, AGENT, 10.0)
    called = []
    wrapped = _meter(db).wrap(lambda p, t="fast": called.append(t) or "x", AGENT)
    with pytest.raises(MonthlyCapExceeded):
        wrapped(Prompt(), "fast")
    assert called == []


# ------------------------------------------------------------------ ledger rows
def test_stream_writes_one_ledger_row_after_the_call(db):
    fake = FakeStream()
    meter = _meter(db)
    stream = meter.stream(fake, Prompt(), "powerful", agent=AGENT, max_tokens=50)
    assert db.cost_breakdown(*_ym()) == []  # nothing until the usage arrives
    list(stream)
    rows = db.cost_breakdown(*_ym())
    assert [(r["agent"], r["calls"]) for r in rows] == [(AGENT, 1)]
    assert rows[0]["usd"] == pytest.approx(usage_cost(fake.usage), abs=1e-12)
    assert fake.kw == {"max_tokens": 50}


def test_embed_writes_a_row_per_request_and_costs_tokens_times_price(db):
    meter = _meter(db)
    result = meter.embed(
        FakeEmbed(batches=(123_457, 98_765)),
        ["x"] * 3,
        "openai/text-embedding-3-small",
        agent=EVAL_AGENT,
    )
    assert result.usage["prompt"] == 222_222
    rows = db.cost_breakdown(*_ym())
    assert rows[0]["agent"] == EVAL_AGENT and rows[0]["calls"] == 2
    expected = 222_222 * 0.02 / 1e6
    assert rows[0]["usd"] == pytest.approx(expected, abs=1e-9)
    assert meter.spent == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("model", "price"),
    [("openai/text-embedding-3-small", 0.02), ("openai/text-embedding-3-large", 0.13)],
)
def test_embedding_cost_is_tokens_times_the_configured_price(model, price):
    tokens = 1_234_567
    assert usage_cost({"model": model, "prompt": tokens}) == pytest.approx(
        tokens * price / 1e6, abs=1e-9
    )
    assert config.rates_for(model) == (price, price, 0.0)


def test_end_to_end_through_the_real_llm_functions(db, monkeypatch, openai_env):
    import litellm

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: iter(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))], usage=None
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=1000, completion_tokens=100, prompt_tokens_details=None
                    ),
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        litellm,
        "embedding",
        lambda **kw: SimpleNamespace(
            data=[{"index": 0, "embedding": [1.0]}], usage={"prompt_tokens": 5000}
        ),
    )
    meter = _meter(db)
    assert list(meter.stream(llm.stream_response, Prompt(), agent=AGENT))[0] == "Hi"
    meter.embed(llm.embed, ["q"], "openai/text-embedding-3-small", agent=AGENT, count_tokens=len)
    by_model = {r["model"]: r["usd"] for r in db.cost_breakdown(*_ym())}
    assert by_model["openai/gpt-5-mini"] == pytest.approx((1000 * 0.25 + 100 * 2.0) / 1e6)
    assert by_model["openai/text-embedding-3-small"] == pytest.approx(5000 * 0.02 / 1e6)
    assert meter.month_spend(AGENT) == pytest.approx(sum(by_model.values()))


# -------------------------------------------------------------- solar unchanged
def test_solar_agents_are_never_capped_by_the_knowledge_caps(db):
    _spend(db, AGENT, 10.0)
    _spend(db, EVAL_AGENT, 10.0)
    meter = _meter(db)
    assert meter.wrap(lambda p, t="powerful": "ran", "solar")(Prompt()) == "ran"
    assert meter.wrap(lambda p, t="powerful": "ran", "triage")(Prompt()) == "ran"


def test_the_solar_monthly_precheck_ignores_knowledge_spend(db):
    today = date.today()
    _spend(db, AGENT, 9.0)
    _spend(db, EVAL_AGENT, 9.0)
    status = monthly_precheck(db, today, 10.0)
    assert status.spent == 0 and not status.skip_agent and status.triage_enabled
    _spend(db, "solar", 8.5)
    db.add_ledger("r", None, {"model": "m"}, 0.1)  # an agent-less row still counts for solar
    status = monthly_precheck(db, today, 10.0)
    assert status.spent == pytest.approx(8.6) and not status.triage_enabled
    # the ledger total (what `hal9000 cost` prints) still includes everything
    assert db.month_cost(today.year, today.month) == pytest.approx(26.6)


def test_month_cost_for_filters(db):
    _spend(db, "solar", 1.0)
    _spend(db, AGENT, 2.0)
    db.add_ledger("r", None, {"model": "m"}, 4.0)
    now = datetime.now(UTC)
    y, m = now.year, now.month
    assert db.month_cost_for(y, m) == pytest.approx(7.0)
    assert db.month_cost_for(y, m, agents=[AGENT]) == pytest.approx(2.0)
    assert db.month_cost_for(y, m, agents=[]) == 0
    assert db.month_cost_for(y, m, exclude=[]) == pytest.approx(7.0)
    assert db.month_cost_for(y, m, exclude=[AGENT]) == pytest.approx(5.0)


def test_monthly_caps_come_from_the_environment(monkeypatch):
    assert monthly_caps() == {AGENT: 10.0, EVAL_AGENT: 10.0}
    monkeypatch.setenv("KB_MONTHLY_CAP_USD", "4.5")
    monkeypatch.setenv("KB_EVAL_CAP_USD", "2")
    assert monthly_caps() == {AGENT: 4.5, EVAL_AGENT: 2.0}


def test_month_spend_uses_the_meter_timezone(db):
    from zoneinfo import ZoneInfo

    meter = Meter(db, "r", 1, 1, tz=ZoneInfo("America/Los_Angeles"), monthly_caps=CAPS)
    meter.record({"model": "openai/text-embedding-3-small", "prompt": 1_000_000}, AGENT)
    assert meter.month_spend(AGENT) == pytest.approx(0.02)


def _ym():
    now = datetime.now(UTC)
    return now.year, now.month
