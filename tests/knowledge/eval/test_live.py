"""Live gold generation (handoff KWP5): 5 items from the fixture corpus, capped at $0.50.

Opt-in with ``pytest -m live`` on the Mac; skips without ``OPENAI_API_KEY``.
The spend is written to the real ledger as ``knowledge-eval``.
"""

from __future__ import annotations

import os

import pytest

from hal9000.knowledge.eval.config import GoldConfig
from hal9000.knowledge.eval.gold import generate_gold, plan_generation
from hal9000.knowledge.eval.text import find_normalized

#: the environment before the autouse fixtures strip provider keys
_ENV_AT_IMPORT = dict(os.environ)


@pytest.mark.live
def test_live_gold_generation_of_five_items(fixture_docs, monkeypatch):
    from dotenv import dotenv_values

    from hal9000 import config as hal_config
    from hal9000.cost.metering import Meter
    from hal9000.knowledge.config import EVAL_AGENT, monthly_caps
    from hal9000.llm import generate_response
    from hal9000.storage.db import Database

    env = {**{k: v for k, v in dotenv_values().items() if v is not None}, **_ENV_AT_IMPORT}
    if not env.get("OPENAI_API_KEY"):
        pytest.skip("no OPENAI_API_KEY")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    cfg = GoldConfig(
        quotas={"factoid": 2, "money": 1, "date": 1, "list": 1},
        overgenerate=1.0,
        window_pages=12,
    )
    docs = [d for d in fixture_docs if d.doc_type == "inspection_report"]
    plan = plan_generation(docs, cfg)
    estimate = plan.estimate()
    print(estimate.describe())
    assert estimate.usd < 0.50
    hal = hal_config.settings()
    database = Database(hal.data_dir / "hal9000.db")
    meter = Meter(database, "kb-eval-live-gold", 0.5, 0.5, tz=hal.tz, monthly_caps=monthly_caps())
    try:
        result = generate_gold(plan, meter.wrap(generate_response, EVAL_AGENT))
    finally:
        database.close()
    print(result.summary(), f"cost=${meter.spent:.4f}")
    assert meter.spent <= 0.50
    assert len(result.items) + sum(result.shortfall.values()) == 5
    assert len(result.items) >= 4
    doc = docs[0]
    for item in result.items:
        for ev in item.evidence:
            assert find_normalized(ev.quote, doc.pages[ev.page]) is not None
