"""The embedding service: cache, batching, retries, metering; and the fake embedder."""

from __future__ import annotations

import numpy as np
import pytest

from hal9000.cost.metering import BudgetExceeded, Meter, MonthlyCapExceeded
from hal9000.knowledge.config import AGENT, EVAL_AGENT
from hal9000.knowledge.index.embedding import (
    EmbeddingError,
    EmbeddingService,
    default_embed_fn,
    is_transient,
    text_sha,
)
from hal9000.knowledge.index.fake import FAKE_EMBED_MODEL, FakeEmbedder, concepts
from hal9000.knowledge.obs.timing import REGISTRY
from hal9000.llm import embed as real_embed
from hal9000.storage.db import Database

TE3S = "openai/text-embedding-3-small"
TE3L = "openai/text-embedding-3-large"


class RateLimitError(Exception):
    """Named like LiteLLM's, which is how transient errors are recognised."""


class Flaky:
    """Fails ``failures`` times with ``error``, then behaves like the fake embedder."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.fake = FakeEmbedder()
        self.seen_dimensions: list[int | None] = []

    def __call__(self, texts, model, dimensions=None, **kw):
        self.seen_dimensions.append(dimensions)
        if self.failures:
            self.failures -= 1
            raise self.error
        return self.fake(texts, model, dimensions, **kw)


def test_the_cache_is_keyed_by_model_dimensions_and_text(kb_db):
    fake = FakeEmbedder()
    service = EmbeddingService(kb_db, embed_fn=fake)
    first = service.embed(["water heater", "garage door", "water heater"], FAKE_EMBED_MODEL, 64)
    assert first.shape == (3, 64) and first.dtype == np.float32
    assert np.array_equal(first[0], first[2])
    assert (service.last.embedded, service.last.cache_hits, fake.texts) == (2, 0, 2)

    again = service.embed(["garage door", "water heater", "attic"], FAKE_EMBED_MODEL, 64)
    assert (service.last.embedded, service.last.cache_hits) == (1, 2)
    assert np.array_equal(again[0], first[1]) and np.array_equal(again[1], first[0])
    assert fake.texts == 3

    service.embed(["water heater"], FAKE_EMBED_MODEL, 32)  # other dimensions: a miss
    service.embed(["water heater"], "fake/other", 64)  # other model: a miss
    assert fake.texts == 5
    rows = kb_db.execute("SELECT model, dimensions, text_sha FROM embed_cache").fetchall()
    assert (FAKE_EMBED_MODEL, 64, text_sha("attic")) in [tuple(r) for r in rows]
    assert len(rows) == 5
    assert service.stats.requested == 3 + 3 + 1 + 1
    assert service.embed([], FAKE_EMBED_MODEL, 64).shape == (0, 64)


def test_batches_are_cached_as_they_arrive(kb_db):
    fake = FakeEmbedder()
    service = EmbeddingService(kb_db, embed_fn=fake, batch_size=3)
    progress = []
    texts = [f"room {name}" for name in "abcdefgh"]
    service.embed(texts, FAKE_EMBED_MODEL, 64, progress=lambda d, t: progress.append((d, t)))
    assert fake.calls == 3 and service.last.calls == 3
    assert progress == [(3, 8), (6, 8), (8, 8)]

    def dies_on_third(batch, model, dims=None, **kw):
        if "room z" in batch:
            raise ValueError("provider rejected the batch")
        return fake(batch, model, dims, **kw)

    other = EmbeddingService(kb_db, embed_fn=dies_on_third, batch_size=2)
    with pytest.raises(ValueError):
        other.embed(["room q", "room r", "room s", "room z"], FAKE_EMBED_MODEL, 64)
    # the first batch is kept, so a retry pays only for the rest
    assert len(other.cached(["room q", "room r", "room s", "room z"], FAKE_EMBED_MODEL, 64)) == 2


def test_transient_errors_are_retried_with_exponential_backoff(kb_db):
    flaky = Flaky(2, RateLimitError("slow down"))
    sleeps: list[float] = []
    service = EmbeddingService(kb_db, embed_fn=flaky, backoff_s=0.5, sleep=sleeps.append)
    vectors = service.embed(["roof"], TE3L, 256)
    assert vectors.shape == (1, 256)
    assert sleeps == [0.5, 1.0] and service.last.retries == 2
    assert flaky.seen_dimensions == [256, 256, 256]  # reduced dimensions go to the provider

    gives_up = EmbeddingService(
        kb_db, embed_fn=Flaky(9, TimeoutError()), max_attempts=3, sleep=sleeps.append
    )
    with pytest.raises(TimeoutError):
        gives_up.embed(["gutter"], FAKE_EMBED_MODEL, 64)


@pytest.mark.parametrize("error", [ValueError("bad request"), BudgetExceeded("cap")])
def test_permanent_errors_are_not_retried(kb_db, error):
    flaky = Flaky(1, error)
    service = EmbeddingService(kb_db, embed_fn=flaky, sleep=lambda s: pytest.fail("slept"))
    with pytest.raises(type(error)):
        service.embed(["fence"], FAKE_EMBED_MODEL, 64)
    assert len(flaky.seen_dimensions) == 1


def test_transient_detection():
    class ServiceUnavailableError(Exception):
        pass

    class WithStatus(Exception):
        status_code = 503

    class BadRequest(Exception):
        status_code = 400

    assert is_transient(ServiceUnavailableError()) and is_transient(WithStatus())
    assert is_transient(ConnectionResetError())
    assert not is_transient(BadRequest()) and not is_transient(MonthlyCapExceeded("x"))


def test_wrong_vector_sizes_are_refused(kb_db):
    def eight_dims(texts, model, dimensions=None, **kw):
        return FakeEmbedder()(texts, model, 8, **kw)

    def one_short(texts, model, dimensions=None, **kw):
        result = FakeEmbedder()(texts, model, dimensions, **kw)
        result.vectors.pop()
        return result

    with pytest.raises(EmbeddingError, match="expects 16"):
        EmbeddingService(kb_db, embed_fn=eight_dims).embed(["deck"], FAKE_EMBED_MODEL, 16)
    with pytest.raises(EmbeddingError, match="returned 1 vectors for 2"):
        EmbeddingService(kb_db, embed_fn=one_short).embed(["deck", "stairs"], FAKE_EMBED_MODEL, 16)
    with pytest.raises(ValueError):
        EmbeddingService(kb_db, agent="solar")
    service = EmbeddingService(kb_db, embed_fn=FakeEmbedder())
    assert service.embed_query("deck", FAKE_EMBED_MODEL, 64).shape == (64,)


def test_paid_calls_go_through_the_meter_with_the_agent(kb_db, tmp_path):
    ledger = Database(tmp_path / "hal9000.db")
    meter = Meter(ledger, "kb-test", 10, 10, monthly_caps={AGENT: 5.0, EVAL_AGENT: 5.0})
    fake = FakeEmbedder()
    tokens = REGISTRY.get_sample_value("kb_tokens_total", {"model": TE3S, "purpose": "embed"}) or 0

    ingest = EmbeddingService(kb_db, embed_fn=fake, meter=meter)
    ingest.embed(["the main water shutoff valve"], TE3S, 64)
    evaluation = EmbeddingService(kb_db, embed_fn=fake, meter=meter, agent=EVAL_AGENT)
    evaluation.embed(["how old is the furnace"], TE3S, 64)
    EmbeddingService(kb_db, embed_fn=fake, meter=meter).embed(["free"], FAKE_EMBED_MODEL, 64)

    rows = ledger._conn.execute("SELECT agent, model, prompt_tokens FROM ledger").fetchall()
    assert [(r[0], r[1]) for r in rows] == [(AGENT, TE3S), (EVAL_AGENT, TE3S)]  # fake never billed
    assert ingest.last.cost_usd == pytest.approx(rows[0][2] * 0.02 / 1e6)
    assert ingest.last.tokens == rows[0][2] > 0
    after = REGISTRY.get_sample_value("kb_tokens_total", {"model": TE3S, "purpose": "embed"})
    assert after == tokens + rows[0][2] + rows[1][2]

    # at the monthly cap the meter refuses before calling out; nothing is cached
    capped = Meter(ledger, "kb-test", 10, 10, monthly_caps={AGENT: 0.0})
    calls = fake.calls
    with pytest.raises(MonthlyCapExceeded):
        EmbeddingService(kb_db, embed_fn=fake, meter=capped).embed(["new text"], TE3S, 64)
    assert fake.calls == calls
    ledger.close()


def test_estimate_counts_only_uncached_texts(kb_db):
    service = EmbeddingService(kb_db, embed_fn=FakeEmbedder())
    service.embed(["already cached text"], TE3S, 64)
    tokens, usd = service.estimate(
        ["already cached text", "brand new text", "brand new text"], TE3S, 64
    )
    assert tokens == 3 and usd == pytest.approx(3 * 0.02 / 1e6)
    assert service.estimate(["x y"], FAKE_EMBED_MODEL, 64)[1] == 0.0


def test_default_embed_fn_picks_the_fake_offline():
    assert isinstance(default_embed_fn(FAKE_EMBED_MODEL), FakeEmbedder)
    assert default_embed_fn(TE3S) is real_embed


# -------------------------------------------------------------- fake embedder
def cosine(a, b) -> float:
    return float(np.dot(a, b))


def test_fake_vectors_are_deterministic_unit_vectors():
    one, two = FakeEmbedder(), FakeEmbedder()
    a = np.array(one.vector("The furnace filter"))
    assert np.array_equal(a, np.array(two.vector("the FURNACE filters!")))
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert len(one.vector("x", 16)) == 16


def test_fake_synonyms_make_paraphrases_close():
    fake = FakeEmbedder()
    question = fake.vector("Is the water heater leaking?")
    finding = fake.vector("The relief valve on the water heater is dripping.")
    unrelated = fake.vector("Garage door opener remote programming")
    assert cosine(question, finding) > 0.5 > cosine(question, unrelated) + 0.3
    assert concepts("furnace heater boiler") == ["heater"] * 3
    assert concepts("costs prices charged fees") == ["cost"] * 4


def test_fake_is_blind_to_identifiers_and_numbers():
    assert concepts("Invoice CHI-INV-20250314 for $625.00 on 3/14/2025") == ["invoic"]
    fake = FakeEmbedder()
    # a query that is only an identifier gets a direction of its own, unrelated to the text
    assert (
        cosine(fake.vector("CHI-INV-20250314"), fake.vector("Invoice number CHI-INV-20250314"))
        < 0.5
    )


def test_fake_has_the_embed_signature():
    fake = FakeEmbedder()
    usages, before = [], []
    result = fake(["a b c", "d"], "fake/embed-64", 8, on_usage=usages.append,
                  before_request=lambda: before.append(1))  # fmt: skip
    assert len(result.vectors) == 2 and len(result.vectors[0]) == 8
    assert result.usage["prompt"] == 4 and usages == [result.usage] and before == [1]
    assert (fake.calls, fake.texts) == (1, 2)
