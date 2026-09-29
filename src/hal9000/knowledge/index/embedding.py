"""The embedding service: cached, batched, retried and metered (plan §5.1 step 8, §13).

Every vector the knowledge base computes goes through :class:`EmbeddingService`:

- **Cache.** ``embed_cache`` is keyed by ``(model, dimensions, sha256(text))``,
  so re-indexing, a profile rebuild with different chunking, a retried job and
  the evaluation sweeps never pay twice for the same text. Identical texts in
  one request are embedded once.
- **Batches** of :attr:`EmbeddingService.batch_size` texts; each batch's
  vectors are committed to the cache as soon as they arrive, so a job that
  dies half-way keeps what it paid for. Within a batch,
  :func:`hal9000.llm.embed` splits requests by the provider's token limits.
- **Retries** with exponential backoff on transient errors (rate limits,
  timeouts, connection errors, 5xx). Budget errors and bad requests are not
  retried.
- **Metering** (rule K9): with a :class:`~hal9000.cost.metering.Meter`, each
  batch goes through :meth:`Meter.embed` with ``agent="knowledge"`` (ingestion,
  queries) or ``agent="knowledge-eval"`` (the evaluation harness), so caps are
  checked before every request and every request writes a ledger row.
  Prometheus ``kb_tokens_total`` / ``kb_cost_usd_total{purpose="embed"}`` are
  counted from the same usage. A paid model without a meter is refused.
  ``fake/*`` models (the offline
  :class:`~hal9000.knowledge.index.fake.FakeEmbedder`, explicit fake mode only) are free and never
  reach the ledger.
- ``dimensions`` is passed to the provider, which is how
  ``text-embedding-3-large`` is reduced to 1,024 or 1,536 dimensions.

Don't call :meth:`EmbeddingService.embed` with a transaction open on the
connection: the cache writes commit.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from hal9000.knowledge.config import AGENT, EVAL_AGENT

__all__ = [
    "AGENT",
    "EVAL_AGENT",
    "EmbedStats",
    "EmbeddingError",
    "EmbeddingService",
    "UnmeteredCall",
    "default_embed_fn",
    "is_fake_model",
    "is_transient",
    "text_sha",
]

EmbedFn = Callable[..., Any]

#: Exception class names (LiteLLM / OpenAI / httpx) worth retrying.
_TRANSIENT_NAMES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "ServiceUnavailableError",
        "InternalServerError",
        "BadGatewayError",
    }
)
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    """The provider returned something unusable (wrong count or size of vectors)."""


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_fake_model(model: str) -> bool:
    return model.startswith("fake/")


def is_transient(error: BaseException) -> bool:
    """Whether ``error`` is worth retrying (never a budget or validation error)."""
    from hal9000.cost.metering import BudgetExceeded

    if isinstance(error, BudgetExceeded):
        return False
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    for cls in type(error).__mro__:
        if cls.__name__ in _TRANSIENT_NAMES:
            return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and status in _TRANSIENT_STATUS


class UnmeteredCall(RuntimeError):
    """A paid embedding model was used without a meter (rule K9)."""


def default_embed_fn(model: str) -> EmbedFn:
    """:class:`FakeEmbedder` for ``fake/*`` models (in fake mode only), else ``llm.embed``."""
    if is_fake_model(model):
        from hal9000.knowledge.index.fake import FakeEmbedder, require_fake_mode

        require_fake_mode(model)
        return FakeEmbedder()
    from hal9000.llm import embed

    return embed


@dataclass
class EmbedStats:
    """Counts for one :meth:`EmbeddingService.embed` call, or cumulative for the service."""

    requested: int = 0  # texts asked for (duplicates included)
    cache_hits: int = 0  # distinct texts found in embed_cache
    embedded: int = 0  # distinct texts sent to the provider
    calls: int = 0  # embed-function calls (one per batch attempt that succeeded)
    retries: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: EmbedStats) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class EmbeddingService:
    """Embeds texts through ``embed_cache``, in batches, with retries and metering.

    Args:
        conn: ``knowledge.db`` (holds ``embed_cache``).
        embed_fn: callable like :func:`hal9000.llm.embed`; ``None`` picks
            :func:`default_embed_fn` per model.
        meter: meters every non-fake call (``Meter.embed``); ``None`` calls directly.
        agent: the ledger agent: ``knowledge``, or ``knowledge-eval`` for evaluation.
        sleep: injectable for tests.
    """

    conn: sqlite3.Connection
    embed_fn: EmbedFn | None = None
    meter: Any | None = None
    agent: str = AGENT
    batch_size: int = 256
    max_attempts: int = 4
    backoff_s: float = 1.0
    sleep: Callable[[float], None] = time.sleep
    stats: EmbedStats = field(default_factory=EmbedStats)
    #: the counts of the most recent :meth:`embed` call
    last: EmbedStats = field(default_factory=EmbedStats)
    _fns: dict[str, EmbedFn] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.agent not in (AGENT, EVAL_AGENT):
            raise ValueError(f"agent must be {AGENT!r} or {EVAL_AGENT!r}, not {self.agent!r}")
        if self.batch_size < 1 or self.max_attempts < 1:
            raise ValueError("batch_size and max_attempts must be positive")

    # ----------------------------------------------------------------- cache
    def cached(self, texts: Sequence[str], model: str, dimensions: int) -> dict[str, np.ndarray]:
        """``sha -> vector`` for the texts already in ``embed_cache``."""
        shas = sorted({text_sha(text) for text in texts})
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(shas), 500):
            part = shas[start : start + 500]
            marks = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"SELECT text_sha, vector FROM embed_cache WHERE model = ? AND dimensions = ? "
                f"AND text_sha IN ({marks})",
                (model, dimensions, *part),
            ).fetchall()
            for sha, blob in rows:
                found[sha] = np.frombuffer(blob, dtype=np.float32)
        return found

    def _store(self, model: str, dimensions: int, vectors: dict[str, np.ndarray]) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO embed_cache (model, dimensions, text_sha, vector, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (model, dimensions, sha, vec.astype(np.float32).tobytes(), now)
                    for sha, vec in vectors.items()
                ],
            )

    # ------------------------------------------------------------- calling
    def _fn(self, model: str) -> EmbedFn:
        if self.embed_fn is not None:
            return self.embed_fn
        if model not in self._fns:
            self._fns[model] = default_embed_fn(model)
        return self._fns[model]

    def _call(self, texts: list[str], model: str, dimensions: int, stats: EmbedStats) -> Any:
        from hal9000.cost.metering import usage_cost
        from hal9000.knowledge.obs.metrics import record_usage

        fake = is_fake_model(model)
        fn = self._fn(model)

        def with_metrics(batch, model_, dims=None, *, on_usage=None, **kw):
            def sink(usage) -> None:
                if on_usage is not None:
                    on_usage(usage)
                cost = 0.0 if fake else usage_cost(dict(usage))
                stats.tokens += int(usage.get("prompt", 0) or 0)
                stats.cost_usd += cost
                record_usage(usage, "embed", cost_usd=cost)

            return fn(batch, model_, dims, on_usage=sink, **kw)

        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.meter is not None and not fake:
                    return self.meter.embed(
                        with_metrics, texts, model, dimensions, agent=self.agent
                    )
                return with_metrics(texts, model, dimensions)
            except Exception as error:
                if attempt == self.max_attempts or not is_transient(error):
                    raise
                stats.retries += 1
                self.sleep(self.backoff_s * 2 ** (attempt - 1))
        raise AssertionError("unreachable")  # pragma: no cover

    def embed(
        self,
        texts: Sequence[str],
        model: str,
        dimensions: int,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Vectors for ``texts`` (``len(texts) × dimensions``, float32), in order.

        ``progress(done, total)`` is called after each batch, counting distinct
        texts that needed the provider. A ``fake/*`` model needs fake mode; any
        other model needs a meter (:class:`UnmeteredCall` otherwise).
        """
        if is_fake_model(model):
            from hal9000.knowledge.index.fake import require_fake_mode

            require_fake_mode(model)
        elif self.meter is None:
            raise UnmeteredCall(
                f"{model} is a paid model: the embedding service needs a Meter (rule K9)"
            )
        stats = EmbedStats(requested=len(texts))
        order = [text_sha(text) for text in texts]
        vectors = self.cached(texts, model, dimensions)
        stats.cache_hits = len(vectors)
        missing: dict[str, str] = {}
        for sha, text in zip(order, texts):
            if sha not in vectors and sha not in missing:
                missing[sha] = text
        pending = list(missing.items())
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            result = self._call([text for _, text in batch], model, dimensions, stats)
            stats.calls += 1
            got = list(result.vectors)
            if len(got) != len(batch):
                raise EmbeddingError(f"{model} returned {len(got)} vectors for {len(batch)} texts")
            fresh: dict[str, np.ndarray] = {}
            for (sha, _), vec in zip(batch, got):
                array = np.asarray(vec, dtype=np.float32)
                if array.shape != (dimensions,):
                    raise EmbeddingError(
                        f"{model} returned a {array.shape[0]}-dimension vector; the profile "
                        f"expects {dimensions}"
                    )
                fresh[sha] = array
            self._store(model, dimensions, fresh)
            vectors.update(fresh)
            stats.embedded += len(batch)
            if progress is not None:
                progress(min(start + len(batch), len(pending)), len(pending))
        self.stats.add(stats)
        self.last = stats
        if not texts:
            return np.zeros((0, dimensions), dtype=np.float32)
        return np.stack([vectors[sha] for sha in order])

    def embed_query(self, text: str, model: str, dimensions: int) -> np.ndarray:
        """One query's vector, cached by its text, timed as ``stage("embed_query")`` (plan §7.1)."""
        from hal9000.knowledge.obs.timing import Stage, stage

        with stage(Stage.EMBED_QUERY):
            return self.embed([text], model, dimensions)[0]

    def estimate(self, texts: Sequence[str], model: str, dimensions: int) -> tuple[int, float]:
        """``(tokens, usd)`` to embed the texts not yet cached: the estimate paid commands print."""
        from hal9000.cost.metering import usage_cost
        from hal9000.knowledge.tokens import count_tokens, encoding_for_model

        cached = self.cached(texts, model, dimensions)
        seen: set[str] = set()
        tokens = 0
        encoding = encoding_for_model(model)
        for text in texts:
            sha = text_sha(text)
            if sha in cached or sha in seen:
                continue
            seen.add(sha)
            tokens += count_tokens(text, encoding)
        if is_fake_model(model):
            return tokens, 0.0
        return tokens, usage_cost({"model": model, "prompt": tokens})
