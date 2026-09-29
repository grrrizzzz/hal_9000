"""A deterministic, offline stand-in for the OpenAI embedding endpoint (tests, ``kb serve --fake``).

:class:`FakeEmbedder` has the call signature of :func:`hal9000.llm.embed`, so
it plugs into :class:`~hal9000.knowledge.index.embedding.EmbeddingService`
(and :meth:`hal9000.cost.metering.Meter.embed`) exactly where the real call
goes. It is a bag-of-words model:

- The text is lower-cased, accents are dropped and it is split into words.
  Stopwords are dropped, and a light suffix stemmer folds ``leaking`` and
  ``leaks`` into ``leak``.
- A small synonym table (:data:`SYNONYMS`) maps words onto one concept, so a
  paraphrase such as "Is the water heater leaking?" lands near "the valve is
  dripping". Everything else is unrelated to everything else.
- Each concept gets a fixed pseudo-random direction seeded by its SHA-256; a
  text's vector is the normalised sum of its concepts' directions.
- **Identifiers and numbers are invisible** (any word containing a digit), the
  way a real embedding model is weak on exact reference numbers. That is the
  gap BM25 exists to close, and the tests rely on it. A text with no concept
  at all gets a direction seeded by the text itself.

The vectors are unit length, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache

import numpy as np

from hal9000.knowledge.chunking.profiles import FAKE_EMBED_DIMENSIONS, FAKE_EMBED_MODEL
from hal9000.llm import Embeddings, Usage

__all__ = [
    "FAKE_EMBED_DIMENSIONS",
    "FAKE_EMBED_MODEL",
    "STOPWORDS",
    "SYNONYMS",
    "FakeEmbedder",
    "concepts",
]

#: word -> concept. Words in one group embed identically.
SYNONYMS: Mapping[str, str] = {
    **dict.fromkeys(("leak", "drip", "seep", "dribble", "weep"), "leak"),
    **dict.fromkeys(("furnace", "heater", "boiler"), "heater"),
    **dict.fromkeys(("cost", "price", "charge", "fee", "pay", "paid", "bill"), "cost"),
    **dict.fromkeys(("fix", "repair", "mend"), "repair"),
    **dict.fromkeys(("house", "home", "property", "residence"), "house"),
    **dict.fromkeys(("old", "age", "aged"), "age"),
    **dict.fromkeys(("broken", "damage", "damaged", "defect", "defective"), "damage"),
    **dict.fromkeys(("shutoff", "shut"), "shutoff"),
}

#: Function words carry no meaning in a bag of words (and are dropped from FTS queries too).
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves s t don doesn didn isn
    wasn aren weren won wouldn shouldn couldn ll re ve d m o y
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+(?:[-/.'][a-z0-9]+)*")
_DIGIT = re.compile(r"\d")
_VOWEL = re.compile(r"[aeiouy]")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(word: str) -> str:
    """A small suffix stripper: enough to fold plurals, -ing, -ed and a final -e."""
    stem = word
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            candidate = word[: len(word) - len(suffix)] + replacement
            if not _VOWEL.search(candidate):
                continue
            if len(candidate) > 3 and candidate[-1] == candidate[-2] and candidate[-1] not in "lsz":
                candidate = candidate[:-1]  # dripp -> drip
            stem = candidate
            break
    if len(stem) > 4 and stem.endswith("e"):
        stem = stem[:-1]  # charge, charged, charges -> charg
    return stem


def concepts(text: str) -> list[str]:
    """The concepts a text contributes to its vector, in order (duplicates kept)."""
    out: list[str] = []
    for match in _WORD.finditer(_fold(text)):
        word = match.group(0).replace("'", "")
        if _DIGIT.search(word):
            continue  # identifiers and numbers are invisible to the fake model
        for part in re.split(r"[-/.]", word):
            if len(part) < 2 or part in STOPWORDS:
                continue
            stem = _stem(part)
            out.append(_LOOKUP.get(part) or _LOOKUP.get(stem) or stem)
    return out


_LOOKUP: dict[str, str] = {
    **{_stem(word): concept for word, concept in SYNONYMS.items()},
    **SYNONYMS,
}


@lru_cache(maxsize=65536)
def _direction(seed: str, dimensions: int) -> np.ndarray:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vector = rng.standard_normal(dimensions)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Callable like :func:`hal9000.llm.embed`; counts its calls for tests.

    Attributes:
        calls: how many times it was called (one call = one ``embed`` batch).
        texts: how many texts it embedded in total.
    """

    def __init__(self, dimensions: int = FAKE_EMBED_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.calls = 0
        self.texts = 0
        self._lock = threading.Lock()

    def vector(self, text: str, dimensions: int | None = None) -> list[float]:
        size = dimensions or self.dimensions
        found = concepts(text)
        if not found:
            return _direction("text:" + _fold(text).strip(), size).tolist()
        total = np.zeros(size)
        for concept in set(found):
            total += _direction("concept:" + concept, size)
        return (total / np.linalg.norm(total)).tolist()

    def __call__(
        self,
        texts: Sequence[str],
        model: str = FAKE_EMBED_MODEL,
        dimensions: int | None = None,
        *,
        on_usage: Callable[[Usage], None] | None = None,
        count_tokens: Callable[[str], int] | None = None,
        before_request: Callable[[], None] | None = None,
    ) -> Embeddings:
        if before_request is not None:
            before_request()
        with self._lock:
            self.calls += 1
            self.texts += len(texts)
        counter = count_tokens or (lambda text: max(1, len(text.split())))
        usage = Usage(
            model=model,
            prompt=sum(counter(text) for text in texts),
            cached=0,
            completion=0,
            reasoning=0,
        )
        vectors = [self.vector(text, dimensions) for text in texts]
        if on_usage is not None:
            on_usage(usage)
        return Embeddings(vectors=vectors, usage=usage, requests=1 if texts else 0)
