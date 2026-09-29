"""``select``: what the prompt receives, within the context budget (plan §7.1, §8.2).

Candidates are walked in final rank order (after fusion and any reranking):

1. **Parent expansion** (``parent_child`` profiles): a child is replaced by its
   parent section, which carries the child's scores and rank. A later child of a
   parent already selected adds nothing, so a parent's tokens are counted once.
2. **Near-duplicate removal**: a later item B is dropped as a copy of an
   already selected item A only when **all three** hold (:func:`is_near_duplicate`):

   - the shingle overlap ``|A ∩ B| / min(|A|, |B|)`` is at least
     :data:`NEAR_DUPLICATE` (0.9), over word 3-gram shingles of the text
     (lower-cased, accents folded, punctuation dropped);
   - B has fewer than :data:`MAX_UNSHARED` (5) shingles that A lacks, so B adds
     at most a few words; and
   - B's numbers and identifiers (words containing a digit) are a subset of A's,
     so a copy with a different amount, date or reference number is kept.

   The same passage in two files (a report and the bundle it was scanned into),
   or a chunk wholly inside one already sent, is dropped. Distinct findings that
   share boilerplate (the same "recommend evaluation by a licensed
   contractor..." paragraph) are kept, and so are neighbouring chunks that share
   a 15% overlap.
3. **Budget**: items are taken while their ``prompt_tokens`` fit in
   ``budget_tokens``. **The first item that would overflow stops selection**:
   a later, smaller item is not squeezed in. This keeps selection
   prefix-consistent: the selection at a smaller budget is exactly the prefix of
   the selection at a larger one, which is how KWP5's scorer truncates one
   ``select`` output to 1k/2k/4k (``budget_prefix``). So the scorer measures
   exactly what the chat would send at each budget. If even the top item
   overflows, nothing is selected (the scorer reports that budget as N/A).

``prompt_tokens`` is the header-plus-text token count (``index_tokens``) when
headers are sent, else the text's ``tokens``, counted in the profile's
embedding encoding, the unit chunk sizes are measured in.
"""

from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from hal9000.knowledge.contracts import RetrievedChunk
from hal9000.knowledge.tokens import count_tokens

__all__ = [
    "CACHE_SIZE",
    "MAX_UNSHARED",
    "NEAR_DUPLICATE",
    "Fingerprint",
    "Selection",
    "fingerprint",
    "is_near_duplicate",
    "overlap",
    "prompt_tokens",
    "select",
    "shingles",
]

#: Shingle overlap at or above which a copy may be a near-duplicate.
NEAR_DUPLICATE = 0.9
#: A near-duplicate adds fewer than this many shingles to the item it copies.
MAX_UNSHARED = 5
_SHINGLE = 3
_WORD = re.compile(r"[a-z0-9]+")
_DIGIT = re.compile(r"\d")
#: Entries kept in each of the per-text caches below (keyed by the text's SHA-256).
CACHE_SIZE = 2048


class _Lru:
    """A small, bounded, thread-safe LRU keyed by a text digest (no chunk text is kept as key)."""

    def __init__(self, size: int = CACHE_SIZE) -> None:
        self.size = size
        self._items: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()

    def get_or(self, key: tuple, make):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = make()
        with self._lock:
            self._items[key] = value
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._items)


_TOKENS = _Lru()
_PRINTS = _Lru()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_tokens(header: str, text: str, tokens: int, *, send_headers: bool, encoding: str) -> int:
    """What a chunk costs in the budget: ``index_tokens`` with headers sent, else ``tokens``."""
    if send_headers and header:
        index_text = f"{header}\n{text}"
        return _TOKENS.get_or(
            (_sha(index_text), encoding), lambda: count_tokens(index_text, encoding)
        )
    return tokens


def _words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    return _WORD.findall("".join(ch for ch in folded if not unicodedata.combining(ch)))


def _shingles(words: list[str]) -> frozenset[tuple[str, ...]]:
    if len(words) < _SHINGLE:
        return frozenset((w,) for w in words)
    return frozenset(tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1))


@dataclass(frozen=True)
class Fingerprint:
    """A text's word 3-gram shingles and its numbers and identifiers (words with a digit)."""

    shingles: frozenset[tuple[str, ...]]
    identifiers: frozenset[str]


def fingerprint(text: str) -> Fingerprint:
    """The text's :class:`Fingerprint`, cached by the text's SHA-256."""

    def make() -> Fingerprint:
        words = _words(text)
        return Fingerprint(_shingles(words), frozenset(w for w in words if _DIGIT.search(w)))

    return _PRINTS.get_or((_sha(text),), make)


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    """Word 3-grams of the normalised text (the words themselves for very short texts)."""
    return fingerprint(text).shingles


def overlap(a: frozenset, b: frozenset) -> float:
    """``|A ∩ B| / min(|A|, |B|)``; 0 when either is empty."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return sum(1 for s in small if s in large) / len(small)


def is_near_duplicate(
    later: Fingerprint, kept: Fingerprint, *, threshold: float = NEAR_DUPLICATE
) -> bool:
    """Whether ``later`` only repeats ``kept`` (see the module docstring for the three tests)."""
    if overlap(later.shingles, kept.shingles) < threshold:
        return False
    if sum(1 for s in later.shingles if s not in kept.shingles) >= MAX_UNSHARED:
        return False
    return later.identifiers <= kept.identifiers


@dataclass(frozen=True)
class Selection:
    """``select``'s output: the items in prompt order, their total tokens and their sources.

    ``sources[i]`` is the candidate (the searched chunk) that brought
    ``items[i]`` in: the item itself, or the child of an expanded parent.
    """

    items: list[RetrievedChunk]
    sources: list[RetrievedChunk]
    tokens: int


def select(
    candidates: Sequence[RetrievedChunk],
    budget_tokens: int,
    *,
    parents: dict[int, RetrievedChunk] | None = None,
    near_duplicate: float = NEAR_DUPLICATE,
) -> Selection:
    """Take candidates in order within the budget, expanding children and dropping duplicates.

    ``parents`` maps a parent chunk ID to the parent as a
    :class:`RetrievedChunk` (text, header, ``prompt_tokens``); a candidate with
    a ``parent_id`` in it is replaced by its parent.
    """
    items: list[RetrievedChunk] = []
    sources: list[RetrievedChunk] = []
    kept: list[Fingerprint] = []
    chosen: set[int] = set()
    used = 0
    for candidate in candidates:
        item = candidate
        if parents and candidate.parent_id is not None and candidate.parent_id in parents:
            parent = parents[candidate.parent_id]
            item = parent.model_copy(
                update={
                    "rank": candidate.rank,
                    "score": candidate.score,
                    "dense_score": candidate.dense_score,
                    "bm25_score": candidate.bm25_score,
                    "rerank_score": candidate.rerank_score,
                    "vector_rank": candidate.vector_rank,
                    "bm25_rank": candidate.bm25_rank,
                    "fused_score": candidate.fused_score,
                }
            )
        if item.chunk_id in chosen:
            continue  # its parent is already in the prompt: no tokens counted twice
        text = fingerprint(item.text)
        if any(is_near_duplicate(text, other, threshold=near_duplicate) for other in kept):
            continue
        if used + item.prompt_tokens > budget_tokens:
            break
        items.append(item)
        sources.append(candidate)
        kept.append(text)
        chosen.add(item.chunk_id)
        used += item.prompt_tokens
    return Selection(items, sources, used)
