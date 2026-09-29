"""``select``: what the prompt receives, within the context budget (plan §7.1, §8.2).

Candidates are walked in final rank order (after fusion and any reranking):

1. **Parent expansion** (``parent_child`` profiles): a child is replaced by its
   parent section, which carries the child's scores and rank. A later child of a
   parent already selected adds nothing, so a parent's tokens are counted once.
2. **Near-duplicate removal**: an item whose normalised text overlaps an
   already selected item by at least :data:`NEAR_DUPLICATE` is dropped. Overlap
   is ``|A ∩ B| / min(|A|, |B|)`` over word 3-gram shingles of the text
   (lower-cased, accents folded, punctuation dropped), so the same passage in two
   files (a report and the bundle it was scanned into) or a chunk wholly inside
   one already sent is caught, while neighbouring chunks that share a 15%
   overlap are not.
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

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from hal9000.knowledge.contracts import RetrievedChunk
from hal9000.knowledge.tokens import count_tokens

__all__ = [
    "NEAR_DUPLICATE",
    "Selection",
    "overlap",
    "prompt_tokens",
    "select",
    "shingles",
]

#: Shingle overlap at or above which two items are near-duplicates.
NEAR_DUPLICATE = 0.8
_SHINGLE = 3
_WORD = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=8192)
def _index_tokens(header: str, text: str, encoding: str) -> int:
    return count_tokens(f"{header}\n{text}", encoding)


def prompt_tokens(header: str, text: str, tokens: int, *, send_headers: bool, encoding: str) -> int:
    """What a chunk costs in the budget: ``index_tokens`` with headers sent, else ``tokens``."""
    if send_headers and header:
        return _index_tokens(header, text, encoding)
    return tokens


@lru_cache(maxsize=8192)
def shingles(text: str) -> frozenset[tuple[str, ...]]:
    """Word 3-grams of the normalised text (the words themselves for very short texts)."""
    folded = unicodedata.normalize("NFKD", text.lower())
    words = _WORD.findall("".join(ch for ch in folded if not unicodedata.combining(ch)))
    if len(words) < _SHINGLE:
        return frozenset((w,) for w in words)
    return frozenset(tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1))


def overlap(a: frozenset, b: frozenset) -> float:
    """``|A ∩ B| / min(|A|, |B|)``; 0 when either is empty."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return sum(1 for s in small if s in large) / len(small)


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
    kept: list[frozenset] = []
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
        text = shingles(item.text)
        if any(overlap(text, other) >= near_duplicate for other in kept):
            continue
        if used + item.prompt_tokens > budget_tokens:
            break
        items.append(item)
        sources.append(candidate)
        kept.append(text)
        chosen.add(item.chunk_id)
        used += item.prompt_tokens
    return Selection(items, sources, used)
