"""Text normalization shared by scoring, gold verification and selection alignment.

Evaluation compares *text*, never offsets (plan §8.2), so the same passage
extracted by two parsers must compare equal. Two forms are used:

- :func:`normalize`: NFKC, lower case, soft hyphens and PDFium's U+FFFE
  hyphen marker removed, ``word-⏎word`` joined, whitespace collapsed. Used for
  the "exact substring, ignoring whitespace and case" quote check (plan §8.1).
- :func:`joined`: only the letters and digits of the normalized text, run
  together, remembering where each word came from. Whitespace, punctuation and
  letter-spacing artifacts (``D ES C R I PT I O N``) all vanish, so fuzzy
  alignment on it is parser-independent. Evidence *tokens* are the words of
  the original text (runs of letters and digits).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "Joined",
    "find_normalized",
    "joined",
    "joined_text",
    "normalize",
    "words",
]

_WORD = re.compile(r"[^\W_]+")
_INVISIBLE = re.compile("[­￾​‌‍﻿]")
_HYPHEN_BREAK = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_HYPHEN_AT = re.compile(r"-[ \t]*\n[ \t]*(?=\w)")
_SPACE = re.compile(r"\s+")


def _fold(text: str) -> str:
    """NFKC, lower case, invisible characters removed, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text)).lower()
    return _SPACE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Lower-cased, NFKC-normalized text with hyphen breaks joined and spaces collapsed."""
    return _fold(_HYPHEN_BREAK.sub("", _INVISIBLE.sub("", text)))


def words(text: str) -> list[str]:
    """The words of ``text`` (runs of letters and digits), normalized."""
    return _WORD.findall(normalize(text))


@lru_cache(maxsize=65536)
def joined_text(text: str) -> str:
    """Only the letters and digits of ``normalize(text)``, run together (cached)."""
    return "".join(_WORD.findall(normalize(text)))


@dataclass(frozen=True)
class Joined:
    """Letters and digits of a text run together, with word spans and a source map.

    ``spans[i]`` is word *i*'s ``[start, end)`` in :attr:`text`; ``source[j]``
    is the index in the original string of joined character *j*.
    """

    text: str
    spans: tuple[tuple[int, int], ...]
    source: tuple[int, ...]

    @property
    def n_words(self) -> int:
        return len(self.spans)


@lru_cache(maxsize=4096)
def joined(text: str) -> Joined:
    """:func:`joined_text` with word spans and a map back to ``text``'s offsets.

    Works character by character so every joined character knows where it
    came from; soft hyphens and U+FFFE are skipped without breaking a word,
    and a hyphen at a line break joins the two halves (as cleaning does).
    """
    out: list[str] = []
    source: list[int] = []
    spans: list[tuple[int, int]] = []
    start: int | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if _INVISIBLE.match(ch):
            i += 1
            continue
        if ch == "-" and start is not None:
            match = _HYPHEN_AT.match(text, i)
            if match is not None:
                i = match.end()
                continue
        produced = [c for c in unicodedata.normalize("NFKC", ch).lower() if _WORD.match(c)]
        if produced:
            if start is None:
                start = len(out)
            for c in produced:
                out.append(c)
                source.append(i)
        elif start is not None:
            spans.append((start, len(out)))
            start = None
        i += 1
    if start is not None:
        spans.append((start, len(out)))
    return Joined("".join(out), tuple(spans), tuple(source))


def find_normalized(needle: str, haystack: str) -> tuple[int, int] | None:
    """Where ``needle`` occurs in ``haystack``, ignoring case and whitespace.

    Returns the ``[start, end)`` span in the *original* ``haystack``, or None.
    Used to check that an evidence quote is an exact substring of its page's
    cleaned text (plan §8.1) and to store the page's own wording.
    """
    target = _fold(needle)
    if not target:
        return None
    # Build the normalized haystack with a map back to original offsets.
    chars: list[str] = []
    origin: list[int] = []
    pending_space = False
    for index, ch in enumerate(haystack):
        if _INVISIBLE.match(ch):
            continue
        if ch.isspace():
            pending_space = bool(chars)
            continue
        folded = unicodedata.normalize("NFKC", ch).lower()
        if pending_space:
            chars.append(" ")
            origin.append(index)
            pending_space = False
        for c in folded:
            chars.append(c)
            origin.append(index)
    flat = "".join(chars)
    at = flat.find(target)
    if at < 0:
        return None
    end_index = origin[at + len(target) - 1] + 1
    return origin[at], end_index
