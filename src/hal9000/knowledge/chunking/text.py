"""Text geometry for the chunkers: token offsets and candidate split points.

A :class:`TextIndex` is built once per (document text, encoding) and cached,
because the evaluation re-chunks the same corpus with many profiles
(plan §8.4). It holds

* the character offset where each token of the whole-text encoding starts,
  which gives a cheap *approximate* token count for any span, and an *exact*
  count (the real tokenizer on the substring, memoized) used to verify every
  chunk that is emitted; and
* sorted split points: word, line, sentence and paragraph starts and ends.

Offsets are Python string indices into the cleaned document text.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from functools import lru_cache

from hal9000.knowledge.tokens import count_tokens, encoding

__all__ = ["TextIndex", "text_index"]

#: Terminal punctuation (with closing quotes or brackets) followed by whitespace or the end.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|\Z)")
_WHITESPACE = re.compile(r"\s+")
_NON_WHITESPACE = re.compile(r"\S")


class TextIndex:
    """Token offsets and split points for one text under one encoding."""

    def __init__(self, text: str, encoding_name: str) -> None:
        self.text = text
        self.length = len(text)
        self.encoding_name = encoding_name
        enc = encoding(encoding_name)
        tokens = enc.encode(text, disallowed_special=())
        self.offsets: list[int] = enc.decode_with_offsets(tokens)[1] if tokens else []
        self.n_tokens = len(tokens)
        self._exact: dict[tuple[int, int], int] = {}

        punct_ends = {m.end() for m in _SENTENCE_END.finditer(text)}
        word_starts: list[int] = []
        word_ends: list[int] = []
        line_starts: list[int] = []
        line_ends: list[int] = []
        para_starts: list[int] = []
        para_ends: list[int] = []
        sent_starts: list[int] = []
        sent_ends: list[int] = []
        for m in _WHITESPACE.finditer(text):
            start, end = m.span()
            newlines = text.count("\n", start, end)
            inner = 0 < start and end < self.length
            if start > 0:
                word_ends.append(start)
            if end < self.length:
                word_starts.append(end)
            if newlines and inner:
                line_ends.append(start)
                line_starts.append(end)
                if newlines >= 2:
                    para_ends.append(start)
                    para_starts.append(end)
            if start in punct_ends:
                sent_ends.append(start)
                if end < self.length:
                    sent_starts.append(end)
        if self.length in punct_ends:
            sent_ends.append(self.length)

        self.word_starts = word_starts
        self.word_ends = word_ends
        self.line_starts = line_starts
        self.line_ends = line_ends
        self.para_starts = para_starts
        #: Where a sentence or a paragraph ends (``fixed`` snaps back to these).
        self.sentence_ends = sorted(set(sent_ends) | set(para_ends))
        #: Where a sentence or a paragraph starts (``recursive`` splits here).
        self.sentence_starts = sorted(set(sent_starts) | set(para_starts))

    # ------------------------------------------------------------ tokens
    def approx(self, start: int, end: int) -> int:
        """Tokens of the whole-text encoding that fall in ``[start, end)``; near-exact."""
        if end <= start:
            return 0
        return bisect_left(self.offsets, end) - bisect_right(self.offsets, start) + 1

    def exact(self, start: int, end: int) -> int:
        """The real token count of ``text[start:end]`` (memoized)."""
        key = (start, end)
        found = self._exact.get(key)
        if found is None:
            found = count_tokens(self.text[start:end], self.encoding_name) if end > start else 0
            self._exact[key] = found
        return found

    def tokens_before(self, pos: int) -> int:
        """How many tokens of the whole-text encoding start before ``pos``."""
        return bisect_left(self.offsets, pos)

    def token_at(self, pos: int) -> int:
        """Index of the token that contains character ``pos``."""
        return max(0, bisect_right(self.offsets, pos) - 1)

    def token_start(self, index: int) -> int:
        """Character offset where token ``index`` starts (the text's end past the last)."""
        return self.offsets[index] if index < self.n_tokens else self.length

    # ------------------------------------------------------- whitespace
    def skip_space(self, pos: int, end: int) -> int:
        """First non-whitespace position in ``[pos, end)``, or ``end``."""
        m = _NON_WHITESPACE.search(self.text, pos, end)
        return m.start() if m else end

    def trim(self, start: int, end: int) -> tuple[int, int]:
        """``[start, end)`` without leading and trailing whitespace (empty stays empty)."""
        start = self.skip_space(start, end)
        text = self.text
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    # ------------------------------------------------------ split points
    @staticmethod
    def last_in(points: list[int], lo: int, hi: int) -> int | None:
        """The largest point in ``[lo, hi]``."""
        i = bisect_right(points, hi) - 1
        return points[i] if i >= 0 and points[i] >= lo else None

    @staticmethod
    def inside(points: list[int], start: int, end: int) -> list[int]:
        """Points strictly inside ``(start, end)``."""
        return points[bisect_right(points, start) : bisect_left(points, end)]


@lru_cache(maxsize=32)
def text_index(text: str, encoding_name: str) -> TextIndex:
    """The cached :class:`TextIndex` for ``text``."""
    return TextIndex(text, encoding_name)
