"""Turning a user's question into a safe FTS5 ``MATCH`` expression (plan §7.1 ``keyword_search``).

A question is never passed to ``MATCH`` raw: punctuation (``$``, ``'``, ``?``,
``-``, ``:``) raises FTS5 syntax errors, and FTS5's implicit AND would require
every word to be present. :func:`build_fts_query` instead:

1. finds **identifiers** first: reference numbers such as ``A-1234-5``,
   ``CHI-INV-20250314`` or ``12345678``, and figures such as ``$1,234.00``.
   Each becomes one quoted *phrase* of its tokens (``"a 1234 5"``), which FTS5
   matches only as consecutive tokens, so ``A-1234-5`` doesn't match a chunk
   that merely contains an ``a`` and a ``5``;
2. tokenizes the rest the way the index tokenizer does
   (``porter unicode61 remove_diacritics 2``: runs of letters and digits,
   case-folded; accents are left for FTS5 to remove on both sides);
3. drops stopwords and single letters;
4. double-quotes every term and joins everything with ``OR``.

Stemming isn't done here: FTS5 applies the ``porter`` stemmer to query terms,
quoted or not. The result contains only quoted strings and ``OR``, so it can't
be a syntax error, whatever the input. ``None`` means nothing searchable was
left (the question was only stopwords or punctuation).
"""

from __future__ import annotations

import re

from hal9000.knowledge.index.fake import STOPWORDS

__all__ = ["QUESTION_WORDS", "build_fts_query", "fts_tokens", "identifiers"]

#: Words that carry no search meaning in a question, on top of :data:`STOPWORDS`.
QUESTION_WORDS: frozenset[str] = frozenset(
    "please tell know show find give many much anything something whats wheres hows".split()
)

# unicode61 token characters: letters and digits (not underscore, which is punctuation)
_TOKEN = re.compile(r"[^\W_]+")
# figures: 1,234 / 1,234.56 / 12.50 (a leading $ or trailing % is punctuation to FTS5)
_FIGURE = re.compile(
    r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w]|[.,]\d)|(?<![\w.])\d+\.\d+(?![\w.])"
)
# a token run joined by - / . or : that contains a digit (A-1234-5, INV/2024/17, 6/22/23)
_JOINED = re.compile(r"(?<![\w])[^\W_]+(?:[-/.:#][^\W_]+)+")
# a long bare number (account, invoice, permit numbers)
_LONG_NUMBER = re.compile(r"(?<![\w])\d{5,}(?![\w])")


def fts_tokens(text: str) -> list[str]:
    """``text`` split and case-folded like the index tokenizer (before stemming)."""
    return [token.casefold() for token in _TOKEN.findall(text)]


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def identifiers(text: str) -> list[tuple[int, int]]:
    """Spans of ``text`` that are identifiers or figures, left to right, not overlapping."""
    spans: list[tuple[int, int]] = []
    for pattern in (_FIGURE, _JOINED, _LONG_NUMBER):
        for match in pattern.finditer(text):
            if pattern is _JOINED and not any(ch.isdigit() for ch in match.group(0)):
                continue  # well-known, e-mail: ordinary words, not an identifier
            start, end = match.span()
            if any(start < b and a < end for a, b in spans):
                continue
            spans.append((start, end))
    return sorted(spans)


def build_fts_query(text: str) -> str | None:
    """A ``MATCH`` expression for ``text``: quoted terms and phrases joined with ``OR``."""
    parts: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        if term and term not in seen:
            seen.add(term)
            parts.append(_quote(term))

    rest: list[str] = []
    cursor = 0
    for start, end in identifiers(text):
        rest.append(text[cursor:start])
        add(" ".join(fts_tokens(text[start:end])))
        cursor = end
    rest.append(text[cursor:])
    for token in fts_tokens(" ".join(rest)):
        if token in STOPWORDS or token in QUESTION_WORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        add(token)
    return " OR ".join(parts) if parts else None
