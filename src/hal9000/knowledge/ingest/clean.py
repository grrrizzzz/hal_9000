"""Cleaning: raw parser text → clean page text with an offset map (plan §5.1 step 5).

Four repairs, in order, on each page:

1. **Running headers and footers** are removed: a line near the top or bottom
   of a page (the *edge zone*) whose normalised form -- case-folded, spaces
   collapsed, every digit run replaced by ``#`` -- appears on at least 30% of
   the file's pages with text (``ingest.header_repeat_fraction``), and on at
   least two. That catches DocuSign envelope IDs, "Page 12 of 58" and an
   inspector's name in the header, while a finding line that happens to repeat
   on two pages stays.
2. **Hyphenation**: PDFium's line-break hyphen marker (U+FFFE), soft hyphens,
   and ``word-⏎word`` at a line end are joined.
3. **Letter spacing**: headings such as ``D ES C R I PT I O N O F`` are rejoined
   (``DESCRIPTION OF``). A run is only rewritten when a small vocabulary can
   segment it, so ordinary text such as ``A B testing`` is untouched.
4. **Whitespace**: tabs, no-break spaces and runs of spaces collapse to one
   space, lines are trimmed, and three or more newlines become two.

Every cleaned character remembers where it came from. Each
:class:`~hal9000.knowledge.contracts.Page` stores that as ``offset_map``:
``(clean_offset, raw_offset)`` pairs at every point where cleaning shifted the
text (:func:`source_offset` reads it). :class:`PagedText` joins cleaned pages
into one document text and maps any offset in it back to its page.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from hal9000.knowledge.contracts import Page
from hal9000.knowledge.ingest.parsers.base import ParsedPage
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = [
    "DEFAULT_REPEAT_FRACTION",
    "DEFAULT_ZONE",
    "PAGE_SEPARATOR",
    "VOCABULARY",
    "CleanResult",
    "PagedText",
    "clean_page_text",
    "clean_pages",
    "normalize_line",
    "repair_letter_spacing",
    "running_lines",
    "source_offset",
]

#: Lines on at least this share of pages are running headers or footers (plan §5.1).
DEFAULT_REPEAT_FRACTION = 0.30
#: How many non-blank lines at each end of a page count as header or footer.
DEFAULT_ZONE = 5
#: Placed between pages when they are joined into one document text.
PAGE_SEPARATOR = "\n\n"

# Words the letter-spacing repair may assemble. Headings on estimates,
# invoices, inspection and pest reports, disclosures and manuals, plus common
# short words. Extend with ``vocabulary=`` rather than by guessing.
VOCABULARY = frozenset(
    """
    a about above access account additional address after age agreement air all amount and any
    appliance appliances approval approved are area as at attic authorization balance basement
    bath bathroom bathrooms be bedroom bid billing bill by call certificate certification charge
    charges check city client code comments company completed completion conditions contact
    contract contractor cost costs crawlspace credit customer damage date deck deposit
    description details disclosure discount distribution doors drain drainage due each
    electrical email equipment estimate estimated evaluation exclusions exterior fax fee fees
    findings fire for foundation from fungus furnace garage gas general grading gutters heater
    heating hours house hvac if in information inspected inspection inspector installation
    installed insulation interior invoice is it item items job kitchen labor labour laundry
    license limitations limited line lot maintenance manual material materials method model
    no not notes notice number of on or order organisms other our owner page paid parts
    payment performed permit pest pests phone plumbing pool price pricing project property
    proposal qty quantity rate re receipt recommendations recommended reference remarks repair
    repairs replace replacement report reports roof roofing room scope section serial service
    services sewer signature site sold standards structural subtotal summary system systems
    tax technician terms the thank this to total unit units upon ventilation video warranty
    water we windows with wood work you your
    """.split()
)

_SINGLE_LETTER_WORDS = frozenset({"a", "i"})
_KEEP_TOKEN_COST = 3.0
_SINGLE_LETTER_COST = 2.0
_WORD_COST = 1.0

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
# 4+ tokens of 1-3 capitals separated by single spaces, on one line
_SPACED_RUN = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{1,3} ){3,}[A-Z]{1,3}(?![A-Za-z0-9])")


# ------------------------------------------------------------ tracked text
@dataclass
class _Tracked:
    """Text plus, for each character, its offset in the raw parser text."""

    text: str
    src: list[int]
    end: int  # raw length, the source of characters inserted at the very end

    @classmethod
    def of(cls, raw: str) -> _Tracked:
        return cls(raw, list(range(len(raw))), len(raw))

    def sub(self, pattern: re.Pattern[str], repl: str | Callable[[re.Match[str]], str]) -> _Tracked:
        parts: list[str] = []
        src: list[int] = []
        last = 0
        for match in pattern.finditer(self.text):
            start, stop = match.span()
            parts.append(self.text[last:start])
            src.extend(self.src[last:start])
            new = repl(match) if callable(repl) else match.expand(repl)
            after = self.src[stop] if stop < len(self.src) else self.end
            src.extend(_align(match.group(0), self.src[start:stop], new, after))
            parts.append(new)
            last = stop
        parts.append(self.text[last:])
        src.extend(self.src[last:])
        return _Tracked("".join(parts), src, self.end)

    def keep(self, spans: Iterable[tuple[int, int]]) -> _Tracked:
        text: list[str] = []
        src: list[int] = []
        for start, stop in spans:
            text.append(self.text[start:stop])
            src.extend(self.src[start:stop])
        return _Tracked("".join(text), src, self.end)

    def offset_map(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for index, source in enumerate(self.src):
            if index == 0 or source != self.src[index - 1] + 1:
                pairs.append((index, source))
        return pairs


def _align(old: str, old_src: list[int], new: str, after: int) -> list[int]:
    """Source offsets for a replacement: kept characters keep theirs, inserted ones borrow."""
    out: list[int] = []
    j = 0
    for char in new:
        k = old.find(char, j)
        if k >= 0:
            out.append(old_src[k])
            j = k + 1
        else:
            out.append(old_src[j] if j < len(old_src) else after)
    return out


def source_offset(offset_map: Sequence[Sequence[int]], clean_offset: int) -> int:
    """The raw-text offset of cleaned character ``clean_offset`` on one page."""
    if not offset_map:
        return clean_offset
    index = bisect.bisect_right([pair[0] for pair in offset_map], clean_offset) - 1
    clean, raw = offset_map[max(index, 0)]
    return raw + (clean_offset - clean)


# ------------------------------------------------------ headers and footers
def normalize_line(line: str) -> str:
    """The comparison key for header/footer detection: case, spacing and digits folded."""
    return _DIGITS.sub("#", _SPACES.sub(" ", line).strip().casefold())


def _line_spans(text: str) -> list[tuple[int, int]]:
    """``(start, stop)`` of each line, including its newline."""
    spans, start = [], 0
    for line in text.splitlines(keepends=True):
        spans.append((start, start + len(line)))
        start += len(line)
    return spans


def _edge_lines(text: str, zone: int) -> list[tuple[int, int, str]]:
    """``(start, stop, key)`` for the first and last ``zone`` non-blank lines."""
    lines = [(a, b, normalize_line(text[a:b])) for a, b in _line_spans(text)]
    lines = [line for line in lines if line[2]]
    picked = lines[:zone] + lines[max(zone, len(lines) - zone) :]
    return picked


def running_lines(texts: Sequence[str], fraction: float, zone: int = DEFAULT_ZONE) -> set[str]:
    """Normalised lines that repeat in the edge zone of ≥ ``fraction`` of the pages with text."""
    with_text = [text.replace("\r\n", "\n").replace("\r", "\n") for text in texts if text.strip()]
    if len(with_text) < 2:
        return set()
    minimum = max(2, math.ceil(fraction * len(with_text) - 1e-9))
    counts: Counter[str] = Counter()
    for text in with_text:
        counts.update({key for _, _, key in _edge_lines(text, zone)})
    return {key for key, count in counts.items() if count >= minimum}


# ------------------------------------------------------------ letter spacing
def _segment(tokens: list[str], vocabulary: frozenset[str]) -> list[str] | None:
    """Best split of the joined tokens into vocabulary words or kept tokens (DP, least cost)."""
    joined = "".join(tokens)
    boundaries = {0}
    position = 0
    token_at: dict[int, str] = {}
    for token in tokens:
        token_at[position] = token
        position += len(token)
        boundaries.add(position)
    n = len(joined)
    best: list[tuple[float, list[str]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])
    longest = max((len(word) for word in vocabulary), default=1)
    for i in range(n):
        if best[i] is None:
            continue
        cost, words = best[i]
        candidates: list[tuple[int, float, str]] = []
        for j in range(i + 1, min(n, i + longest) + 1):
            piece = joined[i:j].casefold()
            if piece in vocabulary and (len(piece) > 1 or piece in _SINGLE_LETTER_WORDS):
                weight = _SINGLE_LETTER_COST if len(piece) == 1 else _WORD_COST
                candidates.append((j, weight, joined[i:j]))
        if i in token_at:
            token = token_at[i]
            candidates.append((i + len(token), _KEEP_TOKEN_COST, token))
        for j, weight, word in candidates:
            if best[j] is None or cost + weight < best[j][0]:
                best[j] = (cost + weight, [*words, word])
    return best[n][1] if best[n] is not None else None


def repair_letter_spacing(text: str, vocabulary: Iterable[str] | None = None) -> str:
    """Rejoin letter-spaced capitals (``D ES C R I PT I O N`` → ``DESCRIPTION``)."""
    return _repair_tracked(_Tracked.of(text), _vocab(vocabulary)).text


def _vocab(vocabulary: Iterable[str] | None) -> frozenset[str]:
    if vocabulary is None:
        return VOCABULARY
    return VOCABULARY | frozenset(word.casefold() for word in vocabulary)


def _repair_tracked(tracked: _Tracked, vocabulary: frozenset[str]) -> _Tracked:
    def fix(match: re.Match[str]) -> str:
        original = match.group(0)
        tokens = original.split(" ")
        if sum(1 for token in tokens if len(token) == 1) < 2:
            return original
        words = _segment(tokens, vocabulary)
        if words is None or len(words) >= len(tokens):
            return original
        return " ".join(words)

    return tracked.sub(_SPACED_RUN, fix)


# ------------------------------------------------------------- the pipeline
_CRLF = re.compile(r"\r\n?")
_BREAK_HYPHEN = re.compile(r"[￾\x02­]\n?")
_LINE_END_HYPHEN = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_ODD_SPACE = re.compile(r"[\t\f\v   -   　]")
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f​﻿￾￿]")
_MULTI_SPACE = re.compile(r" {2,}")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_EDGES = re.compile(r"^\s+|\s+$")


def _clean_tracked(
    tracked: _Tracked,
    running: set[str],
    zone: int,
    vocabulary: frozenset[str],
) -> _Tracked:
    tracked = tracked.sub(_CRLF, "\n")
    if running:
        drop = {(a, b) for a, b, key in _edge_lines(tracked.text, zone) if key in running}
        tracked = tracked.keep(span for span in _line_spans(tracked.text) if span not in drop)
    tracked = tracked.sub(_BREAK_HYPHEN, "")
    tracked = tracked.sub(_LINE_END_HYPHEN, "")
    tracked = tracked.sub(_ODD_SPACE, " ")
    tracked = tracked.sub(_CONTROL, "")
    tracked = _repair_tracked(tracked, vocabulary)
    tracked = tracked.sub(_MULTI_SPACE, " ")
    tracked = tracked.sub(_SPACE_AROUND_NEWLINE, "\n")
    tracked = tracked.sub(_MANY_NEWLINES, "\n\n")
    return tracked.sub(_EDGES, "")


def clean_page_text(
    raw: str,
    running: set[str] | None = None,
    *,
    zone: int = DEFAULT_ZONE,
    vocabulary: Iterable[str] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Clean one page's raw text; returns the text and its offset map."""
    tracked = _clean_tracked(_Tracked.of(raw), running or set(), zone, _vocab(vocabulary))
    return tracked.text, tracked.offset_map()


# ---------------------------------------------------------- joined documents
@dataclass(frozen=True)
class PagedText:
    """Cleaned pages joined with :data:`PAGE_SEPARATOR`, with offset → page lookups.

    Chunk character spans (``Chunk.char_start``/``char_end``) index this text.
    Pages with no text are left out of the join but keep their numbers.
    """

    text: str
    pages: tuple[int, ...]
    starts: tuple[int, ...]
    lengths: tuple[int, ...]

    @classmethod
    def join(cls, pages: Iterable[Page]) -> PagedText:
        numbers: list[int] = []
        starts: list[int] = []
        lengths: list[int] = []
        parts: list[str] = []
        offset = 0
        for page in pages:
            if not page.text:
                continue
            if parts:
                parts.append(PAGE_SEPARATOR)
                offset += len(PAGE_SEPARATOR)
            numbers.append(page.page)
            starts.append(offset)
            lengths.append(len(page.text))
            parts.append(page.text)
            offset += len(page.text)
        return cls("".join(parts), tuple(numbers), tuple(starts), tuple(lengths))

    def page_at(self, offset: int) -> int:
        """The page holding character ``offset``; a separator belongs to the page before it."""
        if not 0 <= offset < len(self.text):
            raise IndexError(f"offset {offset} is outside the text (length {len(self.text)})")
        return self.pages[bisect.bisect_right(self.starts, offset) - 1]

    def start_of(self, page: int) -> int:
        """Where ``page``'s text starts in :attr:`text`."""
        return self.starts[self.pages.index(page)]

    def locate(self, offset: int) -> tuple[int, int]:
        """``(page, offset within that page's cleaned text)``."""
        page = self.page_at(offset)
        return page, offset - self.start_of(page)

    def page_span(self, start: int, end: int) -> tuple[int, int]:
        """First and last page touched by the half-open span ``[start, end)``."""
        return self.page_at(start), self.page_at(max(start, end - 1))


@dataclass(frozen=True)
class CleanResult:
    """Cleaned pages of one file, and the header/footer keys that were removed."""

    pages: list[Page]
    removed: frozenset[str] = field(default_factory=frozenset)

    def document(self, page_start: int | None = None, page_end: int | None = None) -> PagedText:
        """The joined text of pages ``page_start``–``page_end`` (inclusive; default all)."""
        low = page_start if page_start is not None else 1
        high = page_end if page_end is not None else len(self.pages)
        return PagedText.join(page for page in self.pages if low <= page.page <= high)


def clean_pages(
    pages: Sequence[ParsedPage],
    *,
    file_sha: str,
    parser: str,
    repeat_fraction: float = DEFAULT_REPEAT_FRACTION,
    zone: int = DEFAULT_ZONE,
    vocabulary: Iterable[str] | None = None,
) -> CleanResult:
    """Clean every page of one file, timed as ``stage("clean")`` (rule K7).

    Header and footer frequency is counted across the whole file, before it is
    segmented, as plan §5.1 orders the steps.
    """
    with stage(Stage.CLEAN):
        words = _vocab(vocabulary)
        running = running_lines([page.text for page in pages], repeat_fraction, zone)
        cleaned: list[Page] = []
        for page in pages:
            tracked = _clean_tracked(_Tracked.of(page.text), running, zone, words)
            cleaned.append(
                Page(
                    file_sha=file_sha,
                    page=page.page,
                    parser=parser,
                    text=tracked.text,
                    has_text_layer=page.has_text_layer,
                    offset_map=tracked.offset_map(),
                )
            )
        return CleanResult(pages=cleaned, removed=frozenset(running))
