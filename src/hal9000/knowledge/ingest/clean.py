"""Cleaning: raw parser text → clean page text with an offset map (plan §5.1 step 5).

Four repairs, in order, on each page:

1. **Running headers and footers** are removed: a line whose normalised form
   (case-folded, spaces collapsed, UUID and hex IDs replaced by ``<id>``, every
   digit run by ``#``) sits at the *same position* -- the i-th non-blank line
   from the top, or the j-th from the bottom, within the first/last
   :data:`DEFAULT_ZONE` lines -- on at least 30% of the file's pages with text
   (``ingest.header_repeat_fraction``) and on at least three. That catches
   DocuSign envelope IDs, "Page 12 of 58" and an inspector's name in the
   header, while a finding line repeated on two pages, or estimate line items
   that differ only in their numbers, stay.
2. **Hyphenation**: PDFium's line-break hyphen marker (U+FFFE), soft hyphens,
   and ``word-⏎word`` at a line end are joined.
3. **Letter spacing**: headings such as ``D ES C R I PT I O N O F`` are rejoined
   (``DESCRIPTION OF``). A run of short capital groups is segmented against a
   ~60k-word English list with frequency classes (SCOWL, vendored in
   ``wordlists/``) plus house-document words; stretches of single letters that
   no word covers (a vendor's name) are joined whole. A run is only rewritten
   when the result compresses it into clearly fewer words, so ordinary text
   such as ``A B testing`` or ``SEE A OR B`` is untouched.
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
import gzip
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from hal9000.knowledge.contracts import Page
from hal9000.knowledge.ingest.parsers.base import ParsedPage
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = [
    "DEFAULT_REPEAT_FRACTION",
    "DEFAULT_ZONE",
    "DOMAIN_WORDS",
    "PAGE_SEPARATOR",
    "CleanResult",
    "PagedText",
    "clean_page_text",
    "clean_pages",
    "normalize_line",
    "repair_letter_spacing",
    "running_lines",
    "source_offset",
    "word_costs",
]

#: Lines on at least this share of pages are running headers or footers (plan §5.1).
DEFAULT_REPEAT_FRACTION = 0.30
#: How many non-blank lines at each end of a page count as header or footer.
DEFAULT_ZONE = 5
#: Placed between pages when they are joined into one document text.
PAGE_SEPARATOR = "\n\n"

#: House-document words added to the SCOWL list for letter-spacing repair
#: (headings on estimates, invoices, inspection and pest reports, manuals).
DOMAIN_WORDS = frozenset(
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

#: SCOWL words, size levels 10-50, as "word level" lines (see SCOWL-COPYRIGHT.txt).
WORDLIST_PATH = Path(__file__).resolve().parent / "wordlists" / "scowl-words-50.txt.gz"
_LEVEL_COST = {10: 1.0, 20: 1.2, 35: 1.6, 40: 1.8, 50: 2.2}
_DOMAIN_COST = 1.2
_SINGLE_LETTER_COSTS = {"a": 2.0, "i": 2.0}
_MAX_WORD = 24
# A stretch of >= 4 single-letter tokens that no word covers is joined whole.
_UNKNOWN_MIN_LETTERS = 4
_UNKNOWN_BASE = 2.5
_UNKNOWN_PER_LETTER = 0.25
# Rewrite only when there are fewer than 2 words per 3 tokens: a real letter-spaced
# heading compresses far more, ordinary short capital words ("SEE A OR B") don't.
_WORDS, _PER_TOKENS = 2, 3

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
# UUIDs and dashed or long hex IDs (DocuSign envelopes); they must contain a digit and a-f
_HEX_ID = re.compile(
    r"(?<![0-9a-z])(?=[0-9a-f-]*[0-9])(?=[0-9a-f-]*[a-f])"
    r"(?:[0-9a-f]{4,}(?:-[0-9a-f]{2,}){2,}|[0-9a-f]{12,})(?![0-9a-z])"
)
# folded lines that are page numbers: "page # of #", "pg. #", "#/#", "- # -", "#"
_PAGE_NUMBER = re.compile(r"\bp(?:age|g)?\.? ?#|# ?(?:of|/) ?#|^[^a-z#]*#[^a-z#]*$")
# 3+ groups of 1-3 capitals separated by single spaces, on one line
_SPACED_RUN = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{1,3} ){2,}[A-Z]{1,3}(?![A-Za-z0-9])")


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
    """The comparison key for header/footer detection: case, spacing, IDs and digits folded."""
    folded = _SPACES.sub(" ", line).strip().casefold()
    return _DIGITS.sub("#", _HEX_ID.sub("<id>", folded))


def _line_spans(text: str) -> list[tuple[int, int]]:
    """``(start, stop)`` of each line, including its newline."""
    spans, start = [], 0
    for line in text.splitlines(keepends=True):
        spans.append((start, start + len(line)))
        start += len(line)
    return spans


#: A running line: which edge, its position from that edge, and its normalised text.
RunningLine = tuple[str, int, str]


def _edge_lines(text: str, zone: int) -> list[tuple[int, int, RunningLine]]:
    """``(start, stop, (edge, position, key))`` for the first and last ``zone`` non-blank lines.

    A line near both edges of a short page is listed once per edge.
    """
    lines = [(a, b, normalize_line(text[a:b])) for a, b in _line_spans(text)]
    lines = [line for line in lines if line[2]]
    picked = [(a, b, ("top", index, key)) for index, (a, b, key) in enumerate(lines[:zone])]
    bottom = list(reversed(lines))[:zone]
    picked += [(a, b, ("bottom", index, key)) for index, (a, b, key) in enumerate(bottom)]
    return picked


def _exact(line: str) -> str:
    """A line with case, spacing and IDs folded but digits kept."""
    return _HEX_ID.sub("<id>", _SPACES.sub(" ", line).strip().casefold())


def _counts_with_pages(entries: list[tuple[int, str]], minimum: int) -> bool:
    """Whether one number in the line moves in step with the page number on ≥ ``minimum`` pages."""
    numbers = [(page, [int(n) for n in _DIGITS.findall(text)]) for page, text in entries]
    width = min(len(values) for _, values in numbers)
    for index in range(width):
        offsets = Counter(values[index] - page for page, values in numbers)
        if offsets.most_common(1)[0][1] >= minimum:
            return True
    return False


def running_lines(
    texts: Sequence[str], fraction: float, zone: int = DEFAULT_ZONE
) -> set[RunningLine]:
    """Lines at the same edge position on ≥ ``fraction`` of the pages with text, and ≥ 3.

    ``texts`` are the file's pages in order (page *n* is ``texts[n - 1]``). A line
    that is identical on those pages is running. One whose numbers vary (the
    digit folding made them match) is running only if it is a page number
    ("Page # of #", "#", "#/#") or one of its numbers counts up with the page;
    estimate items and totals that happen to share a position are not.
    """
    pages = [
        (number, text.replace("\r\n", "\n").replace("\r", "\n"))
        for number, text in enumerate(texts, start=1)
        if text.strip()
    ]
    minimum = max(3, math.ceil(fraction * len(pages) - 1e-9))
    if len(pages) < minimum:
        return set()
    seen: dict[RunningLine, list[tuple[int, str]]] = {}
    for number, text in pages:
        for start, stop, slot in _edge_lines(text, zone):
            seen.setdefault(slot, []).append((number, _exact(text[start:stop])))
    running: set[RunningLine] = set()
    for slot, entries in seen.items():
        if len(entries) < minimum:
            continue
        if (
            len({text for _, text in entries}) == 1
            or _PAGE_NUMBER.search(slot[2])
            or _counts_with_pages(entries, minimum)
        ):
            running.add(slot)
    return running


@lru_cache(maxsize=1)
def _base_costs() -> dict[str, float]:
    costs: dict[str, float] = dict(_SINGLE_LETTER_COSTS)
    with gzip.open(WORDLIST_PATH, "rt", encoding="ascii") as handle:
        for line in handle:
            word, level = line.split()
            cost = _LEVEL_COST[int(level)]
            if len(word) == 1 or (len(word) == 2 and cost > _LEVEL_COST[35]):
                continue  # stray letters and rare two-letter words only mislead the split
            costs[word] = cost
    for word in DOMAIN_WORDS:
        costs[word] = min(costs.get(word, _DOMAIN_COST), _DOMAIN_COST)
    return costs


def word_costs(vocabulary: Iterable[str] | None = None) -> Mapping[str, float]:
    """Word → cost (lower is more common) for letter-spacing repair, plus extra ``vocabulary``."""
    base = _base_costs()
    if vocabulary is None:
        return base
    extra = {word.casefold(): _DOMAIN_COST for word in vocabulary}
    return {**base, **{w: min(base.get(w, c), c) for w, c in extra.items()}}


def _segment(tokens: list[str], costs: Mapping[str, float]) -> list[str] | None:
    """Least-cost split of the joined tokens into words (DP), or ``None``.

    Pieces are known words, or a stretch of ≥ 4 consecutive single-letter tokens
    joined whole (a name the list doesn't know). Multi-letter fragments such as
    ``ES`` or ``PT`` must be absorbed into words, or there is no split.
    """
    joined = "".join(tokens)
    n = len(joined)
    unknown: dict[int, list[int]] = {}
    position, index = 0, 0
    while index < len(tokens):
        if len(tokens[index]) != 1:
            position += len(tokens[index])
            index += 1
            continue
        starts = []
        while index < len(tokens) and len(tokens[index]) == 1:
            starts.append(position)
            position += 1
            index += 1
        ends = [*starts[1:], position]
        for i, start in enumerate(starts):
            for end in ends[i + _UNKNOWN_MIN_LETTERS - 1 :]:
                unknown.setdefault(start, []).append(end)
    best: list[tuple[float, list[str]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])
    for i in range(n):
        if best[i] is None:
            continue
        cost, words = best[i]
        options = [
            (j, costs[joined[i:j].casefold()])
            for j in range(i + 1, min(n, i + _MAX_WORD) + 1)
            if joined[i:j].casefold() in costs
        ]
        options += [(j, _UNKNOWN_BASE + _UNKNOWN_PER_LETTER * (j - i)) for j in unknown.get(i, [])]
        for j, weight in options:
            if best[j] is None or cost + weight < best[j][0]:
                best[j] = (cost + weight, [*words, joined[i:j]])
    return best[n][1] if best[n] is not None else None


def _is_alphabet_run(tokens: list[str]) -> bool:
    """``A B C D`` or ``W X Y Z``: an option list, not a spaced-out word."""
    return all(len(t) == 1 for t in tokens) and all(
        ord(b) - ord(a) == 1 for a, b in zip(tokens, tokens[1:])
    )


def repair_letter_spacing(text: str, vocabulary: Iterable[str] | None = None) -> str:
    """Rejoin letter-spaced capitals (``D ES C R I PT I O N`` → ``DESCRIPTION``)."""
    return _repair_tracked(_Tracked.of(text), word_costs(vocabulary)).text


def _repair_tracked(tracked: _Tracked, costs: Mapping[str, float]) -> _Tracked:
    def fix(match: re.Match[str]) -> str:
        original = match.group(0)
        tokens = original.split(" ")
        if _is_alphabet_run(tokens):
            return original
        words = _segment(tokens, costs)
        if words is None or _PER_TOKENS * len(words) >= _WORDS * len(tokens):
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
    running: set[RunningLine],
    zone: int,
    costs: Mapping[str, float],
) -> _Tracked:
    tracked = tracked.sub(_CRLF, "\n")
    if running:
        drop = {(a, b) for a, b, slot in _edge_lines(tracked.text, zone) if slot in running}
        tracked = tracked.keep(span for span in _line_spans(tracked.text) if span not in drop)
    tracked = tracked.sub(_BREAK_HYPHEN, "")
    tracked = tracked.sub(_LINE_END_HYPHEN, "")
    tracked = tracked.sub(_ODD_SPACE, " ")
    tracked = tracked.sub(_CONTROL, "")
    tracked = _repair_tracked(tracked, costs)
    tracked = tracked.sub(_MULTI_SPACE, " ")
    tracked = tracked.sub(_SPACE_AROUND_NEWLINE, "\n")
    tracked = tracked.sub(_MANY_NEWLINES, "\n\n")
    return tracked.sub(_EDGES, "")


def clean_page_text(
    raw: str,
    running: set[RunningLine] | None = None,
    *,
    zone: int = DEFAULT_ZONE,
    vocabulary: Iterable[str] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Clean one page's raw text; returns the text and its offset map."""
    tracked = _clean_tracked(_Tracked.of(raw), running or set(), zone, word_costs(vocabulary))
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
    """Cleaned pages of one file, and the normalised header/footer lines that were removed."""

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
    repeat_fraction: float | None = None,
    zone: int = DEFAULT_ZONE,
    vocabulary: Iterable[str] | None = None,
    pseudo_pages: bool = False,
) -> CleanResult:
    """Clean every page of one file, timed as ``stage("clean")`` (rule K7).

    Header and footer frequency is counted across the whole file, before it is
    segmented, as plan §5.1 orders the steps. Pseudo-pages (DOCX, HTML,
    Markdown, TXT; ``ParsedFile.pseudo_pages``) have no running headers, so
    nothing is removed from them for repeating. ``repeat_fraction`` defaults to
    ``ingest.header_repeat_fraction`` in ``config/knowledge.yaml``.
    """
    with stage(Stage.CLEAN):
        if repeat_fraction is None and not pseudo_pages:
            from hal9000.knowledge.config import load_knowledge_file

            repeat_fraction = load_knowledge_file().ingest.header_repeat_fraction
        costs = word_costs(vocabulary)
        running = (
            set()
            if pseudo_pages
            else running_lines([page.text for page in pages], float(repeat_fraction), zone)
        )
        cleaned: list[Page] = []
        for page in pages:
            tracked = _clean_tracked(_Tracked.of(page.text), running, zone, costs)
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
        return CleanResult(pages=cleaned, removed=frozenset(key for _, _, key in running))
