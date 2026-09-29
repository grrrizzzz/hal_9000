"""Verify an answer against the sources it was given (plan §7.1 ``verify``).

Two checks, both deterministic and offline:

- **Citations.** Every ``[S#]`` marker must name a source that was supplied
  (``S1``..``Sn``). ``[S9]`` with five sources is a ``citation`` issue.
- **Figures.** After the markers are stripped (so the ``1`` in ``[S1]`` is never
  a figure), every figure in the answer must appear in a source the answer
  cites. Figures are found with :func:`hal9000.solar.domain.numerals.extract_numerals`
  (digits and number words; ``$1,234`` and ``$1,234.00`` are the same value).
  Dates are normalized on both sides first: ``June 22, 2023``, ``22 June
  2023``, ``2023-06-22``, ``6/22/23`` and ``Jun. 2023`` become (year, month,
  day) claims, and a claim must match a date in a cited source (a claim
  without a day matches any day of that month). A bare year is a figure like
  any other.

The text searched for a source (:func:`haystack`) is its chunk text and header
plus its metadata: title, vendor and ``doc_date`` written as the ISO date,
"June 2023", "6/22/23" and "2023". So "The June 2023 inspection [S1]" passes
when the page itself never says 2023 but the document is dated 2023-06-22.

The result lists each problem once, as text the UI shows and the regeneration
prompt repeats.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal

from hal9000.solar.domain.numerals import UNIT_MONEY, extract_numerals

__all__ = [
    "Issue",
    "IssueKind",
    "SourceFacts",
    "Verification",
    "cited_markers",
    "haystack",
    "strip_markers",
    "verify_answer",
]

IssueKind = Literal["numeral", "citation"]

MONTH_NAMES = (
    "January February March April May June July August September October November December"
).split()

#: ``[S1]``, ``[S1][S2]`` (two groups), ``[S1, S3]`` and ranges ``[S1-S3]`` / ``[S1–3]``.
_ITEM = r"S\d+(?:\s*[-–—]\s*S?\d+)?"
_GROUP = re.compile(rf"\[\s*({_ITEM}(?:\s*[,;]\s*{_ITEM})*)\s*\]")
_MARKER = re.compile(r"S(\d+)(?:\s*[-–—]\s*S?(\d+))?")
#: a range wider than this is read as its two ends only
_MAX_RANGE = 50
#: "1st", "2nd", "23rd": ordinals are positions, not figures
_ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I)


@dataclass(frozen=True)
class SourceFacts:
    """What verify needs of one numbered source: its text and its document's metadata."""

    marker: str
    text: str
    header: str = ""
    title: str = ""
    vendor: str | None = None
    doc_date: date | None = None
    #: the pages the source's text comes from ("page 41" is backed by them)
    pages: tuple[int, ...] = ()


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    message: str


@dataclass(frozen=True)
class Verification:
    issues: tuple[Issue, ...] = ()
    #: the valid markers the answer cites, in order of first use
    cited: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def grounding(self) -> Literal["verified", "failed", "skipped"]:
        """``skipped`` for an answer that cites nothing and states no figure."""
        if not self.ok:
            return "failed"
        return "verified" if self.cited else "skipped"

    @property
    def kinds(self) -> set[str]:
        return {issue.kind for issue in self.issues}

    @property
    def messages(self) -> list[str]:
        return [issue.message for issue in self.issues]


# ------------------------------------------------------------------ markers
def cited_markers(text: str) -> list[str]:
    """Every ``S#`` the text cites, in order of first use, without duplicates."""
    seen: dict[str, None] = {}
    for group in _GROUP.finditer(text):
        for item in _MARKER.finditer(group.group(1)):
            low = int(item.group(1))
            high = int(item.group(2)) if item.group(2) else low
            if high < low or high - low > _MAX_RANGE:
                numbers = [low, high]
            else:
                numbers = list(range(low, high + 1))
            for number in numbers:
                seen.setdefault(f"S{number}", None)
    return list(seen)


def strip_markers(text: str) -> str:
    """The text with every ``[S#]`` group replaced by a space."""
    return _GROUP.sub(" ", text)


# -------------------------------------------------------------------- dates
_MON = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_ORD = r"(?:st|nd|rd|th)?"
_MONTH_INDEX = {name.lower()[:3]: number for number, name in enumerate(MONTH_NAMES, start=1)}

DateClaim = tuple[int | None, int | None, int | None]  # (year, month, day)

#: (pattern, order of the groups), longest forms first; each match is masked out.
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.I), order)
    for pattern, order in (
        (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", "ymd"),
        (r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", "mdy"),
        (rf"\b{_MON}\s+(\d{{1,2}}){_ORD},?\s+(\d{{4}})\b", "Mdy"),
        (rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON},?\s+(\d{{4}})\b", "dMy"),
        (rf"\b{_MON},?\s+(\d{{4}})\b", "My"),
        (rf"\b{_MON}\s+(\d{{1,2}}){_ORD}\b(?!\s*[,:/-]?\s*\d)", "Md"),
        (rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON}\b", "dM"),
    )
)
#: "90 days", "7-day": spaced out so ``extract_numerals`` doesn't discard them as windows.
_DURATION = re.compile(r"(\d)\s*[- ]\s*(days?|weeks?)\b", re.I)


def _claim(groups: Sequence[str], order: str) -> DateClaim | None:
    year = month = day = None
    for value, key in zip(groups, order):
        if key == "y":
            year = int(value)
            if len(value) == 2:
                year += 2000 if year < 70 else 1900
        elif key == "m":
            month = int(value)
        elif key == "M":
            month = _MONTH_INDEX.get(value.lower()[:3])
        elif key == "d":
            day = int(value)
    if month is None or not 1 <= month <= 12 or (day is not None and not 1 <= day <= 31):
        return None
    return (year, month, day)


def _dates(text: str) -> tuple[str, list[tuple[str, DateClaim]]]:
    """Mask every date in ``text``; returns the masked text and ``(written, claim)`` pairs."""
    found: list[tuple[str, DateClaim]] = []
    for pattern, order in _DATE_PATTERNS:

        def mask(match: re.Match[str], order: str = order) -> str:
            claim = _claim(match.groups(), order)
            if claim is None:
                return match.group(0)
            found.append((match.group(0).strip(), claim))
            return " " * len(match.group(0))

        text = pattern.sub(mask, text)
    return text, found


def _matches(claim: DateClaim, known: DateClaim) -> bool:
    return all(want is None or want == have for want, have in zip(claim, known))


# ------------------------------------------------------------------ figures
@dataclass
class _Facts:
    """The normalized contents of a text: figure values, money values and dates."""

    values: set[Decimal] = field(default_factory=set)
    money: set[Decimal] = field(default_factory=set)
    dates: list[DateClaim] = field(default_factory=list)


@dataclass(frozen=True)
class _Figure:
    text: str
    value: Decimal
    money: bool
    spelled: bool


def _figures(text: str) -> tuple[list[_Figure], list[tuple[str, DateClaim]]]:
    masked, dates = _dates(text)
    masked = _DURATION.sub(r"\1  \2", masked)
    figures = [
        _Figure(
            numeral.text,
            numeral.value,
            numeral.unit == UNIT_MONEY,
            not any(ch.isdigit() for ch in numeral.text),
        )
        for numeral in extract_numerals(masked)
    ]
    return figures, dates


def _facts(text: str) -> _Facts:
    figures, dates = _figures(text)
    facts = _Facts(
        values={f.value for f in figures},
        money={f.value for f in figures if f.money},
        dates=[claim for _, claim in dates],
    )
    for year, _month, day in facts.dates:  # "in 2023" is backed by "June 22, 2023"
        for part in (year, day):
            if part is not None:
                facts.values.add(Decimal(part))  # never money: "$2,023" is not a year
    return facts


def _exempt(figure: _Figure) -> bool:
    """A spelled-out count up to ten with no unit ("two issues") isn't a quoted figure."""
    return figure.spelled and not figure.money and Decimal(1) <= figure.value <= Decimal(10)


def _date_strings(when: date) -> list[str]:
    month = MONTH_NAMES[when.month - 1]
    return [
        when.isoformat(),
        f"{month} {when.year}",
        f"{when.month}/{when.day}/{when.year % 100:02d}",
        str(when.year),
    ]


def haystack(source: SourceFacts) -> str:
    """The text a source's figures are looked up in: its text, header and metadata."""
    parts = [source.text, source.header, source.title, source.vendor or ""]
    if source.doc_date is not None:
        parts.extend(_date_strings(source.doc_date))
    parts.extend(f"page {page}" for page in source.pages)
    return "\n".join(part for part in parts if part)


def _cited_list(markers: Iterable[str]) -> str:
    return " or ".join(f"[{marker}]" for marker in markers)


# ---------------------------------------------------------------------- verify
def verify_answer(answer: str, sources: Sequence[SourceFacts]) -> Verification:
    """Check ``answer``'s citations and figures against the numbered ``sources``."""
    by_marker = {source.marker: source for source in sources}
    issues: dict[str, Issue] = {}

    def add(kind: IssueKind, message: str) -> None:
        issues.setdefault(message, Issue(kind, message))

    count = len(sources)
    noun = "source" if count == 1 else "sources"
    valid: list[str] = []
    for marker in cited_markers(answer):
        if marker in by_marker:
            valid.append(marker)
        else:
            add("citation", f"[{marker}] is not one of the {count} {noun} supplied")

    facts = _Facts()
    for marker in valid:
        found = _facts(haystack(by_marker[marker]))
        facts.values |= found.values
        facts.money |= found.money
        facts.dates.extend(found.dates)

    stated = _ORDINAL.sub(lambda m: " " * len(m.group(0)), strip_markers(answer))
    figures, dates = _figures(stated)
    where = _cited_list(valid)
    for figure in figures:
        if _exempt(figure):
            continue
        backing = facts.money if figure.money else facts.values
        if figure.value not in backing:
            add("numeral", _unbacked(figure.text, where))
    for written, claim in dates:
        if not any(_matches(claim, known) for known in facts.dates):
            add("numeral", _unbacked(written, where))
    return Verification(issues=tuple(issues.values()), cited=tuple(valid))


def _unbacked(written: str, where: str) -> str:
    if not where:
        return f"{written} is not backed by a cited source"
    return f"{written} does not appear in {where}"
