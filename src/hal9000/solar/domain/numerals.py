"""Find every number a narrative states, and normalise it for the numeric guard.

Digits are always numbers ("8.2", "$1.37", "37%", "1,024"). Words are numbers
when they form a numeric phrase ("eight point two", "twelve", "one hundred"),
except for a small set of everyday uses ("one of", "no one", "once") that are
not quantities. Normalisation returns :class:`~decimal.Decimal` so "8.20" and
"8.2" compare equal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

__all__ = [
    "Numeral",
    "UNIT_ENERGY",
    "UNIT_MONEY",
    "UNIT_PERCENT",
    "extract_numerals",
    "mask_times_and_dates",
    "to_decimal",
]

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_SCALES = {"hundred": 100, "thousand": 1000}
_WORD_NUMBER = set(_UNITS) | set(_TENS) | set(_SCALES) | {"point", "and"}

_DIGIT_RE = re.compile(r"(?<![\w.])\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])\$?\d+(?:\.\d+)?%?")

# "one" is a number only when it isn't one of these idioms.
_ONE_IDIOMS = re.compile(
    r"\b(one of|one more|no one|anyone|everyone|someone|one another|one day"
    r"|the one|this one|that one)\b",
    re.I,
)


_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_NUMBER_WORD = "|".join(sorted(list(_UNITS) + list(_TENS), key=len, reverse=True))
_ORDINAL = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
    "thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|"
    "thirtieth"
)
_CLOCK = r"(?:a\.m\.|p\.m\.|am\b|pm\b|o'clock)"
_NUM = rf"(?:\d{{1,2}}|(?:{_NUMBER_WORD})(?:[ -](?:{_NUMBER_WORD}))?)"
_TIME_PATTERNS = [
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?", re.I),
    # ranges: "4 to 9 pm", "between 4 and 9 pm", "4-9 pm", "four to nine p.m."
    re.compile(
        rf"(?<![:\d])\b{_NUM}\s*(?:-|–|to|and|through|until)\s*{_NUM}(?::\d{{2}})?\s*{_CLOCK}",
        re.I,
    ),
    re.compile(rf"\b{_NUM}\s*{_CLOCK}", re.I),
    # dates: "September 22, 2026", "September twenty-seventh", "2026-09-22"
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?", re.I),
    re.compile(rf"\b(?:{_MONTHS})\s+(?:the\s+)?(?:(?:twenty|thirty)[ -])?(?:{_ORDINAL})\b", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # durations of a window: "7-day average", "last 30 days", "seven days"
    re.compile(rf"\b{_NUM}(?:[ -]day\b| days\b| weeks?\b)", re.I),
]

UNIT_MONEY = "money"
UNIT_PERCENT = "percent"
UNIT_ENERGY = "energy"
_UNIT_WORDS = [
    (re.compile(r"^\s*(?:%|percent\b|per cent\b)", re.I), UNIT_PERCENT),
    (re.compile(r"^\s*(?:dollars?\b|cents?\b|bucks?\b)", re.I), UNIT_MONEY),
    (re.compile(r"^\s*(?:kilowatt|kwh\b|kw\b|watt|megawatt)", re.I), UNIT_ENERGY),
]


def mask_times_and_dates(text: str) -> str:
    """Blank out clock times, time ranges, dates and window lengths.

    They are not quantities, so they neither need backing nor back anything.
    """
    for pattern in _TIME_PATTERNS:
        text = pattern.sub(lambda m: " " * len(m.group()), text)
    return text


@dataclass(frozen=True)
class Numeral:
    text: str
    value: Decimal
    unit: str | None = None  # money | percent | energy | None (bare number)


def to_decimal(text: str) -> Decimal | None:
    """Parse a digit numeral ("$1,024.50", "37%") into a Decimal."""
    cleaned = text.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _unit_after(text: str, end: int) -> str | None:
    tail = text[end : end + 30]
    for pattern, unit in _UNIT_WORDS:
        if pattern.match(tail):
            return unit
    return None


def _words_to_number(words: list[str]) -> Decimal | None:
    total, current = 0, 0
    decimal_digits: list[int] | None = None
    for word in words:
        if word == "and":
            continue
        if word == "point":
            decimal_digits = []
            continue
        if decimal_digits is not None:
            if word in _UNITS and _UNITS[word] < 10:
                decimal_digits.append(_UNITS[word])
                continue
            return None
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word in _SCALES:
            current = max(current, 1) * _SCALES[word]
            if word == "thousand":
                total += current
                current = 0
        else:
            return None
    value = Decimal(total + current)
    if decimal_digits:
        value += Decimal("0." + "".join(str(d) for d in decimal_digits))
    return value


def extract_numerals(text: str) -> list[Numeral]:
    """All quantities in ``text``, digits and words, with their unit when stated.

    Clock times, time ranges, dates and window lengths are skipped.
    """
    text = mask_times_and_dates(text)
    found: list[Numeral] = []
    for match in _DIGIT_RE.finditer(text):
        token = match.group()
        value = to_decimal(token)
        if value is None:
            continue
        if token.startswith("$"):
            unit = UNIT_MONEY
        elif token.endswith("%"):
            unit = UNIT_PERCENT
        else:
            unit = _unit_after(text, match.end())
        found.append(Numeral(token, value, unit))

    masked = _ONE_IDIOMS.sub(lambda m: " " * len(m.group()), text)
    masked = _DIGIT_RE.sub(lambda m: " " * len(m.group()), masked)
    masked = masked.replace("-", " ")
    run: list[tuple[str, int]] = []

    def flush():
        while run and run[-1][0] in {"and", "point"}:
            run.pop()
        while run and run[0][0] in {"and", "point"}:
            run.pop(0)
        if run:
            value = _words_to_number([w for w, _ in run])
            if value is not None:
                found.append(
                    Numeral(" ".join(w for w, _ in run), value, _unit_after(masked, run[-1][1]))
                )
        run.clear()

    for match in re.finditer(r"[A-Za-z]+|[^\sA-Za-z]", masked):
        token = match.group().lower()
        if token in _WORD_NUMBER:
            run.append((token, match.end()))
        else:
            flush()
    flush()
    return found
