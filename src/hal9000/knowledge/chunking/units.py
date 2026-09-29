"""Structural units: a deterministic detector and the unit tree (plan §6.1 ``structure``).

The detector works on cleaned text without KWP1's segmentation. It
recognises, one line at a time:

* numbered headings and findings: ``3.1.2 Plumbing – Water Heater: Valve Leak``
  (level = number of components). A dotted number needs a title starting with
  a capital; a single number (``3 PLUMBING``, ``4. Heating and Cooling``) needs
  an upper-case or title-case title without end punctuation, so numbered list
  items (``1. Replace the valve.``) are not headings;
* Markdown headings: ``## Water Heater`` (level = number of ``#``);
* upper-case lines: ``PLUMBING SYSTEM``. After a blank line (or at the top)
  they are level-1 headings; directly under a numbered or Markdown unit they
  are labels inside it (``RECOMMENDATION``), one level deeper.

:func:`build_tree` nests units by containment. A unit without an explicit end
runs to the next unit of the same or a higher level, like a heading.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from hal9000.knowledge.chunking.inputs import Unit

__all__ = ["Node", "build_tree", "detect_units"]

_NUMBERED = re.compile(r"(?P<num>\d{1,3}(?:\.\d{1,3})*)\.?[ \t]+(?P<title>\S.*)")
_MARKDOWN = re.compile(r"(?P<hashes>#{1,6})[ \t]+(?P<title>\S.*)")
_SMALL_WORDS = frozenset("a an and as at by for in of on or the to vs with".split())
_END_PUNCTUATION = tuple(".,;:!?")
_MAX_HEADING_CHARS = 160


def _is_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _is_title_case(text: str) -> bool:
    words = text.split()
    if not words or not words[0][0].isupper():
        return False
    return all(w[0].isupper() or not w[0].isalpha() or w.lower() in _SMALL_WORDS for w in words)


def _numbered(line: str) -> tuple[int, str] | None:
    m = _NUMBERED.fullmatch(line)
    if m is None:
        return None
    depth = m["num"].count(".") + 1
    title = m["title"].rstrip()
    if depth > 1:
        return (depth, "finding") if title[0].isupper() else None
    short = len(title) <= 60 and not title.endswith(_END_PUNCTUATION)
    if _is_caps(title) or (short and _is_title_case(title)):
        return 1, "heading"
    return None


def detect_units(text: str) -> list[Unit]:
    """Headings and numbered findings in ``text``, in order (see the module docstring)."""
    units: list[Unit] = []
    context: int | None = None  # level of the last numbered or Markdown unit
    after_blank = True
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = pos + len(line) - len(line.lstrip())
        pos += len(line)
        if not stripped:
            after_blank = True
            continue
        was_blank, after_blank = after_blank, False
        if len(stripped) > _MAX_HEADING_CHARS:
            continue
        if m := _MARKDOWN.fullmatch(stripped):
            level = len(m["hashes"])
            units.append(Unit(start=start, title=m["title"].strip(), level=level))
            context = level
        elif found := _numbered(stripped):
            level, kind = found
            if kind == "finding" and ":" not in stripped and " – " not in stripped:
                kind = "heading"
            units.append(Unit(start=start, title=stripped, level=level, kind=kind))  # type: ignore[arg-type]
            context = level
        elif _is_caps(stripped) and len(stripped) <= 80 and not stripped.endswith("."):
            if was_blank or context is None:
                units.append(Unit(start=start, title=stripped, level=1))
                context = 1
            else:
                units.append(Unit(start=start, title=stripped, level=context + 1))
    return units


@dataclass(eq=False)
class Node:
    """A unit in the tree; the root is the whole document (level 0)."""

    start: int
    end: int
    title: str
    level: int
    children: list[Node] = field(default_factory=list)
    _starts: list[int] | None = field(default=None, repr=False)

    def child_at(self, pos: int) -> Node | None:
        if self._starts is None:
            self._starts = [c.start for c in self.children]
        i = bisect_right(self._starts, pos) - 1
        if i >= 0 and self.children[i].start <= pos < self.children[i].end:
            return self.children[i]
        return None

    def path_at(self, pos: int) -> tuple[str, ...]:
        """Titles of the units containing ``pos``, outermost first."""
        path: list[str] = []
        node: Node | None = self.child_at(pos)
        while node is not None:
            path.append(node.title)
            node = node.child_at(pos)
        return tuple(path)

    def walk(self) -> Iterator[Node]:
        for child in self.children:
            yield child
            yield from child.walk()


def build_tree(units: Sequence[Unit], length: int) -> Node:
    """Nest ``units`` by containment under a root spanning ``[0, length)``."""
    root = Node(0, length, "", 0)
    ordered = sorted(units, key=lambda u: (u.start, u.level))
    ends: dict[int, int] = {}
    open_units: list[int] = []  # indexes of units still waiting for an implicit end
    for i, u in enumerate(ordered):
        while open_units and ordered[open_units[-1]].level >= u.level:
            ends[open_units.pop()] = u.start
        if u.end is None:
            open_units.append(i)
        else:
            ends[i] = min(u.end, length)
    for i in open_units:
        ends[i] = length
    spans = [(u.start, ends[i], u) for i, u in enumerate(ordered)]
    spans.sort(key=lambda t: (t[0], -t[1]))
    stack = [root]
    for start, end, u in spans:
        while len(stack) > 1 and start >= stack[-1].end:
            stack.pop()
        parent = stack[-1]
        end = min(end, parent.end)
        if end <= start:
            continue
        node = Node(start, end, u.title, u.level)
        parent.children.append(node)
        stack.append(node)
    return root
