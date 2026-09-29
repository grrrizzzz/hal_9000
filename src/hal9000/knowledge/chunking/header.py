"""The contextual header (plan §6.1).

``"{title} ({doc_type}, {vendor}, {doc_date}) › {section path}"``, built from
metadata alone (no LLM call). It is prefixed to what gets embedded and
keyword-indexed, never to the text a citation displays. Missing parts are left
out rather than printed as blanks.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

__all__ = ["SEPARATOR", "build_header"]

SEPARATOR = " › "


def _one_line(value: object) -> str:
    return " ".join(str(value).split())


def build_header(
    title: str,
    doc_type: str,
    vendor: str | None,
    doc_date: date | None,
    section_path: Sequence[str] = (),
) -> str:
    """The header line for a chunk of a document with this metadata and section path."""
    meta = [
        _one_line(doc_type.replace("_", " ")) if doc_type else "",
        _one_line(vendor) if vendor else "",
        doc_date.isoformat() if doc_date else "",
    ]
    meta = [m for m in meta if m]
    head = _one_line(title)
    if meta:
        head = f"{head} ({', '.join(meta)})".strip()
    parts = [head] if head else []
    parts += [p for p in (_one_line(s) for s in section_path) if p]
    return SEPARATOR.join(parts)
