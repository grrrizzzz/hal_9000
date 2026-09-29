"""Chunkers and index profiles (plan §6; handoff KWP2).

``chunk_document(input, profile)`` turns one document's cleaned text into
``ChunkDraft`` objects under the ``chunk`` stage timer; ``chunk()`` is the same
with explicit parameters. Profile names round-trip through
``profile_name``/``parse_profile_name``.
"""

from __future__ import annotations

from hal9000.knowledge.chunking.chunkers import CHUNKERS, MIN_SIZE, chunk, chunk_document
from hal9000.knowledge.chunking.header import build_header
from hal9000.knowledge.chunking.inputs import PAGE_JOINER, ChunkDraft, ChunkInput, Unit
from hal9000.knowledge.chunking.profiles import (
    EMBED_ALIASES,
    make_profile,
    overlap_tokens,
    parse_profile_name,
    profile_name,
)
from hal9000.knowledge.chunking.units import detect_units

__all__ = [
    "CHUNKERS",
    "EMBED_ALIASES",
    "MIN_SIZE",
    "PAGE_JOINER",
    "ChunkDraft",
    "ChunkInput",
    "Unit",
    "build_header",
    "chunk",
    "chunk_document",
    "detect_units",
    "make_profile",
    "overlap_tokens",
    "parse_profile_name",
    "profile_name",
]
