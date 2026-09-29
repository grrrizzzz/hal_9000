"""Index profile names (plan §6.2).

A profile is named
``{parser}.{chunker}-{size}[-p{parent}][-o{overlap}].{hdr|nohdr}.{embed}-{dims}``,
for example ``pdfium.structure-768.hdr.te3s-1536`` or ``pdfium.fixed-512-o15.nohdr.te3l-1024``.

* The overlap is written as a whole percentage of the size (``o15`` = 15% of
  512 = 77 tokens, rounded half up) when that reproduces the token count
  exactly, otherwise as tokens (``o151t``). No overlap is written by omission.
* ``p1500`` is the parent size, present for ``parent_child`` only.
* The embedding model is a short alias (:data:`EMBED_ALIASES`).

Only canonical names parse, so every profile has exactly one name and
``parse_profile_name(profile_name(p)) == p`` for any profile built with
:func:`make_profile` (``id`` and ``active`` are storage state, not naming).
"""

from __future__ import annotations

import re

from hal9000.knowledge.contracts import IndexProfile

__all__ = [
    "EMBED_ALIASES",
    "make_profile",
    "overlap_tokens",
    "parse_profile_name",
    "profile_name",
]

#: Short names for the embedding models a profile may use (LiteLLM model strings).
EMBED_ALIASES: dict[str, str] = {
    "te3s": "openai/text-embedding-3-small",
    "te3l": "openai/text-embedding-3-large",
}

_PARSER = re.compile(r"[a-z0-9_]+")
_NAME = re.compile(
    r"(?P<parser>[a-z0-9_]+)\."
    r"(?P<chunker>fixed|recursive|structure|parent_child)-(?P<size>\d+)"
    r"(?:-p(?P<parent>\d+))?"
    r"(?:-o(?P<overlap>\d+)(?P<in_tokens>t?))?\."
    r"(?P<header>hdr|nohdr)\."
    r"(?P<embed>[a-z0-9]+)-(?P<dims>\d+)"
)


def overlap_tokens(size: int, percent: int) -> int:
    """``percent``% of ``size`` tokens, rounded half up."""
    return (size * percent + 50) // 100


def _overlap_label(size: int, overlap: int) -> str:
    if overlap == 0:
        return ""
    percent = (overlap * 200 + size) // (2 * size)
    if 0 < percent < 100 and overlap_tokens(size, percent) == overlap:
        return f"-o{percent}"
    return f"-o{overlap}t"


def profile_name(profile: IndexProfile) -> str:
    """The canonical name of ``profile``'s fields (its ``name`` field is ignored)."""
    if not _PARSER.fullmatch(profile.parser):
        raise ValueError(f"parser {profile.parser!r} can't be named: use [a-z0-9_]+")
    alias = next((a for a, m in EMBED_ALIASES.items() if m == profile.embed_model), None)
    if alias is None:
        known = ", ".join(f"{a}={m}" for a, m in EMBED_ALIASES.items())
        raise ValueError(f"no alias for embedding model {profile.embed_model!r} (known: {known})")
    parent = f"-p{profile.parent_size}" if profile.parent_size is not None else ""
    overlap = _overlap_label(profile.size, profile.overlap)
    header = "hdr" if profile.header else "nohdr"
    return (
        f"{profile.parser}.{profile.chunker}-{profile.size}{parent}{overlap}"
        f".{header}.{alias}-{profile.dimensions}"
    )


def parse_profile_name(name: str) -> IndexProfile:
    """The profile a canonical name describes; raises ``ValueError`` for anything else."""
    m = _NAME.fullmatch(name)
    if m is None:
        raise ValueError(
            f"{name!r} is not a profile name "
            "({parser}.{chunker}-{size}[-p{parent}][-o{overlap}].{hdr|nohdr}.{embed}-{dims})"
        )
    model = EMBED_ALIASES.get(m["embed"])
    if model is None:
        raise ValueError(f"unknown embedding alias {m['embed']!r} in {name!r}")
    size = int(m["size"])
    if m["overlap"] is None:
        overlap = 0
    elif m["in_tokens"]:
        overlap = int(m["overlap"])
    else:
        overlap = overlap_tokens(size, int(m["overlap"]))
    profile = IndexProfile(
        name=name,
        parser=m["parser"],
        chunker=m["chunker"],
        size=size,
        overlap=overlap,
        parent_size=int(m["parent"]) if m["parent"] else None,
        header=m["header"] == "hdr",
        embed_model=model,
        dimensions=int(m["dims"]),
    )
    canonical = profile_name(profile)
    if canonical != name:
        raise ValueError(f"{name!r} is not canonical; the name for this profile is {canonical!r}")
    return profile


def make_profile(**fields) -> IndexProfile:
    """An ``IndexProfile`` from its fields, named canonically."""
    draft = IndexProfile(name="-", **fields)
    return draft.model_copy(update={"name": profile_name(draft)})
