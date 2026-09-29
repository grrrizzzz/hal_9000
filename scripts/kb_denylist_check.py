#!/usr/bin/env python3
"""Fail if any tracked file contains a string from the house-data denylist (plan §14, rule K2).

The denylist holds distinctive strings from the house documents: reference
numbers, people's names, the street, amounts. It is never committed. It is read
from, in order:

1. the ``KB_DENYLIST`` environment variable (CI reads the GitHub secret of the
   same name);
2. the ``KB_DENYLIST`` key in ``.env`` at the repository root (the pre-commit
   hook).

Either way, entries are separated by ``|`` or newlines. Every path ``git
ls-files`` lists, and every such file's content, is scanned case-insensitively.
A hit prints the file, the line and the *number* of the denylist entry, never
the string, so the check's own output can't leak what it protects. Exit codes:
0 clean or no list configured (with a warning; a ``::warning::`` annotation
under GitHub Actions), 1 a hit, 2 usage error.

    uv run python scripts/kb_denylist_check.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

#: Third-party files verified by hash; their base64 would only give chance hits.
EXCLUDED_PREFIXES = ("vendor/tiktoken/",)
#: Shorter entries would match almost anything, so they are skipped with a warning.
MIN_LENGTH = 3
#: Line number reported when a tracked file's *path* contains an entry.
PATH_HIT = -1
NO_LIST = "no house-data denylist configured (KB_DENYLIST); nothing was checked."


def _split(raw: str) -> list[str]:
    """Entries separated by ``|`` or newlines, trimmed, blanks dropped."""
    return [p.strip() for p in raw.replace("|", "\n").splitlines() if p.strip()]


def _from_env_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        from dotenv import dotenv_values
    except ImportError:  # a bare interpreter (pre-commit's system python): KEY=value lines
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip().removeprefix("export ").strip() == "KB_DENYLIST":
                return value.strip().strip("'\"")
        return ""
    return dotenv_values(path).get("KB_DENYLIST") or ""


def load_denylist(env: dict[str, str], env_file: Path) -> tuple[list[str], str]:
    """The entries and where they came from (``""`` when none are configured)."""
    raw = env.get("KB_DENYLIST", "")
    if raw.strip():
        return _split(raw), "KB_DENYLIST environment variable"
    raw = _from_env_file(env_file)
    if raw.strip():
        return _split(raw), f"KB_DENYLIST in {env_file.name}"
    return [], ""


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def scan(root: Path, files: list[str], entries: list[str]) -> list[tuple[str, int, int]]:
    """(path, line number, entry number) for every hit.

    Line ``PATH_HIT`` means the file's path itself matched; line 0 means a
    match inside a binary file.
    """
    needles = [(n, e.casefold()) for n, e in enumerate(entries, 1) if len(e) >= MIN_LENGTH]
    hits: list[tuple[str, int, int]] = []
    for rel in files:
        hits.extend((rel, PATH_HIT, n) for n, needle in needles if needle in rel.casefold())
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        path = root / rel
        if not path.is_file():
            continue  # deleted in the working tree, or a submodule
        data = path.read_bytes()
        binary = b"\0" in data[:8192]
        text = data.decode("utf-8", "replace").casefold()
        if binary:
            hits.extend((rel, 0, n) for n, needle in needles if needle in text)
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            hits.extend((rel, lineno, n) for n, needle in needles if needle in line)
    return hits


def redact(text: str, entries: list[str]) -> str:
    """``text`` with every (case-insensitive) occurrence of an entry replaced by ``***``."""
    for entry in entries:
        if len(entry) < MIN_LENGTH:
            continue
        folded, needle = text.casefold(), entry.casefold()
        start = folded.find(needle)
        while start != -1:
            # casefold can change lengths for a few scripts; fall back to masking the rest
            end = start + len(entry) if len(folded) == len(text) else len(text)
            text = text[:start] + "***" + text[end:]
            folded = text.casefold()
            start = folded.find(needle)
    return text


def main(argv: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan tracked files for denylisted strings.")
    parser.add_argument("--root", type=Path, default=None, help="repository root")
    parser.add_argument("--env-file", type=Path, default=None, help="default: <root>/.env")
    args = parser.parse_args(argv)
    env = dict(os.environ) if env is None else env

    root = args.root
    if root is None:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        )
        if top.returncode != 0:
            print("kb_denylist_check: not inside a git repository", file=sys.stderr)
            return 2
        root = Path(top.stdout.strip())
    env_file = args.env_file or root / ".env"

    entries, source = load_denylist(env, env_file)
    if not entries:
        print(f"WARNING: {NO_LIST}", file=sys.stderr)
        if env.get("GITHUB_ACTIONS") == "true":
            print(f"::warning title=House-data denylist::{NO_LIST}")
        return 0
    short = [n for n, e in enumerate(entries, 1) if len(e) < MIN_LENGTH]
    for n in short:
        print(f"WARNING: denylist entry #{n} is under {MIN_LENGTH} characters; skipped.")

    files = tracked_files(root)
    hits = scan(root, files, entries)
    if not hits:
        print(f"denylist: {len(files)} tracked files clean ({len(entries)} entries, {source}).")
        return 0
    for rel, lineno, n in hits:
        rel = redact(rel, entries)  # a path can itself contain an entry
        if lineno == PATH_HIT:
            print(f"{rel}: file path contains denylist entry #{n}", file=sys.stderr)
            continue
        where = f"{rel}:{lineno}" if lineno else f"{rel} (binary)"
        print(f"{where}: contains denylist entry #{n}", file=sys.stderr)
    print(
        f"denylist: {len(hits)} hit(s). House data must never be committed (rule K2).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
