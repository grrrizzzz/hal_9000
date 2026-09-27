"""The guestbook itself: a JSON file, confined to one directory, written atomically.

Three properties are worth more than the twenty lines they cost, and all three
are MATE's *environmental safety* in miniature:

**Confinement.** Every path is resolved and checked against a root directory. A
name that climbs out with ``..``, a symlink that points away, an absolute path
from elsewhere -- all raise :class:`GuestbookPathError` before anything is opened.
The model never chooses a path today; the guard is what keeps that true after
someone adds a tool that lets it.

**Atomic replace.** A write goes to a temporary file in the same directory, then
:func:`os.replace` swaps it in -- atomic on POSIX and Windows alike. An
interrupted run leaves the previous guestbook intact, not a truncated one.

**Serialised read-modify-write.** Appending is read, mutate, write -- three steps
two threads can interleave into a lost entry and a duplicated ``entry_id``. A lock
keyed on the resolved path serialises them, so two stores sharing a file take turns.

The document on disk is ``{"entries": [...], "next_entry_id": N}``. Keeping the
counter in the file rather than deriving it from the entries is what keeps ids
monotonic across a deletion: undo entry 3 and the next signature is 4, never a
recycled 3.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

from hello_agent import config

__all__ = ["FIRST_ENTRY_ID", "GuestbookPathError", "GuestbookStore"]

#: The first entry is number 1, so an unwritten guestbook starts here.
FIRST_ENTRY_ID = 1

# One lock per resolved path, shared by every store that points at that file.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class GuestbookPathError(ValueError):
    """Raised when a path would escape the guestbook's directory."""


def _lock_for(path: Path) -> threading.Lock:
    """The lock guarding ``path``, created once and shared process-wide."""
    key = str(path)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


class GuestbookStore:
    """A JSON-file guestbook, confined to ``root``.

    Args:
        path: The guestbook file. Relative paths resolve against ``root``.
            Defaults to :func:`hello_agent.config.guestbook_path`.
        root: The directory the file must stay inside. Defaults to the parent of
            ``path``, which trusts the operator's configured location -- it comes
            from the environment, not from the model. Pass ``root`` explicitly
            whenever the name comes from anywhere less trustworthy.

    Raises:
        GuestbookPathError: if ``path`` resolves outside ``root``.

    Example:
        >>> import tempfile
        >>> directory = tempfile.mkdtemp()
        >>> store = GuestbookStore("guestbook.json", root=directory)
        >>> store.append({"name": "Ada"})["entry_id"]
        1
        >>> try:
        ...     GuestbookStore("../elsewhere.json", root=directory)
        ... except GuestbookPathError:
        ...     print("refused")
        refused
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        root: str | os.PathLike | None = None,
    ):
        configured = Path(path) if path is not None else Path(config.guestbook_path())
        if root is not None:
            self.root = Path(root).expanduser().resolve()
        else:
            self.root = (Path.cwd() / configured.expanduser()).resolve().parent
        self.path = self._confine(configured)

    def _confine(self, candidate: Path) -> Path:
        """Resolve ``candidate`` inside :attr:`root`, or refuse to."""
        resolved = (self.root / candidate.expanduser()).resolve()
        if self.root not in resolved.parents:
            raise GuestbookPathError(
                f"Refusing to use {str(candidate)!r}: it resolves to {resolved}, "
                f"outside the guestbook directory {self.root}. "
                "Use a plain file name with no directory separators."
            )
        return resolved

    # --- reading ----------------------------------------------------------

    def read(self) -> dict:
        """The whole document. A missing or unreadable file reads as empty."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # A fresh dict every call: a shared default would be mutated by the
            # first append and inherited by every store after it.
            return {"entries": [], "next_entry_id": FIRST_ENTRY_ID}
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"The guestbook at {self.path} is not valid JSON ({error}). "
                "Move it aside and a fresh one will be created on the next signing."
            ) from error
        entries = document.get("entries", [])
        next_id = document.get("next_entry_id") or _next_after(entries)
        return {"entries": list(entries), "next_entry_id": next_id}

    def entries(self, limit: int | None = None) -> list[dict]:
        """Entries oldest-first, or the last ``limit`` of them if given."""
        entries = self.read()["entries"]
        return entries[-limit:] if limit else entries

    def count(self) -> int:
        """How many entries the guestbook holds."""
        return len(self.read()["entries"])

    def exists(self) -> bool:
        """Has the file been created yet? (It is created on first write.)"""
        return self.path.exists()

    # --- writing ----------------------------------------------------------

    def append(self, entry: dict) -> dict:
        """Add ``entry``, stamping it with a monotonic ``entry_id`` and a time.

        Returns:
            The stored entry, including the fields this method added.
        """
        with _lock_for(self.path):
            document = self.read()
            stored = {
                "entry_id": document["next_entry_id"],
                "signed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **entry,
            }
            document["entries"].append(stored)
            document["next_entry_id"] = stored["entry_id"] + 1
            self._write(document)
            return dict(stored)

    def remove_entry(self, entry_id: int) -> dict | None:
        """Delete the entry with ``entry_id``. Returns it, or ``None`` if absent.

        ``next_entry_id`` is left alone, so ids are never reused.
        """
        with _lock_for(self.path):
            document = self.read()
            for index, entry in enumerate(document["entries"]):
                if entry.get("entry_id") == entry_id:
                    removed = document["entries"].pop(index)
                    self._write(document)
                    return removed
            return None

    def remove_last(self) -> dict | None:
        """Delete the most recent entry. Returns it, or ``None`` if empty."""
        with _lock_for(self.path):
            document = self.read()
            if not document["entries"]:
                return None
            removed = document["entries"].pop()
            self._write(document)
            return removed

    def _write(self, document: dict) -> None:
        """Replace the file atomically, creating the directory if need be."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(document, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def __repr__(self) -> str:
        return f"GuestbookStore(path={str(self.path)!r})"


def _next_after(entries: list[dict]) -> int:
    """Recover a counter from entries alone, for a file written by an older version."""
    return max((entry.get("entry_id", 0) for entry in entries), default=0) + 1
