"""Persistence for the guestbook -- the one place this template writes to disk.

The store is deliberately the *only* thing that knows the file format, the file
path, or how a write is made safe. Tools call methods on it; they never open a
file. Swap this module for a database and nothing above it changes.
"""

from __future__ import annotations

from hello_agent.storage.guestbook import GuestbookPathError, GuestbookStore

__all__ = ["GuestbookPathError", "GuestbookStore"]
