"""M is for Memory: the agent's record of what it has done and what happened.

An LLM has no memory. Every call is answered from the messages you hand it and
nothing else, so "the agent remembers the file read failed" really means "we put
the failure back into the next prompt". :class:`Memory` is that store.

It stays a thin wrapper around a list on purpose. The interface -- append an
item, read back the last *n* -- is narrow enough that a subclass can put the
items in a database, summarise them, or rank them by relevance without the agent
loop noticing. What cannot change is the shape at the boundary: whatever the
storage, the agent language must be able to render memories as a list of
messages.

Each item is a ``dict`` with a ``type`` and a ``content``:

- ``user`` -- the task, and the results of actions fed back to the model
- ``assistant`` -- what the agent decided
- ``system`` -- framing injected by the framework or a capability, not
  conversation (see :meth:`Memory.copy_without_system_memories`)
"""

from __future__ import annotations

from typing import Any

__all__ = ["Memory"]


class Memory:
    """Working memory: an ordered list of conversation items.

    Example:
        >>> m = Memory()
        >>> m.add_memory({"type": "user", "content": "hello"})
        >>> m.get_memories()
        [{'type': 'user', 'content': 'hello'}]
    """

    def __init__(self, items: list[dict[str, Any]] | None = None):
        #: the raw items, oldest first. Read it directly only when you mean to.
        self.items: list[dict[str, Any]] = list(items or [])

    def add_memory(self, memory: dict) -> None:
        """Append one memory item. Items are dicts with ``type`` and ``content``."""
        self.items.append(memory)

    def get_memories(self, limit: int | None = None) -> list[dict]:
        """Return memory items, oldest first, at most ``limit`` of them.

        ``limit`` returns the *first* ``limit`` items, matching the framework in
        the course notes. Token efficiency -- MATE's T -- is why this argument
        exists at all: a long-running agent should not re-send its entire history
        on every turn.
        """
        return self.items[:limit]

    def copy_without_system_memories(self) -> Memory:
        """A new :class:`Memory` with every ``system`` item dropped.

        Used when one agent hands its memory to another: the caller's framing
        ("you are a greeter", "right now it is 9am") is noise -- or worse, a
        contradiction -- inside the callee, while the actual conversation is the
        context worth sharing. The copy is shallow; the items themselves are not
        duplicated.
        """
        filtered = [item for item in self.items if item.get("type") != "system"]
        return Memory(items=filtered)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"Memory(items={len(self.items)})"
