"""Memory: the agent's record of what it did and what happened."""

from __future__ import annotations

from hal9000.core.memory import Memory


def test_memory_starts_empty():
    memory = Memory()
    assert memory.get_memories() == []
    assert len(memory) == 0


def test_add_memory_appends_in_order():
    memory = Memory()
    memory.add_memory({"type": "user", "content": "greet Ada"})
    memory.add_memory({"type": "assistant", "content": "on it"})

    assert [item["content"] for item in memory.get_memories()] == ["greet Ada", "on it"]


def test_limit_returns_the_first_n_items():
    """Token efficiency -- MATE's T -- is why the limit exists at all."""
    memory = Memory(items=[{"type": "user", "content": str(i)} for i in range(5)])

    assert [item["content"] for item in memory.get_memories(limit=2)] == ["0", "1"]
    assert len(memory.get_memories(limit=None)) == 5


def test_copy_without_system_memories_drops_only_system_items():
    memory = Memory()
    memory.add_memory({"type": "system", "content": "you are a greeter"})
    memory.add_memory({"type": "user", "content": "greet Ada"})
    memory.add_memory({"type": "assistant", "content": "welcome"})
    memory.add_memory({"type": "system", "content": "it is 9am"})

    shared = memory.copy_without_system_memories()

    assert [item["type"] for item in shared.get_memories()] == ["user", "assistant"]
    assert [item["content"] for item in shared.get_memories()] == ["greet Ada", "welcome"]


def test_copy_without_system_memories_leaves_the_original_alone():
    memory = Memory(items=[{"type": "system", "content": "framing"}])
    copy = memory.copy_without_system_memories()

    copy.add_memory({"type": "user", "content": "new"})

    assert len(memory) == 1
    assert isinstance(copy, Memory) and copy is not memory


def test_memory_can_be_seeded_from_existing_items():
    """A sub-agent receives a memory; it must not alias the caller's list."""
    items = [{"type": "user", "content": "task"}]
    memory = Memory(items=items)
    memory.add_memory({"type": "assistant", "content": "done"})

    assert len(items) == 1
    assert len(memory) == 2


def test_repr_is_informative():
    assert repr(Memory(items=[{"type": "user", "content": "x"}])) == "Memory(items=1)"
