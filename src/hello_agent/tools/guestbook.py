"""The guestbook tools -- and the reason there is no ``write_file`` in this template.

``sign_guestbook`` is the "single safe tool" from ``environment_safety.md`` and
the "action specificity" half of MATE, in one function. A generic
``write_file(path, content)`` would hand the model three ways to be wrong -- the
wrong path, the wrong format, a clobbered file -- and no way to be checked. This
tool owns all three: the path comes from configuration, the format from
:mod:`hello_agent.storage.guestbook`, and the ordering from a lock. The model
supplies a name, a greeting, and an optional note. That is the entire attack
surface, and every one of those three is validated here.

The store arrives as ``_guestbook`` -- ``@register_tool`` strips underscore
parameters from the schema, so the model cannot point the tool at another file
even by accident. When nothing was injected the tool falls back to a store built
from :func:`hello_agent.config.guestbook_path`, which keeps it callable from a
test or a REPL without an agent around it.

Two error styles, chosen deliberately:

- **Invalid arguments raise.** The environment turns the exception into
  ``{"tool_executed": False, "error": ...}``, which tells the agent it did
  something wrong and hands it the message explaining what.
- **An empty guestbook returns.** Emptiness is a legitimate state, not a
  malfunction, so it comes back as data -- carrying an instruction about what to
  do next, the injected-instruction pattern from
  ``tool_design_best_practices.md``.

The reversal tool that pairs with ``sign_guestbook`` belongs in this module too,
carrying the ``safety`` tag so it stays off unless a flag adds it. Everything it
needs is already here: :meth:`GuestbookStore.remove_entry` undoes a signing by
the ``entry_id`` this tool returned.
"""

from __future__ import annotations

from hello_agent.storage.guestbook import GuestbookStore
from hello_agent.tools.decorators import register_tool

__all__ = [
    "MAX_GREETING_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_NOTE_LENGTH",
    "read_guestbook",
    "sign_guestbook",
    "undo_last_guestbook_entry",
]

#: Field ceilings. A guestbook entry is a line in a book, not a document.
MAX_NAME_LENGTH = 120
MAX_GREETING_LENGTH = 2000
MAX_NOTE_LENGTH = 500

_EMPTY_BOOK_INSTRUCTION = (
    "The guestbook is empty -- nobody has signed it yet. This is not a failure. "
    "Do not call read_guestbook again expecting a different answer. If you are "
    "greeting a visitor, carry on: extract their details, compose a greeting, and "
    "call sign_guestbook to create the first entry."
)


def _store(injected: GuestbookStore | None) -> GuestbookStore:
    """The injected store, or a fresh one at the configured path."""
    return injected if injected is not None else GuestbookStore()


def _clean_text(value: object, field: str, maximum: int, required: bool) -> str:
    """Validate one free-text field, explaining the fix in any error it raises."""
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(
            f"The {field!r} argument must be a string, not {type(value).__name__}. "
            "Pass the text itself, not an object or a list."
        )
    text = value.strip()
    if required and not text:
        raise ValueError(
            f"The {field!r} argument was empty. Supply the visitor's {field} as text; "
            "if you do not know it, ask the user rather than inventing one."
        )
    if len(text) > maximum:
        raise ValueError(
            f"The {field!r} argument is {len(text)} characters, over the {maximum}-character "
            f"limit for a guestbook entry. Shorten the {field} and call sign_guestbook again."
        )
    return text


@register_tool(tags=["guestbook"])
def sign_guestbook(
    name: str,
    greeting: str,
    note: str = "",
    _guestbook: GuestbookStore | None = None,
) -> dict:
    """Record a visit in the guestbook. Call this once per visitor, after you have
    composed their greeting.

    This is the only tool that writes to the guestbook. It validates the entry,
    assigns the next entry number, and saves the file safely -- you do not need
    to read the guestbook first, and you must not try to write the file yourself.

    Args:
        name: The visitor's name as you would address them, e.g. "Dr. Maria Chen".
        greeting: The greeting you composed for them, in full.
        note: Optional. Anything worth remembering about this visit -- their role,
            where they travelled from, what they came to do.

    Returns:
        ``{"status": "signed", "entry_id": N, "name": ..., "signed_at": ...}``.
        ``entry_id`` is that visit's number in the book; quote it to the user.
    """
    store = _store(_guestbook)
    entry = store.append(
        {
            "name": _clean_text(name, "name", MAX_NAME_LENGTH, required=True),
            "greeting": _clean_text(greeting, "greeting", MAX_GREETING_LENGTH, required=True),
            "note": _clean_text(note, "note", MAX_NOTE_LENGTH, required=False),
        }
    )
    return {
        "status": "signed",
        "entry_id": entry["entry_id"],
        "name": entry["name"],
        "signed_at": entry["signed_at"],
    }


@register_tool(tags=["guestbook"])
def read_guestbook(limit: int = 5, _guestbook: GuestbookStore | None = None) -> dict:
    """Read the most recent guestbook entries, newest last.

    Use this when you need to know whether someone has visited before, or what was
    said to them last time. You do not need to call it before signing.

    Args:
        limit: How many of the most recent entries to return. Defaults to 5.

    Returns:
        ``{"entry_count": N, "entries": [...]}`` when the book has been signed, or
        ``{"error": ...}`` explaining what to do instead when it is empty.
    """
    store = _store(_guestbook)
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 5

    document = store.read()
    entries = document["entries"]
    if not entries:
        return {"error": _EMPTY_BOOK_INSTRUCTION, "entry_count": 0, "entries": []}

    return {
        "entry_count": len(entries),
        "showing": min(limit, len(entries)),
        "entries": entries[-limit:],
    }


_NOTHING_TO_UNDO_INSTRUCTION = (
    "There is nothing to undo -- the guestbook has no entries. Nothing was changed. "
    "Do not call undo_last_guestbook_entry again; if you meant to record a visit, call "
    "sign_guestbook instead."
)


@register_tool(tags=["safety"])
def undo_last_guestbook_entry(_guestbook: GuestbookStore | None = None) -> dict:
    """Remove the most recent guestbook entry, undoing the last signing.

    Use this only to correct a mistake you just made -- you signed the wrong name,
    or signed twice for one visitor. It removes the newest entry and nothing else,
    so it cannot be used to edit history further back than your own last action.
    Entry numbers are never reused: undoing entry 3 means the next signing is 4.

    Returns:
        ``{"status": "undone", "entry_id": N, "name": ..., "removed": {...}}`` when
        an entry was removed, or ``{"status": "nothing_to_undo", "error": ...}``
        explaining what to do instead when the guestbook is empty.
    """
    store = _store(_guestbook)
    removed = store.remove_last()
    if removed is None:
        return {"status": "nothing_to_undo", "error": _NOTHING_TO_UNDO_INSTRUCTION}
    return {
        "status": "undone",
        "entry_id": removed.get("entry_id"),
        "name": removed.get("name", ""),
        "removed": removed,
    }
