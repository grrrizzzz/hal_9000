"""Conversation storage, migration 4 (feedback), and the passcode/session primitives."""

from __future__ import annotations

import os
import sqlite3
import stat
from datetime import UTC, datetime, timedelta

import pytest

from hal9000.knowledge.answer import store
from hal9000.knowledge.auth import Auth, RateLimiter, load_secret
from hal9000.knowledge.contracts import FeedbackRequest
from hal9000.knowledge.storage import migrate, open_db
from hal9000.knowledge.storage.db import statements
from hal9000.knowledge.storage.schema import MIGRATIONS, SCHEMA_VERSION


# ------------------------------------------------------------------ store
def test_history_is_the_last_turns_oldest_first(kb_db):
    conversation = store.create_conversation(kb_db, "s1", "  A  long   title " + "x" * 200)
    for n in range(8):
        store.add_message(kb_db, conversation, "user" if n % 2 == 0 else "assistant", f"m{n}")
    assert store.history(kb_db, conversation, 6) == [
        ("user", "m2"), ("assistant", "m3"), ("user", "m4"),
        ("assistant", "m5"), ("user", "m6"), ("assistant", "m7"),
    ]  # fmt: skip
    assert store.history(kb_db, conversation, 0) == []
    title = store.list_conversations(kb_db, "s1")[0].title
    assert title.startswith("A long title x") and len(title) == store.TITLE_CHARS
    assert store.list_conversations(kb_db, "s2") == []
    assert store.message_session(kb_db, 1) == "s1" and store.message_session(kb_db, 999) is None


def test_feedback_without_a_citation_makes_an_unanswerable_draft(kb_db):
    conversation = store.create_conversation(kb_db, "s1", "q")
    store.add_message(kb_db, conversation, "user", "Is there a pool?")
    answer = store.add_message(
        kb_db, conversation, "assistant", "I couldn't find that.", outcome="abstained"
    )
    result = store.save_feedback(
        kb_db, FeedbackRequest(message_id=answer.id, add_to_gold=True, comment="Right call.")
    )
    row = kb_db.execute("SELECT * FROM gold_items WHERE id = ?", (result.gold_item_id,)).fetchone()
    assert row["category"] == "unanswerable" and row["evidence"] == "[]"
    assert row["question"] == "Is there a pool?" and row["origin"] == "feedback"
    feedback = kb_db.execute("SELECT rating, add_to_gold FROM feedback").fetchone()
    assert feedback["rating"] is None and feedback["add_to_gold"] == 1


def test_the_database_refuses_feedback_with_neither_rating_nor_gold(kb_db):
    conversation = store.create_conversation(kb_db, "s1", "q")
    message = store.add_message(kb_db, conversation, "assistant", "a")
    with pytest.raises(sqlite3.IntegrityError):
        kb_db.execute(
            "INSERT INTO feedback (message_id, rating, add_to_gold, created_at, updated_at) "
            "VALUES (?, NULL, 0, 'now', 'now')",
            (message.id,),
        )


def test_migration_4_keeps_the_latest_feedback_per_message(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)")
    for version, script in MIGRATIONS[:3]:
        for statement in statements(script):
            conn.execute(statement)
        conn.execute("INSERT INTO schema_version VALUES (?, 'then')", (version,))
    conn.execute(
        "INSERT INTO conversations (id, session_id, created_at, updated_at) VALUES (1, 's', 'a', 'a')"
    )
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) "
        "VALUES (7, 1, 'assistant', 'x', 'a')"
    )
    conn.executemany(
        "INSERT INTO feedback (message_id, rating, comment, created_at) VALUES (7, ?, ?, 'a')",
        [(1, "first"), (-1, "second")],
    )
    conn.commit()
    assert migrate(conn) == SCHEMA_VERSION
    rows = conn.execute("SELECT rating, comment, updated_at FROM feedback").fetchall()
    assert [tuple(r) for r in rows] == [(-1, "second", "a")]
    columns = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    assert {"grounding", "issues"} <= columns
    conn.close()
    open_db(tmp_path / "old.db").close()  # idempotent afterwards


# ------------------------------------------------------------------- auth
def test_rate_limiter_windows_slide():
    now = [0.0]
    limiter = RateLimiter(2, 60.0, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")  # per key
    assert limiter.retry_after("a") == 61
    now[0] = 59.9
    assert limiter.blocked("a")
    now[0] = 60.0
    assert not limiter.blocked("a") and limiter.retry_after("a") == 0
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_sessions_expire_and_follow_the_passcode(tmp_path, monkeypatch):
    secret = load_secret(tmp_path / "session.key")
    auth = Auth("a-synthetic-passcode", secret)
    assert auth.check_passcode("a-synthetic-passcode") and not auth.check_passcode("nope")
    token, session, expires = auth.issue()
    assert auth.session(token) == session
    assert timedelta(days=29) < expires - datetime.now(UTC) <= timedelta(days=30)
    assert auth.session(None) is None and auth.session("") is None
    assert Auth("another-passcode", secret).session(token) is None  # new passcode: signed out
    assert Auth("a-synthetic-passcode", secret).session(token) == session  # survives a restart
    expired = Auth("a-synthetic-passcode", secret, max_age_s=-1)
    assert expired.session(token) is None
    with pytest.raises(ValueError):
        Auth("", secret)


def test_the_secret_is_created_once_and_private(tmp_path):
    path = tmp_path / "k" / "session.key"
    first = load_secret(path)
    assert len(first) == 32 and load_secret(path) == first
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    path.write_bytes(b"short")
    assert len(load_secret(path)) == 32  # a truncated key is replaced
