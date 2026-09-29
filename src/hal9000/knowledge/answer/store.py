"""Conversations, messages and feedback in ``knowledge.db`` (plan §7.3, §12).

- Conversations belong to one browser session (the signed cookie's session
  ID); every read and delete is scoped to it.
- Assistant messages carry their citations, trace ID, outcome, grounding and
  verify issues.
- :func:`history` gives the last turns for condensing a follow-up
  (``retrieval.history_turns``, six by default).
- **Feedback** is one row per message: a later post for the same message
  updates the rating, comment and "add to question set". Adding to the
  question set creates a *draft* gold item (origin ``feedback``) from the
  question, the answer and the cited passages, once.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime

from hal9000.knowledge.contracts import (
    Citation,
    ConversationDetail,
    ConversationSummary,
    EvidenceQuote,
    FeedbackRequest,
    FeedbackResponse,
    GoldItem,
    MessageOut,
)

__all__ = [
    "TITLE_CHARS",
    "add_message",
    "create_conversation",
    "delete_conversation",
    "delete_conversations",
    "get_conversation",
    "history",
    "list_conversations",
    "message_session",
    "save_feedback",
]

TITLE_CHARS = 80


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _summary(row: sqlite3.Row) -> ConversationSummary:
    return ConversationSummary(
        id=row["id"],
        title=row["title"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _message(row: sqlite3.Row) -> MessageOut:
    return MessageOut(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        citations=[Citation.model_validate(c) for c in json.loads(row["citations"] or "[]")],
        trace_id=row["trace_id"],
        outcome=row["outcome"],
        grounding=row["grounding"],
        issues=json.loads(row["issues"] or "[]"),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def create_conversation(conn: sqlite3.Connection, session_id: str, title: str) -> int:
    now = _now()
    text = " ".join(title.split())
    if len(text) > TITLE_CHARS:
        text = text[: TITLE_CHARS - 1].rstrip() + "…"
    with conn:
        cur = conn.execute(
            "INSERT INTO conversations (session_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, text, now, now),
        )
    return int(cur.lastrowid)


def _owned(conn: sqlite3.Connection, session_id: str, conversation_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM conversations WHERE id = ? AND session_id = ?",
        (conversation_id, session_id),
    ).fetchone()


def list_conversations(conn: sqlite3.Connection, session_id: str) -> list[ConversationSummary]:
    rows = conn.execute(
        "SELECT * FROM conversations WHERE session_id = ? ORDER BY updated_at DESC, id DESC",
        (session_id,),
    ).fetchall()
    return [_summary(row) for row in rows]


def get_conversation(
    conn: sqlite3.Connection, session_id: str, conversation_id: int
) -> ConversationDetail | None:
    row = _owned(conn, session_id, conversation_id)
    if row is None:
        return None
    messages = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
    ).fetchall()
    return ConversationDetail(
        **_summary(row).model_dump(), messages=[_message(m) for m in messages]
    )


def delete_conversation(conn: sqlite3.Connection, session_id: str, conversation_id: int) -> bool:
    with conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND session_id = ?",
            (conversation_id, session_id),
        )
    return cur.rowcount > 0


def delete_conversations(conn: sqlite3.Connection, session_id: str) -> int:
    with conn:
        cur = conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    return cur.rowcount


def add_message(
    conn: sqlite3.Connection,
    conversation_id: int,
    role: str,
    content: str,
    *,
    citations: Sequence[Citation] = (),
    trace_id: str | None = None,
    outcome: str | None = None,
    grounding: str | None = None,
    issues: Sequence[str] = (),
) -> MessageOut:
    now = _now()
    with conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, citations, trace_id, outcome, "
            "grounding, issues, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                json.dumps([c.model_dump(mode="json") for c in citations]),
                trace_id,
                outcome,
                grounding,
                json.dumps(list(issues)),
                now,
            ),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _message(row)


def history(conn: sqlite3.Connection, conversation_id: int, turns: int) -> list[tuple[str, str]]:
    """The last ``turns`` messages as ``(role, text)``, oldest first (for condensing)."""
    if turns <= 0:
        return []
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? AND content != '' "
        "ORDER BY id DESC LIMIT ?",
        (conversation_id, turns),
    ).fetchall()
    return [(row["role"], row["content"]) for row in reversed(rows)]


def message_session(conn: sqlite3.Connection, message_id: int) -> str | None:
    row = conn.execute(
        "SELECT c.session_id FROM messages m JOIN conversations c ON c.id = m.conversation_id "
        "WHERE m.id = ?",
        (message_id,),
    ).fetchone()
    return row[0] if row else None


# ----------------------------------------------------------------- feedback
def _quote(conn: sqlite3.Connection, citation: Citation) -> str:
    row = conn.execute("SELECT text FROM chunks WHERE id = ?", (citation.chunk_id,)).fetchone()
    text = (row[0] if row else "") or citation.snippet or citation.title
    return " ".join(text.split()) or citation.title


def _draft_gold(conn: sqlite3.Connection, message_id: int, comment: str | None) -> int:
    from hal9000.knowledge.eval.store import insert_gold

    answer = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    question = conn.execute(
        "SELECT content FROM messages WHERE conversation_id = ? AND id < ? AND role = 'user' "
        "ORDER BY id DESC LIMIT 1",
        (answer["conversation_id"], message_id),
    ).fetchone()
    citations = [Citation.model_validate(c) for c in json.loads(answer["citations"] or "[]")]
    evidence = [
        EvidenceQuote(quote=_quote(conn, c), page=c.page, file_sha=c.file_sha) for c in citations
    ]
    answered = answer["outcome"] == "answered" and evidence
    item = GoldItem(
        question=(question[0] if question else "").strip() or "(question not found)",
        reference_answer=answer["content"] or None,
        category="factoid" if answered else "unanswerable",
        evidence=evidence if answered else [],
        status="draft",
        reviewer_note=comment,
        origin="feedback",
    )
    return int(insert_gold(conn, [item])[0].id)  # type: ignore[arg-type]


def save_feedback(conn: sqlite3.Connection, request: FeedbackRequest) -> FeedbackResponse:
    """Create or update the message's one feedback row (and its draft gold item, once)."""
    from hal9000.knowledge.obs.metrics import record_feedback

    now = _now()
    row = conn.execute(
        "SELECT * FROM feedback WHERE message_id = ?", (request.message_id,)
    ).fetchone()
    rating = request.rating if request.rating is not None else (row["rating"] if row else None)
    comment = request.comment if request.comment is not None else (row["comment"] if row else None)
    add = bool(request.add_to_gold or (row and row["add_to_gold"]))
    gold_id = row["gold_item_id"] if row else None
    if request.add_to_gold and gold_id is None:
        gold_id = _draft_gold(conn, request.message_id, comment)
    with conn:
        if row is None:
            cur = conn.execute(
                "INSERT INTO feedback (message_id, rating, comment, add_to_gold, gold_item_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (request.message_id, rating, comment, int(add), gold_id, now, now),
            )
            feedback_id = int(cur.lastrowid)
        else:
            feedback_id = int(row["id"])
            conn.execute(
                "UPDATE feedback SET rating = ?, comment = ?, add_to_gold = ?, gold_item_id = ?, "
                "updated_at = ? WHERE id = ?",
                (rating, comment, int(add), gold_id, now, feedback_id),
            )
    if request.rating is not None and (row is None or row["rating"] != request.rating):
        record_feedback(request.rating)
    return FeedbackResponse(id=feedback_id, gold_item_id=gold_id)
