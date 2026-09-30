"""``gold_items`` and ``judge_labels`` in ``knowledge.db`` (plan §12), and JSONL exchange.

The question set never enters git (plan §8.1, rule K2): JSONL export and
import are refused anywhere inside a git work tree except its ``data/``
directory (:func:`check_data_path`).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from hal9000.knowledge.contracts import (
    Citation,
    EvidenceQuote,
    GoldItem,
    JudgeLabel,
    RubricLabels,
)

__all__ = [
    "UnsafePath",
    "check_data_path",
    "export_jsonl",
    "get_gold",
    "import_jsonl",
    "insert_gold",
    "insert_judge_label",
    "list_gold",
    "list_judge_labels",
    "set_check_note",
    "set_gold_status",
    "set_human_labels",
    "set_judge_labels",
    "unreviewed_gold",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------ gold items
def _row_to_gold(row: sqlite3.Row) -> GoldItem:
    return GoldItem(
        id=row["id"],
        question=row["question"],
        reference_answer=row["reference_answer"],
        category=row["category"],
        difficulty=row["difficulty"],
        paraphrased=bool(row["paraphrased"]),
        evidence=[EvidenceQuote.model_validate(e) for e in json.loads(row["evidence"] or "[]")],
        status=row["status"],
        split=row["split"],
        reviewer_note=row["reviewer_note"],
        origin=row["origin"],
    )


def _values(item: GoldItem) -> tuple:
    return (
        item.question,
        item.reference_answer,
        item.category,
        item.difficulty,
        int(item.paraphrased),
        json.dumps([e.model_dump(mode="json") for e in item.evidence]),
        item.status,
        item.split,
        item.reviewer_note,
        item.origin,
    )


_COLUMNS = (
    "question, reference_answer, category, difficulty, paraphrased, evidence, status, split, "
    "reviewer_note, origin"
)


def insert_gold(conn: sqlite3.Connection, items: Iterable[GoldItem]) -> list[GoldItem]:
    """Store new items; returns them with their IDs."""
    stored: list[GoldItem] = []
    now = _now()
    with conn:
        for item in items:
            cur = conn.execute(
                f"INSERT INTO gold_items ({_COLUMNS}, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*_values(item), now, now),
            )
            stored.append(item.model_copy(update={"id": cur.lastrowid}))
    return stored


def list_gold(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    split: str | None = None,
    category: str | None = None,
) -> list[GoldItem]:
    clauses, params = [], []
    for column, value in (("status", status), ("split", split), ("category", category)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM gold_items{where} ORDER BY id", params).fetchall()
    return [_row_to_gold(r) for r in rows]


def unreviewed_gold(conn: sqlite3.Connection, *, category: str) -> list[GoldItem]:
    """Drafts of ``category`` nobody has edited or reviewed yet (``updated_at = created_at``)."""
    rows = conn.execute(
        "SELECT * FROM gold_items WHERE category = ? AND status = 'draft' "
        "AND updated_at = created_at ORDER BY id",
        (category,),
    ).fetchall()
    return [_row_to_gold(r) for r in rows]


def set_check_note(
    conn: sqlite3.Connection, item_id: int, old_note: str | None, new_note: str | None
) -> bool:
    """Replace an unreviewed draft's reviewer note with a (re)check result.

    Compare-and-set: only if the item is still an unedited draft with
    ``old_note``, so a review decision made meanwhile is never overwritten.
    ``updated_at`` is left alone: it records review changes, and a check is
    not one. Returns whether the note was replaced.
    """
    with conn:
        cur = conn.execute(
            "UPDATE gold_items SET reviewer_note = ? WHERE id = ? AND status = 'draft' "
            "AND updated_at = created_at AND reviewer_note IS ?",
            (new_note, item_id, old_note),
        )
    return cur.rowcount == 1


def get_gold(conn: sqlite3.Connection, item_id: int) -> GoldItem | None:
    row = conn.execute("SELECT * FROM gold_items WHERE id = ?", (item_id,)).fetchone()
    return _row_to_gold(row) if row else None


def set_gold_status(conn: sqlite3.Connection, item_ids: Sequence[int], status: str) -> None:
    with conn:
        conn.executemany(
            "UPDATE gold_items SET status = ?, updated_at = ? WHERE id = ?",
            [(status, _now(), i) for i in item_ids],
        )


# ------------------------------------------------------------------- JSONL
class UnsafePath(ValueError):
    """A question-set file would land where git could pick it up (rule K2)."""


def _git_root(path: Path) -> Path | None:
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return parent
    return None


def check_data_path(path: Path | str, data_dir: Path | str) -> Path:
    """Resolve ``path`` and refuse it inside a git work tree unless it is under ``data/``.

    Allowed: anywhere under the configured ``data_dir`` or a work tree's
    ``data/`` (git-ignored), or outside every git work tree. The file must be
    ``.jsonl``.
    """
    target = Path(path).expanduser().resolve()
    if target.suffix != ".jsonl":
        raise UnsafePath(f"{target} must be a .jsonl file")
    data = Path(data_dir).expanduser().resolve()
    if target.is_relative_to(data):
        return target
    root = _git_root(target.parent)
    if root is not None and not target.is_relative_to(root / "data"):
        raise UnsafePath(
            f"{target} is inside the git work tree {root} but not under its data/ directory; "
            "the question set must never be committed (keep it under data/)"
        )
    return target


def export_jsonl(conn: sqlite3.Connection, path: Path) -> int:
    """Write every gold item as one JSON object per line. Returns the count."""
    items = list_gold(conn)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    return len(items)


def import_jsonl(conn: sqlite3.Connection, path: Path) -> tuple[int, int]:
    """Read gold items from JSONL: an ``id`` already stored is updated, anything else inserted.

    Every line is validated before anything is written. Returns (inserted, updated).
    """
    items: list[GoldItem] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                items.append(GoldItem.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from None
    inserted = updated = 0
    now = _now()
    with conn:
        for item in items:
            exists = (
                item.id is not None
                and conn.execute("SELECT 1 FROM gold_items WHERE id = ?", (item.id,)).fetchone()
            )
            if exists:
                conn.execute(
                    "UPDATE gold_items SET question = ?, reference_answer = ?, category = ?, "
                    "difficulty = ?, paraphrased = ?, evidence = ?, status = ?, split = ?, "
                    "reviewer_note = ?, origin = ?, updated_at = ? WHERE id = ?",
                    (*_values(item), now, item.id),
                )
                updated += 1
            else:
                conn.execute(
                    f"INSERT INTO gold_items ({_COLUMNS}, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*_values(item), now, now),
                )
                inserted += 1
    return inserted, updated


# ---------------------------------------------------------------- judge labels
def _labels(raw: str | None) -> RubricLabels | None:
    return RubricLabels.model_validate_json(raw) if raw else None


def insert_judge_label(
    conn: sqlite3.Connection,
    gold_item_id: int,
    answer: str,
    *,
    eval_run_id: str | None = None,
    judge: RubricLabels | None = None,
    human: RubricLabels | None = None,
    citations: Sequence[Citation] = (),
) -> int:
    now = _now()
    with conn:
        cur = conn.execute(
            "INSERT INTO judge_labels (gold_item_id, eval_run_id, answer, human_labels, "
            "judge_labels, citations, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                gold_item_id,
                eval_run_id,
                answer,
                human.model_dump_json() if human else None,
                judge.model_dump_json() if judge else None,
                json.dumps([c.model_dump(mode="json") for c in citations]),
                now,
                now,
            ),
        )
    return int(cur.lastrowid)


def _set(conn: sqlite3.Connection, column: str, label_id: int, labels: RubricLabels) -> None:
    with conn:
        cur = conn.execute(
            f"UPDATE judge_labels SET {column} = ?, updated_at = ? WHERE id = ?",
            (labels.model_dump_json(), _now(), label_id),
        )
    if cur.rowcount == 0:
        raise KeyError(f"no judge label {label_id}")


def set_human_labels(conn: sqlite3.Connection, label_id: int, labels: RubricLabels) -> None:
    """Mike's labels from the Judge calibration tab (gate G-K1b)."""
    _set(conn, "human_labels", label_id, labels)


def set_judge_labels(conn: sqlite3.Connection, label_id: int, labels: RubricLabels) -> None:
    _set(conn, "judge_labels", label_id, labels)


def list_judge_labels(
    conn: sqlite3.Connection, *, eval_run_id: str | None = None
) -> list[JudgeLabel]:
    where, params = ("WHERE j.eval_run_id = ?", [eval_run_id]) if eval_run_id else ("", [])
    rows = conn.execute(
        "SELECT j.*, g.question FROM judge_labels j JOIN gold_items g ON g.id = j.gold_item_id "
        f"{where} ORDER BY j.id",
        params,
    ).fetchall()
    return [
        JudgeLabel(
            id=r["id"],
            gold_item_id=r["gold_item_id"],
            eval_run_id=r["eval_run_id"],
            question=r["question"],
            answer=r["answer"],
            citations=[Citation.model_validate(c) for c in json.loads(r["citations"] or "[]")],
            human=_labels(r["human_labels"]),
            judge=_labels(r["judge_labels"]),
        )
        for r in rows
    ]
