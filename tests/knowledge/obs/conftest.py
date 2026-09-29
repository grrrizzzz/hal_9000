"""Fixtures for the observability suite: a seeded knowledge.db and a clean OTel/collector state."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from hal9000.knowledge.obs import metrics, otel

SHA = "a" * 64


def seed_corpus(conn, profiles=("fixed-512", "recursive-1024")) -> dict[str, int]:
    """Two synthetic profiles with chunks over two synthetic documents."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "INSERT INTO files (sha256, original_name, relative_path, mime_type, bytes, uploaded_at)"
            " VALUES (?, 'synthetic.pdf', 'synthetic.pdf', 'application/pdf', 10, ?)",
            (SHA, now),
        )
        docs = [
            conn.execute(
                "INSERT INTO documents (file_sha, page_start, page_end, title) VALUES (?, ?, ?, ?)",
                (SHA, page, page, f"Synthetic doc {page}"),
            ).lastrowid
            for page in (1, 2)
        ]
        ids: dict[str, int] = {}
        for n, name in enumerate(profiles):
            pid = conn.execute(
                "INSERT INTO profiles (name, parser, chunker, size, header, embed_model,"
                " dimensions, created_at) VALUES (?, 'pypdfium2', 'fixed', 512, 1, 'fake', 4, ?)",
                (name, now),
            ).lastrowid
            ids[name] = pid
            # profile n gets chunks over the first n+1 documents, three per document
            for doc in docs[: n + 1]:
                for i in range(3):
                    conn.execute(
                        "INSERT INTO chunks (profile_id, document_id, page_start, page_end,"
                        " char_start, char_end, text, tokens) VALUES (?, ?, 1, 1, ?, ?, 'x', 1)",
                        (pid, doc, i, i + 1),
                    )
    return ids


def add_eval_run(conn, run_id, profile, metrics_json, *, finished="2026-09-01T00:00:00+00:00"):
    config = {"profile": profile} if profile is not None else {}
    with conn:
        conn.execute(
            "INSERT INTO eval_runs (id, stage, config, metrics, started_at, finished_at)"
            " VALUES (?, 'S1a', ?, ?, '2026-09-01T00:00:00+00:00', ?)",
            (run_id, json.dumps(config), json.dumps(metrics_json), finished),
        )


@pytest.fixture
def db_path(tmp_path, kb_db):
    """The path of the migrated ``kb_db``."""
    return tmp_path / "knowledge.db"


@pytest.fixture(autouse=True)
def isolated_obs(monkeypatch):
    """No knowledge.db collector, OTel provider or OTLP endpoint leaks between tests."""
    monkeypatch.delenv(otel.ENDPOINT_ENV, raising=False)
    metrics.install_db_collector(None)
    otel.reset()
    yield
    metrics.install_db_collector(None)
    otel.reset()
