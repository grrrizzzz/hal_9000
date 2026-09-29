"""Fixtures for the knowledge-base suite.

Every test starts with no ``KB_*`` variables, so a developer's ``.env`` or shell
can't change what the suite sees (the root conftest does the same for ``HAL_*``).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def clean_kb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [key for key in os.environ if key.startswith("KB_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def kb_db(tmp_path):
    """A migrated ``knowledge.db`` in a temp dir, sqlite-vec loaded."""
    from hal9000.knowledge.storage import open_db

    conn = open_db(tmp_path / "knowledge.db")
    yield conn
    conn.close()
