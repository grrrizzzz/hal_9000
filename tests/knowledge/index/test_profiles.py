"""Profile builds from stored pages and cached embeddings, and atomic activation (plan §6.2)."""

from __future__ import annotations

import threading

import pytest

from hal9000.knowledge.chunking import parse_profile_name
from hal9000.knowledge.index.embedding import EmbeddingService
from hal9000.knowledge.index.fake import FakeEmbedder
from hal9000.knowledge.index.profiles import (
    ProfileError,
    ProfileNotReady,
    activate_profile,
    active_profile,
    build_profile,
    ensure_profile,
    get_profile,
    list_profiles,
    profile_status,
)
from hal9000.knowledge.index.search import keyword_search
from hal9000.knowledge.storage import open_db
from tests.knowledge.fixtures import corpus
from tests.knowledge.index.conftest import PROFILE, ingest_all

OTHER = "pdfium.structure-512.hdr.fake-64"
FILES = [corpus.INSPECTION, corpus.ESTIMATE, corpus.INVOICE, corpus.MARKDOWN, corpus.TEXT]


def index_texts(conn, name: str) -> set[str]:
    profile = get_profile(conn, name)
    return {
        row[0]
        for row in conn.execute(
            "SELECT CASE WHEN header = '' THEN text ELSE header || char(10) || text END "
            "FROM chunks WHERE profile_id = ?",
            (profile.id,),
        )
    }


def tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_a_build_with_changed_chunking_embeds_only_new_texts(conn, make_pipeline, fake):
    pipeline = make_pipeline()
    ingest_all(conn, pipeline, FILES)
    first = index_texts(conn, PROFILE)
    calls_before = fake.texts

    report = pipeline.build(conn, OTHER)
    second = index_texts(conn, OTHER)
    new, reused = second - first, second & first
    assert new and reused  # the check below is meaningful in both directions
    assert report.embedded == len(new)
    assert report.cache_hits == len(reused)
    assert fake.texts - calls_before == len(new)
    assert profile_status(conn, OTHER) == "ready" and not get_profile(conn, OTHER).active

    # building it again is a no-op; rebuilding embeds nothing new at all
    assert pipeline.build(conn, OTHER).skipped
    rebuilt = pipeline.build(conn, OTHER, rebuild=True)
    assert (rebuilt.embedded, rebuilt.cache_hits) == (0, len(second))
    assert fake.texts - calls_before == len(new)


class Gate:
    """Blocks the build thread inside a step until the test lets it go (or fails it)."""

    def __init__(self, fail: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    def wait(self) -> None:
        self.entered.set()
        assert self.release.wait(30), "the test never released the build"
        if self.fail:
            raise RuntimeError("embedding provider down")


def run_build(db_path, profile_name, embed_fn=None, pages_for=None, **kw):
    """Build on its own connection in a thread; returns (thread, outcome dict)."""
    outcome: dict = {}

    def target() -> None:
        conn = open_db(db_path)
        try:
            args = {"pages_for": pages_for} if pages_for else {}
            outcome["report"] = build_profile(
                conn, parse_profile_name(profile_name),
                EmbeddingService(conn, embed_fn=embed_fn or FakeEmbedder()), **args, **kw,
            )  # fmt: skip
        except BaseException as error:
            outcome["error"] = error
        finally:
            conn.close()

    thread = threading.Thread(target=target)
    thread.start()
    return thread, outcome


def gated_embedder(gate: Gate):
    fake = FakeEmbedder()

    def embed(texts, model, dimensions=None, **kw):
        gate.wait()
        return fake(texts, model, dimensions, **kw)

    return embed


def test_queries_keep_hitting_the_old_profile_until_the_new_one_is_complete(
    conn, db_path, make_pipeline
):
    ingest_all(conn, make_pipeline(), [corpus.INSPECTION, corpus.TEXT])
    old = active_profile(conn)
    question = "water heater relief valve"
    answer_before = keyword_search(conn, old, question, k=5)

    gate = Gate()
    thread, outcome = run_build(db_path, OTHER, embed_fn=gated_embedder(gate), activate=True)
    assert gate.entered.wait(30)
    # mid-build: the old profile serves, the new one can't be activated
    assert active_profile(conn).name == PROFILE
    assert keyword_search(conn, active_profile(conn), question, k=5) == answer_before
    assert profile_status(conn, OTHER) == "building"
    with pytest.raises(ProfileNotReady):
        activate_profile(conn, OTHER)
    gate.release.set()
    thread.join(30)
    assert "error" not in outcome, outcome.get("error")
    now = active_profile(conn)
    assert now.name == OTHER and now.active
    assert not get_profile(conn, PROFILE).active
    assert conn.execute("SELECT COUNT(*) FROM profiles WHERE active = 1").fetchone()[0] == 1


def test_a_failed_build_leaves_the_old_profile_active(conn, db_path, make_pipeline):
    ingest_all(conn, make_pipeline(), [corpus.INSPECTION])
    gate = Gate(fail=True)
    thread, outcome = run_build(db_path, OTHER, embed_fn=gated_embedder(gate), activate=True)
    assert gate.entered.wait(30)
    gate.release.set()
    thread.join(30)
    assert isinstance(outcome.get("error"), RuntimeError)
    assert active_profile(conn).name == PROFILE
    failed = get_profile(conn, OTHER)
    assert profile_status(conn, OTHER) == "failed"
    assert f"vec_{failed.id}" not in tables(conn) and f"fts_{failed.id}" not in tables(conn)
    assert (
        conn.execute("SELECT COUNT(*) FROM chunks WHERE profile_id = ?", (failed.id,)).fetchone()[0]
        == 0
    )
    # a later build of the failed profile starts clean and succeeds
    report = build_profile(
        conn, parse_profile_name(OTHER), EmbeddingService(conn, embed_fn=FakeEmbedder())
    )
    assert profile_status(conn, OTHER) == "ready" and report.chunks > 0


def test_rebuilding_the_active_profile_swaps_it_in_atomically(conn, db_path, make_pipeline):
    ingest_all(conn, make_pipeline(), [corpus.INSPECTION, corpus.ESTIMATE])
    old = active_profile(conn)
    chunks_before = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE profile_id = ?", (old.id,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO abstain_thresholds (profile_id, reranker, threshold, calibrated_at) "
        "VALUES (?, '', 0.42, 'now')",
        (old.id,),
    )
    conn.commit()
    question = "plumbing estimate total"
    before = [h.score for h in keyword_search(conn, old, question, k=5)]

    gate = Gate()
    from hal9000.knowledge.index.profiles import stored_pages

    def pages_for(c, file, parser):
        gate.wait()
        return stored_pages(c, file, parser)

    thread, outcome = run_build(db_path, PROFILE, pages_for=pages_for, rebuild=True)
    assert gate.entered.wait(30)
    assert active_profile(conn).id == old.id  # still the old build
    assert [h.score for h in keyword_search(conn, active_profile(conn), question, k=5)] == before
    assert [p.name for p in list_profiles(conn)] == [PROFILE]  # the shadow build is hidden
    gate.release.set()
    thread.join(30)
    assert "error" not in outcome, outcome.get("error")

    new = active_profile(conn)
    assert new.name == PROFILE and new.id != old.id
    assert f"vec_{old.id}" not in tables(conn) and f"fts_{old.id}" not in tables(conn)
    assert (
        conn.execute("SELECT COUNT(*) FROM chunks WHERE profile_id = ?", (new.id,)).fetchone()[0]
        == chunks_before
    )
    assert conn.execute("SELECT profile_id, threshold FROM abstain_thresholds").fetchall()[0][
        :
    ] == (new.id, 0.42)
    assert [h.score for h in keyword_search(conn, new, question, k=5)] == before


def test_activation_rules(conn):
    with pytest.raises(ProfileError, match="no index profile"):
        activate_profile(conn, PROFILE)
    stored = ensure_profile(conn, parse_profile_name(PROFILE))
    assert profile_status(conn, PROFILE) == "building"
    with pytest.raises(ProfileNotReady):
        activate_profile(conn, PROFILE)
    conn.execute("UPDATE profiles SET status = 'ready' WHERE id = ?", (stored.id,))
    conn.commit()
    assert activate_profile(conn, PROFILE).active
    assert ensure_profile(conn, parse_profile_name(PROFILE)).id == stored.id
    changed = parse_profile_name(PROFILE).model_copy(update={"size": 300})
    with pytest.raises(ProfileError, match="different settings"):
        ensure_profile(conn, changed)
    # a profile built over an empty library is complete at once
    report = build_profile(conn, parse_profile_name(OTHER), EmbeddingService(conn), activate=True)
    assert report.files == 0 and active_profile(conn).name == OTHER


def test_building_needs_stored_pages_or_a_parser(conn, make_pipeline):
    pipeline = make_pipeline()
    ingest_all(conn, pipeline, [corpus.ESTIMATE])
    conn.execute("DELETE FROM pages")
    conn.commit()
    with pytest.raises(ProfileError, match="no pdfium pages"):
        build_profile(
            conn, parse_profile_name(OTHER), EmbeddingService(conn, embed_fn=FakeEmbedder())
        )
    # the pipeline's page source re-parses the stored original instead
    report = pipeline.build(conn, OTHER)
    assert report.chunks > 0
    assert conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == corpus.ESTIMATE.page_count
