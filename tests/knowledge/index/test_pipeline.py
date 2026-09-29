"""The ingestion pipeline (plan §5.1 steps 2–10) end to end on the synthetic fixture corpus."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hal9000.knowledge.index.library import (
    documents_for,
    get_file,
    is_superseded,
    latest_files,
    load_pages,
)
from hal9000.knowledge.index.profiles import active_profile, get_profile
from hal9000.knowledge.index.search import keyword_search
from hal9000.knowledge.ingest.pipeline import (
    IngestPipeline,
    default_collection,
    normalize_relative_path,
    screen,
)
from hal9000.knowledge.obs.timing import INGEST_STAGES, REGISTRY
from hal9000.knowledge.storage import fts_table, vec_table
from tests.knowledge.fixtures import corpus
from tests.knowledge.index.conftest import PROFILE, ingest_all, table_counts


def stage_count(name: str) -> float:
    return REGISTRY.get_sample_value("kb_stage_seconds_count", {"stage": name}) or 0.0


def files_total(status: str) -> float:
    return REGISTRY.get_sample_value("kb_ingest_files_total", {"status": status}) or 0.0


# ------------------------------------------------------------------ the corpus
def test_the_whole_fixture_corpus_ingests_under_five_seconds(conn, make_pipeline, fake):
    constructed = time.perf_counter()
    # the default configuration: parsing isolated with a time limit; the pipeline warms up
    # when it is built (the parse process, the token encoding), as the service does at start
    pipeline = make_pipeline()
    assert pipeline.isolate_parse and pipeline.parse_process.starts == 1
    before = {name: stage_count(name) for name in INGEST_STAGES}
    started = time.perf_counter()
    jobs = ingest_all(conn, pipeline)
    elapsed = time.perf_counter() - started
    print(
        f"fixture corpus ingested in {elapsed:.2f} s "
        f"(pipeline start-up {started - constructed:.2f} s before that)"
    )
    assert [job.status for job in jobs] == ["done"] * len(corpus.ALL_FILES), [j.error for j in jobs]
    assert elapsed < 5.0, f"ingesting the fixture corpus took {elapsed:.2f} s"

    profile = active_profile(conn)
    assert profile is not None and profile.name == PROFILE
    counts = table_counts(conn)
    assert counts["files"] == len(corpus.ALL_FILES)
    expected_docs = sum(len(item.documents) for item in corpus.ALL_FILES)
    assert counts["documents"] == expected_docs
    assert counts["pages"] == sum(item.page_count for item in corpus.ALL_FILES)
    assert counts[fts_table(profile.id)] == counts[vec_table(profile.id)] == counts["chunks"] > 0
    # every ingestion stage was timed under its plan name (rule K7)
    for name in (
        "screen",
        "store",
        "parse",
        "clean",
        "segment",
        "chunk",
        "embed",
        "index",
        "regression",
    ):
        assert stage_count(name) > before[name], name
    # each distinct chunk text was embedded once (the bundle repeats three files)
    assert fake.texts == len(
        {row[0] for row in conn.execute("SELECT header || char(10) || text FROM chunks")}
    )


def test_documents_pages_and_chunks_match_the_ground_truth(conn, make_pipeline):
    pipeline = make_pipeline()
    ingest_all(conn, pipeline, [corpus.BUNDLE, corpus.MARKDOWN])
    bundle = get_file(conn, _sha(corpus.BUNDLE))
    docs = documents_for(conn, bundle.sha256)
    assert [(d.page_start, d.page_end, d.doc_type) for d in docs] == [
        (d.page_start, d.page_end, d.doc_type) for d in corpus.BUNDLE.documents
    ]
    pages = load_pages(conn, bundle.sha256, "pdfium")
    assert [p.page for p in pages] == list(range(1, 16))
    assert [p.page for p in pages if not p.has_text_layer] == list(corpus.BUNDLE.no_text_pages)
    for line in corpus.RUNNING_LINES:  # cleaned text is what was stored and chunked
        assert not any(line in p.text for p in pages)
    chunks = conn.execute(
        "SELECT c.document_id, c.page_start, c.page_end, c.header FROM chunks c "
        "JOIN documents d ON d.id = c.document_id WHERE d.file_sha = ?",
        (bundle.sha256,),
    ).fetchall()
    by_doc = {d.id: d for d in docs}
    for document_id, start, end, header in chunks:
        doc = by_doc[document_id]
        assert doc.page_start <= start <= end <= doc.page_end
        assert doc.title in header  # the contextual header is on (".hdr.")
    markdown = get_file(conn, _sha(corpus.MARKDOWN))
    assert len(load_pages(conn, markdown.sha256, "text")) == corpus.MARKDOWN.page_count


def _sha(item) -> str:
    import hashlib

    return hashlib.sha256(item.path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ duplicates
def test_reingesting_identical_bytes_is_a_duplicate_with_no_rows_and_no_embedding(
    conn, make_pipeline, fake
):
    pipeline = make_pipeline()
    ingest_all(conn, pipeline, [corpus.INVOICE, corpus.ESTIMATE])
    before, calls = table_counts(conn), fake.calls
    duplicates = files_total("duplicate")

    again = pipeline.submit(conn, corpus.INVOICE.path, "Elsewhere/renamed.pdf")
    as_bytes = pipeline.submit(conn, corpus.ESTIMATE.path.read_bytes(), "Docs/estimate.pdf")

    assert again.status == as_bytes.status == "duplicate"
    assert again.reason == "already in library as Docs/invoice.pdf"
    assert again.job_id is None
    assert pipeline.worker(":unused:").run_pending(conn) == []
    assert table_counts(conn) == before
    assert fake.calls - calls == 0
    assert files_total("duplicate") == duplicates + 2


def test_an_unfinished_file_uploaded_again_is_retried_not_called_a_duplicate(conn, make_pipeline):
    pipeline = make_pipeline()
    result = pipeline.submit(conn, corpus.TEXT.path, "pool_log.txt")
    conn.execute("UPDATE jobs SET status = 'failed', error = 'boom' WHERE id = ?", (result.job_id,))
    conn.commit()
    again = pipeline.submit(conn, corpus.TEXT.path, "pool_log.txt")
    assert again.status == "added" and again.job_id == result.job_id
    assert "retrying" in again.reason
    assert pipeline.worker(":unused:").run_job(conn, again.job_id).status == "done"
    assert pipeline.submit(conn, corpus.TEXT.path, "pool_log.txt").status == "duplicate"


# --------------------------------------------------------------------- screen
ELF = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\x3e\x00"
PDF_BYTES = corpus.INVOICE.path.read_bytes()


@pytest.mark.parametrize(
    ("data", "name", "status", "reason"),
    [
        (b"\xff\xd8\xff\xe0 jpeg", "photos/IMG_1.jpg", "skipped", ".jpg is not a supported type"),
        (b"no extension", "README", "skipped", "files without an extension"),
        (b"hello", "fake.pdf", "failed", "does not start with %PDF-"),
        (b"see page 3 %PDF-1.7", "late.pdf", "failed", "does not start with %PDF-"),
        (b"PK\x03\x04 not really a zip", "fake.docx", "failed", "not a Word document"),
        (b"just text", "fake.docx", "failed", "not a Word document"),
        (b"text\x00with nul", "notes.txt", "failed", "binary data"),
        (ELF, "program.txt", "failed", "binary data"),
        (bytes(range(256)) * 4, "blob.md", "failed", "binary data"),
        (b"caf\xe9 au lait", "latin1.txt", "failed", "not UTF-8"),
        (b"%PDF-1.7\n<html>not really</html>", "trick.html", "failed", "PDF or ZIP"),
        (b"%PDF-1.4 but called text", "trick.md", "failed", "PDF or ZIP"),
        (PDF_BYTES, "invoice.html", "failed", "not a HTML file"),
        (b"plain words, no markup", "page.html", "failed", "does not start with <"),
        (b"", "empty.md", "failed", "empty"),
    ],
)
def test_screening_refuses_what_is_not_on_the_allow_list(
    conn, make_pipeline, data, name, status, reason
):
    pipeline = make_pipeline()
    before = files_total(status)
    result = pipeline.submit(conn, data, name)
    assert result.status == status and reason in result.reason, result
    assert result.job_id is None and result.sha256 is None
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert files_total(status) == before + 1


@pytest.mark.parametrize(
    ("data", "name"),
    [
        (b"\xef\xbb\xbf \n%PDF-1.7\n" + PDF_BYTES[8:], "bom.pdf"),
        (b"\xef\xbb\xbf\n  <!doctype html><p>Filter: every 90 days</p>", "bom.html"),
        ("Café notes\tline\r\n\x0cnext page\n".encode(), "notes.txt"),
    ],
)
def test_screening_accepts_a_bom_whitespace_and_ordinary_text(data, name):
    assert screen(data, name, max_bytes=10**8).status == "ok"


def test_type_and_size_are_checked_before_reading(tmp_path, monkeypatch):
    from hal9000.knowledge.ingest import pipeline as module

    def no_reading(source):
        raise AssertionError("the content was read")

    monkeypatch.setattr(module, "_chunks", no_reading)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.7\n" + b"0" * 2000)
    verdict = screen(big, "big.pdf", max_bytes=1000)
    assert verdict.status == "failed" and "too large" in verdict.reason
    assert screen(big, "big.exe", max_bytes=10**6).status == "skipped"


def test_every_allowed_type_passes_screening():
    for item in corpus.ALL_FILES:
        verdict = screen(item.path, item.name, max_bytes=10**8)
        assert verdict.status == "ok", (item.name, verdict.reason)
        assert verdict.mime_type == item.mime_type
        assert verdict.sha256 == _sha(item)


def test_relative_paths_are_normalised_and_give_the_collection(conn, make_pipeline, tmp_path):
    assert normalize_relative_path("../../etc/Appliances\\Furnace/./manual.pdf") == (
        "etc/Appliances/Furnace/manual.pdf"
    )
    assert default_collection("Appliances/Furnace/manual.pdf") == "Appliances/Furnace/"
    assert default_collection("manual.pdf") == ""
    for bad in ("../", "a\x00b.pdf", "tab\there.md", "del\x7f.txt"):
        with pytest.raises(ValueError):
            normalize_relative_path(bad)
    pipeline = make_pipeline()
    result = pipeline.submit(conn, corpus.MARKDOWN.path, "/Garage/Door/garage_door.md")
    assert (result.relative_path, result.collection) == (
        "Garage/Door/garage_door.md",
        "Garage/Door/",
    )
    named = pipeline.submit(
        conn, corpus.TEXT.path, "Pool/pool_log.txt", collection=" Outdoor\\Pool "
    )
    assert named.collection == "Outdoor/Pool/"
    assert get_file(conn, named.sha256).collection == "Outdoor/Pool/"
    stored = tmp_path / "files" / named.sha256  # stored by hash, not by the user's path
    assert stored.read_bytes() == corpus.TEXT.path.read_bytes()
    assert pipeline.submit(conn, b"x", "../").status == "failed"
    nul = pipeline.submit(conn, b"# Notes\n", "evil\x00name.md")
    assert nul.status == "failed" and "control characters" in nul.reason
    assert "\x00" not in nul.relative_path
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 2


def test_a_long_multibyte_name_is_parsed_under_a_safe_link_name(conn, make_pipeline):
    name = "é" * 147 + ".txt"  # 151 characters, 298 bytes: too long for one file name
    assert len(name) == 151 and len(name.encode()) > 255
    pipeline = make_pipeline()
    result = pipeline.submit(conn, corpus.TEXT.path, f"Pool/{name}")
    assert result.status == "added"
    job = pipeline.worker(":unused:").run_job(conn, result.job_id)
    assert job.status == "done", job.error
    assert get_file(conn, result.sha256).original_name == name


# ------------------------------------------------------------------- versions
def _chunks_of(conn, sha: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.file_sha = ?",
        (sha,),
    ).fetchone()[0]


def _searchable(conn, text: str) -> bool:
    profile = active_profile(conn)
    return any(
        text in conn.execute("SELECT text FROM chunks WHERE id = ?", (h.chunk_id,)).fetchone()[0]
        for h in keyword_search(conn, profile, text, k=50)
    )


V1 = b"# Pool\n\nThe pump runs six hours a day in summer.\n"
V2 = b"# Pool\n\nThe pump runs eight hours a day in summer.\n"


def test_a_new_version_supersedes_the_old_once_it_is_indexed(conn, make_pipeline):
    pipeline = make_pipeline()
    worker = pipeline.worker(":unused:")
    first = pipeline.submit(conn, V1, "Pool/notes.md")
    worker.run_job(conn, first.job_id)
    second = pipeline.submit(conn, V2, "Pool/notes.md")
    assert second.status == "added" and second.reason == "new version of an earlier file"
    assert get_file(conn, second.sha256).version_of == first.sha256

    # until the new version is indexed, the old one stays searchable
    assert _searchable(conn, "six") and not is_superseded(conn, first.sha256)
    assert [f.sha256 for f in latest_files(conn)] == [second.sha256]

    job = worker.run_job(conn, second.job_id)
    assert job.status == "done" and "replaces 1 earlier version" in job.detail
    assert is_superseded(conn, first.sha256) and not is_superseded(conn, second.sha256)
    assert _chunks_of(conn, first.sha256) == 0 and _chunks_of(conn, second.sha256) > 0
    assert _searchable(conn, "eight") and not _searchable(conn, "six")
    profile = active_profile(conn)
    vec_ids = {r[0] for r in conn.execute(f"SELECT chunk_id FROM {vec_table(profile.id)}")}
    fts_ids = {r[0] for r in conn.execute(f"SELECT rowid FROM {fts_table(profile.id)}")}
    chunk_ids = {r[0] for r in conn.execute("SELECT id FROM chunks")}
    assert vec_ids == fts_ids == chunk_ids  # no orphaned index rows
    # the old version's rows stay, ready to be restored
    assert get_file(conn, first.sha256) is not None
    assert load_pages(conn, first.sha256, "text") and documents_for(conn, first.sha256)
    assert [f.sha256 for f in latest_files(conn)] == [second.sha256]
    # re-uploading the current bytes is still a duplicate
    assert pipeline.submit(conn, V2, "Pool/notes.md").status == "duplicate"


def test_a_failed_new_version_leaves_the_old_one_searchable(conn, make_pipeline):
    pipeline = make_pipeline()
    worker = pipeline.worker(":unused:")
    worker.run_job(conn, pipeline.submit(conn, corpus.INVOICE.path, "Bills/invoice.pdf").job_id)
    broken = pipeline.submit(conn, b"%PDF-1.7\nnot really\n", "Bills/invoice.pdf")
    assert worker.run_job(conn, broken.job_id).status == "failed"
    assert _chunks_of(conn, _sha(corpus.INVOICE)) > 0 and _searchable(conn, "BE-10442")
    assert not is_superseded(conn, _sha(corpus.INVOICE))
    assert [f.sha256 for f in latest_files(conn)] == [broken.sha256]  # shown, with its error


def test_reuploading_an_old_version_restores_it(conn, make_pipeline):
    pipeline = make_pipeline()
    worker = pipeline.worker(":unused:")
    first = pipeline.submit(conn, V1, "Pool/notes.md")
    worker.run_job(conn, first.job_id)
    second = pipeline.submit(conn, V2, "Pool/notes.md")
    worker.run_job(conn, second.job_id)

    back = pipeline.submit(conn, V1, "Pool/notes.md")
    assert back.status == "added" and "restoring an earlier version" in back.reason
    assert back.sha256 == first.sha256
    assert _searchable(conn, "eight")  # the current version serves until the restore is indexed
    assert worker.run_job(conn, back.job_id).status == "done"
    assert not is_superseded(conn, first.sha256) and is_superseded(conn, second.sha256)
    assert _searchable(conn, "six") and not _searchable(conn, "eight")
    assert [f.sha256 for f in latest_files(conn)] == [first.sha256]
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 2
    assert pipeline.submit(conn, V1, "Pool/notes.md").status == "duplicate"


def test_profile_builds_skip_superseded_versions(conn, make_pipeline):
    pipeline = make_pipeline()
    worker = pipeline.worker(":unused:")
    worker.run_job(conn, pipeline.submit(conn, V1, "Pool/notes.md").job_id)
    worker.run_job(conn, pipeline.submit(conn, V2, "Pool/notes.md").job_id)
    report = pipeline.build(conn, "pdfium.fixed-64.hdr.fake-64")
    assert report.files == 1
    other = get_profile(conn, "pdfium.fixed-64.hdr.fake-64")
    texts = [
        r[0] for r in conn.execute("SELECT text FROM chunks WHERE profile_id = ?", (other.id,))
    ]
    assert any("eight" in t for t in texts) and not any("six" in t for t in texts)


# ------------------------------------------------------------ atomicity
def test_the_index_step_is_atomic(conn, make_pipeline, monkeypatch):
    pipeline = make_pipeline()
    result = pipeline.submit(conn, corpus.ESTIMATE.path, "estimate.pdf")
    pipeline.worker(":unused:").run_job(conn, result.job_id)
    before = table_counts(conn)
    documents_before = documents_for(conn, result.sha256)

    from hal9000.knowledge.index.vector import SqliteVecIndex

    def broken_add(self, rows):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SqliteVecIndex, "add", broken_add)
    with pytest.raises(sqlite3.OperationalError):
        pipeline.process(conn, result.sha256)
    # the earlier delete of this file's rows and the new inserts were rolled back together
    assert not conn.in_transaction
    assert table_counts(conn) == before
    assert documents_for(conn, result.sha256) == documents_before
    assert _searchable(conn, "estimate")


# --------------------------------------------------------------- parsing
def test_one_parse_process_serves_every_file(conn, make_pipeline):
    pipeline = make_pipeline()
    ingest_all(conn, pipeline, [corpus.INVOICE, corpus.ESTIMATE, corpus.DOCX, corpus.HTML])
    assert pipeline.parse_process.starts == 1


class SlowParser:
    """Picklable; sleeps in the parse process to trip the time limit."""

    name = "text"

    def parse(self, path):
        import time as _time

        _time.sleep(30)


def test_a_parse_over_its_time_limit_kills_and_restarts_the_process(tmp_path):
    from hal9000.knowledge.ingest.parsers import ParseTimeout, get_parser
    from hal9000.knowledge.ingest.pipeline import ParseProcess

    process = ParseProcess()
    try:
        started = time.perf_counter()
        with pytest.raises(ParseTimeout, match="longer than 1 s"):
            process.parse(corpus.TEXT.path, SlowParser(), timeout_s=1, max_chars=3000)
        assert time.perf_counter() - started < 10
        parsed = process.parse(corpus.TEXT.path, get_parser("text"), timeout_s=30, max_chars=3000)
        assert parsed.page_count == corpus.TEXT.page_count
        assert process.starts == 2
    finally:
        process.close()


# ------------------------------------------------------------ segmentation
def _llm_reply(documents: list[dict]) -> str:
    return json.dumps({"tool": "record_documents", "args": {"documents": documents}})


def test_llm_labels_are_cached_and_user_edits_survive_reingestion(conn, make_pipeline):
    calls: list[str] = []

    def generate(prompt, tier):
        calls.append(tier)
        return _llm_reply(
            [{"page_start": 1, "page_end": 1, "doc_type": "invoice", "title": "Electrical invoice",
              "vendor": "Brightline Electric", "doc_date": "2025-03-20"}]
        )  # fmt: skip

    pipeline = make_pipeline(generate=generate, use_llm=True)
    result = pipeline.submit(conn, corpus.INVOICE.path, "invoice.pdf")
    job = pipeline.worker(":unused:").run_job(conn, result.job_id)
    assert job.status == "done", job.error
    assert calls == ["fast"]
    [doc] = documents_for(conn, result.sha256)
    assert (doc.title, doc.meta_source) == ("Electrical invoice", "auto")

    # the user fixes the title in the Library, then the file is processed again
    conn.execute(
        "UPDATE documents SET title = 'Panel repair invoice', meta_source = 'user' WHERE id = ?",
        (doc.id,),
    )
    conn.commit()
    outcome = pipeline.process(conn, result.sha256, use_llm=True)
    [kept] = documents_for(conn, result.sha256)
    assert (kept.id, kept.title, kept.meta_source) == (doc.id, "Panel repair invoice", "user")
    assert outcome.segmentation == "user" and calls == ["fast"]
    headers = [row[0] for row in conn.execute("SELECT header FROM chunks")]
    assert headers and all("Panel repair invoice" in h for h in headers)

    # without the user edit, a re-run takes the labels from the persistent cache: no call
    conn.execute("UPDATE documents SET meta_source = 'auto'")
    conn.commit()
    outcome = pipeline.process(conn, result.sha256, use_llm=True)
    assert outcome.segmentation == "cache" and calls == ["fast"]
    assert [d.title for d in documents_for(conn, result.sha256)] == ["Electrical invoice"]
    assert conn.execute("SELECT COUNT(*) FROM segment_cache").fetchone()[0] == 1


def test_no_llm_uses_heuristics_and_never_calls(conn, make_pipeline):
    def generate(prompt, tier):  # pragma: no cover - must not be called
        raise AssertionError("--no-llm must not call the LLM")

    pipeline = make_pipeline(generate=generate, use_llm=True)
    result = pipeline.submit(conn, corpus.UNNUMBERED.path, "u.pdf", use_llm=False)
    job = pipeline.worker(":unused:").run_job(conn, result.job_id)
    assert job.status == "done"
    assert len(documents_for(conn, result.sha256)) == len(corpus.UNNUMBERED.documents)


# --------------------------------------------------------------- the profile
def test_parent_child_parents_are_stored_first_and_never_searched(conn, make_pipeline, fake):
    pipeline = make_pipeline(active="pdfium.parent_child-64-p256.hdr.fake-64")
    ingest_all(conn, pipeline, [corpus.INSPECTION])
    profile = active_profile(conn)
    rows = conn.execute("SELECT id, parent_id FROM chunks ORDER BY id").fetchall()
    parents = {cid for cid, parent in rows if parent is None}
    children = [(cid, parent) for cid, parent in rows if parent is not None]
    assert parents and children
    assert all(parent in parents and parent < cid for cid, parent in children)
    indexed = {row[0] for row in conn.execute(f"SELECT rowid FROM {fts_table(profile.id)}")}
    vectors = {row[0] for row in conn.execute(f"SELECT chunk_id FROM {vec_table(profile.id)}")}
    assert indexed == vectors == {cid for cid, _ in children}
    distinct_children = conn.execute(
        "SELECT COUNT(DISTINCT header || char(10) || text) FROM chunks WHERE parent_id IS NOT NULL"
    ).fetchone()[0]
    assert fake.texts == distinct_children  # parents were never embedded


def test_without_an_active_profile_files_are_stored_but_not_chunked(conn, make_pipeline, fake):
    pipeline = make_pipeline(active=None)
    [job] = ingest_all(conn, pipeline, [corpus.ESTIMATE])
    assert job.status == "done" and "not chunked yet" in job.detail
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert fake.calls == 0 and active_profile(conn) is None
    # a profile build indexes what was stored, from stored pages
    report = pipeline.build(conn, PROFILE, activate=True)
    assert report.files == 1 and report.chunks > 0
    assert active_profile(conn).name == PROFILE


def test_extra_profiles_are_indexed_too(conn, make_pipeline):
    extra = "pdfium.fixed-64.nohdr.fake-64"
    pipeline = make_pipeline(extra_profiles=[extra])
    [job] = ingest_all(conn, pipeline, [corpus.INSPECTION])
    assert job.status == "done"
    active, other = get_profile(conn, PROFILE), get_profile(conn, extra)
    assert active.active and not other.active
    counts = dict(
        conn.execute("SELECT profile_id, COUNT(*) FROM chunks GROUP BY profile_id").fetchall()
    )
    assert counts[other.id] > counts[active.id] > 0
    assert pipeline.outcomes[job.id].chunks == {PROFILE: counts[active.id], extra: counts[other.id]}


def test_the_regression_hook_runs_after_ingestion_and_cannot_fail_it(conn, make_pipeline):
    seen = []
    pipeline = make_pipeline(regression=lambda c, profile: seen.append(profile.name))
    ingest_all(conn, pipeline, [corpus.INVOICE])
    assert seen == [PROFILE]

    def broken(c, profile):
        raise RuntimeError("scorer down")

    pipeline = make_pipeline(regression=broken)
    [job] = ingest_all(conn, pipeline, [corpus.ESTIMATE])
    assert job.status == "done"
    assert pipeline.outcomes[job.id].warnings == ["regression check failed: RuntimeError"]


def test_a_parse_failure_fails_the_job_with_the_original_name(conn, make_pipeline):
    pipeline = make_pipeline()
    broken = b"%PDF-1.7\nthis is not really a pdf\n%%EOF\n"
    result = pipeline.submit(conn, broken, "Scans/broken.pdf")
    failed = files_total("failed")
    job = pipeline.worker(":unused:").run_job(conn, result.job_id)
    assert job.status == "failed"
    assert "broken.pdf" in job.error and result.sha256 not in job.error
    assert files_total("failed") == failed + 1
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_a_missing_original_fails_clearly(conn, make_pipeline, tmp_path):
    pipeline = make_pipeline()
    result = pipeline.submit(conn, corpus.TEXT.path, "pool_log.txt")
    (tmp_path / "files" / result.sha256).unlink()
    job = pipeline.worker(":unused:").run_job(conn, result.job_id)
    assert job.status == "failed" and "stored original is missing" in job.error


def test_the_pipeline_reads_its_limits_from_knowledge_yaml(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "knowledge.yaml").write_text("ingest:\n  max_upload_bytes: 10\n")
    monkeypatch.setenv("HAL_CONFIG_DIR", str(config))
    pipeline = IngestPipeline(tmp_path / "files")
    assert pipeline.max_bytes == 10


def test_uploading_again_recovers_a_job_left_running_by_a_dead_process(conn, make_pipeline):
    import subprocess
    import sys

    from hal9000.knowledge.ingest.jobs import worker_id

    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    pipeline = make_pipeline()
    first = pipeline.submit(conn, corpus.TEXT.path, "pool_log.txt")
    owner = f"{worker_id().rpartition(':')[0]}:{dead}"
    conn.execute(
        "UPDATE jobs SET status = 'running', params = json_set(params, '$._worker', ?) WHERE id = ?",
        (owner, first.job_id),
    )
    conn.commit()
    again = pipeline.submit(conn, corpus.TEXT.path, "pool_log.txt")
    assert again.status == "added" and again.job_id == first.job_id and "retrying" in again.reason
    assert pipeline.worker(":unused:").run_job(conn, again.job_id).status == "done"
