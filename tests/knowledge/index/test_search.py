"""Vector and BM25 search (plan §7.1): exactness, filters inside the KNN, per-profile BM25."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from hal9000.knowledge.chunking import parse_profile_name
from hal9000.knowledge.contracts import DocumentMeta, FileRecord, SearchFilters
from hal9000.knowledge.index.fake import FakeEmbedder
from hal9000.knowledge.index.fusion import rrf
from hal9000.knowledge.index.keyword import FtsIndex
from hal9000.knowledge.index.library import insert_documents, insert_file
from hal9000.knowledge.index.profiles import active_profile, ensure_profile, get_profile
from hal9000.knowledge.index.search import as_scored, keyword_search, vector_search
from hal9000.knowledge.index.vector import ChunkMeta, SqliteVecIndex, VectorIndex, VectorRow
from hal9000.knowledge.obs.timing import start_trace
from tests.knowledge.fixtures import corpus
from tests.knowledge.index.conftest import PROFILE, ingest_all

DIMS = 16
TYPES = ("invoice", "estimate", "manual", "inspection_report")
COLLECTIONS = ("Appliances/", "Pool/", "")
DATES = ("2021-05-01", "2023-06-22", "2024-11-30", "")


def build_synthetic(conn, n: int = 240, seed: int = 7, collections=COLLECTIONS):
    """``n`` chunks with random unit vectors and cycling metadata, in one stored profile."""
    profile = ensure_profile(conn, parse_profile_name("pdfium.fixed-64.nohdr.fake-64").model_copy(
        update={"name": "synthetic", "dimensions": DIMS}
    ))  # fmt: skip
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, DIMS)).astype(np.float32)
    metas: dict[int, ChunkMeta] = {}
    shas = [f"{i:064x}" for i in range(1, len(collections) + 1)]  # one file per collection
    with conn:
        for sha, collection in zip(shas, collections):
            insert_file(conn, FileRecord(sha256=sha, original_name="s.txt", relative_path="s.txt",
                                         collection=collection, mime_type="text/plain", bytes=1))  # fmt: skip
        index = SqliteVecIndex(conn, profile.id, DIMS)
        fts = FtsIndex(conn, profile.id)
        rows = []
        for i in range(n):
            which = i % len(collections)
            meta = ChunkMeta(TYPES[i % 4], collections[which], DATES[i % 4])
            [doc] = insert_documents(conn, [DocumentMeta(
                file_sha=shas[which], page_start=1, page_end=1, title=f"doc {i}", doc_type=meta.doc_type,
                doc_date=date.fromisoformat(meta.doc_date) if meta.doc_date else None,
            )])  # fmt: skip
            cid = conn.execute(
                "INSERT INTO chunks (profile_id, document_id, page_start, page_end, char_start, "
                "char_end, text, header, tokens) VALUES (?, ?, 1, 1, 0, 1, ?, '', 1)",
                (profile.id, doc.id, f"common word chunk{i}"),
            ).lastrowid
            metas[cid] = meta
            rows.append(VectorRow(cid, vectors[i], meta))
        index.add(rows)
        fts.add((cid, "", f"common word chunk{cid}") for cid in metas)
    return profile, index, vectors, metas


def brute_force(vectors: np.ndarray, ids: list[int], query: np.ndarray, k: int):
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ (query / np.linalg.norm(query))
    order = np.argsort(-sims, kind="stable")[:k]
    return [ids[i] for i in order], [float(sims[i]) for i in order]


def test_knn_equals_a_numpy_brute_force(kb_db):
    profile, index, vectors, metas = build_synthetic(kb_db)
    assert isinstance(index, VectorIndex) and index.count() == len(metas)
    ids = list(metas)
    rng = np.random.default_rng(1)
    for _ in range(5):
        query = rng.standard_normal(DIMS)
        hits = vector_search(kb_db, profile, query, k=25)
        want_ids, want_sims = brute_force(vectors, ids, query, 25)
        assert [h.chunk_id for h in hits] == want_ids
        assert [h.similarity for h in hits] == pytest.approx(want_sims, abs=1e-5)
        assert [h.rank for h in hits] == list(range(1, 26))


def test_a_filtered_vector_search_still_returns_k_results(kb_db):
    profile, _, vectors, metas = build_synthetic(kb_db)
    invoices = [cid for cid, meta in metas.items() if meta.doc_type == "invoice"]
    assert len(invoices) >= 30
    query = np.random.default_rng(3).standard_normal(DIMS)
    hits = vector_search(kb_db, profile, query, k=30, filters=SearchFilters(doc_type="invoice"))
    assert len(hits) == 30
    assert {metas[h.chunk_id].doc_type for h in hits} == {"invoice"}
    # exactly the 30 nearest *invoices*: the filter is inside the KNN, not after it
    subset = [list(metas).index(cid) for cid in invoices]
    want, _ = brute_force(vectors[subset], invoices, query, 30)
    assert [h.chunk_id for h in hits] == want


def test_collection_and_date_filters_apply_to_vectors(kb_db):
    profile, _, _, metas = build_synthetic(kb_db)
    query = np.ones(DIMS)
    pool = vector_search(kb_db, profile, query, k=500, filters=SearchFilters(collection="Pool"))
    assert len(pool) == sum(1 for m in metas.values() if m.collection == "Pool/")
    assert {metas[h.chunk_id].collection for h in pool} == {"Pool/"}
    window = SearchFilters(date_from=date(2022, 1, 1), date_to=date(2024, 1, 1))
    dated = vector_search(kb_db, profile, query, k=500, filters=window)
    assert {metas[h.chunk_id].doc_date for h in dated} == {"2023-06-22"}
    until = vector_search(
        kb_db, profile, query, k=500, filters=SearchFilters(date_to=date(2022, 1, 1))
    )
    assert {metas[h.chunk_id].doc_date for h in until} == {"2021-05-01"}  # unknown dates excluded
    since = vector_search(
        kb_db, profile, query, k=500, filters=SearchFilters(date_from=date(2024, 1, 1))
    )
    assert {metas[h.chunk_id].doc_date for h in since} == {"2024-11-30"}
    assert vector_search(kb_db, profile, query, k=0) == []


def test_filters_restrict_bm25_too(kb_db):
    profile, _, _, metas = build_synthetic(kb_db)
    everything = keyword_search(kb_db, profile, "common", k=500)
    assert len(everything) == len(metas)
    manuals = keyword_search(
        kb_db, profile, "common", k=500, filters=SearchFilters(doc_type="manual")
    )
    assert manuals and {metas[h.chunk_id].doc_type for h in manuals} == {"manual"}
    dated = keyword_search(kb_db, profile, "common", k=500,
                           filters=SearchFilters(date_from=date(2022, 1, 1), date_to=date(2024, 1, 1)))  # fmt: skip
    assert dated and {metas[h.chunk_id].doc_date for h in dated} == {"2023-06-22"}
    until = keyword_search(
        kb_db, profile, "common", k=500, filters=SearchFilters(date_to=date(2022, 1, 1))
    )
    assert {metas[h.chunk_id].doc_date for h in until} == {"2021-05-01"}
    pool = keyword_search(kb_db, profile, "common", k=500, filters=SearchFilters(collection="Pool"))
    assert {h.chunk_id for h in pool} == {c for c, m in metas.items() if m.collection == "Pool/"}
    assert (
        keyword_search(kb_db, profile, "common", k=500, filters=SearchFilters(collection="Nope"))
        == []
    )
    assert keyword_search(kb_db, profile, "What is it?", k=5) == []


NESTED = (
    "Appliances/",
    "Appliances/Furnace/",
    "Appliances/Water heater and softener/",  # a value of 12 bytes or more
    "Appliances-old/",  # sorts before "Appliances/": must not match it
    "AppliancesX/",  # sorts after "Appliances0": must not match either
    "🏠 House/",  # outside the BMP
    "🏠 House/Garage door/",
    "Pool/",
    "",
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "Appliances",
            {"Appliances/", "Appliances/Furnace/", "Appliances/Water heater and softener/"},
        ),
        ("/Appliances/Water heater and softener/", {"Appliances/Water heater and softener/"}),
        ("🏠 House", {"🏠 House/", "🏠 House/Garage door/"}),
        ("🏠 House\\Garage door", {"🏠 House/Garage door/"}),
        ("Appliances/Fur", set()),  # a prefix of a name is not its folder
        ("Nowhere", set()),
        ("", set(NESTED)),  # the root: no filter
    ],
)
def test_a_collection_filter_includes_its_sub_folders_in_both_searches(kb_db, value, expected):
    profile, _, vectors, metas = build_synthetic(kb_db, n=270, collections=NESTED)
    wanted = [cid for cid, meta in metas.items() if meta.collection in expected]
    query = np.random.default_rng(5).standard_normal(DIMS)
    only = SearchFilters(collection=value)
    dense = vector_search(kb_db, profile, query, k=40, filters=only)
    subset = [list(metas).index(cid) for cid in wanted]
    if wanted:
        want, sims = brute_force(vectors[subset], wanted, query, 40)
        assert [h.chunk_id for h in dense] == want  # exact, filtered inside the KNN
        assert [h.similarity for h in dense] == pytest.approx(sims, abs=1e-5)
    else:
        assert dense == []
    keyword = keyword_search(kb_db, profile, "common", k=500, filters=only)
    assert {h.chunk_id for h in keyword} == set(wanted)


def test_collections_are_normalised():
    from hal9000.knowledge.index.filters import collection_bounds, normalize_collection

    assert normalize_collection(" Appliances \\ Furnace ") == "Appliances/Furnace/"
    assert normalize_collection("./a//b/../") == "a/b/"
    assert normalize_collection("") == normalize_collection(None) == normalize_collection("/") == ""
    assert collection_bounds("🏠 House") == ("🏠 House/", "🏠 House0")
    assert collection_bounds("") is None
    with pytest.raises(ValueError):
        normalize_collection("bad\x07name")


def test_long_doc_types_filter_inside_the_knn(kb_db):
    profile, _, vectors, metas = build_synthetic(kb_db)
    reports = [cid for cid, meta in metas.items() if meta.doc_type == "inspection_report"]
    query = np.random.default_rng(9).standard_normal(DIMS)
    hits = vector_search(
        kb_db, profile, query, k=30, filters=SearchFilters(doc_type="inspection_report")
    )
    want, _ = brute_force(vectors[[list(metas).index(c) for c in reports]], reports, query, 30)
    assert [h.chunk_id for h in hits] == want


def test_searches_are_timed_as_their_plan_stages(kb_db):
    profile, _, _, _ = build_synthetic(kb_db, n=8)
    with start_trace() as trace:
        vector_search(kb_db, profile, np.ones(DIMS), k=3)
        keyword_search(kb_db, profile, "common", k=3)
    assert [s.stage for s in trace.stages] == ["vector", "bm25"]


def test_vector_metadata_can_be_rewritten_and_rows_deleted(kb_db):
    profile, index, _, metas = build_synthetic(kb_db, n=12)
    first = next(iter(metas))
    with kb_db:
        index.update_meta([first], ChunkMeta("permit", "Garage", "2020-01-01"))
    hits = vector_search(
        kb_db, profile, np.ones(DIMS), k=5, filters=SearchFilters(doc_type="permit")
    )
    assert [h.chunk_id for h in hits] == [first]
    with kb_db:
        index.delete([first])
    assert index.count() == 11
    with pytest.raises(ValueError, match="16-dimension"):
        index.search(np.ones(4), 3)


# ---------------------------------------------------------- on the corpus
@pytest.fixture
def library(conn, make_pipeline):
    ingest_all(conn, make_pipeline())
    return conn, active_profile(conn)


def text_of(conn, chunk_id: int) -> str:
    return conn.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]


def test_bm25_finds_an_exact_invoice_number_that_dense_search_misses(library):
    conn, profile = library
    number = "CHI-INV-20250314"
    fake = FakeEmbedder()
    dense = vector_search(conn, profile, fake.vector(number), k=5)
    keyword = keyword_search(conn, profile, number, k=5)
    assert keyword and number in text_of(conn, keyword[0].chunk_id)
    assert all(number in text_of(conn, h.chunk_id) for h in keyword)  # the phrase, nothing looser
    assert not any(number in text_of(conn, h.chunk_id) for h in dense)
    # fused, the keyword hit comes back into the candidates
    fused = rrf(as_scored(dense), as_scored(keyword))
    assert any(number in text_of(conn, h.chunk_id) for h in fused[:3])


def test_a_paraphrase_finds_its_finding_through_the_fake_synonyms(library):
    conn, profile = library
    fake = FakeEmbedder()
    hits = vector_search(conn, profile, fake.vector("Is the water heater leaking?"), k=3)
    assert any("dripping" in text_of(conn, h.chunk_id) for h in hits)


def test_filters_restrict_both_searches_on_the_corpus(library):
    conn, profile = library
    fake = FakeEmbedder()
    only = SearchFilters(doc_type="invoice")
    dense = vector_search(conn, profile, fake.vector("total due"), k=30, filters=only)
    keyword = keyword_search(conn, profile, "total due", k=30, filters=only)
    for hits in (dense, keyword):
        types = {
            conn.execute(
                "SELECT d.doc_type FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.id = ?", (h.chunk_id,),
            ).fetchone()[0]
            for h in hits
        }  # fmt: skip
        assert types == {"invoice"}


def test_bm25_statistics_are_per_profile(conn, make_pipeline):
    pipeline = make_pipeline()
    ingest_all(conn, pipeline, [corpus.INSPECTION, corpus.ESTIMATE, corpus.TEXT])
    profile = get_profile(conn, PROFILE)
    question = "water heater relief valve"
    before = [(h.chunk_id, h.score) for h in keyword_search(conn, profile, question, k=10)]
    assert before

    report = pipeline.build(conn, "pdfium.fixed-32.hdr.fake-64")  # much smaller chunks
    assert (
        report.chunks
        > 2
        * conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE profile_id = ?", (profile.id,)
        ).fetchone()[0]
    )
    after = [(h.chunk_id, h.score) for h in keyword_search(conn, profile, question, k=10)]
    assert after == before  # exactly the same scores: no pooled term statistics
