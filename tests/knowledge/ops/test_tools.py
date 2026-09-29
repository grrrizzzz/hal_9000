"""``search_house_docs``: HAL's knowledge tool (plan §16, handoff KWP10)."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from hal9000.core.action_context import ActionContext
from hal9000.core.actions import PythonActionRegistry
from hal9000.core.environment import PythonEnvironment
from hal9000.cost.metering import MonthlyCapExceeded
from hal9000.knowledge import tools
from hal9000.knowledge.contracts import RetrievedChunk
from hal9000.knowledge.index.fake import FakeEmbedder
from hal9000.knowledge.storage import open_db
from hal9000.knowledge.tokens import CHAT_ENCODING, EMBEDDING_ENCODING, count_tokens
from hal9000.knowledge.tools import (
    DOC_TYPES,
    HouseDocs,
    KnowledgeIndexMissing,
    KnowledgeToolInputError,
    citation,
    fit,
    search_house_docs,
)

CITATION = re.compile(r"^.+ \(\d{4}-\d{2}-\d{2}\), pp?\. \d+(-\d+)?$|^.+, pp?\. \d+(-\d+)?$")


def docs(data_dir: Path, **kw) -> HouseDocs:
    return HouseDocs(db_path=data_dir / "knowledge.db", embed_fn=FakeEmbedder(), **kw)


def result_tokens(body: dict) -> int:
    text = json.dumps(body, ensure_ascii=False)
    return max(count_tokens(text, CHAT_ENCODING), count_tokens(text, EMBEDDING_ENCODING))


# ------------------------------------------------------------------ citations
def _chunk(**kw) -> RetrievedChunk:
    base = {
        "chunk_id": 1,
        "document_id": 1,
        "text": "The water heater is 13 years old.",
        "page_start": 12,
        "page_end": 12,
        "tokens": 8,
        "rank": 1,
        "score": 0.5,
        "title": "Home inspection report",
        "doc_type": "inspection_report",
        "doc_date": date(2023, 6, 22),
    }
    return RetrievedChunk(**{**base, **kw})


def test_citation_strings():
    assert citation(_chunk()) == "Home inspection report (2023-06-22), p. 12"
    assert citation(_chunk(page_end=13)) == "Home inspection report (2023-06-22), pp. 12-13"
    assert citation(_chunk(doc_date=None, title="Pool log", page_start=1, page_end=1)) == (
        "Pool log, p. 1"
    )
    assert citation(_chunk(title="")) == "Untitled document (2023-06-22), p. 12"


# ------------------------------------------------------------------ searching
def test_returns_cited_passages_from_the_active_profile(data_dir):
    body = docs(data_dir).search("How old is the water heater?")
    assert body["found"] is True and body["search"] == "hybrid"
    assert 1 <= len(body["passages"]) <= tools.MAX_PASSAGES
    for passage in body["passages"]:
        assert set(passage) >= {"citation", "title", "doc_type", "vendor", "date", "page", "text"}
        assert CITATION.match(passage["citation"]), passage["citation"]
        assert passage["doc_type"] in DOC_TYPES
    hit = next(p for p in body["passages"] if "13 years old" in p["text"])
    assert hit["doc_type"] == "inspection_report" and hit["title"] == "Home Inspection Report"
    assert hit["citation"] == f"Home Inspection Report ({hit['date']}), p. {hit['page']}"
    assert hit["page"] == 7 and hit["vendor"]


def test_the_same_retrieval_as_the_chat_search(data_dir):
    """The tool's passages are KWP4's ``search_documents`` selection, in order."""
    from hal9000.knowledge.config import load_knowledge_file
    from hal9000.knowledge.index.embedding import EmbeddingService
    from hal9000.knowledge.retrieve import RetrievalConfig, search_documents

    body = docs(data_dir).search("Where is the main water shutoff valve?")
    conn = open_db(data_dir / "knowledge.db")
    try:
        config = RetrievalConfig.from_settings(load_knowledge_file(None))
        expected = search_documents(
            conn,
            EmbeddingService(conn, embed_fn=FakeEmbedder()),
            "Where is the main water shutoff valve?",
            k=tools.MAX_PASSAGES,
            config=config,
        )
    finally:
        conn.close()
    assert [p["text"] for p in body["passages"]] == [c.text for c in expected.selected]


def test_doc_type_filters_the_documents(data_dir):
    body = docs(data_dir).search("total", doc_type="estimate")
    assert body["passages"] and body["doc_type"] == "estimate"
    assert {p["doc_type"] for p in body["passages"]} == {"estimate"}
    unfiltered = docs(data_dir).search("total")
    assert {p["doc_type"] for p in unfiltered["passages"]} != {"estimate"}


@pytest.mark.parametrize("value", ["", "any", "all"])
def test_an_empty_doc_type_means_everything(data_dir, value):
    assert docs(data_dir).search("water heater", doc_type=value)["doc_type"] is None


def test_a_doc_type_with_nothing_indexed_finds_nothing(data_dir):
    body = docs(data_dir).search("water heater", doc_type="permit")
    assert body["found"] is False and body["passages"] == []
    assert "Nothing" in body["note"]


def test_bad_arguments_are_recoverable_input_errors(data_dir):
    with pytest.raises(KnowledgeToolInputError, match="inspection_report") as error:
        docs(data_dir).search("water heater", doc_type="receipt")
    assert error.value.recoverable and error.value.error_type == "InvalidArguments"
    with pytest.raises(KnowledgeToolInputError, match="empty"):
        docs(data_dir).search("   ")


def test_a_low_score_is_reported_as_not_found(data_dir):
    from hal9000.knowledge.retrieve import set_threshold

    conn = open_db(data_dir / "knowledge.db")
    try:
        profile_id = conn.execute("SELECT id FROM profiles WHERE active = 1").fetchone()[0]
        set_threshold(conn, profile_id, None, 0.9999)
    finally:
        conn.close()
    body = docs(data_dir).search("How old is the water heater?")
    assert body["passages"] and body["found"] is False
    assert "scored low" in body["note"]


# ------------------------------------------------------------------ the budget
def test_the_result_stays_within_about_2000_tokens(data_dir):
    body = docs(data_dir, max_passages=50).search("inspection report findings")
    assert len(body["passages"]) >= 2
    assert result_tokens(body) <= tools.MAX_RESULT_TOKENS


def test_a_small_budget_drops_passages_from_the_end(data_dir):
    full = docs(data_dir).search("water heater relief valve")
    small = docs(data_dir, max_tokens=300).search("water heater relief valve")
    assert result_tokens(small) <= 300 and small["truncated"] is True
    assert 1 <= len(small["passages"]) < len(full["passages"])
    kept = [p["text"] for p in small["passages"]]
    assert kept[:-1] == [p["text"] for p in full["passages"]][: len(kept) - 1]


def test_fit_shortens_a_lone_oversized_passage():
    body = {"query": "q", "passages": [{"citation": "X, p. 1", "text": "word " * 5000}]}
    fitted = fit(body, 500)
    assert result_tokens(fitted) <= 500 and fitted["truncated"] is True
    assert fitted["passages"][0]["text"].endswith("[...]")
    assert fitted["passages"][0]["text"].startswith("word word")
    small = {"query": "q", "passages": [{"citation": "X, p. 1", "text": "short"}]}
    assert fit(dict(small), 500) == small  # nothing to do


# ------------------------------------------------------------------ degraded modes
def test_at_the_monthly_cap_it_falls_back_to_keywords(data_dir):
    def capped(*args, **kw):
        raise MonthlyCapExceeded("knowledge has spent $10.00 this month")

    body = HouseDocs(db_path=data_dir / "knowledge.db", embed_fn=capped).search(
        "How old is the water heater?"
    )
    assert body["search"] == "keyword" and body["passages"]
    assert "budget" in body["note"]


def test_an_embedding_failure_degrades_to_keywords(data_dir):
    def down(*args, **kw):
        raise RuntimeError("provider down")

    body = HouseDocs(db_path=data_dir / "knowledge.db", embed_fn=down).search("water heater")
    assert body["search"] == "keyword" and body["passages"]
    assert "Semantic search is unavailable" in body["note"]


def test_no_database_is_a_clear_recoverable_error(tmp_path):
    with pytest.raises(KnowledgeIndexMissing, match="no index yet") as error:
        HouseDocs(db_path=tmp_path / "knowledge.db").search("water heater")
    assert error.value.recoverable and error.value.error_type == "KnowledgeIndexMissing"
    assert "Do not call this tool again" in error.value.guidance
    assert not (tmp_path / "knowledge.db").exists()  # a read never creates the database


def test_no_active_profile_is_a_clear_recoverable_error(data_dir):
    conn = sqlite3.connect(data_dir / "knowledge.db")
    with conn:
        conn.execute("UPDATE profiles SET active = 0")
    conn.close()
    with pytest.raises(KnowledgeIndexMissing, match="no active index"):
        docs(data_dir).search("water heater")


def test_a_newer_schema_is_reported_not_guessed(data_dir):
    conn = sqlite3.connect(data_dir / "knowledge.db")
    with conn:
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (999, 'x')")
    conn.close()
    with pytest.raises(tools.KnowledgeToolError, match="can't be read"):
        docs(data_dir).search("water heater")


# ------------------------------------------------------------------ settings, meter
def test_without_an_injected_dependency_it_reads_the_kb_settings(data_dir):
    body = search_house_docs("How old is the water heater?")
    assert body["passages"] and body["found"]


def test_the_meter_is_built_only_for_a_paid_model(tmp_path):
    built = []

    def factory():
        built.append(1)
        return "meter"

    house = HouseDocs(db_path=tmp_path / "k.db", meter_factory=factory)
    assert house._meter_for("fake/embedding-64") is None and built == []
    assert house._meter_for("openai/text-embedding-3-small") == "meter"
    assert house._meter_for("openai/text-embedding-3-small") == "meter" and built == [1]


def test_from_settings_meters_through_the_shared_ledger(data_dir, monkeypatch):
    from hal9000.knowledge.api import services

    made = []
    monkeypatch.setattr(services, "_meter", lambda settings: made.append(settings) or "m")
    house = HouseDocs.from_settings()
    assert house.db_path == data_dir / "knowledge.db" and house.meter is None
    assert house._meter_for("openai/text-embedding-3-small") == "m"
    assert made[0].monthly_cap_usd == 10.0


# ------------------------------------------------------------------ through the framework
def test_the_environment_injects_house_docs_and_the_model_supplies_only_the_query(data_dir):
    registry = tools.knowledge_registry()
    action = registry.get_action("search_house_docs")
    assert [a.name for a in registry.get_actions()] == ["search_house_docs"]
    context = ActionContext({"house_docs": docs(data_dir)})
    result = PythonEnvironment().execute_action(
        None, context, action, {"query": "How old is the water heater?"}
    )
    assert result["tool_executed"] is True
    assert result["result"]["passages"][0]["citation"]


def test_a_forged_dependency_is_overridden_by_the_context(data_dir, tmp_path):
    action = tools.knowledge_registry().get_action("search_house_docs")
    context = ActionContext({"house_docs": docs(data_dir)})
    forged = HouseDocs(db_path=tmp_path / "elsewhere.db")
    result = PythonEnvironment().execute_action(
        None, context, action, {"query": "water heater", "_house_docs": forged}
    )
    assert result["tool_executed"] is True and result["result"]["passages"]


def test_a_missing_index_reaches_the_model_as_a_tool_error(tmp_path):
    action = tools.knowledge_registry().get_action("search_house_docs")
    context = ActionContext({"house_docs": HouseDocs(db_path=tmp_path / "none.db")})
    result = PythonEnvironment().execute_action(None, context, action, {"query": "x"})
    assert result["tool_executed"] is False and "no index yet" in result["error"]


def test_ensure_registered_restores_a_cleared_catalogue(clean_tool_registry):
    from hal9000.core import registry

    assert "search_house_docs" not in registry.tools
    names = [a.name for a in tools.knowledge_registry().get_actions()]
    assert names == ["search_house_docs"]
    assert registry.tools_by_tag[tools.KNOWLEDGE_READ] == ["search_house_docs"]
    assert PythonActionRegistry(tags=[tools.KNOWLEDGE_READ]).get_action("search_house_docs")


# ------------------------------------------------------------------ the ceiling holds for any input
def test_a_long_query_is_refused_and_one_at_the_limit_fits(data_dir):
    with pytest.raises(KnowledgeToolInputError, match="too long") as error:
        docs(data_dir).search("water heater " * 3000)
    assert error.value.recoverable
    at_limit = " ".join(["water heater"] * (tools.MAX_QUERY_TOKENS // 2))
    assert count_tokens(at_limit) <= tools.MAX_QUERY_TOKENS
    body = docs(data_dir, max_passages=50).search(at_limit)
    assert body["passages"] and body["query"] == at_limit
    assert result_tokens(body) <= tools.MAX_RESULT_TOKENS


def test_long_titles_and_vendors_are_cut_and_the_result_fits(data_dir):
    conn = sqlite3.connect(data_dir / "knowledge.db")
    with conn:
        conn.execute(
            "UPDATE documents SET title = ?, vendor = ?",
            ("Very long synthetic title " * 200, "Vendor name " * 300),
        )
    conn.close()
    body = docs(data_dir, max_passages=50).search("water heater relief valve")
    assert body["passages"]
    assert result_tokens(body) <= tools.MAX_RESULT_TOKENS
    for passage in body["passages"]:
        assert len(passage["title"]) <= tools.MAX_FIELD_CHARS
        assert len(passage["vendor"]) <= tools.MAX_FIELD_CHARS
        assert passage["title"].endswith("...")
        assert len(passage["citation"]) <= tools.MAX_FIELD_CHARS + 40


def test_the_guidance_is_in_the_message_the_base_environment_passes_on(tmp_path, data_dir):
    action = tools.knowledge_registry().get_action("search_house_docs")
    missing = ActionContext({"house_docs": HouseDocs(db_path=tmp_path / "none.db")})
    result = PythonEnvironment().execute_action(None, missing, action, {"query": "x"})
    assert KnowledgeIndexMissing.guidance in result["error"]
    context = ActionContext({"house_docs": docs(data_dir)})
    result = PythonEnvironment().execute_action(
        None, context, action, {"query": "x", "doc_type": "receipt"}
    )
    assert KnowledgeToolInputError.guidance in result["error"]
    error = KnowledgeIndexMissing("no index")
    assert error.detail == "no index" and str(error) == f"no index {error.guidance}"
