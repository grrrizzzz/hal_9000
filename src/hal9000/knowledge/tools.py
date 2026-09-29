"""HAL's knowledge tool: ``search_house_docs`` (plan §16, handoff KWP10).

A GAME tool, registered with the tag ``knowledge_read`` and with no other tag,
so it is **not** in the solar or triage toolkits (their registries are built
from other tags; ``tests/knowledge/test_integration.py`` compares them with a
snapshot taken before the knowledge base existed). It is ready for the
follow-on Telegram query agent: ``PythonActionRegistry(tags=[KNOWLEDGE_READ])``
or :func:`knowledge_registry`.

**What it does.** Retrieval only, never generation: the same retrieval code as
the chat's ``GET /api/search`` (KWP4's :func:`~hal9000.knowledge.retrieve.search_documents`
over the active index profile, with ``config/knowledge.yaml``'s retrieval
settings). It returns the top passages with the document's title, type,
vendor, date and page, and a citation string such as
``"Home inspection report (2023-06-22), p. 12"``. The whole result stays within
:data:`MAX_RESULT_TOKENS` (about 2,000 tokens): passages are dropped from the
end, and a lone oversized passage is shortened, until it fits.

**Dependencies are injected, never shown to the model.** The only model-facing
parameters are ``query`` and ``doc_type``; ``_house_docs`` is filled by
:class:`~hal9000.core.environment.PythonEnvironment` from the action context's
``house_docs`` property (a :class:`HouseDocs`), and ``@register_tool`` strips
it from the schema. Without one, the tool reads the ``KB_*`` settings.

**Money** (rule K9): the one paid call is the query embedding, metered as
agent ``knowledge`` through the shared :class:`~hal9000.cost.metering.Meter`,
so it counts toward ``KB_MONTHLY_CAP_USD``. At the cap the tool falls back to
keyword (BM25) search, like the chat's retrieval-only mode, and says so.

**Errors the model can act on.** No index yet, or no active profile, raises
:class:`KnowledgeIndexMissing` (``recoverable``, with guidance: answer without
the house documents). A bad ``doc_type`` or an empty query raises
:class:`KnowledgeToolInputError` naming the allowed values. Both carry the
``error_type`` / ``recoverable`` / ``guidance`` attributes the environments
turn into structured tool results.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, get_args

from hal9000.knowledge.contracts import DocType, RetrievalResult, RetrievedChunk, SearchFilters
from hal9000.tools.decorators import register_tool

__all__ = [
    "DOC_TYPES",
    "KNOWLEDGE_READ",
    "MAX_RESULT_TOKENS",
    "HouseDocs",
    "KnowledgeIndexMissing",
    "KnowledgeToolError",
    "KnowledgeToolInputError",
    "citation",
    "ensure_registered",
    "knowledge_registry",
    "search_house_docs",
]

log = logging.getLogger(__name__)

#: The tag the tool is registered with (plan §16). No solar or triage agent asks for it.
KNOWLEDGE_READ = "knowledge_read"
#: The whole tool result (JSON) stays within this many tokens.
MAX_RESULT_TOKENS = 2000
#: How many passages the tool returns at most.
MAX_PASSAGES = 5
#: The document types the model may filter by.
DOC_TYPES: tuple[str, ...] = get_args(DocType)


class KnowledgeToolError(RuntimeError):
    """A failure the calling agent should read and work around."""

    error_type = "KnowledgeUnavailable"
    recoverable = True
    guidance = "Answer without the house documents, and say they could not be searched."


class KnowledgeIndexMissing(KnowledgeToolError):
    """There is no knowledge base index to search yet."""

    error_type = "KnowledgeIndexMissing"
    guidance = (
        "The house documents have not been indexed yet. Do not call this tool again in this "
        "run; answer without them and say the documents are not searchable yet."
    )


class KnowledgeToolInputError(ValueError):
    """The model's arguments are wrong; the message says how."""

    error_type = "InvalidArguments"
    recoverable = True
    guidance = "Fix the arguments as the error describes and call the tool again."


# ------------------------------------------------------------------ citations
def _pages(chunk: RetrievedChunk) -> str:
    if chunk.page_end > chunk.page_start:
        return f"pp. {chunk.page_start}-{chunk.page_end}"
    return f"p. {chunk.page_start}"


def citation(chunk: RetrievedChunk) -> str:
    """``"Home inspection report (2023-06-22), p. 12"``; undated: ``"Pool log, pp. 1-2"``."""
    title = chunk.title or "Untitled document"
    dated = f"{title} ({chunk.doc_date.isoformat()})" if chunk.doc_date else title
    return f"{dated}, {_pages(chunk)}"


def _passage(chunk: RetrievedChunk, text: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "citation": citation(chunk),
        "title": chunk.title,
        "doc_type": chunk.doc_type,
        "vendor": chunk.vendor,
        "date": chunk.doc_date.isoformat() if chunk.doc_date else None,
        "page": chunk.page_start,
    }
    if chunk.page_end > chunk.page_start:
        body["page_end"] = chunk.page_end
    body["text"] = chunk.text if text is None else text
    return body


def _tokens(body: dict[str, Any]) -> int:
    """Tokens of the JSON result, by the larger of the two encodings a HAL model may use."""
    from hal9000.knowledge.tokens import CHAT_ENCODING, EMBEDDING_ENCODING, count_tokens

    text = json.dumps(body, ensure_ascii=False)
    return max(count_tokens(text, CHAT_ENCODING), count_tokens(text, EMBEDDING_ENCODING))


def _shorten(text: str, tokens: int) -> str:
    """The first ``tokens`` tokens of ``text`` (cl100k), marked as cut."""
    from hal9000.knowledge.tokens import encoding

    ids = encoding().encode(text, disallowed_special=())
    if len(ids) <= tokens:
        return text
    return encoding().decode(ids[: max(tokens, 0)]).rstrip() + " [...]"


def fit(body: dict[str, Any], limit: int = MAX_RESULT_TOKENS) -> dict[str, Any]:
    """Drop passages from the end, then shorten a lone one, until ``body`` is within ``limit``."""
    passages: list[dict[str, Any]] = body["passages"]
    while len(passages) > 1 and _tokens(body) > limit:
        passages.pop()
        body["truncated"] = True
    if passages and _tokens(body) > limit:
        body["truncated"] = True
        text = passages[0]["text"]
        passages[0]["text"] = ""
        room = limit - _tokens(body) - 8  # the " [...]" marker and JSON escaping slack
        while True:
            passages[0]["text"] = _shorten(text, max(room, 0))
            if room <= 0 or _tokens(body) <= limit:
                break
            room -= max(_tokens(body) - limit, 4)
    return body


# ------------------------------------------------------------------ the search
@dataclass
class HouseDocs:
    """What :func:`search_house_docs` needs; pass one as the ``house_docs`` context property.

    Args:
        db_path: ``data/knowledge.db``.
        yaml_path: ``config/knowledge.yaml`` (retrieval settings), read per call;
            ``None`` uses the defaults.
        meter: meters the query embedding as agent ``knowledge``. ``None``: built
            by ``meter_factory`` the first time a paid model needs it.
        meter_factory: builds the meter lazily (so fake mode never opens the ledger).
        embed_fn: the embedding callable; ``None`` picks one by the profile's model.
        max_passages: how many passages at most.
        max_tokens: the result's token ceiling.
    """

    db_path: Path
    yaml_path: Path | None = None
    meter: Any | None = None
    meter_factory: Callable[[], Any] | None = None
    embed_fn: Callable[..., Any] | None = None
    max_passages: int = MAX_PASSAGES
    max_tokens: int = MAX_RESULT_TOKENS

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> HouseDocs:
        """From the ``KB_*`` settings, metered through the shared ledger."""
        from hal9000.knowledge.config import settings as knowledge_settings

        current = settings or knowledge_settings()

        def meter() -> Any:
            from hal9000.knowledge.api.services import _meter

            return _meter(current)

        return cls(db_path=current.db_path, yaml_path=current.yaml_path, meter_factory=meter)

    # -------------------------------------------------------------- helpers
    def _meter_for(self, model: str) -> Any | None:
        from hal9000.knowledge.index.embedding import is_fake_model

        if self.meter is None and self.meter_factory is not None and not is_fake_model(model):
            self.meter = self.meter_factory()
        return self.meter

    def _connect(self) -> sqlite3.Connection:
        from hal9000.knowledge.storage import SchemaTooNew, open_db

        if not Path(self.db_path).is_file():
            raise KnowledgeIndexMissing(
                "The house knowledge base has no index yet: no documents have been added."
            )
        try:
            return open_db(self.db_path)
        except SchemaTooNew as error:
            raise KnowledgeToolError(f"The house knowledge base can't be read: {error}") from error

    # --------------------------------------------------------------- search
    def search(self, query: str, doc_type: str | None = None) -> dict[str, Any]:
        """Retrieval over the active profile; see :func:`search_house_docs` for the result."""
        from hal9000.cost.metering import BudgetExceeded
        from hal9000.knowledge.config import load_knowledge_file
        from hal9000.knowledge.index.embedding import EmbeddingService
        from hal9000.knowledge.index.profiles import active_profile
        from hal9000.knowledge.retrieve import RetrievalConfig, search_documents

        query = (query or "").strip()
        if not query:
            raise KnowledgeToolInputError("query is empty: say what to look for.")
        if doc_type in ("", "any", "all"):
            doc_type = None
        if doc_type is not None and doc_type not in DOC_TYPES:
            raise KnowledgeToolInputError(
                f"doc_type {doc_type!r} is not one of: {', '.join(DOC_TYPES)}. "
                "Leave it out to search every document."
            )
        filters = SearchFilters(doc_type=doc_type) if doc_type else None

        conn = self._connect()
        try:
            profile = active_profile(conn)
            if profile is None:
                raise KnowledgeIndexMissing(
                    "The house knowledge base has no active index yet, so there is nothing "
                    "to search."
                )
            config = RetrievalConfig.from_settings(load_knowledge_file(self.yaml_path))
            config = replace(config, budget_tokens=min(config.budget_tokens, self.max_tokens))
            embeddings = EmbeddingService(
                conn,
                embed_fn=self.embed_fn,
                meter=self._meter_for(profile.embed_model),
                max_attempts=2,
                backoff_s=0.25,
            )
            note = None
            try:
                result = search_documents(
                    conn,
                    embeddings,
                    query,
                    k=self.max_passages,
                    filters=filters,
                    config=config,
                    profile=profile,
                )
            except BudgetExceeded:
                log.warning("knowledge budget reached: search_house_docs falls back to keywords")
                note = (
                    "This month's knowledge-base budget is used up, so these passages come "
                    "from keyword search only."
                )
                result = search_documents(
                    conn,
                    embeddings,
                    query,
                    k=self.max_passages,
                    filters=filters,
                    config=replace(config, mode="bm25", rerank="none"),
                    profile=profile,
                )
        finally:
            conn.close()
        return fit(self._result(result, doc_type, note), self.max_tokens)

    @staticmethod
    def _result(result: RetrievalResult, doc_type: str | None, note: str | None) -> dict[str, Any]:
        passages = [_passage(chunk) for chunk in result.selected]
        if result.degraded == "embedding_unavailable" and note is None:
            note = (
                "Semantic search is unavailable, so these passages come from keyword search only."
            )
        weak = bool(passages) and result.abstained
        if not passages:
            note = note or "Nothing in the house documents matches this query."
        elif weak:
            note = (
                (note + " ") if note else ""
            ) + "The best matches scored low: they may not answer the question."
        body: dict[str, Any] = {
            "query": result.query,
            "doc_type": doc_type,
            "search": "keyword" if result.mode == "bm25" else result.mode,
            "found": bool(passages) and not weak,
            "passages": passages,
        }
        if note:
            body["note"] = note
        return body


@register_tool(tags=[KNOWLEDGE_READ])
def search_house_docs(
    query: str, doc_type: str | None = None, _house_docs: HouseDocs | None = None
) -> dict:
    """Search the house's documents (inspection reports, estimates, invoices, pest reports,
    permits, appliance manuals, warranties, disclosures) for passages about a question.

    Returns up to five passages, best first, each with its document's title, type, vendor,
    date, page, the passage text and a citation such as "Home inspection report (2023-06-22),
    p. 12". Quote figures only from the passage text and cite them. "found" is false when
    nothing relevant was found; do not guess then.

    Args:
        query: What to look for, in plain words (e.g. "age of the water heater").
        doc_type: Only search one kind of document: inspection_report, estimate, invoice,
            pest_report, completion_notice, disclosure_form, manual, warranty, permit or
            other. Leave it out to search everything.
    """
    docs = _house_docs if _house_docs is not None else HouseDocs.from_settings()
    return docs.search(query, doc_type)


def ensure_registered() -> None:
    """Re-register the tool if the global catalogue was cleared (tests, reloads)."""
    from hal9000.core import registry

    if "search_house_docs" not in registry.tools:
        register_tool(tags=[KNOWLEDGE_READ])(search_house_docs)


def knowledge_registry() -> Any:
    """The ``knowledge_read`` toolkit, for the follow-on query agent."""
    from hal9000.core.actions import PythonActionRegistry

    ensure_registered()
    return PythonActionRegistry(tags=[KNOWLEDGE_READ])
