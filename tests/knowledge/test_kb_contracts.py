"""Contracts round-trip JSON and enforce their invariants (handoff KWP0)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from hal9000.knowledge import contracts as c

SHA = "0123456789abcdef" * 4
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

DOC = c.DocumentMeta(
    id=3,
    file_sha=SHA,
    page_start=3,
    page_end=60,
    title="Synthetic Home Inspection",
    doc_type="inspection_report",
    vendor="Example Inspections LLC",
    doc_date=date(2023, 6, 22),
)
PROFILE = c.IndexProfile(
    id=1,
    name="pdfium.structure-768.hdr.te3s-1536",
    parser="pdfium",
    chunker="structure",
    size=768,
    overlap=0,
    embed_model="openai/text-embedding-3-small",
    dimensions=1536,
    active=True,
)
HIT = c.RetrievedChunk(
    chunk_id=11,
    document_id=3,
    text="7.2 Water heater: the relief valve drips.",
    header="Synthetic Home Inspection › Plumbing",
    page_start=41,
    page_end=41,
    tokens=12,
    rank=1,
    score=0.032,
    dense_score=0.71,
    bm25_score=8.2,
    vector_rank=1,
    bm25_rank=2,
    document=DOC,
)
CITATION = c.Citation(
    marker="S1",
    chunk_id=11,
    document_id=3,
    file_sha=SHA,
    page=41,
    title="Synthetic Home Inspection",
    doc_type="inspection_report",
    doc_date=date(2023, 6, 22),
)
ANSWER = c.Answer(
    text="The June 2023 inspection noted a dripping relief valve [S1].",
    citations=[CITATION],
    outcome="answered",
    grounding="verified",
    trace_id="t1",
    cost_usd=0.0012,
)
EVIDENCE = c.EvidenceQuote(quote="the relief valve drips", page=41, file_sha=SHA)
CI = c.MetricCI(value=0.9, low=0.85, high=0.94)
METRICS = c.EvalMetrics(
    n_questions=72,
    budgeted_recall={1000: CI, 2000: CI},
    full_support_hit={5: CI},
    mrr=CI,
    context_precision={2000: c.MetricCI(value=0.2, low=0.1, high=0.3)},
    by_category={"factoid": {"recall_2k": 0.93}},
)
JOB = c.Job(id=1, kind="ingest", status="running", progress=0.5, file_sha=SHA, created_at=NOW)
TRACE = c.Trace(
    id="t1",
    created_at=NOW,
    question="Is the water heater leaking?",
    profile=PROFILE.name,
    stages=[c.StageTiming(stage="vector", started_ms=1.0, duration_ms=3.5)],
    hits=[c.TraceHit(chunk_id=11, rank=1, score=0.03, dense_score=0.7, selected=True)],
    tokens_by_model={"openai/gpt-5-mini": {"prompt": 900, "completion": 80}},
    outcome="answered",
)
RUBRIC = c.RubricLabels(correctness="correct", faithful=True, citations_correct=True)

SAMPLES: list[BaseModel] = [
    c.FileRecord(
        sha256=SHA,
        original_name="inspection.pdf",
        relative_path="House/inspection.pdf",
        collection="House",
        mime_type="application/pdf",
        bytes=1234,
        uploaded_at=NOW,
    ),
    c.Page(
        file_sha=SHA,
        page=1,
        parser="pdfium",
        text="Page one",
        has_text_layer=True,
        offset_map=[(0, 0), (5, 9)],
    ),
    DOC,
    PROFILE,
    c.IndexProfile(
        name="pc",
        parser="pdfium",
        chunker="parent_child",
        size=256,
        parent_size=1500,
        embed_model="m",
        dimensions=256,
    ),
    c.Chunk(
        id=11,
        profile_id=1,
        document_id=3,
        page_start=41,
        page_end=41,
        char_start=10,
        char_end=52,
        text=HIT.text,
        header=HIT.header,
        tokens=12,
    ),
    EVIDENCE,
    c.GoldItem(
        question="Is the water heater leaking?",
        reference_answer="Yes, it drips.",
        category="factoid",
        evidence=[EVIDENCE],
        paraphrased=True,
        status="approved",
        split="dev",
    ),
    c.GoldItem(question="When was the pool resurfaced?", category="unanswerable"),
    HIT,
    c.RetrievalResult(
        query="q",
        standalone_query="q",
        profile=PROFILE.name,
        candidates=[HIT],
        selected=[HIT],
        context_tokens=12,
        abstain_score=0.71,
        threshold=0.3,
    ),
    CITATION,
    ANSWER,
    TRACE.stages[0],
    TRACE.hits[0],
    TRACE,
    JOB,
    CI,
    METRICS,
    c.EvalRun(
        id="r1",
        stage="S1a",
        config={"profiles": [PROFILE.name]},
        metrics=METRICS,
        host="mac",
        cost_usd=0.05,
        started_at=NOW,
        finished_at=NOW,
        results_path="data/knowledge/eval/runs/r1",
    ),
    # API shapes
    c.SessionRequest(passcode="x"),
    c.SessionResponse(ok=True, expires_at=NOW),
    c.UploadResult(relative_path="House/a.pdf", status="added", sha256=SHA, job_id=1),
    c.UploadResponse(
        results=[
            c.UploadResult(relative_path="x.jpg", status="skipped", reason="images are not indexed")
        ]
    ),
    c.DocumentSummary(document=DOC, original_name="i.pdf", relative_path="i.pdf", pages=58),
    c.DocumentDetail(
        document=DOC,
        original_name="i.pdf",
        relative_path="i.pdf",
        pages=58,
        file=c.FileRecord(
            sha256=SHA,
            original_name="i.pdf",
            relative_path="i.pdf",
            mime_type="application/pdf",
            bytes=1,
            uploaded_at=NOW,
        ),
    ),
    c.DocumentPatch(vendor="Other Vendor", doc_date=date(2023, 7, 1)),
    c.ChunkSpan(chunk_id=11, char_start=0, char_end=10),
    c.PageView(
        file_sha=SHA,
        page=41,
        page_count=89,
        text="t",
        has_text_layer=True,
        spans=[c.ChunkSpan(chunk_id=11, char_start=0, char_end=1)],
    ),
    c.SearchFilters(doc_type="invoice", date_from=date(2023, 1, 1)),
    c.ChatRequest(message="How old is the roof?", filters=c.SearchFilters(collection="House")),
    c.ChatStatusEvent(stage="vector"),
    c.ChatSourcesEvent(sources=[CITATION]),
    c.ChatTokenEvent(text="The "),
    c.ChatResetEvent(reason="figure 1,324 not in sources"),
    c.ChatVerifyEvent(grounding="failed", issues=["1,324"]),
    c.ChatDoneEvent(answer=ANSWER),
    c.ChatEventEnvelope(event=c.ChatTokenEvent(text="x")),
    c.MessageOut(
        id=1,
        role="assistant",
        content="x",
        citations=[CITATION],
        grounding="failed",
        issues=["$1,324.00 does not appear in [S1]"],
        created_at=NOW,
    ),
    c.ConversationSummary(id=1, title="Roof", created_at=NOW, updated_at=NOW),
    c.ConversationDetail(id=1, title="Roof", created_at=NOW, updated_at=NOW),
    c.FeedbackRequest(message_id=1, rating=-1, add_to_gold=True),
    c.FeedbackResponse(id=1, gold_item_id=2),
    c.GoldPatch(status="rejected", reviewer_note="ambiguous"),
    RUBRIC,
    c.JudgeLabel(
        id=1,
        gold_item_id=2,
        question="q",
        answer="a",
        citations=[CITATION],
        human=RUBRIC,
        judge=RUBRIC,
    ),
    c.EvalResultsPage(
        items=[
            c.EvalQuestionResult(
                gold_item_id=1,
                question="q",
                category="money",
                recall={"2000": 1.0},
                full_support=True,
                first_hit_rank=1,
                abstain_score=0.62,
                abstained=False,
            )
        ],
        total=1,
        offset=0,
        limit=25,
    ),
    c.EvalQuestionResult(
        gold_item_id=2,
        question="q",
        category="unanswerable",
        recall={},
        full_support=False,
        first_hit_rank=None,
        abstain_score=None,
        abstained=True,
    ),
    c.JudgeLabelUpdate(human=RUBRIC),
    c.ProfileState(active=PROFILE, profiles=[PROFILE], reindex_job=JOB),
    c.ProfileActivate(name=PROFILE.name),
    c.TraceSummary(id="t1", created_at=NOW, outcome="cancelled", total_ms=812.0, question="q"),
    c.Health(status="ok", version="0.2.0", active_profile=PROFILE.name),
    c.ErrorResponse(detail="not implemented yet"),
]


def _exported_models() -> set[type]:
    return {
        obj
        for name in c.__all__
        if isinstance(obj := getattr(c, name), type) and issubclass(obj, BaseModel)
    }


def test_every_exported_model_has_a_sample():
    assert _exported_models() == {type(s) for s in SAMPLES}


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: type(s).__name__)
def test_contracts_round_trip_json(sample):
    model = type(sample)
    assert model.model_validate_json(sample.model_dump_json()) == sample


def test_chat_events_discriminate_on_event():
    adapter = TypeAdapter(c.ChatEvent)
    event = adapter.validate_json('{"event": "reset", "reason": "numeral"}')
    assert isinstance(event, c.ChatResetEvent)


@pytest.mark.parametrize("quote", ["", "   ", "\n\t"])
def test_evidence_quote_requires_text(quote):
    with pytest.raises(ValidationError):
        c.EvidenceQuote(quote=quote, page=1)


@pytest.mark.parametrize("page", [0, -1])
def test_evidence_quote_requires_page_at_least_one(page):
    with pytest.raises(ValidationError):
        c.EvidenceQuote(quote="valve", page=page)


def test_gold_evidence_must_match_the_category():
    with pytest.raises(ValidationError, match="needs at least one evidence"):
        c.GoldItem(question="q", category="money")
    with pytest.raises(ValidationError, match="carries no evidence"):
        c.GoldItem(question="q", category="unanswerable", evidence=[EVIDENCE])


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (c.DocumentMeta, dict(file_sha=SHA, page_start=5, page_end=4, title="t")),
        (c.DocumentMeta, dict(file_sha="not-a-sha", page_start=1, page_end=1, title="t")),
        (
            c.IndexProfile,
            dict(
                name="p",
                parser="x",
                chunker="fixed",
                size=100,
                overlap=100,
                embed_model="m",
                dimensions=8,
            ),
        ),
        (
            c.IndexProfile,
            dict(
                name="p",
                parser="x",
                chunker="parent_child",
                size=100,
                embed_model="m",
                dimensions=8,
            ),
        ),
        (
            c.IndexProfile,
            dict(
                name="p",
                parser="x",
                chunker="fixed",
                size=100,
                parent_size=900,
                embed_model="m",
                dimensions=8,
            ),
        ),
        (
            c.Chunk,
            dict(
                profile_id=1,
                document_id=1,
                page_start=1,
                page_end=1,
                char_start=9,
                char_end=3,
                text="t",
                tokens=1,
            ),
        ),
        (c.MetricCI, dict(value=0.5, low=0.6, high=0.9)),
        (c.Citation, dict(marker="S0", chunk_id=1, document_id=1, file_sha=SHA, page=1, title="t")),
        (c.Job, dict(kind="ingest", progress=1.5)),
        (c.FeedbackRequest, dict(message_id=1, rating=0)),
        (c.FeedbackRequest, dict(message_id=1, comment="no rating and not for the question set")),
        (c.Answer, dict(text="t", outcome="answered", trace_id="t", unknown_field=1)),
    ],
)
def test_invalid_contracts_are_rejected(model, fields):
    with pytest.raises(ValidationError):
        model(**fields)
