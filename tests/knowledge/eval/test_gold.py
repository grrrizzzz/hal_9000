"""Gold set generation (plan §8.1) against the synthetic fixture corpus, offline.

The LLM is scripted: :class:`tests.fakes.FakeLLM` with literal replies where
the order is simple, and :class:`GoldLLM` (a FakeLLM that quotes the prompt's
own page text) for whole-corpus runs.
"""

from __future__ import annotations

import random
from datetime import date

import pytest

from hal9000.cost.metering import BudgetExceeded
from hal9000.knowledge.contracts import EvidenceQuote, GoldItem
from hal9000.knowledge.eval.config import CATEGORIES, GoldConfig, load_eval_config
from hal9000.knowledge.eval.gold import (
    GenerationPlan,
    LexicalRetriever,
    SourceDocument,
    align_selection,
    assign_splits,
    generate_gold,
    is_paraphrased,
    jaccard,
    link_keys,
    load_corpus,
    parse_tool_reply,
    plan_generation,
    propose_link_sets,
)
from hal9000.knowledge.eval.text import find_normalized
from tests.fakes import FakeLLM, tool_call
from tests.knowledge.eval.fakes import FakeRetrieve, GoldLLM, prompt_pages, quotable_lines
from tests.knowledge.fixtures import corpus

SHA = "a" * 64
SMALL = {
    "factoid": 6,
    "money": 3,
    "date": 2,
    "list": 2,
    "cross_document": 2,
    "unanswerable": 4,
}


def cfg(**kw) -> GoldConfig:
    kw.setdefault("quotas", dict(SMALL))
    return GoldConfig(**kw)


def one_window(fixture_docs, requested=None) -> GenerationPlan:
    """A plan with a single window task over the inspection's first pages."""
    doc = fixture_docs[0]
    plan = plan_generation([doc], cfg(quotas={"factoid": 2}, overgenerate=1.0, window_pages=12))
    assert [t.kind for t in plan.tasks] == ["window"]
    return plan


def draft(question, quote, page, category="factoid", document=1):
    return {
        "question": question,
        "reference_answer": "x",
        "category": category,
        "difficulty": "easy",
        "evidence": [{"quote": quote, "page": page, "document": document}],
    }


# --------------------------------------------------------------- quote checks
def test_a_quote_not_on_the_page_is_rejected_regenerated_once_then_dropped(fixture_docs):
    plan = one_window(fixture_docs)
    fact = corpus.QA_FACTS[1]  # "Total due: $625.00", inspection page 11
    good = draft("What was the inspection bill?", fact.quote, fact.page)
    bad = draft("What colour is the roof?", "The roof is bright purple", 3)
    still_bad = draft("What colour is the roof?", "The roof is purple", 3)
    llm = FakeLLM(
        [
            tool_call("record_questions", items=[good, bad]),
            tool_call("record_questions", items=[still_bad]),
        ]
    )
    result = generate_gold(plan, llm)
    assert [i.question for i in result.items] == ["What was the inspection bill?"]
    assert result.items[0].evidence[0].quote == fact.quote
    assert result.items[0].evidence[0].file_sha == fixture_docs[0].file_sha
    assert len(llm.calls) == 2  # one generation, exactly one regeneration
    assert "is not on page 3" in llm.calls[1].messages[-1]["content"]
    assert llm.tiers == ["powerful", "powerful"]
    assert result.dropped == [("What colour is the roof?", result.dropped[0][1])]
    assert "is not on page 3" in result.dropped[0][1]
    assert result.shortfall == {"factoid": 1}


def test_a_regenerated_item_that_passes_is_kept(fixture_docs):
    plan = one_window(fixture_docs)
    fact = corpus.QA_FACTS[2]
    wrong_page = draft("Is the relief valve OK?", fact.quote, fact.page + 1)
    fixed = draft("Is the relief valve OK?", fact.quote, fact.page)
    llm = FakeLLM(
        [
            tool_call("record_questions", items=[wrong_page]),
            tool_call("record_questions", items=[fixed]),
        ]
    )
    result = generate_gold(plan, llm)
    assert [i.question for i in result.items] == ["Is the relief valve OK?"]
    assert result.dropped == []


def test_quote_matching_ignores_whitespace_and_case_and_stores_the_page_wording(fixture_docs):
    plan = one_window(fixture_docs)
    fact = corpus.QA_FACTS[4]
    shouted = "  " + fact.quote.upper().replace(" ", "   ") + " "
    llm = FakeLLM(
        [tool_call("record_questions", items=[draft("Where is the shutoff?", shouted, fact.page)])]
    )
    item = generate_gold(plan, llm).items[0]
    assert item.evidence[0].quote.lower() == fact.quote.lower()
    page_text = fixture_docs[0].pages[fact.page]
    assert item.evidence[0].quote in page_text


def test_invalid_replies_and_disallowed_categories(fixture_docs):
    plan = one_window(fixture_docs)
    fact = corpus.QA_FACTS[1]
    wrong_category = draft("Who fixed it?", fact.quote, fact.page, category="cross_document")
    llm = FakeLLM(["not json at all", tool_call("record_questions", items=[wrong_category])])
    result = generate_gold(plan, llm)
    assert result.items == []
    assert any("was not asked for" in reason for _, reason in result.dropped)
    llm = FakeLLM(["{}", '```json\n{"items": 3}\n```'])
    result = generate_gold(plan, llm)
    assert result.dropped == [("(whole reply)", result.dropped[0][1])]
    assert "does not match the schema" in result.dropped[0][1]


def test_evidence_must_name_a_real_document_and_page(fixture_docs):
    plan = one_window(fixture_docs)
    fact = corpus.QA_FACTS[1]
    llm = FakeLLM(
        [
            tool_call(
                "record_questions",
                items=[
                    draft("Item A?", fact.quote, fact.page, document=2),
                    draft("Item B?", fact.quote, 99),
                ],
            ),
            tool_call("record_questions", items=[]),
        ]
    )
    result = generate_gold(plan, llm)
    reasons = " ".join(r for _, r in result.dropped)
    assert "document 2 does not exist" in reasons and "page 99 is not a page" in reasons


def test_parse_tool_reply_accepts_bare_and_fenced_json():
    from hal9000.knowledge.eval.gold import _CheckReply

    assert parse_tool_reply('{"answered": "no", "reason": "r"}', _CheckReply).answered == "no"
    fenced = '```json\n{"tool": "record_check", "args": {"answered": "yes", "reason": "r"}}\n```'
    assert parse_tool_reply(fenced, _CheckReply).answered == "yes"
    with pytest.raises(ValueError, match="does not match"):
        parse_tool_reply('{"answered": "maybe", "reason": "r"}', _CheckReply)


# ------------------------------------------------------------------ quotas
def test_category_quotas_are_met_on_the_fixture_corpus(fixture_docs):
    plan = plan_generation(fixture_docs, cfg())
    result = generate_gold(plan, GoldLLM(), retrieve=FakeRetrieve())
    assert result.selected_by_category == SMALL
    assert result.shortfall == {} and result.paraphrase_shortfall == 0
    assert all(i.status == "draft" and i.origin == "generated" for i in result.items)
    for item in result.items:
        for ev in item.evidence:
            doc = next(d for d in fixture_docs if d.file_sha == ev.file_sha and ev.page in d.pages)
            assert find_normalized(ev.quote, doc.pages[ev.page]) is not None
    cross = [i for i in result.items if i.category == "cross_document"]
    assert all(len({(e.file_sha, e.page) for e in i.evidence}) >= 2 for i in cross)
    assert "Harborline" not in plan.tasks[0].prompt.messages[0]["content"]


def test_a_shortfall_is_reported_when_the_llm_returns_too_few(fixture_docs):
    plan = plan_generation(fixture_docs, cfg(overgenerate=1.0))
    result = generate_gold(plan, GoldLLM(yield_fraction=0.5), retrieve=FakeRetrieve())
    assert result.shortfall  # at least one category short
    for category, short in result.shortfall.items():
        assert result.selected_by_category[category] + short == SMALL[category]
    assert "SHORTFALL" in result.summary()


def test_the_default_quotas_are_the_plans_120_item_mix():
    quotas = load_eval_config().gold.quotas
    assert quotas == {
        "factoid": 34, "money": 18, "date": 12, "list": 16, "cross_document": 16, "unanswerable": 24,
    }  # fmt: skip
    assert sum(quotas.values()) == 120
    assert GoldConfig().quotas == quotas


# -------------------------------------------------------------- paraphrase
def test_jaccard_and_the_paraphrase_threshold():
    quote = "flexible gas connector routed through cabinet"
    assert jaccard("Is anything wrong with the gas line to the stove?", [quote]) < 0.3
    # content words: {gas, connector, cabinet} vs {flexible, gas, connector, routed, cabinet}
    # -> 3/5 = 0.6
    assert jaccard("the gas connector cabinet", [quote]) == pytest.approx(0.6)
    # exactly 0.3 is NOT paraphrased
    q = "alpha bravo charlie"
    e = ["alpha bravo charlie delta echo foxtrot golf hotel india juliet"]
    assert jaccard(q, e) == pytest.approx(0.3)
    assert not is_paraphrased(q, e, 0.3)
    assert is_paraphrased(q + " kilo", e, 0.3)  # 3/11
    assert jaccard("", []) == 0.0


def test_copied_wording_is_not_counted_as_paraphrased(fixture_docs):
    plan = plan_generation(fixture_docs, cfg())
    result = generate_gold(plan, GoldLLM(paraphrase_every=2), retrieve=FakeRetrieve())
    for item in result.items:
        if item.category == "unanswerable":
            continue
        quotes = [e.quote for e in item.evidence]
        assert item.paraphrased == (jaccard(item.question, quotes) < 0.3)
        if item.question.startswith("What does it say:"):
            assert not item.paraphrased
    assert result.paraphrased_share >= 0.3


def test_the_paraphrase_quota_swaps_in_paraphrased_items(fixture_docs):
    """Only 1 in 5 generated items is paraphrased; spare ones are swapped in to reach 30%."""
    plan = plan_generation(fixture_docs, cfg(overgenerate=3.0))
    result = generate_gold(plan, GoldLLM(paraphrase_every=5), retrieve=FakeRetrieve())
    assert result.selected_by_category == SMALL
    assert result.paraphrased_share >= 0.3
    assert result.paraphrase_shortfall == 0


def test_a_paraphrase_shortfall_is_reported(fixture_docs):
    plan = plan_generation(fixture_docs, cfg(overgenerate=1.0))
    result = generate_gold(plan, GoldLLM(paraphrase_every=1000), retrieve=FakeRetrieve())
    assert result.paraphrased_share == 0.0
    assert result.paraphrase_shortfall == 5  # ceil(0.3 * 15)
    assert "paraphrase quota" in result.summary()


# ------------------------------------------------------------- unanswerable
def test_an_unanswerable_draft_that_retrieval_finds_answered_is_flagged(fixture_docs):
    answered = "What is the invoice number for the home inspection?"
    plan = plan_generation(fixture_docs, cfg())
    retrieve = FakeRetrieve(["Invoice number: CHI-INV-20250314"])
    llm = GoldLLM(answered={answered}, extra_drafts=[answered])
    result = generate_gold(plan, llm, retrieve=retrieve)
    drafts = [p for p in llm.calls if p.metadata["purpose"] == "kb-gold-unanswerable"]
    assert len(drafts) == 1
    assert "Document" not in drafts[0].messages[1]["content"]  # drafted without documents
    checks = [p for p in llm.calls if p.metadata["purpose"] == "kb-gold-check"]
    assert answered in retrieve.questions
    assert len(checks) == len(retrieve.questions) == 6  # every draft is checked
    assert "Invoice number: CHI-INV-20250314" in checks[0].messages[1]["content"]
    unanswerable = [i for i in result.items if i.category == "unanswerable"]
    assert len(unanswerable) == 4
    # clean drafts fill the quota first; the flagged one is left out of a full quota
    assert all(i.reviewer_note is None for i in unanswerable)
    assert answered not in [i.question for i in unanswerable]


def test_flagged_drafts_fill_a_quota_only_after_clean_ones_and_keep_the_flag(fixture_docs):
    flagged = [f"Question {w} about the thing?" for w in ("one", "two", "three")]
    plan = plan_generation(fixture_docs, cfg(overgenerate=1.0))  # asks for exactly 4 drafts
    llm = GoldLLM(answered=set(flagged), extra_drafts=flagged)
    result = generate_gold(plan, llm, retrieve=FakeRetrieve())
    unanswerable = [i for i in result.items if i.category == "unanswerable"]
    assert len(unanswerable) == 4
    notes = [i.reviewer_note for i in unanswerable]
    assert notes[0] is None
    assert all(n and n.startswith("FLAGGED: retrieval found it answered") for n in notes[1:])
    assert result.flagged == 3


def test_a_draft_that_cannot_be_checked_is_flagged_not_assumed(fixture_docs):
    plan = plan_generation(fixture_docs, cfg(overgenerate=1.0))
    result = generate_gold(plan, GoldLLM(), retrieve=FakeRetrieve(fail=True))
    unanswerable = [i for i in result.items if i.category == "unanswerable"]
    assert unanswerable and all(
        "could not be checked" in (i.reviewer_note or "") for i in unanswerable
    )


def test_the_lexical_fallback_retriever_checks_against_the_whole_corpus(fixture_docs):
    retrieve = LexicalRetriever(fixture_docs, k=3)
    top = retrieve("How often should the heat pump filter be replaced?").selected
    assert len(top) == 3
    assert "every 90 days" in top[0].text
    assert retrieve("zzqx").selected == []


# --------------------------------------------------------------- dedup/split
def test_duplicates_and_existing_questions_are_dropped(fixture_docs):
    plan = one_window(fixture_docs)
    fact = corpus.QA_FACTS[1]
    items = [
        draft("How much was the inspection?", fact.quote, fact.page),
        draft("how much was the inspection ?", fact.quote, fact.page),
        draft("What did the inspection cost?", fact.quote, fact.page),
    ]
    existing = [GoldItem(id=1, question="What did the inspection cost", category="unanswerable")]
    result = generate_gold(
        plan, FakeLLM([tool_call("record_questions", items=items)]), existing=existing
    )
    assert [i.question for i in result.items] == ["How much was the inspection?"]
    assert result.duplicates == 2


def test_the_split_is_stable_and_every_category_is_in_both_halves(fixture_docs):
    def run():
        plan = plan_generation(fixture_docs, cfg())
        return generate_gold(plan, GoldLLM(), retrieve=FakeRetrieve()).items

    first, second = run(), run()
    assert [(i.question, i.split) for i in first] == [(i.question, i.split) for i in second]
    for category in CATEGORIES:
        splits = [i.split for i in first if i.category == category]
        assert "dev" in splits and "test" in splits, category
        assert splits.count("dev") == round(0.6 * len(splits)) or len(splits) == 2


def _items(n: int, category: str = "factoid") -> list[GoldItem]:
    return [
        GoldItem(
            question=f"Question number {i} about {category}?",
            category=category,
            evidence=[EvidenceQuote(quote="q", page=1)] if category != "unanswerable" else [],
        )
        for i in range(n)
    ]


def test_assign_splits_is_order_independent_and_stratified():
    items = _items(10) + _items(5, "money") + _items(2, "date") + _items(1, "list")
    a = {i.question: i.split for i in assign_splits(items)}
    shuffled = list(items)
    random.Random(4).shuffle(shuffled)
    assert {i.question: i.split for i in assign_splits(shuffled)} == a
    by = lambda c: [a[i.question] for i in items if i.category == c]  # noqa: E731
    assert by("factoid").count("dev") == 6 and by("money").count("dev") == 3
    assert sorted(by("date")) == ["dev", "test"]
    assert by("list") == ["dev"]


def test_existing_splits_are_kept_and_count_toward_the_ratio():
    existing = [i.model_copy(update={"split": "dev"}) for i in _items(6)]
    new = [
        GoldItem(
            question=f"Another question {w}?",
            category="factoid",
            evidence=[EvidenceQuote(quote="q", page=1)],
        )
        for w in ("alpha", "bravo", "charlie", "delta")
    ]
    out = assign_splits(new, 0.6, existing)
    assert [i.split for i in out].count("test") == 4  # 10 total -> 6 dev, all already dev
    kept = assign_splits([existing[0]])
    assert kept[0].split == "dev"


# ------------------------------------------------------------- cross-document
def test_link_keys_and_link_sets(fixture_docs):
    keys = link_keys(fixture_docs[0])
    assert "vendor:coastline home inspections" in keys
    assert "date:2025-03-14" in keys
    assert "ref:CHI-INV-20250314" in keys
    sets = propose_link_sets(fixture_docs)
    assert sets
    by_title = {d.title: i for i, d in enumerate(fixture_docs)}
    unnumbered = {
        i
        for i, d in enumerate(fixture_docs)
        if d.file_sha == fixture_docs[by_title["Invoice DPS-1001"]].file_sha
    }
    two_invoices = tuple(sorted(i for i in unnumbered if fixture_docs[i].doc_type == "invoice"))
    assert any(docs == two_invoices for docs, _ in sets)


def test_a_document_naming_another_documents_vendor_links_them():
    a = SourceDocument(
        1,
        SHA,
        "Pest report",
        "pest_report",
        "Acme Termite Control",
        date(2024, 6, 1),
        {1: "Termite damage at the deck."},
    )
    b = SourceDocument(
        2,
        "b" * 64,
        "Repair invoice",
        "invoice",
        "Bolt Builders",
        date(2024, 8, 1),
        {1: "Repaired the damage noted by Acme Termite Control."},
    )
    c = SourceDocument(3, "c" * 64, "Manual", "manual", None, None, {1: "Unrelated text."})
    sets = propose_link_sets([a, b, c])
    assert sets == [((0, 1), ("vendor:acme termite control",))]
    assert propose_link_sets([a, b, c], max_sets=0) == []


def test_cross_document_items_need_two_documents(fixture_docs):
    plan = plan_generation(fixture_docs, cfg(quotas={"cross_document": 1}, overgenerate=1.0))
    (task,) = plan.tasks
    assert task.kind == "cross" and len(task.documents) >= 2
    pages = prompt_pages(task.prompt)
    (doc, page), text = next(iter(sorted(pages.items())))
    line = quotable_lines(text)[0]
    one_doc = draft("Was it fixed?", line, page, category="cross_document", document=doc)
    llm = FakeLLM(
        [tool_call("record_questions", items=[one_doc]), tool_call("record_questions", items=[])]
    )
    result = generate_gold(plan, llm)
    assert "at least two documents" in result.dropped[0][1]


# ------------------------------------------------------------------- limits
def test_budget_exceeded_stops_and_keeps_partial_results(fixture_docs):
    plan = plan_generation(fixture_docs, cfg())
    inner = GoldLLM()
    calls = {"n": 0}

    def capped(prompt, tier):
        calls["n"] += 1
        if calls["n"] > 3:
            raise BudgetExceeded("Run cost $2.01 reached the $2.00 hard cap.")
        return inner(prompt, tier)

    result = generate_gold(plan, capped, retrieve=FakeRetrieve())
    assert calls["n"] == 4  # nothing is called after the cap
    assert result.stopped and "hard cap" in result.stopped
    assert 0 < len(result.items) < sum(SMALL.values())
    assert "STOPPED EARLY" in result.summary()


def test_provider_errors_are_recorded_and_generation_continues(fixture_docs):
    plan = one_window(fixture_docs)

    def broken(prompt, tier):
        raise ConnectionError("provider down")

    result = generate_gold(plan, broken)
    assert result.items == [] and result.errors and "provider down" in result.errors[0]


def test_the_estimate_counts_the_real_prompts_and_prices_the_tier(fixture_docs, monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_POWERFUL", "openai/gpt-5")
    plan = plan_generation(fixture_docs, GoldConfig())
    estimate = plan.estimate()
    assert estimate.model == "openai/gpt-5"
    assert estimate.calls >= len(plan.tasks) + plan.n_unanswerable
    assert estimate.prompt_tokens > 10_000
    assert 0.1 < estimate.usd < 2.0  # plan §8.1: about $1
    assert "estimated $" in estimate.describe()
    cheap = plan_generation(fixture_docs, GoldConfig(quotas={"factoid": 1}))
    assert cheap.estimate().usd < estimate.usd


def test_the_default_plan_on_the_fixture_corpus(fixture_docs):
    plan = plan_generation(fixture_docs, GoldConfig())
    kinds = {t.kind for t in plan.tasks}
    assert kinds == {"window", "cross", "unanswerable"}
    requested = {}
    for t in plan.tasks:
        for c, n in t.requested.items():
            requested[c] = requested.get(c, 0) + n
    assert requested == {
        "factoid": 51,
        "money": 27,
        "date": 18,
        "list": 24,
        "cross_document": 24,
        "unanswerable": 36,
    }


# ------------------------------------------------------------ corpus loading
def _store_corpus(conn):
    from hal9000.knowledge.contracts import FileRecord

    rec = FileRecord(
        sha256=SHA,
        original_name="x.pdf",
        relative_path="x.pdf",
        mime_type="application/pdf",
        bytes=1,
    )
    with conn:
        conn.execute(
            "INSERT INTO files (sha256, original_name, relative_path, mime_type, bytes, uploaded_at) VALUES (?,?,?,?,?,?)",
            (
                rec.sha256,
                rec.original_name,
                rec.relative_path,
                rec.mime_type,
                rec.bytes,
                "2026-01-01",
            ),
        )
        conn.execute(
            "INSERT INTO documents (file_sha, page_start, page_end, title, doc_type, vendor, doc_date) VALUES (?,?,?,?,?,?,?)",
            (SHA, 1, 2, "Estimate", "estimate", "Tidewater", "2025-03-20"),
        )
        conn.execute(
            "INSERT INTO documents (file_sha, page_start, page_end, title, doc_type) VALUES (?,?,?,?,?)",
            (SHA, 3, 3, "Blank", "other"),
        )
        for parser, page, text in (
            ("pdfium", 1, "pdfium page one"),
            ("pdfium", 2, "pdfium page two"),
            ("docling", 1, "docling page one"),
            ("pdfium", 3, "   "),
        ):
            conn.execute(
                "INSERT INTO pages (file_sha, page, parser, text, has_text_layer) VALUES (?,?,?,?,1)",
                (SHA, page, parser, text),
            )


def test_load_corpus_reads_documents_and_prefers_the_parser(kb_db):
    _store_corpus(kb_db)
    (doc,) = load_corpus(kb_db)
    assert doc.pages == {1: "pdfium page one", 2: "pdfium page two"}
    assert doc.doc_date == date(2025, 3, 20) and doc.vendor == "Tidewater"
    assert "Tidewater" in doc.label() and "2025-03-20" in doc.label()
    (doc,) = load_corpus(kb_db, parser="docling")
    assert doc.pages == {1: "docling page one"}
    (doc,) = load_corpus(kb_db, parser="docx")  # falls back to a non-Docling parser
    assert doc.pages[1] == "pdfium page one"


# ------------------------------------------------------------ edited evidence
def test_align_selection_maps_a_letter_spaced_hyphenated_selection_to_the_cleaned_quote(
    fixture_docs,
):
    estimate = next(d for d in fixture_docs if d.doc_type == "estimate")
    page = estimate.pages[1]
    spaced, repaired = next(iter(corpus.LETTER_SPACED.items()))
    assert repaired in page
    selection = f"{spaced}"
    quote = align_selection(selection, page)
    assert quote is not None and repaired.lower() in quote.lower()
    assert find_normalized(quote, page) is not None
    fact = corpus.QA_FACTS[3]  # "manufactured in 2012 and is 13 years old"
    inspection = fixture_docs[0].pages[fact.page]
    raw = "manu-\nfactured  in 2012\nand is 13 years ol"
    quote = align_selection(raw, inspection)
    assert quote == fact.quote
    assert align_selection("A sentence from some other page entirely.", inspection) is None
    assert align_selection("", inspection) is None
    assert align_selection(inspection + " and more words", inspection) is None


def _unanswerable_plan(fixture_docs, n=2):
    plan = plan_generation(fixture_docs, cfg(quotas={"unanswerable": n}, overgenerate=1.0))
    assert [t.kind for t in plan.tasks] == ["unanswerable"]
    return plan


def test_unanswerable_drafts_are_regenerated_once_when_invalid(fixture_docs):
    plan = _unanswerable_plan(fixture_docs)
    drafts = tool_call(
        "record_drafts",
        questions=[{"question": "Who painted the fence?", "difficulty": "easy"},
                   {"question": "When was the chimney swept?", "difficulty": "easy"}],
    )  # fmt: skip
    partly = tool_call("record_check", answered="partly", reason="P2 hints at it")
    no = tool_call("record_check", answered="no", reason="none")
    llm = FakeLLM(["oops", drafts, partly, no])
    result = generate_gold(plan, llm, retrieve=FakeRetrieve())
    assert "That reply was invalid" in llm.calls[1].messages[-1]["content"]
    notes = {i.question: i.reviewer_note for i in result.items}
    assert notes["When was the chimney swept?"] is None
    assert notes["Who painted the fence?"].startswith("FLAGGED: retrieval found it partly answered")
    llm = FakeLLM(["oops", "still oops"])
    result = generate_gold(plan, llm, retrieve=FakeRetrieve())
    assert result.items == [] and result.dropped[0][0] == "(unanswerable drafts)"


def test_an_unreadable_or_failed_check_flags_the_draft(fixture_docs):
    plan = _unanswerable_plan(fixture_docs, n=1)
    drafts = tool_call(
        "record_drafts", questions=[{"question": "Who painted the fence?", "difficulty": "easy"}]
    )
    result = generate_gold(plan, FakeLLM([drafts, "not json"]), retrieve=FakeRetrieve())
    assert "could not be checked (the reply is not JSON" in result.items[0].reviewer_note

    replies = iter([drafts])

    def flaky(prompt, tier):
        reply = next(replies, None)
        if reply is None:
            raise TimeoutError("slow")
        return reply

    result = generate_gold(plan, flaky, retrieve=FakeRetrieve())
    assert "the check call failed" in result.items[0].reviewer_note
    result = generate_gold(
        plan, lambda p, t: (_ for _ in ()).throw(TimeoutError()), retrieve=FakeRetrieve()
    )
    assert result.items == [] and result.errors
