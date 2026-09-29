"""The FTS query builder (plan §7.1 keyword_search): no syntax errors, OR semantics, phrases."""

from __future__ import annotations

import sqlite3

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hal9000.knowledge.index.fts_query import build_fts_query, fts_tokens, identifiers
from hal9000.knowledge.storage import FTS_TOKENIZER

ROWS = {
    1: "Invoice for the $1,234 gutter repair, due in 30 days.",
    2: "Quote from O'Brien's Roofing for the ridge vent.",
    3: "Permit A-1234-5 was issued for the water heater replacement.",
    4: "Account 12345678 is billed monthly.",
    5: "Section a 9999 1234 lists 5 findings.",  # a, 1234 and 5 present but not as a phrase
    6: "The attic insulation is thin.",
}


@pytest.fixture(scope="module")
def fts():
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(header, text, tokenize = '{FTS_TOKENIZER}')")
    conn.executemany("INSERT INTO t (rowid, header, text) VALUES (?, '', ?)", ROWS.items())
    yield conn
    conn.close()


def search(conn, question: str) -> list[int]:
    expression = build_fts_query(question)
    if expression is None:
        return []
    return [
        r[0]
        for r in conn.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY rank", (expression,))
    ]


@pytest.mark.parametrize(
    ("question", "expected_first"),
    [
        ("What's the $1,234 invoice for?", 1),
        ("O'Brien's quote?", 2),
        ("A-1234-5", 3),
    ],
)
def test_the_plan_examples_run_without_a_syntax_error_and_find_their_row(
    fts, question, expected_first
):
    # the raw text is a syntax error (or an unintended AND) in FTS5
    with pytest.raises(sqlite3.OperationalError):
        fts.execute("SELECT rowid FROM t WHERE t MATCH ?", (question,)).fetchall()
    assert search(fts, question)[0] == expected_first


def test_one_matching_word_is_enough_or_not_and(fts):
    # only "insulation" occurs in the corpus; AND semantics would return nothing
    assert search(fts, "Is the insulation in the crawlspace rated R-30 by the county?") == [6]


def test_an_identifier_is_matched_as_a_phrase_not_as_loose_tokens(fts):
    assert build_fts_query("A-1234-5") == '"a 1234 5"'
    # row 5 has "a", "1234" and "5", but not consecutively
    assert search(fts, "A-1234-5") == [3]
    assert search(fts, "12345678") == [4]


def test_the_expression_is_quoted_terms_joined_with_or():
    assert build_fts_query("water heater leak") == '"water" OR "heater" OR "leak"'
    assert build_fts_query("What is the permit A-1234-5?") == '"a 1234 5" OR "permit"'
    assert build_fts_query("How much was the $1,234.50 bill?") == '"1 234 50" OR "bill"'
    assert build_fts_query("water water WATER") == '"water"'


def test_only_stopwords_or_punctuation_give_none():
    assert build_fts_query("What is it?") is None
    assert build_fts_query("?!$%") is None
    assert build_fts_query("") is None


def test_operators_and_quotes_in_the_question_are_inert(fts):
    for question in [
        "AND OR NOT",
        "NEAR(a b)",
        '"unbalanced',
        "col:value",
        "x*",
        "^start",
        "a + b",
    ]:
        search(fts, question)  # no OperationalError


def test_tokens_follow_unicode61():
    assert fts_tokens("Café O'Brien_x 12,50") == ["café", "o", "brien", "x", "12", "50"]
    assert identifiers("see 6/22/23 and well-known") == [(4, 11)]


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=80))
def test_arbitrary_text_is_never_a_syntax_error(text):
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(header, text, tokenize = '{FTS_TOKENIZER}')")
    conn.execute("INSERT INTO t (rowid, header, text) VALUES (1, '', 'water heater')")
    expression = build_fts_query(text)
    if expression is not None:
        conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expression,)).fetchall()
    conn.close()
