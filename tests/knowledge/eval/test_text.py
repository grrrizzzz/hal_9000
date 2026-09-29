"""Text normalization shared by scoring, gold checks and selection alignment."""

from __future__ import annotations

from hal9000.knowledge.eval.text import find_normalized, joined, joined_text, normalize, words


def test_normalize_folds_case_space_hyphen_breaks_and_invisibles():
    assert normalize("  The Tempera-\n  ture­  VALVE￾\n\tis ﬁne ") == "the temperature valve is fine"


def test_words_and_joined_text_ignore_punctuation_and_letter_spacing():
    assert words("Total due: $625.00") == ["total", "due", "625", "00"]
    assert joined_text("D ES C R I PT I O N  of work") == joined_text("DESCRIPTION OF WORK")


def test_joined_keeps_word_spans_and_maps_back_to_the_source():
    text = "Re-\nplace the ﬁlter, now."
    j = joined(text)
    assert j.text == "replacethefilternow"
    assert [j.text[a:b] for a, b in j.spans] == ["replace", "the", "filter", "now"]
    assert text[j.source[0]] == "R"
    assert text[j.source[2]] == "p"  # after the hyphen break
    assert text[j.source[j.spans[2][0]]] == "ﬁ"
    assert len(j.source) == len(j.text) and j.n_words == 4


def test_joined_ignores_invisibles_and_a_plain_hyphen_splits_words():
    assert joined("soft­hyphen").spans == ((0, 10),)
    assert joined("well-known").spans == ((0, 4), (4, 9))
    assert joined("").n_words == 0


def test_find_normalized_ignores_whitespace_and_case_and_returns_the_page_span():
    page = "Summary\n\nThe  main WATER shutoff\nvalve is at the front."
    span = find_normalized("the main water shutoff valve", page)
    assert span is not None
    assert page[span[0] : span[1]] == "The  main WATER shutoff\nvalve"
    assert find_normalized("main water shut off valve", page) is None
    assert find_normalized("   ", page) is None
    assert find_normalized("shut off", "shut\u00ad off") == (0, 9)
    assert find_normalized("front.", page) == (len(page) - 6, len(page))
