"""Scripted stand-ins for gold generation, the judge and retrieval (offline, deterministic).

:class:`GoldLLM` is a :class:`tests.fakes.FakeLLM` whose replies are computed
from each prompt: it reads the page text in a generation prompt and quotes
lines of it verbatim, so what it returns passes (or, when told to, fails)
the exact-substring check. Every call and tier is recorded like FakeLLM's.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable

from hal9000.core.prompt import Prompt
from hal9000.knowledge.eval.types import RetrievalOutput, SelectedItem
from tests.fakes import FakeLLM, tool_call

_BLOCK = re.compile(
    r"^--- document (\d+), page (\d+) ---\n(.*?)(?=^--- document |^=== |\Z)", re.M | re.S
)
_SYLLABLES = (
    "zor",
    "vak",
    "quil",
    "met",
    "brun",
    "tosh",
    "plix",
    "dran",
    "yem",
    "kov",
    "sarn",
    "ful",
)


def pseudo(i: int) -> str:
    """A made-up word unique to ``i`` (shares no content word with any page)."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i, len(_SYLLABLES))
        out += _SYLLABLES[r]
    return out + "ix"


def prompt_pages(prompt: Prompt) -> dict[tuple[int, int], str]:
    """``(document, page) -> text`` for the page blocks in a generation prompt."""
    text = prompt.messages[1]["content"]
    return {(int(d), int(p)): body.strip() for d, p, body in _BLOCK.findall(text)}


def quotable_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if len(line.split()) >= 5]


class GoldLLM(FakeLLM):
    """Answers gold-generation, unanswerable-draft and check prompts from their content.

    Args:
        paraphrase_every: every n-th item is worded with made-up words (paraphrased);
            the others copy their quote's wording.
        yield_fraction: share of the requested items actually returned.
        bad_quotes: questions whose first-try quote is corrupted.
        fix_on_repair: a repair call returns the corrected item (else still corrupt).
        answered: unanswerable drafts that the check call says are answered.
        extra_drafts: questions added to the unanswerable drafts.
    """

    def __init__(
        self,
        *,
        paraphrase_every: int = 2,
        yield_fraction: float = 1.0,
        bad_quotes: int = 0,
        fix_on_repair: bool = True,
        answered: Iterable[str] = (),
        extra_drafts: Iterable[str] = (),
    ) -> None:
        super().__init__([])
        self.paraphrase_every = paraphrase_every
        self.yield_fraction = yield_fraction
        self.bad_left = bad_quotes
        self.fix_on_repair = fix_on_repair
        self.answered = set(answered)
        self.extra_drafts = list(extra_drafts)
        self.serial = 0
        self.handlers: dict[str, Callable[[Prompt], str]] = {
            "kb-gold-window": self._questions,
            "kb-gold-cross": self._questions,
            "kb-gold-unanswerable": self._drafts,
            "kb-gold-check": self._check,
        }

    def __call__(self, prompt: Prompt, tier: str = "powerful") -> str:
        self.calls.append(prompt)
        self.tiers.append(tier)
        return self.handlers[prompt.metadata["purpose"]](prompt)

    # -------------------------------------------------------------- handlers
    def _item(self, category: str, evidence: list[dict]) -> dict:
        self.serial += 1
        quote_words = " ".join(e["quote"] for e in evidence)
        if self.serial % self.paraphrase_every == 0:
            question = (
                f"Could you tell me about the {pseudo(self.serial)} {pseudo(self.serial + 500)}?"
            )
        else:
            question = f"What does it say: {quote_words}?"
        return {
            "question": question,
            "reference_answer": "per the document",
            "category": category,
            "difficulty": "medium",
            "evidence": evidence,
        }

    def _questions(self, prompt: Prompt) -> str:
        pages = prompt_pages(prompt)
        repair = bool(prompt.metadata.get("repair"))
        if repair:
            # replace the corrupted items: re-quote their lines correctly (or not)
            items = []
            for item in self._bad_items:
                item = json.loads(json.dumps(item))
                for ev in item["evidence"]:
                    good = ev.pop("_good")
                    if self.fix_on_repair:
                        ev["quote"] = good
                items.append(item)
            self._bad_items = []
            return tool_call("record_questions", items=items)
        lines = [
            (doc, page, line)
            for (doc, page), text in sorted(pages.items())
            for line in quotable_lines(text)
        ]
        docs = sorted({d for d, _, _ in lines})
        requested = prompt.metadata["requested"]
        items = []
        self._bad_items = []
        cursor = 0
        for category, n in requested.items():
            for _ in range(int(n * self.yield_fraction + 1e-9)):
                if category == "cross_document":
                    evidence = []
                    for d in docs[:2]:
                        own = [x for x in lines if x[0] == d]
                        doc, page, line = own[cursor % len(own)]
                        evidence.append({"quote": line, "page": page, "document": doc})
                else:
                    doc, page, line = lines[cursor % len(lines)]
                    evidence = [{"quote": line, "page": page, "document": doc}]
                cursor += 1
                item = self._item(category, evidence)
                if self.bad_left:
                    self.bad_left -= 1
                    for ev in item["evidence"]:
                        ev["_good"] = ev["quote"]
                        ev["quote"] = "This sentence appears on no page at all anywhere"
                    self._bad_items.append(json.loads(json.dumps(item)))
                    for ev in item["evidence"]:
                        ev.pop("_good")
                items.append(item)
        return tool_call("record_questions", items=items)

    def _drafts(self, prompt: Prompt) -> str:
        n = prompt.metadata["requested"]["unanswerable"]
        questions = [{"question": q, "difficulty": "easy"} for q in self.extra_drafts]
        for i in range(n - len(questions)):
            questions.append(
                {
                    "question": f"When was the {pseudo(900 + i)} {pseudo(1900 + i)} last serviced?",
                    "difficulty": "medium",
                }
            )
        return tool_call("record_drafts", questions=questions)

    def _check(self, prompt: Prompt) -> str:
        question = prompt.metadata["question"]
        if question in self.answered:
            return tool_call("record_check", answered="yes", reason="P1 states it")
        return tool_call("record_check", answered="no", reason="no passage mentions it")


class FakeRetrieve:
    """A retrieve callable returning fixed passages; records the questions it saw."""

    def __init__(self, texts: Iterable[str] = ("A passage.",), *, fail: bool = False) -> None:
        self.texts = list(texts)
        self.fail = fail
        self.questions: list[str] = []

    def __call__(self, question: str) -> RetrievalOutput:
        self.questions.append(question)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return RetrievalOutput(
            selected=[
                SelectedItem(chunk_id=i, text=t, prompt_tokens=len(t.split()))
                for i, t in enumerate(self.texts)
            ],
            abstain_score=0.5,
        )
