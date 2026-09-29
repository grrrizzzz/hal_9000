"""What the evaluation harness needs from the rest of the knowledge base.

Retrieval (KWP4) and answering (KWP6) are built on other branches, so the
harness never imports them. It depends only on these small protocols, and
the CLI connects the real implementations through
:mod:`hal9000.knowledge.eval.wiring`.

- :class:`RetrieveFn`: ``question -> RetrievalOutput``, the **exact** output of
  ``select`` (after parent expansion and de-duplication), in rank order, with
  each item's prompt token count (``index_tokens`` when headers are sent).
  Scoring truncates it to each budget itself, so the retriever should select
  with a budget at least as large as the largest scored budget.
- :class:`AnswerFn`: ``question -> AnswerOutput`` (judge calibration, S5).
- :data:`Generate`: ``(Prompt, tier) -> str``, already metered as
  ``knowledge-eval`` (rule K9).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from hal9000.core.prompt import Prompt
from hal9000.knowledge.contracts import GoldCategory

__all__ = [
    "AnswerFn",
    "AnswerOutput",
    "EvalQuestionResult",
    "EvalResultsPage",
    "Generate",
    "RetrieveFn",
    "RetrievalOutput",
    "SelectedItem",
    "Source",
]

#: ``(prompt, tier) -> reply``; a tool call comes back as ``{"tool": ..., "args": ...}``.
Generate = Callable[[Prompt, str], str]


@dataclass(frozen=True)
class SelectedItem:
    """One item of ``select``'s output: exactly what the prompt would receive.

    ``prompt_tokens`` is what the item costs in the context budget (the
    chunk's ``index_tokens`` when headers are sent, else its ``tokens``; the
    parent's for an expanded parent-child hit). Offsets are deliberately
    absent: scoring aligns text, so it can't depend on a parser's offsets.
    """

    chunk_id: int
    text: str
    prompt_tokens: int
    header: str = ""


@dataclass(frozen=True)
class RetrievalOutput:
    """``select``'s ordered output for one question, and the abstain decision (plan §7.1)."""

    selected: Sequence[SelectedItem]
    #: top dense cosine or reranker score -- never the RRF score; None when there is none
    abstain_score: float | None = None
    abstained: bool = False
    latency_ms: float | None = None


class RetrieveFn(Protocol):
    def __call__(self, question: str) -> RetrievalOutput: ...  # pragma: no cover


@dataclass(frozen=True)
class Source:
    """A numbered source an answer could cite (``[S1]`` ...)."""

    marker: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class AnswerOutput:
    """A generated answer with the sources it was given."""

    text: str
    sources: Sequence[Source] = field(default_factory=tuple)
    abstained: bool = False
    cost_usd: float = 0.0


class AnswerFn(Protocol):
    def __call__(self, question: str) -> AnswerOutput: ...  # pragma: no cover


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalQuestionResult(_Model):
    """One question's result in a run: ``GET /api/eval/runs/{id}/results`` items.

    ``recall`` is keyed by the token budget as a string (``"2000"``); a missing
    key means N/A (the chunk size exceeds the budget, or the item is
    unanswerable). Matches the approved contract addition in
    ``web/src/api/pending-contract.ts`` (item 4).
    """

    gold_item_id: int
    question: str
    category: GoldCategory
    recall: dict[str, float]
    full_support: bool
    first_hit_rank: int | None
    abstain_score: float | None
    abstained: bool


class EvalResultsPage(_Model):
    items: list[EvalQuestionResult]
    total: int
    offset: int
    limit: int
