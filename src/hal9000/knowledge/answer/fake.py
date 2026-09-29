"""Offline chat models: :class:`ScriptedStream` (tests) and :class:`FakeChat` (``--fake``).

Both have the call signature of :func:`hal9000.llm.stream_response` (text
deltas, then one :class:`~hal9000.llm.Usage` from a ``fake/`` model, which
costs nothing and never reaches the ledger).

:class:`FakeChat` answers extractively from the sources in the prompt, so its
answers verify, and reacts to the UI's fake-mode trigger phrases (``web/README.md``,
the same table the Mock Service Worker handlers follow):

======================  ==================================================================
phrase in the question  behaviour
======================  ==================================================================
``force a reset``       a first draft with a wrong figure, then (asked again) the right one
``retrieval only``      the monthly cap counts as reached: sources only, no chat call
``embedding down``      the query embedding fails: BM25 only, with the banner
``unverified``          both drafts carry a wrong figure: ``verify`` fails, the badge shows
``swimming pool``       ``NOT_IN_SOURCES``: the answer abstains
======================  ==================================================================

The first two are wired by :func:`fake_deps` (a budget check and an
embedder), the rest by the chat model itself.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from hal9000.core.prompt import Prompt
from hal9000.cost.metering import MonthlyCapExceeded
from hal9000.knowledge.answer.prompt import NOT_IN_SOURCES, PromptSource, parse_prompt
from hal9000.knowledge.index.fake import FakeEmbedder, concepts
from hal9000.llm import Usage

__all__ = [
    "FAKE_CHAT_MODEL",
    "TRIGGERS",
    "FakeChat",
    "ScriptedStream",
    "TriggerEmbedder",
    "fake_deps",
    "perturb_first_figure",
]

FAKE_CHAT_MODEL = "fake/chat"

TRIGGERS = {
    "reset": "force a reset",
    "retrieval_only": "retrieval only",
    "embedding_down": "embedding down",
    "unverified": "unverified",
    "unanswerable": "swimming pool",
}


def _words(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


def _usage(prompt: Prompt, text: str) -> Usage:
    prompt_words = sum(len(str(m.get("content", "")).split()) for m in prompt.messages)
    return Usage(
        model=FAKE_CHAT_MODEL,
        prompt=prompt_words,
        cached=0,
        completion=len(text.split()),
        reasoning=0,
    )


class ScriptedStream:
    """A scripted ``stream_response``: each call streams the next reply word by word.

    A reply may be a string, a callable ``prompt -> str``, or an exception to
    raise (after ``fail_after`` deltas). ``calls`` records every prompt.
    """

    model = FAKE_CHAT_MODEL

    def __init__(
        self,
        replies: Sequence[str | Callable[[Prompt], str] | BaseException],
        *,
        fail_after: int = 0,
        on_delta: Callable[[int], None] | None = None,
    ) -> None:
        self.replies = list(replies)
        self.fail_after = fail_after
        self.on_delta = on_delta
        self.calls: list[Prompt] = []
        self.kwargs: list[dict[str, Any]] = []
        self.closed = 0

    def __call__(
        self,
        prompt: Prompt,
        tier: str = "fast",
        *,
        on_usage: Callable[[Usage], None] | None = None,
        **kw: Any,
    ) -> Iterator[str | Usage]:
        self.calls.append(prompt)
        self.kwargs.append(dict(kw))
        if not self.replies:
            raise AssertionError(f"ScriptedStream ran out of replies on call {len(self.calls)}")
        reply = self.replies.pop(0)
        return self._stream(prompt, reply, on_usage)

    def _stream(self, prompt, reply, on_usage) -> Iterator[str | Usage]:
        try:
            if isinstance(reply, BaseException):
                for n in range(self.fail_after):
                    yield f"partial{n} "
                raise reply
            text = reply(prompt) if callable(reply) else reply
            for n, word in enumerate(_words(text)):
                if self.on_delta is not None:
                    self.on_delta(n)
                yield word
            usage = _usage(prompt, text)
            if on_usage is not None:
                on_usage(usage)
            yield usage
        except GeneratorExit:
            self.closed += 1
            raise


# ------------------------------------------------------------ the fake model
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_FIGURE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
_DATED = re.compile(r"dated (\w+) \d{1,2}, (\d{4})")


def perturb_first_figure(text: str) -> str | None:
    """``$1,234.00`` → ``$1,324.00``; ``2009`` → ``2008``. ``None`` when there is no figure."""
    match = _FIGURE.search(text)
    if match is None:
        return None
    token = match.group(0)
    digits = [i for i, ch in enumerate(token) if ch.isdigit()]
    chars = list(token)
    if len(digits) >= 3 and chars[digits[1]] != chars[digits[2]]:
        chars[digits[1]], chars[digits[2]] = chars[digits[2]], chars[digits[1]]
    else:
        last = digits[-1]
        chars[last] = str((int(chars[last]) - 1) % 10)
    return text[: match.start()] + "".join(chars) + text[match.end() :]


def _question_concepts(question: str) -> set[str]:
    lower = question.lower()
    for phrase in TRIGGERS.values():
        lower = lower.replace(phrase, " ")
    return set(concepts(lower))


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        out.extend(part.strip() for part in _SENTENCE.split(line) if part.strip())
    return out


def _label(source: PromptSource) -> str:
    title = source.meta_line.split(" · ", 1)[0].strip()
    dated = _DATED.search(source.meta_line)
    return f"{title} ({dated.group(1)} {dated.group(2)})" if dated else title


class FakeChat:
    """The extractive chat model of ``kb serve --fake`` (see the module notes)."""

    model = FAKE_CHAT_MODEL

    def __init__(self, *, delay_s: float = 0.02) -> None:
        self.delay_s = delay_s
        self.calls: list[Prompt] = []

    def reply(self, prompt: Prompt) -> str:
        question, sources, issues = parse_prompt(prompt)
        lower = question.lower()
        if TRIGGERS["unanswerable"] in lower:
            return NOT_IN_SOURCES
        wanted = _question_concepts(question)
        best: tuple[PromptSource, list[str]] | None = None
        for source in sources:  # the top-ranked source that mentions the question
            sentences = _sentences(source.text)
            scores = [len(wanted & set(concepts(s))) for s in sentences]
            if scores and max(scores) > 0:
                i = scores.index(max(scores))
                best = (source, sentences[i : i + 2])
                break
        if best is None:
            return NOT_IN_SOURCES
        source, chosen = best
        body = " ".join(chosen)
        cite = f"[{source.marker}]"
        right = f"{_label(source)}: {body} {cite}"
        wrong_body = perturb_first_figure(body)
        if TRIGGERS["reset"] in lower and not issues:
            figure = _FIGURE.search(wrong_body or "")
            return f"The amount was {figure.group(0)} {cite}." if figure else f"{body} [S99]"
        if TRIGGERS["unverified"] in lower:
            return f"{_label(source)}: {wrong_body} {cite}" if wrong_body else f"{body} [S99]"
        return right

    def __call__(
        self,
        prompt: Prompt,
        tier: str = "fast",
        *,
        on_usage: Callable[[Usage], None] | None = None,
        **kw: Any,
    ) -> Iterator[str | Usage]:
        self.calls.append(prompt)
        text = self.reply(prompt)
        return self._stream(prompt, text, on_usage)

    def _stream(self, prompt, text, on_usage) -> Iterator[str | Usage]:
        for word in _words(text):
            if self.delay_s:
                time.sleep(self.delay_s)
            yield word
        usage = _usage(prompt, text)
        if on_usage is not None:
            on_usage(usage)
        yield usage

    def generate(
        self,
        prompt: Prompt,
        tier: str = "fast",
        *,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> str:
        """Condensing: the last question, as asked (the fake has no memory to resolve)."""
        user = next(m["content"] for m in reversed(prompt.messages) if m["role"] == "user")
        text = str(user).rsplit("Last question:", 1)[-1].strip()
        if on_usage is not None:
            on_usage(_usage(prompt, text))
        return text


class TriggerEmbedder(FakeEmbedder):
    """The fake embedder, failing for a query containing ``embedding down``."""

    def __call__(self, texts, model=None, dimensions=None, **kw):  # type: ignore[override]
        if any(TRIGGERS["embedding_down"] in text.lower() for text in texts):
            raise RuntimeError("fake mode: the embedding service is unavailable")
        return super().__call__(texts, model, dimensions, **kw)


def _budget_check(question: str) -> None:
    if TRIGGERS["retrieval_only"] in question.lower():
        raise MonthlyCapExceeded("fake mode: the monthly cap counts as reached")


def fake_deps(*, delay_s: float = 0.02):
    """:class:`~hal9000.knowledge.answer.pipeline.AnswerDeps` for ``kb serve --fake``."""
    from hal9000.knowledge.answer.pipeline import AnswerDeps

    chat = FakeChat(delay_s=delay_s)
    return AnswerDeps(
        stream_fn=chat,
        generate_fn=chat.generate,
        meter=None,
        embed_fn=TriggerEmbedder(),
        budget_check=_budget_check,
    )
