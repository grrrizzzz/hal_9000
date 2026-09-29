"""Where the CLI gets retrieval (KWP4) and answering (KWP6) from, without importing them.

The harness only knows the protocols in :mod:`hal9000.knowledge.eval.types`.
A package that can provide one registers a factory here, or exposes a
``binding`` function in the optional modules below, which are imported
lazily if they exist:

- ``hal9000.knowledge.retrieve.eval_binding.binding(conn, settings, *, meter, run)
  -> RetrievalBinding``: retrieval over the active profile (or the profile of
  ``run``, an :class:`~hal9000.knowledge.contracts.EvalRun`, when given).
- ``hal9000.knowledge.answer.eval_binding.binding(conn, settings, *, meter, run)
  -> AnswerFn``: answers with the configuration of ``run``.

``meter`` is the command's :class:`~hal9000.cost.metering.Meter` (or None):
any paid call the binding makes must go through it with
``agent="knowledge-eval"`` (rule K9).

Until one exists the commands that need it say so and exit.
"""

from __future__ import annotations

import importlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from hal9000.knowledge.eval.types import AnswerFn, RetrieveFn

__all__ = [
    "NotWired",
    "RetrievalBinding",
    "answerer_for",
    "register_answerer",
    "register_retriever",
    "retriever_for",
]

RETRIEVE_MODULE = "hal9000.knowledge.retrieve.eval_binding"
ANSWER_MODULE = "hal9000.knowledge.answer.eval_binding"


class NotWired(RuntimeError):
    """No implementation of retrieval or answering is available yet."""


@dataclass(frozen=True)
class RetrievalBinding:
    """Retrieval over one index profile, as the harness uses it."""

    retrieve: RetrieveFn
    profile_id: int | None
    profile_name: str
    #: the profile's chunk size (the parent size for parent-child): budgets below it are N/A
    chunk_size: int | None = None


RetrieverFactory = Callable[..., RetrievalBinding]
AnswererFactory = Callable[..., AnswerFn]

_retriever: RetrieverFactory | None = None
_answerer: AnswererFactory | None = None


def register_retriever(factory: RetrieverFactory | None) -> None:
    global _retriever
    _retriever = factory


def register_answerer(factory: AnswererFactory | None) -> None:
    global _answerer
    _answerer = factory


def _module_binding(name: str):
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and name.startswith(error.name):
            return None
        raise
    return getattr(module, "binding", None)


def retriever_for(
    conn: sqlite3.Connection, settings: object, *, meter: object = None, run: object = None
) -> RetrievalBinding:
    factory = _retriever or _module_binding(RETRIEVE_MODULE)
    if factory is None:
        raise NotWired("retrieval is not available yet (KWP4 has not been merged)")
    return factory(conn, settings, meter=meter, run=run)


def answerer_for(
    conn: sqlite3.Connection, settings: object, *, meter: object = None, run: object = None
) -> AnswerFn:
    factory = _answerer or _module_binding(ANSWER_MODULE)
    if factory is None:
        raise NotWired("answering is not available yet (KWP6 has not been merged)")
    return factory(conn, settings, meter=meter, run=run)
