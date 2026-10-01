"""The answer function the evaluation harness uses (KWP5: judge-prep and S5).

``binding(conn, settings, *, meter, run)`` returns ``question -> AnswerOutput``
(:mod:`hal9000.knowledge.eval.types`) over the same pipeline as the chat:

- every paid call (query embedding, condensing, the answer) goes through the
  command's ``meter`` as agent ``knowledge-eval`` (rule K9), so evaluation
  spend never counts toward the chat's monthly cap;
- the index profile is ``run.config["profile"]`` when the run names one (it
  must be built), else the active profile;
- evaluation answers are not written to the ``traces`` table (they are not
  household questions);
- the sources handed back are the numbered sources with their full text,
  which the judge grades faithfulness against, and the answer's citations,
  which the Judge calibration tab opens.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from hal9000.knowledge.answer.pipeline import AnswerDeps, AnswerPipeline
from hal9000.knowledge.answer.prompt import NumberedSource
from hal9000.knowledge.config import EVAL_AGENT, ConfigMissing
from hal9000.knowledge.eval.types import AnswerFn, AnswerOutput, Source

__all__ = ["binding"]


def _profile(conn: sqlite3.Connection, run: Any):
    from hal9000.knowledge.index.profiles import get_profile, profile_status

    name = (getattr(run, "config", None) or {}).get("profile") if run is not None else None
    if not name:
        return None
    profile = get_profile(conn, name)
    if profile is None or profile_status(conn, name) != "ready":
        raise ConfigMissing(
            f"the run's index profile {name!r} is not built; build it before answering with it"
        )
    return profile


def binding(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    meter: Any = None,
    run: Any = None,
    deps: AnswerDeps | None = None,
) -> AnswerFn:
    """``question -> AnswerOutput`` with the configuration of ``run`` (see the module notes)."""
    pipeline = AnswerPipeline(
        settings,
        deps or AnswerDeps(meter=meter, agent=EVAL_AGENT),
        profile=_profile(conn, run),
        record_traces=False,
    )

    def answer(question: str) -> AnswerOutput:
        kept: list[NumberedSource] = []
        result = pipeline.ask(conn, question, emit=lambda event: None, on_sources=kept.extend)
        return AnswerOutput(
            text=result.text,
            sources=tuple(
                Source(marker=s.marker, text=s.text, title=s.document.title, meta=s.meta_line)
                for s in kept
            ),
            abstained=result.outcome == "abstained",
            cost_usd=result.cost_usd,
            citations=tuple(result.citations),
        )

    answer.pipeline = pipeline  # type: ignore[attr-defined]  (for inspection and tests)
    return answer
