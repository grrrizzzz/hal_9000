"""Answering questions about the house (plan §7): prompt, streamed generation, verify.

- ``prompt``: the numbered, fenced sources and the policy read fresh per question.
- ``generate``: metered streaming generation with cancellation and a deadline.
- ``verify``: citation validity and figure grounding.
- ``pipeline``: retrieve → abstain → generate → verify (→ reset, regenerate once),
  degraded modes, every question observed as a trace.
- ``store``: conversations, messages and feedback.
- ``fake``: the offline chat model behind ``kb serve --fake``.
- ``eval_binding``: the answer function the evaluation harness uses.

Nothing is imported here, so importing one module stays light.
"""
