"""The four capabilities this template ships. Every one of them is off by default.

A capability is how you change the way an agent *runs* without editing the loop.
The loop already calls all ten hooks on every run
(:mod:`hal9000.core.capabilities`), so switching one on is a list entry::

    Agent(..., capabilities=[TimeAwareCapability()])

and in the assembled greeter it is one environment variable, because
:class:`~hal9000.config.Features` maps a flag to a constructor:

====================  ===============================  ============================
``Features`` flag     Capability                       Cost per run
====================  ===============================  ============================
``time_aware``        :class:`TimeAwareCapability`     none -- no LLM calls
``plan_first``        :class:`PlanFirstCapability`     one extra LLM call
``progress_tracking`` :class:`ProgressTrackingCapability`  one per tracked iteration
``verbose_logging``   :class:`LoggingCapability`       none -- no LLM calls
====================  ===============================  ============================

Order matters when you enable more than one: the ``process_*`` hooks compose
left to right, so the *last* capability in the list sees what the earlier ones
produced. Putting :class:`LoggingCapability` last is usually what you want --
it then records the prompt the LLM actually receives, rather than the one
before the other capabilities got to it.
"""

from __future__ import annotations

from hal9000.capabilities.logging import LoggingCapability
from hal9000.capabilities.plan_first import PlanFirstCapability
from hal9000.capabilities.progress_tracking import ProgressTrackingCapability
from hal9000.capabilities.time_aware import TimeAwareCapability

__all__ = [
    "LoggingCapability",
    "PlanFirstCapability",
    "ProgressTrackingCapability",
    "TimeAwareCapability",
]
