"""Planning tools: the two LLM calls that make an agent think before and between.

Both tools here answer a question the agent loop never asks on its own. The loop
is memoryless about *strategy*: every iteration it re-reads memory and picks one
tool. That works for three-step tasks and drifts badly for thirty-step ones.

- :func:`create_plan` runs once, before the first iteration, and writes a
  numbered plan into memory so every later prompt carries it.
- :func:`track_progress` runs after an iteration and writes a short assessment
  of what just happened and what is left.

**What they cost.** Each call is a full extra LLM round-trip, and the plan or
report it returns stays in memory, so every *subsequent* prompt is longer too.
``create_plan`` costs one call per run. ``track_progress`` costs one call per
tracked iteration -- at the default ``track_frequency=1``, that doubles the
number of LLM calls a run makes. MATE's token efficiency says spend that only
where the task is long enough to drift; the lean greeter is not, which is why
the ``planning`` tag is absent from the default registry and both capabilities
that drive these tools are off by default.

Both carry the ``planning`` tag. They are registered as tools -- the agent can
call them itself when the tag is enabled -- but their usual callers are
:class:`~hello_agent.capabilities.plan_first.PlanFirstCapability` and
:class:`~hello_agent.capabilities.progress_tracking.ProgressTrackingCapability`,
which call them as plain Python functions.
"""

from __future__ import annotations

from hello_agent.core.action_context import ActionContext
from hello_agent.core.actions import ActionRegistry
from hello_agent.core.memory import Memory
from hello_agent.tools.decorators import register_tool
from hello_agent.tools.self_prompting import prompt_llm

__all__ = ["create_plan", "track_progress"]

#: Memory item types worth showing a planner: the task, and framing about it.
PLANNING_MEMORY_TYPES = ("user", "system")

_PLAN_TEMPLATE = """\
Given the task in memory and the available tools, create a detailed plan.
Think through this step by step:

1. First, identify the key components of the task
2. Consider what tools you have available
3. Break down the task into logical steps
4. For each step, specify:
   - What needs to be done
   - What tool(s) will be used
   - What information is needed
   - What the expected outcome is

Write your plan in clear, numbered steps. Each step should be specific and actionable.

Available tools:
{tools}

Task context from memory:
{memory}

Create a plan that accomplishes this task effectively."""

_PROGRESS_TEMPLATE = """\
Given the current task and available tools, generate a progress report.
Think through this step by step:

1. Identify the key components of the task and the intended outcome.
2. Assess the progress made so far based on available information.
3. Identify any blockers or issues preventing completion.
4. Suggest the next steps to move forward efficiently.
5. Recommend any tool usage that might help complete the task.

Write your progress report in clear, structured points.

Available tools:
{tools}

Task context from memory:
{memory}

Provide a well-organized report on the current progress and next steps."""


def _tool_descriptions(action_registry: ActionRegistry | None) -> str:
    """One ``- name: description`` line per available tool."""
    if action_registry is None:
        return "(no tools are available)"
    return "\n".join(
        f"- {action.name}: {action.description}" for action in action_registry.get_actions()
    )


def _memory_context(memory: Memory | None) -> str:
    """The task and its framing, flattened for a prompt.

    Assistant turns are left out on purpose: a planner wants the *task*, not a
    transcript of the agent arguing with itself. That omission is also the
    cheapest token saving available here.
    """
    if memory is None:
        return "(no memory is available)"
    return "\n".join(
        f"{item['type']}: {item['content']}"
        for item in memory.get_memories()
        if item.get("type") in PLANNING_MEMORY_TYPES
    )


@register_tool(tags=["planning"])
def create_plan(
    action_context: ActionContext,
    _memory: Memory,
    _action_registry: ActionRegistry,
) -> str:
    """Write a detailed, numbered execution plan for the current task.

    Reads the task from memory and the tools from the registry, then returns a
    step-by-step plan naming, for each step, what to do, which tool to use, what
    information it needs, and what it should produce.

    Costs one extra LLM call, and the returned plan lengthens every later prompt.
    Worth it for multi-step work; wasteful for a task you can finish in two calls.

    Returns:
        The plan, as numbered prose.
    """
    prompt = _PLAN_TEMPLATE.format(
        tools=_tool_descriptions(_action_registry), memory=_memory_context(_memory)
    )
    return prompt_llm(action_context=action_context, prompt=prompt)


@register_tool(tags=["planning"])
def track_progress(
    action_context: ActionContext,
    _memory: Memory,
    _action_registry: ActionRegistry,
) -> str:
    """Assess how far the current task has got, and what should happen next.

    Reviews the task and what has been done so far, then reports what is
    complete, what is blocking, and which tool to reach for next.

    Costs one extra LLM call every time it runs, and the report it returns stays
    in memory. Driven automatically it is the single most expensive thing in this
    template -- see ProgressTrackingCapability's track_frequency.

    Returns:
        The progress report, as structured prose.
    """
    prompt = _PROGRESS_TEMPLATE.format(
        tools=_tool_descriptions(_action_registry), memory=_memory_context(_memory)
    )
    return prompt_llm(action_context=action_context, prompt=prompt)
