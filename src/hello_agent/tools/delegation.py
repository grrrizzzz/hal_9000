"""Delegation: one agent calling another, and the four ways they share memory.

A sub-agent is not a new kind of thing. It is a tool whose implementation happens
to be another agent loop, which is why delegation needs no change to
:class:`~hello_agent.core.agent.Agent` at all -- only these four functions and
the :class:`~hello_agent.core.registry.AgentRegistry` they look agents up in.

What *does* need deciding is memory: the callee must start from some history, and
the caller must end up with some record. The notes give four answers, and all
four ship because that is a design choice, not a default.

- :func:`call_agent` -- fresh memory down, last item back. An email.
- :func:`call_agent_with_reflection` -- fresh memory down, *everything* back,
  tagged with its source. Ask for the working, not just the answer.
- :func:`hand_off_to_agent` -- the caller's own memory object, written into.
  Taking over, not consulting.
- :func:`call_agent_with_selected_context` -- an LLM picks what is relevant, and
  its reasoning is kept.

**The registry never travels.** The caller's context properties are forwarded
minus :data:`NOT_INHERITED`, which contains ``agent_registry``. A callee has no
directory to look agents up in, so it cannot delegate, so two agents cannot call
each other forever. It is the same mechanism that keeps delegation off by
default: ``agent_registry`` is absent from
:func:`~hello_agent.core.action_context.default_properties` and appears only when
an assembler passes it to ``Agent.run(action_context_props=...)``.

All four carry the ``delegation`` tag and are off by default -- an agent that can
call other agents is a bigger, slower, costlier agent.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from hello_agent.core.action_context import ActionContext
from hello_agent.core.memory import Memory
from hello_agent.tools.decorators import register_tool
from hello_agent.tools.self_prompting import prompt_llm_for_json

__all__ = [
    "NOT_INHERITED",
    "SELECTION_SCHEMA",
    "call_agent",
    "call_agent_with_reflection",
    "call_agent_with_selected_context",
    "hand_off_to_agent",
]

#: Context properties a callee must **not** inherit from its caller.
#: ``agent_registry`` is the load-bearing one -- without it the callee cannot
#: delegate. The rest are per-run resources ``Agent.run`` rebuilds for itself;
#: forwarding them would override the callee's own memory, tools and LLM tiers
#: with the caller's, since caller-supplied props win. Everything else *is*
#: forwarded, which is how an auth token or a house policy reaches a sub-agent.
NOT_INHERITED = frozenset(
    {"agent_registry", "memory", "action_registry", "llm", "powerful_llm", "fast_llm"}
)

#: What the LLM must answer with when choosing memories to share. The
#: ``reasoning`` field is not decoration: it is written back into the caller's
#: memory so there is a record of *why* context was or was not shared.
SELECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "selected_memories": {
            "type": "array",
            "items": {"type": "string", "description": "ID of a memory to include"},
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of why these memories were selected",
        },
    },
    "required": ["selected_memories", "reasoning"],
}

_SELECTION_TEMPLATE = """\
Review these memories and select the ones relevant for this task:

Task: {task}

Available Memories:
{memory_text}

Select memories that provide important context or information for this specific \
task. Explain your selection process."""


def _inherited_props(action_context: ActionContext) -> dict:
    """The caller's context properties, minus the ones in :data:`NOT_INHERITED`."""
    return {
        key: value for key, value in action_context.properties.items() if key not in NOT_INHERITED
    }


def _failure(agent_name: str, error: str) -> dict:
    """A refusal the caller can read and act on, rather than an exception.

    A bad agent name and a sub-agent that fell over are both things the model can
    recover from, so both come back as results.
    """
    return {"success": False, "agent": agent_name, "error": error}


def _resolve_agent(action_context: ActionContext, agent_name: str) -> tuple[Callable, dict]:
    """Look ``agent_name`` up. Returns ``(run_function, None)`` or ``(None, failure)``."""
    agent_registry = action_context.get_agent_registry()
    if agent_registry is None:
        return None, _failure(
            agent_name,
            "No agent registry is available, so there is nobody to delegate to. "
            "Do not retry this tool; complete the task yourself.",
        )
    agent_run = agent_registry.get_agent(agent_name)
    if agent_run is None:
        known = ", ".join(agent_registry.list_agents()) or "none"
        return None, _failure(
            agent_name,
            f"There is no agent named {agent_name!r}. Available agents: {known}. "
            "Call one of those instead, or do the work yourself.",
        )
    return agent_run, None


def _delegate(
    action_context: ActionContext, agent_name: str, task: str, memory: Memory
) -> tuple[Memory, dict]:
    """Run ``agent_name`` on ``task`` with ``memory``.

    ``(result_memory, None)``, or ``(None, failure)``. The sub-agent's whole run
    is contained here, so a failure inside it never escapes the caller's loop.
    """
    agent_run, failure = _resolve_agent(action_context, agent_name)
    if failure is not None:
        return None, failure
    try:
        result_memory = agent_run(
            user_input=task,
            memory=memory,
            action_context_props=_inherited_props(action_context),
        )
    except Exception as error:  # a sub-agent crash is the caller's news, not its death
        return None, _failure(agent_name, f"{type(error).__name__}: {error}")
    if result_memory is None or not result_memory.items:
        return None, _failure(agent_name, "The agent ran but produced no memories.")
    return result_memory, None


def _last_content(memory: Memory) -> str:
    """The final thing the sub-agent said -- its answer, by convention.

    The notes stop at ``items[-1]["content"]``. Here that last item is the
    environment's JSON result envelope, ``{"tool_executed": true, "result":
    ...}``, because ``Agent.update_memory`` writes tool results back as memories.
    Passing that up would force every delegating agent to know the framework's
    internal shapes, so the envelope is opened here; anything else passes through.
    """
    content = memory.items[-1].get("content", "No result")
    if isinstance(content, str):
        try:
            envelope = json.loads(content)
        except ValueError:
            return content
        if isinstance(envelope, dict) and "tool_executed" in envelope:
            result = envelope.get("result", envelope.get("error", "No result"))
            return result if isinstance(result, str) else json.dumps(result, default=str)
    return content


@register_tool(tags=["delegation"])
def call_agent(action_context: ActionContext, agent_name: str, task: str) -> dict:
    """Ask another agent to do a task for you, and get back its answer.

    The other agent starts fresh: it knows only the task you write here, so put
    everything it needs into `task`. You get its conclusion, not its reasoning.
    Use this when you want an answer and do not care how it was reached.

    Args:
        agent_name: Which agent to ask. Use a name you were told exists.
        task: The complete, self-contained request for that agent.

    Returns:
        ``{"success": True, "agent": ..., "result": ...}``, or a result with
        ``"success": False`` and an ``"error"`` explaining what to do instead.
    """
    result_memory, failure = _delegate(action_context, agent_name, task, Memory())
    if failure is not None:
        return failure
    return {"success": True, "agent": agent_name, "result": _last_content(result_memory)}


@register_tool(tags=["delegation"])
def call_agent_with_reflection(action_context: ActionContext, agent_name: str, task: str) -> dict:
    """Ask another agent to do a task, and see how it reached its answer.

    Like call_agent, but everything the other agent thought along the way is
    copied into your own memory, labelled with whose thought it was. Use this
    when the reasoning matters to you and not just the conclusion -- it costs
    you a longer memory on every turn afterwards.

    Args:
        agent_name: Which agent to ask.
        task: The complete, self-contained request for that agent.

    Returns:
        ``{"success": True, "agent": ..., "result": ..., "memories_added": n}``.
    """
    result_memory, failure = _delegate(action_context, agent_name, task, Memory())
    if failure is not None:
        return failure

    caller_memory = action_context.get_memory()
    for item in result_memory.items:
        caller_memory.add_memory(
            # The type marks the source, so a later prompt renders these as the
            # sub-agent's turns rather than as the caller's own decisions.
            {"type": f"{agent_name}_thought", "content": item.get("content")}
        )

    return {
        "success": True,
        "agent": agent_name,
        "result": _last_content(result_memory),
        "memories_added": len(result_memory.items),
    }


@register_tool(tags=["delegation"])
def hand_off_to_agent(action_context: ActionContext, agent_name: str, task: str) -> dict:
    """Hand the whole conversation over to another agent to continue.

    The other agent works in *your* memory: it sees everything that has happened
    so far and what it does is recorded alongside what you did. Use this when
    the task is a continuation rather than a question -- a specialist taking
    over, not a colleague being consulted.

    Args:
        agent_name: Which agent takes over.
        task: What they should do next, given everything already in memory.

    Returns:
        ``{"success": True, "agent": ..., "result": ..., "memory_id": ...}``.
        ``memory_id`` is the shared memory's identity -- proof the handoff was
        not a copy.
    """
    caller_memory = action_context.get_memory()
    result_memory, failure = _delegate(action_context, agent_name, task, caller_memory)
    if failure is not None:
        return failure
    return {
        "success": True,
        "agent": agent_name,
        "result": _last_content(result_memory),
        "memory_id": id(result_memory),
    }


def _select_memories(action_context: ActionContext, task: str) -> tuple[Memory, str]:
    """Ask the LLM which of the caller's memories matter for ``task``.

    Each memory is labelled ``mem_0``, ``mem_1``, ... so the model can refer to
    them precisely; the ids are stripped again before the chosen items are
    handed on. Returns the filtered memory and the model's stated reasoning.
    """
    caller_memory = action_context.get_memory()
    memory_with_ids = [
        {**item, "memory_id": f"mem_{index}"} for index, item in enumerate(caller_memory.items)
    ]
    memory_text = "\n".join(f"Memory {m['memory_id']}: {m.get('content')}" for m in memory_with_ids)

    selection = prompt_llm_for_json(
        action_context=action_context,
        schema=SELECTION_SCHEMA,
        prompt=_SELECTION_TEMPLATE.format(task=task, memory_text=memory_text),
    )

    selected_ids = set(selection.get("selected_memories") or [])
    filtered = Memory()
    for item in memory_with_ids:
        if item["memory_id"] in selected_ids:
            filtered.add_memory({k: v for k, v in item.items() if k != "memory_id"})
    return filtered, selection.get("reasoning", "")


@register_tool(tags=["delegation"])
def call_agent_with_selected_context(
    action_context: ActionContext, agent_name: str, task: str
) -> dict:
    """Delegate a task, sharing only the parts of the conversation that matter.

    A language model reads your memory, picks the items relevant to this task,
    and the other agent starts from just those. Use this when your history is
    long, or when some of it is none of the other agent's business. Its work,
    and the record of what was shared and why, come back into your memory.

    Args:
        agent_name: Which agent to delegate to.
        task: The complete request for that agent.

    Returns:
        ``{"success": True, "agent": ..., "result": ..., "shared_memories": n,
        "selection_reasoning": ...}``.
    """
    # Resolve first: choosing memories costs an LLM call, and there is no point
    # paying for it on the way to "no such agent".
    _, failure = _resolve_agent(action_context, agent_name)
    if failure is not None:
        return failure

    filtered_memory, reasoning = _select_memories(action_context, task)
    # Counted before the run: the sub-agent writes into this same object, so
    # afterwards its length is "shared plus produced" rather than "shared".
    shared_count = len(filtered_memory.items)
    result_memory, failure = _delegate(action_context, agent_name, task, filtered_memory)
    if failure is not None:
        return failure

    caller_memory = action_context.get_memory()
    caller_memory.add_memory(
        {"type": "system", "content": f"Memory selection reasoning: {reasoning}"}
    )
    # Every item, including the ones that were shared in the first place -- the
    # note is explicit about accepting that duplication, and the alternative
    # (diffing memories by identity) buys tidiness at the cost of a subtlety.
    for item in result_memory.items:
        caller_memory.add_memory(item)

    return {
        "success": True,
        "agent": agent_name,
        "result": _last_content(result_memory),
        "shared_memories": shared_count,
        "selection_reasoning": reasoning,
    }
