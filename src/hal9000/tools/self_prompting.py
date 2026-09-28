"""Self-prompting tools: the agent calling an LLM on purpose, not by accident.

Every turn of the agent loop already calls an LLM -- that is how the agent
decides what to do. These four tools are different: they let the agent call an
LLM as a *unit of work*, on a sub-problem it identified. Deciding "I need this
prose turned into fields" is reasoning; turning it into fields is a tool call
that happens to be implemented with a language model instead of a parser. That
split is the whole subject of ``separation_of_reasoning.md``.

All four read the powerful tier from ``action_context.get("llm")``, and the
model never sees that parameter -- ``@register_tool`` strips ``action_context``
from the schema. All four carry the ``prompts`` tag and are **off by default**:
a large, general capability the lean greeter does not need. ``tools/greeting.py``
calls two of them as ordinary Python functions instead, which is rather the
point -- a tool is still just a function.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from hal9000.core.action_context import ActionContext
from hal9000.core.prompt import Prompt
from hal9000.tools.decorators import register_tool

__all__ = [
    "create_and_consult_expert",
    "prompt_expert",
    "prompt_llm",
    "prompt_llm_for_json",
]

#: How many times :func:`prompt_llm_for_json` asks before giving up.
JSON_ATTEMPTS = 3

_PERSONA_TEMPLATE = """\
Create a detailed description of an expert in {domain} who would be ideally \
suited to address the following problem:

{problem}

Your description should include: the expert's background and experience; their \
specific areas of specialization within {domain}; their approach to \
problem-solving; and the unique perspective they bring to this type of challenge."""

_CONSULTATION_TEMPLATE = """\
Create a detailed consultation prompt for an expert in {domain} addressing the \
following problem:

{problem}

The prompt should guide the expert to provide comprehensive insights and \
actionable recommendations specific to this problem."""


def _require_llm(action_context: ActionContext) -> Callable:
    """Fetch the powerful-tier LLM, or fail with something the agent can act on."""
    generate_response = action_context.get("llm")
    if generate_response is None:
        raise ValueError(
            "No LLM is available in this ActionContext under the key 'llm'. "
            "Do not retry this tool; report the problem and terminate."
        )
    return generate_response


def _strip_code_fence(text: str) -> str:
    """Return the body of a Markdown code fence, or ``text`` unchanged.

    Models wrap JSON in ```` ```json ```` fences however firmly you ask them not
    to. Searching forward for the opening fence and back for the closing one
    survives a reply that wraps the fence in commentary.
    """
    stripped = text.strip()
    if "```json" in stripped:
        start = stripped.find("```json") + len("```json")
        end = stripped.rfind("```")
        return stripped[start:end].strip() if end > start else stripped[start:].strip()
    if stripped.startswith("```"):
        end = stripped.rfind("```")
        body = stripped[3:end] if end > 3 else stripped[3:]
        return body.split("\n", 1)[-1].strip() if "\n" in body else body.strip()
    return stripped


def _ask(generate_response: Callable, content: str) -> str:
    """One user-role turn -- the whole LLM contract, in one line."""
    return generate_response(Prompt(messages=[{"role": "user", "content": content}]))


@register_tool(tags=["prompts"])
def prompt_llm(action_context: ActionContext, prompt: str) -> str:
    """Ask a language model a free-form question and get its answer as text.

    Use this to think something through, summarise, rewrite, or translate, when
    no other tool fits. If you need the answer as structured fields rather than
    prose, use prompt_llm_for_json instead.

    Args:
        prompt: The complete question or instruction to send.

    Returns:
        The model's reply, as plain text.
    """
    return _ask(_require_llm(action_context), prompt)


@register_tool(tags=["prompts"])
def prompt_llm_for_json(action_context: ActionContext, schema: dict, prompt: str) -> dict:
    """Ask a language model for JSON matching a schema. Use this whenever you
    need structured data rather than prose.

    The schema is sent as a system instruction and the reply is parsed. If the
    reply is not valid JSON the request is retried, three attempts in total.

    Args:
        schema: A JSON Schema describing the object you want back.
        prompt: What to extract, and the text to extract it from.

    Returns:
        The parsed object.
    """
    generate_response = _require_llm(action_context)
    instruction = (
        "You MUST produce output that adheres to the following JSON schema:"
        f"\n\n{json.dumps(schema)}"
    )
    last_error: Exception | None = None

    for _attempt in range(JSON_ATTEMPTS):
        try:
            response = generate_response(
                Prompt(
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt},
                    ]
                )
            )
            return json.loads(_strip_code_fence(response))
        except Exception as error:
            # Broad, following the source note: a malformed reply and a flaky
            # transport are both worth one more try. The error *type* goes into
            # the final message so a real bug is not disguised as bad JSON.
            last_error = error

    raise ValueError(
        f"The language model did not return valid JSON after {JSON_ATTEMPTS} attempts "
        f"(last error: {type(last_error).__name__}: {last_error}). Try again with a "
        "simpler, flatter schema and a prompt that asks for fewer fields at once."
    ) from last_error


@register_tool(tags=["prompts"])
def prompt_expert(action_context: ActionContext, description_of_expert: str, prompt: str) -> str:
    """Consult a named expert persona and get their answer.

    The persona is installed as a system message, so the model answers from
    inside that expertise rather than describing it from outside.

    Args:
        description_of_expert: Who the expert is, in enough detail to shape the
            answer. "A concierge of thirty years, warm but never cloying" beats
            "a greeting expert".
        prompt: The specific question or task for them, inside their domain.

    Returns:
        The expert's response, as plain text.
    """
    generate_response = _require_llm(action_context)
    persona = f"Act as the following expert and respond accordingly: {description_of_expert}"
    return generate_response(
        Prompt(
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": prompt},
            ]
        )
    )


@register_tool(tags=["prompts"])
def create_and_consult_expert(
    action_context: ActionContext, expertise_domain: str, problem_description: str
) -> str:
    """Invent an expert suited to a problem, then consult them about it.

    Three model calls: write the persona, write the consultation prompt, then
    ask. Use this when no pre-defined expert fits -- it is the difference
    between a fixed roster of consultants and being able to hire one.

    Args:
        expertise_domain: The field the problem needs, e.g. "conference
            interpreting".
        problem_description: The problem, in full. Both the persona and the
            question are written from this, so vagueness here costs twice.

    Returns:
        The invented expert's insights and recommendations.
    """
    generate_response = _require_llm(action_context)
    fill = {"domain": expertise_domain, "problem": problem_description}

    persona_description = _ask(generate_response, _PERSONA_TEMPLATE.format(**fill))
    consultation_prompt = _ask(generate_response, _CONSULTATION_TEMPLATE.format(**fill))

    return prompt_expert(
        action_context=action_context,
        description_of_expert=persona_description,
        prompt=consultation_prompt,
    )
