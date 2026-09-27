"""The greeting tools: a fixed-schema extractor and a concierge persona.

Two patterns from the self-prompting notes, and the MATE model-efficiency demo.

**Fixed-schema extraction.** ``extract_visitor_details`` is the specialised
extractor from ``structured_data_extraction.md``: rather than letting the agent
invent a schema per visitor with the general ``prompt_llm_for_json`` tool, the
schema is written once, here, and every visitor comes back shaped the same way.
The agent supplies prose and gets fields. It never sees the schema, so it cannot
drift.

**Model efficiency.** Pulling five obvious fields out of a sentence is pawn work,
so it runs on ``fast_llm``. Writing something a stranger will read as welcoming
is queen work, so ``compose_greeting`` runs on ``llm``, the powerful tier. Same
run, same context, two models -- that is MATE's first letter, and the whole cost
of it is which key these two functions read.

**Policy as prompt.** ``compose_greeting`` reads ``config/greeting_policy.txt``
*at call time* and pastes it into the prompt. Editing that file changes how the
agent greets people, with no code change, no restart and no redeploy -- the
``purchasing_rules.txt`` pattern from ``invoice_processing_with_experts.md``. A
missing file is not an error; a default stands in.

Both tools call :mod:`hello_agent.tools.self_prompting` as plain functions, which
is why the ``prompts`` tools can stay flag-gated while these two ship by default.
A tool is still just a function.
"""

from __future__ import annotations

from pathlib import Path

from hello_agent.core.action_context import ActionContext
from hello_agent.tools.decorators import register_tool
from hello_agent.tools.self_prompting import prompt_expert, prompt_llm_for_json

__all__ = [
    "DEFAULT_POLICY",
    "POLICY_FILE",
    "VISITOR_FIELDS",
    "VISITOR_SCHEMA",
    "compose_greeting",
    "extract_visitor_details",
    "policy_paths",
    "read_greeting_policy",
]

#: Looked for relative to the working directory first, then to the repo root.
POLICY_FILE = "config/greeting_policy.txt"

#: src/hello_agent/tools/greeting.py -> the checkout root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_POLICY = (
    "No house rules are on file at config/greeting_policy.txt. Fall back on good "
    "manners: be warm, be brief, use the visitor's name, mention one specific thing "
    "the visitor's own details support, and never invent facts about them."
)

#: field -> what the model should put in it. Add a line here and every visitor
#: from then on carries that field; the agent's goals do not change.
VISITOR_FIELDS: dict[str, str] = {
    "name": "The visitor's name, with any title or honorific they use.",
    "role": "Their profession, or the role they are here in, if stated.",
    "affiliation": "Their organisation, institution, or employer, if stated.",
    "origin": "Where they have travelled from, if stated.",
    "purpose": "What brings them here, in a short phrase, if stated.",
    "previous_visits": "What the text says about earlier visits, e.g. 'second this year'.",
    "language": "The language they would prefer to be greeted in, if stated.",
    "notes": "Anything else worth knowing that no other field captures.",
}

#: The one shape every visitor comes back in. Fixed on purpose -- see the module
#: docstring. ``name`` is the only field worth failing over.
VISITOR_SCHEMA: dict = {
    "type": "object",
    "required": ["name"],
    "properties": {
        field: {"type": "string", "description": text} for field, text in VISITOR_FIELDS.items()
    },
}

_EXTRACTION_TEMPLATE = """\
Extract the visitor's details from the text below.

Record only what the text actually says. Leave a field out entirely rather than \
guessing at it, and never invent a name, a title, or a place. If a detail is \
implied rather than stated -- "back again" implying a previous visit -- record \
the implication in the words the text used.

Text:
{document_text}"""

_CONCIERGE = """\
A hotel concierge of thirty years' standing, the kind guests ask for by name. \
You have greeted heads of state, exhausted graduate students, and people who \
arrived at the wrong building, and every one of them left feeling expected. You \
are warm without being effusive, specific without being intrusive, and you know \
that the fastest way to make someone feel unwelcome is to say something generic \
at them. You never invent a fact about a guest; you work with what you are told."""

_GREETING_TEMPLATE = """\
Greet this visitor, who has just arrived.

What we know about them:
{visitor}

The house rules for greeting visitors here, which you must follow:

{policy}

Write only the greeting itself -- the words you would say aloud. No preamble, no \
stage directions, no explanation of your choices."""


def policy_paths() -> list[Path]:
    """Where :func:`read_greeting_policy` looks, in order.

    The working directory comes first, so a deployment can override the policy
    shipped in the checkout simply by having its own ``config/`` directory.
    """
    return [Path(POLICY_FILE), _REPO_ROOT / POLICY_FILE]


def read_greeting_policy() -> str:
    """Read the house rules from disk, or return :data:`DEFAULT_POLICY`.

    Called on every :func:`compose_greeting`, never cached: the point of keeping
    the policy in a text file is that editing it takes effect immediately.
    """
    for path in policy_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if text.strip():
            return text.strip()
    return DEFAULT_POLICY


@register_tool(tags=["greeting"])
def extract_visitor_details(action_context: ActionContext, document_text: str) -> dict:
    """Pull a visitor's details out of free-form text into structured fields.

    Give it whatever you were told about the visitor, in whatever shape it
    arrived -- a sentence, an email, a scribbled note. It returns the parts that
    matter, ready to hand to compose_greeting and sign_guestbook.

    Args:
        document_text: Everything you know about the visitor, as text. Pass it
            whole; do not summarise it first.

    Returns:
        ``{"name": ..., "role": ..., "origin": ..., ...}``. Only ``name`` is
        guaranteed; fields the text did not mention are simply absent.
    """
    # MATE, model efficiency: mechanical extraction runs on the cheap tier. The
    # fast callable is handed to prompt_llm_for_json as its "llm" so the retry
    # and fence-stripping logic is reused rather than reimplemented here.
    fast = action_context.get("fast_llm") or action_context.get("llm")
    details = prompt_llm_for_json(
        action_context=ActionContext({"llm": fast}),
        schema=VISITOR_SCHEMA,
        prompt=_EXTRACTION_TEMPLATE.format(document_text=document_text),
    )

    if not isinstance(details, dict) or not str(details.get("name", "")).strip():
        raise ValueError(
            "The text does not name the visitor, and a guestbook entry needs a name. "
            "Ask the user who is visiting, then call extract_visitor_details again "
            "with their answer included."
        )
    return {key: value for key, value in details.items() if value not in (None, "", [], {})}


@register_tool(tags=["greeting"])
def compose_greeting(
    action_context: ActionContext,
    visitor: dict,
    _greeting_policy: str | None = None,
) -> str:
    """Write a greeting for a visitor, following this house's greeting policy.

    Consults a concierge persona and returns the greeting itself, ready to say
    aloud and to pass to sign_guestbook. Call it after extract_visitor_details,
    with the details that tool returned.

    Args:
        visitor: The visitor's details, as returned by extract_visitor_details.
            The more fields it carries, the more specific the greeting.

    Returns:
        The greeting, as plain text.
    """
    policy = _greeting_policy if _greeting_policy else read_greeting_policy()
    return prompt_expert(
        action_context=action_context,
        description_of_expert=_CONCIERGE,
        prompt=_GREETING_TEMPLATE.format(visitor=visitor, policy=policy),
    )
