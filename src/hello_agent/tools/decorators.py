"""``@register_tool``: the decorator that turns a function into an agent tool.

A tool needs three things the model can read -- a name, a description, and a
JSON Schema for its arguments -- and one thing only Python needs: the function.
Written out by hand that is four places to edit and four places to drift, so we
don't: the function *is* the source of truth. ``__name__`` becomes the tool
name, ``__doc__`` the model-facing description, and the signature plus type
hints the JSON Schema. Registration is a side effect of import, which is why
:mod:`hello_agent.tools` imports every tool module.

The subtle part is what the schema leaves *out*. Three kinds of parameter are
dependencies, supplied by :class:`~hello_agent.core.environment.PythonEnvironment`
rather than by the model, so they are stripped from the schema entirely:
``action_context`` (the :class:`~hello_agent.core.action_context.ActionContext`),
``action_agent`` (the running agent), and ``_foo`` (``action_context.get("foo")``).

The model never sees them, so it never has to invent an auth token, a database
handle, or a reference to itself -- and it cannot forge one either. That single
exclusion rule is what makes dependency injection invisible at the prompt.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_origin, get_type_hints

from hello_agent.core import registry

__all__ = ["get_json_type", "get_tool_metadata", "register_tool"]

#: Parameters the environment injects by name, never supplied by the model.
RESERVED_PARAMETERS = ("action_context", "action_agent")

#: Python type -> JSON Schema type; anything absent falls back to ``"string"``.
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def get_json_type(python_type: Any) -> str:
    """Map a Python type hint onto a JSON Schema type name.

    Generic aliases are reduced to their origin, so ``list[str]``, ``List[str]``
    and bare ``list`` all answer ``"array"``. Anything unrecognised -- a custom
    class, ``Optional[str]``, a missing hint -- answers ``"string"``.

    Example:
        >>> get_json_type(int), get_json_type(list[str]), get_json_type(object)
        ('integer', 'array', 'string')
    """
    origin = get_origin(python_type)
    if origin is not None:
        python_type = origin
    try:
        return _JSON_TYPES.get(python_type, "string")
    except TypeError:  # unhashable annotation, e.g. a list literal
        return "string"


def _is_dependency(param_name: str) -> bool:
    """Is this parameter injected rather than model-supplied?

    The most load-bearing predicate in the tool layer: it decides what the model
    is shown. See the module docstring for the three shapes.
    """
    return param_name in RESERVED_PARAMETERS or param_name.startswith("_")


def _resolve_hints(func: Callable) -> dict[str, Any]:
    """Type hints for ``func``, or ``{}`` if they cannot be resolved.

    ``get_type_hints`` evaluates string annotations, which raises if a hint names
    something the module never imported at runtime -- a common consequence of
    ``if TYPE_CHECKING``. A tool with an unresolvable hint must still register;
    its parameters simply default to ``"string"``.
    """
    try:
        return get_type_hints(func)
    except Exception:
        return {}


def _build_schema(func: Callable) -> dict:
    """Derive the model-facing JSON Schema from ``func``'s signature.

    Dependencies are skipped, as are ``*args``/``**kwargs`` (the model cannot
    meaningfully fill them). No default means required; no type hint means string.
    """
    signature = inspect.signature(func)
    type_hints = _resolve_hints(func)
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for param_name, param in signature.parameters.items():
        if _is_dependency(param_name):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema["properties"][param_name] = {"type": get_json_type(type_hints.get(param_name, str))}
        if param.default is inspect.Parameter.empty:
            schema["required"].append(param_name)

    return schema


def get_tool_metadata(
    func: Callable,
    tool_name: str | None = None,
    description: str | None = None,
    parameters_override: dict | None = None,
    terminal: bool = False,
    tags: list[str] | None = None,
) -> dict:
    """Extract everything needed to register ``func`` as a tool.

    Args:
        func: The function to describe.
        tool_name: Overrides ``func.__name__``.
        description: Overrides the docstring.
        parameters_override: A hand-written JSON Schema. When given, **no**
            introspection happens at all -- including the dependency exclusion,
            so an override must not mention injected parameters.
        terminal: ``True`` if running this tool ends the agent loop.
        tags: Labels used to slice the catalogue into per-agent registries.

    Returns:
        ``{"tool_name", "description", "parameters", "function", "terminal",
        "tags"}``. The key is ``tool_name``, not ``name``; the entry stored in
        :data:`hello_agent.core.registry.tools` drops it, being the dict key there.
    """
    docstring = inspect.cleandoc(func.__doc__ or "No description provided.")
    return {
        "tool_name": tool_name or func.__name__,
        "description": description or docstring,
        "parameters": _build_schema(func) if parameters_override is None else parameters_override,
        "function": func,
        "terminal": terminal,
        "tags": list(tags or []),
    }


def _index_by_tag(tool_name: str, tags: list[str]) -> None:
    """Point every tag at ``tool_name``, exactly once.

    ``tools[name] = ...`` de-duplicates for free; a list does not. Re-importing
    a module (or reloading it in a test) must not leave a tag holding the same
    name twice, so stale entries are dropped before the new ones go in.
    """
    for names in registry.tools_by_tag.values():
        if tool_name in names:
            names.remove(tool_name)
    for tag in tags:
        registry.tools_by_tag.setdefault(tag, []).append(tool_name)


def register_tool(
    tool_name: str | None = None,
    description: str | None = None,
    parameters_override: dict | None = None,
    terminal: bool = False,
    tags: list[str] | None = None,
) -> Callable[[Callable], Callable]:
    """Publish the decorated function to the global tool catalogue.

    The function is returned unchanged, so it stays an ordinary callable you can
    unit-test or call directly.

    Example:
        >>> from hello_agent.core import registry
        >>> @register_tool(tags=["demo"])
        ... def shout(text: str, times: int = 1) -> str:
        ...     '''Repeat TEXT in upper case.'''
        ...     return (text.upper() + " ") * times
        >>> registry.tools["shout"]["parameters"]["required"]
        ['text']
        >>> registry.tools_by_tag["demo"]
        ['shout']
        >>> shout("hi"), registry.clear_tools()
        ('HI ', None)
    """

    def decorator(func: Callable) -> Callable:
        metadata = get_tool_metadata(
            func=func,
            tool_name=tool_name,
            description=description,
            parameters_override=parameters_override,
            terminal=terminal,
            tags=tags,
        )
        name = metadata["tool_name"]
        registry.tools[name] = {
            "description": metadata["description"],
            "parameters": metadata["parameters"],
            "function": metadata["function"],
            "terminal": metadata["terminal"],
            "tags": metadata["tags"],
        }
        _index_by_tag(name, metadata["tags"])
        return func

    return decorator
