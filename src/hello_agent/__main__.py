"""The command line: ``python -m hello_agent "Dr. Maria Chen is here ..."``.

This module is thin on purpose. It does four things and delegates the rest:
read the flags, fail fast and legibly if no provider is configured, run the
greeter, and print what the agent produced.

**Nothing here decides agent behaviour.** Which tools exist, which capabilities
run, which environment executes -- all of that is
:func:`~hello_agent.agents.greeter.build_greeter_agent` reading a
:class:`~hello_agent.config.Features`. The CLI's only contribution is where that
``Features`` comes from: ``AGENT_ENABLE_*`` first, then any ``--flag`` given
here on top. So a flag can be switched on for one run without exporting
anything, and switched on for every run without typing anything.

**Exit codes.** ``0`` success, ``2`` for anything the user can fix by changing
their invocation or their environment -- no key, no visitor text, an unusable
stdin -- and ``1`` if the agent ran but produced nothing to print. A missing API
key is a configuration mistake, not a crash, so it gets one line on stderr and
no traceback.

**Staged review and stdin.** ``--staged-review`` sends every write to a reviewer,
and the default reviewer asks a human on stdin. Piping the visitor's details in
on that same stdin would leave nobody to answer, so this CLI requires the text as
an argument when staged review is on. In CI, or anywhere without a terminal, the
human reviewer refuses by design -- use ``--reviewer llm`` (or ``AGENT_REVIEWER=llm``)
if you want writes to go through unattended.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable

from dotenv import find_dotenv, load_dotenv

from hello_agent import __version__, config
from hello_agent.agents.greeter import build_greeter_agent, greeter_context_props
from hello_agent.config import Features
from hello_agent.core.memory import Memory

__all__ = ["build_parser", "main"]

PROMPT = "Tell me about the visitor: "

#: flag -> what it does and what it costs, one line each, for ``--help``. The
#: environment variable is appended automatically, so the two cannot drift.
FLAG_HELP = {
    "time_aware": "the agent knows the date and time, refreshed every turn. No extra calls.",
    "plan_first": "write a plan before the first move. One extra LLM call per run.",
    "progress_tracking": "report progress each turn. One extra LLM call per turn.",
    "verbose_logging": "one structured log line per loop event, on stderr. No extra calls.",
    "delegation": (
        "add call_agent and a translator sub-agent, so a visitor who named a "
        "language is greeted in it. Costs a whole sub-agent run when used."
    ),
    "staged_review": (
        "environmental safety: guestbook writes are held for review before they "
        "happen, and undo_last_guestbook_entry joins the toolkit. See the note below."
    ),
    "dynamic_experts": (
        "add the four self-prompting tools, including create_and_consult_expert. "
        "Four more tools for the model to choose between."
    ),
}

_EPILOG = """\
environment variables:
  ANTHROPIC_API_KEY / OPENAI_API_KEY   one of these is required
  LLM_PROVIDER              force "anthropic" or "openai" regardless of keys
  AGENT_MODEL_POWERFUL      override the powerful tier's model string
  AGENT_MODEL_FAST          override the fast tier's model string
  AGENT_MAX_TOKENS          output ceiling per LLM call (default 1024)
  AGENT_MAX_ITERATIONS      agent loop ceiling (default 10)
  AGENT_MAX_DURATION_SECONDS  wall-clock ceiling for a run (default 180)
  AGENT_GUESTBOOK_PATH      where the guestbook lives (default guestbook.json)
  AGENT_TIMEZONE            used by --time-aware (default America/Chicago)
  AGENT_REVIEWER            "human" or "llm" for --staged-review (default human)

examples:
  python -m hello_agent "Dr. Maria Chen is here, marine biologist, from Osaka."
  echo "Sam, here about the reef samples." | python -m hello_agent
  python -m hello_agent --delegation "Yuki Tanaka, prefers Japanese."
  AGENT_REVIEWER=llm python -m hello_agent --staged-review "Ada Lovelace, visiting."

note: with --staged-review the visitor's details must be an argument, not piped
in, because the human reviewer needs stdin to answer on. Without a terminal the
reviewer refuses every write -- that is fail-closed behaviour, not a bug; pass
--reviewer llm for unattended runs.
"""


class UsageError(RuntimeError):
    """Something the user can fix by invoking the command differently."""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, including the ``--help`` text that documents the flags."""
    parser = argparse.ArgumentParser(
        prog="python -m hello_agent",
        description=(
            "Greet a visitor and sign the guestbook. Describe the visitor in your own "
            "words; the agent extracts their details, composes a greeting, records the "
            "visit, and prints the greeting."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "visitor",
        nargs="?",
        help="What you know about the visitor, in free-form prose. "
        "Omit it to read from stdin, or to be prompted at a terminal.",
    )
    parser.add_argument("--version", action="version", version=f"hello-agent {__version__}")
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Print only the agent's final message, without the tool-by-tool trace.",
    )
    parser.add_argument(
        "--reviewer",
        choices=("human", "llm"),
        help="Who reviews staged writes under --staged-review. Sets AGENT_REVIEWER.",
    )
    flags = parser.add_argument_group(
        "feature flags",
        "All off by default. Each is also settable with the environment variable "
        "named beside it, and the flag wins when both are given.",
    )
    for name in Features().__dataclass_fields__:
        flags.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            action="store_true",
            help=f"[{Features.env_var(name)}=1] {FLAG_HELP[name]}",
        )
    return parser


def features_from(args: argparse.Namespace) -> Features:
    """``AGENT_ENABLE_*`` first, then any ``--flag`` on the command line on top.

    Flags only ever turn things **on**: there is no ``--no-delegation``, because
    a run you can reproduce by exporting nothing is worth more than symmetry.
    """
    features = Features.from_env()
    for name in features.__dataclass_fields__:
        if getattr(args, name, False):
            setattr(features, name, True)
    return features


def configure_logging() -> None:
    """Make ``LoggingCapability`` visible on stderr.

    Scoped to the ``hello_agent`` logger rather than ``basicConfig``, so turning
    the agent's own logging on does not also turn on every library's.
    """
    logger = logging.getLogger("hello_agent")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[log] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def read_visitor_text(text: str | None, features: Features, stdin=None) -> str:
    """The visitor's details, from the argument, from stdin, or from a prompt."""
    if text and text.strip():
        return text.strip()

    if features.staged_review:
        raise UsageError(
            "with --staged-review, pass the visitor's details as an argument rather "
            "than on stdin -- the human reviewer needs stdin to answer on. "
            'Try: python -m hello_agent --staged-review "Ada Lovelace, visiting."'
        )

    stream = sys.stdin if stdin is None else stdin
    try:
        interactive = stream.isatty()
    except Exception:
        interactive = False

    piped = "" if interactive else (stream.read() or "")
    if interactive:
        try:
            piped = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            piped = ""

    if not piped.strip():
        raise UsageError(
            "no visitor to greet. Pass their details as an argument, or pipe them in: "
            'python -m hello_agent "Dr. Maria Chen is here, marine biologist, from Osaka."'
        )
    return piped.strip()


def _parsed(item: dict) -> dict | None:
    """A memory item's content as a dict, or ``None`` if it is not JSON at all."""
    content = item.get("content")
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _outcome(envelope: dict) -> str:
    """One short line describing what a tool result envelope carried."""
    value = envelope.get("result") if envelope.get("tool_executed") else envelope.get("error")
    # ensure_ascii=False: a greeting translated into Japanese should read as
    # Japanese in the trace, not as a row of \uXXXX escapes.
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(str(text).split())
    return text if len(text) <= 100 else text[:100] + "..."


def tool_trace(memory: Memory) -> list[tuple[str, str]]:
    """``(tool, outcome)`` for each call the run made, reconstructed from memory.

    Memory is the only record the loop keeps, which is the point: anything the
    CLI can show you, a capability or a later run could read just as well.
    """
    trace: list[tuple[str, str]] = []
    pending: str | None = None
    for item in memory.get_memories():
        parsed = _parsed(item)
        if parsed is None:
            continue
        if item.get("type") == "assistant" and "tool" in parsed:
            pending = str(parsed["tool"])
        elif pending is not None and "tool_executed" in parsed:
            trace.append((pending, _outcome(parsed)))
            pending = None
    return trace


def final_message(memory: Memory) -> str:
    """What the agent produced for the user: the last tool result, as text.

    On a completed run that is ``terminate``'s message, unchanged (ruling R7).
    On a run that hit its iteration ceiling it is whatever the last tool said,
    which is more useful to print than nothing.
    """
    for item in reversed(memory.get_memories()):
        parsed = _parsed(item)
        if parsed is not None and "tool_executed" in parsed:
            return _full_result(parsed)
    return ""


def _full_result(envelope: dict) -> str:
    """The untruncated payload of a result envelope."""
    value = envelope.get("result") if envelope.get("tool_executed") else envelope.get("error")
    if value is None:
        return ""
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, indent=2, default=str, ensure_ascii=False)
    )


def _is_auth_error(error: BaseException) -> bool:
    """True if ``error`` is the provider rejecting our credentials.

    Imported lazily and matched narrowly: a wrong key is the second thing a new
    user hits after a missing one, and a stack trace is the wrong answer to it.
    Everything else propagates, because a traceback is exactly the right answer
    to a real bug.
    """
    try:
        from litellm.exceptions import AuthenticationError
    except ImportError:  # pragma: no cover -- litellm is a hard dependency
        return False
    return isinstance(error, AuthenticationError)


def main(argv: list[str] | None = None, generate_response: Callable | None = None) -> int:
    """Run the greeter once. Returns the process exit code.

    ``generate_response`` exists for the test suite: pass a ``FakeLLM`` and the
    whole command runs with no key, no network, and no provider resolution.
    """
    # Before anything reads the environment. `usecwd=True` searches from the
    # directory the user is standing in rather than from this file's location,
    # which is what makes `cp .env.example .env && python -m hello_agent` work
    # in a fresh clone. `override=False` means a real exported variable always
    # beats a stale .env, and a missing file is not an error -- exporting keys
    # directly stays perfectly valid.
    load_dotenv(find_dotenv(usecwd=True), override=False)

    args = build_parser().parse_args(argv)
    features = features_from(args)

    if args.reviewer:
        # Read at call time by safety.review.reviewer_for, so setting it here --
        # before the environment is constructed -- is enough.
        os.environ["AGENT_REVIEWER"] = args.reviewer
    if features.verbose_logging:
        configure_logging()

    try:
        if generate_response is None:
            config.resolve_provider()  # fail fast, before any work is done
        visitor = read_visitor_text(args.visitor, features)
    except (config.ConfigError, UsageError) as error:
        print(f"hello-agent: {error}", file=sys.stderr)
        return 2

    agent = build_greeter_agent(features, generate_response)
    try:
        memory = agent.run(
            visitor,
            action_context_props=greeter_context_props(features, generate_response),
        )
    except Exception as error:  # noqa: BLE001 -- narrowed by _is_auth_error
        if not _is_auth_error(error):
            raise
        provider = config.resolve_provider()
        print(
            f"hello-agent: the {provider} API rejected the credentials "
            f"({type(error).__name__}). The key was found, so this is a bad or "
            f"expired key rather than a missing one -- check "
            f"{'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'}.",
            file=sys.stderr,
        )
        return 2

    if not args.quiet:
        for tool, outcome in tool_trace(memory):
            print(f"[greeter] {tool:<24} -> {outcome}")
        print()

    message = final_message(memory)
    if not message.strip():
        print(
            "hello-agent: the agent stopped without producing a final message. "
            "Raise AGENT_MAX_ITERATIONS, or run with --verbose-logging to see why.",
            file=sys.stderr,
        )
        return 1
    print(message)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
