"""The long-lived parse process of an ingestion pipeline (plan §5.1 step 4).

It lives here, apart from :mod:`hal9000.knowledge.ingest.pipeline`, so the
child imports only the parsers. Where the platform has it (Linux, macOS), the
child is forked from multiprocessing's *fork server*, which has these parsers
(and PDFium) preloaded and never imports the program's ``__main__``: the fork
server starts once per process, and each parse process after it starts in
milliseconds. Elsewhere it is spawned.

multiprocessing normally makes every child import the program's ``__main__``
(the CLI, or pytest), which costs more than the fork itself. The parse
process runs only this module's code, so it is started with ``__main__``
hidden for the instant of ``start()`` (:func:`_without_main`).
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
import types
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hal9000.knowledge.ingest.parsers import (
    ParsedFile,
    ParseError,
    Parser,
    ParseTimeout,
    parse_file,
)
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = ["ParseProcess"]

#: imported once by the fork server, so no parse pays for them (``stage()`` needs the contracts)
_PRELOAD = [
    "hal9000.knowledge.index.parsing",
    "hal9000.knowledge.contracts",
    "pypdfium2",
    "docx",
    "selectolax.parser",
]
_context = None
_READY_TIMEOUT_S = 120.0


def _start_context():
    """The fork server (parsers preloaded, no ``__main__`` import), else spawn."""
    global _context
    if _context is None:
        if "forkserver" in multiprocessing.get_all_start_methods():
            context = multiprocessing.get_context("forkserver")
            context.set_forkserver_preload(_PRELOAD)
        else:  # pragma: no cover - Windows
            context = multiprocessing.get_context("spawn")
        _context = context
    return _context


@contextmanager
def _without_main() -> Iterator[None]:
    """Hide ``__main__`` while a child is started, so the child doesn't import it."""
    real = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        if real is not None:
            sys.modules["__main__"] = real


def _parse_server(conn) -> None:  # pragma: no cover - runs in the child process
    """The long-lived parse process: say it is ready, then parse until the pipe closes."""
    try:
        conn.send(("ready", None))
    except OSError:
        return
    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            return
        if request is None:
            return
        parser, path, max_chars = request
        try:
            result = parse_file(path, parser=parser, isolate=False, max_chars=max_chars)
            conn.send(("ok", result))
        except ParseError as error:
            conn.send(("parse_error", str(error)))
        except BaseException as error:  # noqa: BLE001 - reported to the parent
            conn.send(("error", f"{type(error).__name__}: {error}"))


def _terminate(process, conn) -> None:
    try:
        conn.close()
    except OSError:  # pragma: no cover
        pass
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():  # pragma: no cover - terminate is normally enough
            process.kill()
            process.join()


class ParseProcess:
    """One long-lived child process that parses files, reused across files.

    Starting a process per file costs about 0.3 s (plan §5.1 step 4 wants
    isolation and a time limit, not a new process each time). A parse that
    runs past its limit, or crashes the child, kills it; the next parse starts
    a fresh one. ``starts`` counts the processes started.
    """

    def __init__(self) -> None:
        self._process = None
        self._conn = None
        self._finalizer: weakref.finalize | None = None
        self._lock = threading.Lock()
        self.starts = 0

    def warm(self) -> None:
        """Start the process now (and wait until it is ready), not on the first parse."""
        with self._lock:
            self._ensure()

    def _ensure(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._kill()
        context = _start_context()
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=_parse_server, args=(child,), name="kb-parse", daemon=True)
        with _without_main():
            process.start()
        child.close()
        self._process, self._conn = process, parent
        self._finalizer = weakref.finalize(self, _terminate, process, parent)
        self.starts += 1
        # wait for the child's own start-up (it imports the program's modules), so the
        # first parse's time limit measures parsing only
        try:
            ready = parent.poll(_READY_TIMEOUT_S) and parent.recv()[0] == "ready"
        except (EOFError, OSError):
            ready = False
        if not ready:
            self._kill()
            raise ParseError("Could not start the parser process.")

    def _kill(self) -> None:
        if self._finalizer is not None:
            self._finalizer()  # terminates the process and closes the pipe
        self._process = self._conn = self._finalizer = None

    def close(self) -> None:
        with self._lock:
            self._kill()

    def parse(self, path: Path, parser: Parser, *, timeout_s: float, max_chars: int) -> ParsedFile:
        """Parse ``path`` in the child, timed as ``stage("parse")``."""
        with stage(Stage.PARSE), self._lock:
            self._ensure()
            assert self._conn is not None and self._process is not None
            self._conn.send((parser, str(path), max_chars))
            if not self._conn.poll(timeout_s):
                self._kill()
                raise ParseTimeout(
                    f"Could not parse {path.name}: parsing took longer than {timeout_s:g} s and "
                    "was stopped."
                )
            try:
                kind, payload = self._conn.recv()
            except (EOFError, OSError):
                code = self._process.exitcode
                self._kill()
                raise ParseError(
                    f"Could not parse {path.name}: the {parser.name} parser crashed "
                    f"(exit code {code})."
                ) from None
        if kind == "ok":
            return payload
        if kind == "parse_error":
            raise ParseError(payload)
        raise ParseError(f"Could not parse {path.name}: {payload}")
