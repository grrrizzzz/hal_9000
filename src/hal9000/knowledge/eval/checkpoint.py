"""Checkpoints for ``kb gold generate``: nothing paid for is lost, nothing is paid for twice.

A checkpoint is a folder under ``data/knowledge/eval/gold/<id>/`` (git-ignored,
rule K2) holding:

- ``manifest.json``: the plan (one descriptor per generation task), the
  per-category targets the selection fills, which items were already stored
  (question key -> gold item ID) and the status (``running``, ``stopped``,
  ``complete``). Rewritten atomically.
- ``events.jsonl``: one line per finished unit of paid work, appended and
  fsynced as it finishes: a generation task's validated items (``task``), the
  unanswerable drafts (``drafts``) and each checked draft (``check``). A
  truncated last line (the process was killed mid-write) is ignored.

:class:`StopSignals` turns SIGINT (Ctrl-C) and SIGTERM into
:class:`GenerationInterrupted` so the command can save before it exits, and
defers a signal that arrives while a checkpoint line or ledger row is being
written until the write is done.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hal9000.knowledge.contracts import GoldItem

__all__ = [
    "Checkpoint",
    "CheckpointError",
    "GenerationInterrupted",
    "StopSignals",
]

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
EVENTS = "events.jsonl"
FINISHED = "complete"


class CheckpointError(ValueError):
    """No such checkpoint, or it cannot be read."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_id() -> str:
    return f"gold-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


Shield = Callable[[], AbstractContextManager[Any]]


class Checkpoint:
    """The resumable state of one gold generation (see the module docstring).

    ``directory`` None keeps everything in memory (library use and tests).
    """

    def __init__(
        self, directory: Path | None, manifest: dict[str, Any], shield: Shield | None = None
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.shield: Shield = shield or nullcontext
        #: task key -> validated items of a finished generation task
        self.done: dict[str, list[GoldItem]] = {}
        #: unanswerable task key -> the drafts to check ({"question", "difficulty"})
        self.drafts: dict[str, list[dict[str, str]]] = {}
        #: unanswerable task key -> draft index -> the checked item
        self.checks: dict[str, dict[int, GoldItem]] = {}
        self.dropped: list[tuple[str, str]] = []
        self.duplicates = 0

    # ------------------------------------------------------------ create/load
    @classmethod
    def memory(cls, *, tasks: Sequence[Mapping], targets: Mapping[str, int]) -> Checkpoint:
        return cls(None, cls._manifest("memory", tasks, targets, (), False))

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        tasks: Sequence[Mapping],
        targets: Mapping[str, int],
        only: Sequence[str] = (),
        keyword_only: bool = False,
    ) -> Checkpoint:
        checkpoint_id = _new_id()
        directory = root / checkpoint_id
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = cls(
            directory, cls._manifest(checkpoint_id, tasks, targets, only, keyword_only)
        )
        (directory / EVENTS).touch()
        checkpoint._write_manifest()
        return checkpoint

    @staticmethod
    def _manifest(
        checkpoint_id: str,
        tasks: Sequence[Mapping],
        targets: Mapping[str, int],
        only: Sequence[str],
        keyword_only: bool,
    ) -> dict[str, Any]:
        return {
            "id": checkpoint_id,
            "created_at": _now(),
            "updated_at": _now(),
            "status": "running",
            "only": list(only),
            "keyword_only": keyword_only,
            "targets": {c: int(n) for c, n in targets.items()},
            "tasks": [dict(t) for t in tasks],
            "stored": {},
            "spent_usd": 0.0,
            "calls": 0,
            "sessions": 0,
        }

    @classmethod
    def load(cls, root: Path, checkpoint_id: str | None = None) -> Checkpoint:
        """The checkpoint ``checkpoint_id``, or the latest one that isn't complete."""
        if checkpoint_id is None:
            unfinished = [c for c in cls.manifests(root) if c["status"] != FINISHED]
            if not unfinished:
                raise CheckpointError(f"no unfinished gold generation to resume under {root}")
            checkpoint_id = unfinished[-1]["id"]
        if Path(checkpoint_id).name != checkpoint_id or not checkpoint_id:
            raise CheckpointError(f"{checkpoint_id!r} is not a checkpoint ID")
        directory = root / checkpoint_id
        try:
            manifest = json.loads((directory / MANIFEST).read_text())
        except FileNotFoundError:
            raise CheckpointError(f"no checkpoint {checkpoint_id} under {root}") from None
        except (OSError, ValueError) as error:
            raise CheckpointError(f"checkpoint {checkpoint_id} is unreadable: {error}") from None
        checkpoint = cls(directory, manifest)
        checkpoint._replay()
        return checkpoint

    @staticmethod
    def manifests(root: Path) -> list[dict[str, Any]]:
        """Every readable checkpoint's manifest, oldest first."""
        found = []
        for path in sorted(root.glob(f"gold-*/{MANIFEST}")) if root.exists() else []:
            try:
                found.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                log.warning("skipping unreadable checkpoint manifest %s", path)
        return sorted(found, key=lambda m: (m.get("created_at", ""), m.get("id", "")))

    def _replay(self) -> None:
        assert self.directory is not None
        path = self.directory / EVENTS
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                self._apply(json.loads(line))
            except (ValueError, KeyError, TypeError) as error:
                if number == len(lines):  # killed mid-write: that unit of work is redone
                    log.warning("ignoring a truncated last checkpoint line in %s", path)
                    continue
                raise CheckpointError(f"{path}:{number}: {error}") from None

    def _apply(self, event: Mapping[str, Any]) -> None:
        kind = event["type"]
        if kind == "task":
            self.done[event["key"]] = [GoldItem.model_validate(i) for i in event["items"]]
            self.dropped.extend((q, r) for q, r in event.get("dropped", []))
        elif kind == "drafts":
            self.drafts[event["key"]] = [
                {"question": str(d["question"]), "difficulty": str(d["difficulty"])}
                for d in event["drafts"]
            ]
            self.dropped.extend((q, r) for q, r in event.get("dropped", []))
            self.duplicates += int(event.get("duplicates", 0))
        elif kind == "check":
            item = GoldItem.model_validate(event["item"])
            self.checks.setdefault(event["key"], {})[int(event["index"])] = item
        else:
            raise ValueError(f"unknown checkpoint event {kind!r}")

    # --------------------------------------------------------------- writes
    def _append(self, event: dict[str, Any]) -> None:
        with self.shield():
            self._apply(event)
            if self.directory is None:
                return
            line = json.dumps(event, ensure_ascii=False) + "\n"
            with (self.directory / EVENTS).open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

    def _write_manifest(self) -> None:
        if self.directory is None:
            return
        with self.shield():
            self.manifest["updated_at"] = _now()
            target = self.directory / MANIFEST
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.manifest, indent=1, ensure_ascii=False))
            os.replace(tmp, target)

    def record_task(
        self, key: str, items: Sequence[GoldItem], dropped: Sequence[tuple[str, str]]
    ) -> None:
        self._append(
            {
                "type": "task",
                "key": key,
                "items": [i.model_dump(mode="json") for i in items],
                "dropped": [list(d) for d in dropped],
            }
        )

    def record_drafts(
        self,
        key: str,
        drafts: Sequence[Mapping[str, str]],
        dropped: Sequence[tuple[str, str]] = (),
        duplicates: int = 0,
    ) -> None:
        self._append(
            {
                "type": "drafts",
                "key": key,
                "drafts": [dict(d) for d in drafts],
                "dropped": [list(d) for d in dropped],
                "duplicates": duplicates,
            }
        )

    def record_check(self, key: str, index: int, item: GoldItem) -> None:
        self._append(
            {"type": "check", "key": key, "index": index, "item": item.model_dump(mode="json")}
        )

    def record_stored(self, stored: Sequence[tuple[str, int]]) -> None:
        """Items stored in ``gold_items`` from this checkpoint (question key, ID)."""
        self.manifest["stored"].update({key: int(i) for key, i in stored})
        self._write_manifest()

    def close_session(self, *, status: str, spent: float, calls: int) -> None:
        self.manifest["status"] = status
        self.manifest["spent_usd"] = round(float(self.manifest.get("spent_usd", 0)) + spent, 6)
        self.manifest["calls"] = int(self.manifest.get("calls", 0)) + calls
        self.manifest["sessions"] = int(self.manifest.get("sessions", 0)) + 1
        self._write_manifest()

    def reopen(self) -> None:
        self.manifest["status"] = "running"
        self._write_manifest()

    # ---------------------------------------------------------------- reads
    @property
    def id(self) -> str:
        return str(self.manifest["id"])

    @property
    def status(self) -> str:
        return str(self.manifest["status"])

    @property
    def tasks(self) -> list[dict[str, Any]]:
        return list(self.manifest["tasks"])

    @property
    def targets(self) -> dict[str, int]:
        return dict(self.manifest["targets"])

    @property
    def only(self) -> list[str]:
        return list(self.manifest.get("only", []))

    @property
    def stored(self) -> dict[str, int]:
        return dict(self.manifest["stored"])

    def n_items(self) -> int:
        return sum(len(v) for v in self.done.values()) + sum(len(v) for v in self.checks.values())


# -------------------------------------------------------------------- signals
class GenerationInterrupted(KeyboardInterrupt):
    """Ctrl-C or SIGTERM during generation: save what is done, then exit."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum

    @property
    def exit_code(self) -> int:
        return 128 + self.signum

    def describe(self) -> str:
        return "interrupted (Ctrl-C)" if self.signum == signal.SIGINT else "stopped by SIGTERM"


class StopSignals:
    """SIGINT and SIGTERM raise :class:`GenerationInterrupted`, except inside :meth:`shield`.

    A signal inside a shield is raised when the outermost shield exits (so a
    checkpoint line or ledger row is never half-written). During
    :meth:`saving` signals are noted and ignored: the save always finishes.
    Only the main thread can own signal handlers; elsewhere this is inert.
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(self, out=None) -> None:
        self.depth = 0
        self.pending: int | None = None
        self.in_save = False
        self.out = out
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> StopSignals:
        if threading.current_thread() is threading.main_thread():
            for signum in self.SIGNALS:
                self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()

    def _handle(self, signum: int, frame: object) -> None:
        if self.in_save:
            print("hal9000 kb: saving what is done, please wait", file=self.out or sys.stderr)
            return
        if self.depth:
            self.pending = signum
            return
        raise GenerationInterrupted(signum)

    @contextmanager
    def shield(self) -> Iterator[None]:
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
        if self.depth == 0 and self.pending is not None and not self.in_save:
            signum, self.pending = self.pending, None
            raise GenerationInterrupted(signum)

    @contextmanager
    def saving(self) -> Iterator[None]:
        self.in_save = True
        try:
            yield
        finally:
            self.in_save = False
            self.pending = None
