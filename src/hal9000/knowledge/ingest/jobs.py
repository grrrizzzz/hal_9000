"""Background jobs: the ``jobs`` table, one worker thread and progress events (plan §5.1).

Ingestion and profile builds run as jobs, one at a time, on a single
background thread (:class:`JobWorker`), so a slow parse never blocks a
question and two jobs never write the same profile at once.

- **Lifecycle:** ``queued`` → ``running`` → ``done`` or ``failed`` (with its
  error). Claiming a job is one atomic ``UPDATE``, so a CLI process and the
  service can share a database without running a job twice.
- **Crashes:** a job left ``running`` by a process that is gone (killed, power
  cut) is marked ``failed`` when a worker starts (:func:`recover_interrupted`).
  A job still owned by another live process on this host is left alone.
- **Retry:** :func:`retry_job` puts a ``failed`` job back in the queue with
  its original parameters (``jobs.params``); the Library calls it through
  ``POST /api/jobs/{id}/retry``. Ingestion is idempotent, so a retried job
  replaces any partial state and reuses cached embeddings and labels.
- **Progress:** every change to a job is published on a :class:`ProgressBus`
  as the :class:`~hal9000.knowledge.contracts.Job` contract. KWP6's SSE
  endpoint subscribes (:meth:`ProgressBus.asyncio_queue` bridges into an
  event loop); the CLI prints from it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import socket
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hal9000.knowledge.contracts import Job

__all__ = [
    "BUS",
    "INTERRUPTED",
    "JobHandler",
    "JobNotFound",
    "JobNotRetryable",
    "JobWorker",
    "ProgressBus",
    "Reporter",
    "WriterLock",
    "create_job",
    "get_job",
    "job_params",
    "list_jobs",
    "recover_interrupted",
    "retry_job",
    "update_job",
    "worker_id",
]

log = logging.getLogger(__name__)

INTERRUPTED = "interrupted: the process running this job stopped before it finished"
_ERROR_CHARS = 500

Reporter = Callable[[float, str], None]
#: ``handler(conn, job, report) -> final detail``; raising fails the job.
JobHandler = Callable[[sqlite3.Connection, Job, Reporter], str | None]


class JobNotFound(LookupError):
    pass


class JobNotRetryable(RuntimeError):
    """Only a ``failed`` job can be retried."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


# ------------------------------------------------------------------- pub/sub
class ProgressBus:
    """In-process publish/subscribe of :class:`Job` updates (thread-safe)."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Job], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Job], None]) -> Callable[[], None]:
        """Call ``callback(job)`` on every update; returns the unsubscribe function."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(self, job: Job) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(job)
            except Exception:  # a broken listener never breaks a job
                log.exception("job progress subscriber failed")

    @contextmanager
    def listen(self, maxsize: int = 10_000) -> Iterator[queue.Queue[Job]]:
        """A queue receiving updates for the duration of the block (oldest dropped when full)."""
        inbox: queue.Queue[Job] = queue.Queue(maxsize=maxsize)

        def put(job: Job) -> None:
            try:
                inbox.put_nowait(job)
            except queue.Full:
                try:
                    inbox.get_nowait()
                except queue.Empty:  # pragma: no cover - raced with the consumer
                    pass
                inbox.put_nowait(job)

        unsubscribe = self.subscribe(put)
        try:
            yield inbox
        finally:
            unsubscribe()

    def asyncio_queue(
        self, loop: asyncio.AbstractEventLoop | None = None, *, maxsize: int = 1000
    ) -> tuple[asyncio.Queue[Job], Callable[[], None]]:
        """A bounded ``asyncio.Queue`` fed from the worker thread, and its unsubscribe (SSE).

        When the queue is full the oldest update is dropped (the newest state of
        a job is what matters). Once the loop has closed, the subscription
        removes itself.
        """
        loop = loop or asyncio.get_running_loop()
        inbox: asyncio.Queue[Job] = asyncio.Queue(maxsize=maxsize)

        def deliver(job: Job) -> None:  # runs on the loop
            if inbox.full():
                inbox.get_nowait()
            inbox.put_nowait(job)

        def put(job: Job) -> None:
            try:
                loop.call_soon_threadsafe(deliver, job)
            except RuntimeError:  # the loop is closed: this listener is gone
                unsubscribe()

        unsubscribe = self.subscribe(put)
        return inbox, unsubscribe


#: The process-wide bus the service's worker publishes on and SSE subscribes to.
BUS = ProgressBus()


# --------------------------------------------------------------------- rows
_COLUMNS = "id, kind, status, progress, detail, error, file_sha, created_at, updated_at"


def _job(row: sqlite3.Row | tuple) -> Job:
    id_, kind, status, progress, detail, error, file_sha, created, updated = tuple(row)
    return Job(
        id=id_,
        kind=kind,
        status=status,
        progress=min(max(float(progress), 0.0), 1.0),
        detail=detail,
        error=error,
        file_sha=file_sha,
        created_at=datetime.fromisoformat(created),
        updated_at=datetime.fromisoformat(updated),
    )


def create_job(
    conn: sqlite3.Connection,
    kind: str,
    *,
    file_sha: str | None = None,
    params: Mapping[str, Any] | None = None,
    detail: str = "queued",
    bus: ProgressBus | None = BUS,
) -> Job:
    """Queue a job (commits) and publish it."""
    now = _now()
    with conn:
        cur = conn.execute(
            "INSERT INTO jobs (kind, status, progress, detail, file_sha, params, created_at, "
            "updated_at) VALUES (?, 'queued', 0, ?, ?, ?, ?, ?)",
            (kind, detail, file_sha, json.dumps(dict(params or {})), now, now),
        )
    job = get_job(conn, int(cur.lastrowid))
    assert job is not None
    if bus is not None:
        bus.publish(job)
    return job


def get_job(conn: sqlite3.Connection, job_id: int) -> Job | None:
    row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _job(row) if row else None


def job_params(conn: sqlite3.Connection, job_id: int) -> dict[str, Any]:
    row = conn.execute("SELECT params FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise JobNotFound(f"no job {job_id}")
    return {k: v for k, v in json.loads(row[0] or "{}").items() if not k.startswith("_")}


def list_jobs(
    conn: sqlite3.Connection, *, status: str | None = None, limit: int = 100
) -> list[Job]:
    """Newest first."""
    if status is None:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM jobs WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    return [_job(row) for row in rows]


def update_job(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    bus: ProgressBus | None = BUS,
    **fields: Any,
) -> Job:
    """Set ``status``, ``progress``, ``detail`` and/or ``error`` (commits) and publish."""
    allowed = {"status", "progress", "detail", "error"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"cannot update job fields {sorted(unknown)}")
    if "progress" in fields:
        fields["progress"] = min(max(float(fields["progress"]), 0.0), 1.0)
    sets = ", ".join(f"{name} = ?" for name in fields)
    with conn:
        conn.execute(
            f"UPDATE jobs SET {sets}, updated_at = ? WHERE id = ?",
            (*fields.values(), _now(), job_id),
        )
    job = get_job(conn, job_id)
    if job is None:
        raise JobNotFound(f"no job {job_id}")
    if bus is not None:
        bus.publish(job)
    return job


def _owner(params: str) -> str:
    try:
        return str(json.loads(params or "{}").get("_worker", ""))
    except ValueError:
        return ""


def _alive_elsewhere(owner: str) -> bool:
    """Whether ``owner`` (``host:pid``) is another process that is still running here."""
    host, _, pid = owner.rpartition(":")
    if host != socket.gethostname() or not pid.isdigit() or int(pid) == os.getpid():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists, owned by another user
        return True
    return True


def recover_interrupted(conn: sqlite3.Connection, *, bus: ProgressBus | None = BUS) -> list[int]:
    """Mark jobs left ``running`` by a dead process as ``failed``; returns their IDs."""
    rows = conn.execute("SELECT id, params FROM jobs WHERE status = 'running'").fetchall()
    failed: list[int] = []
    for job_id, params in rows:
        if _alive_elsewhere(_owner(params)):
            continue
        update_job(conn, job_id, bus=bus, status="failed", error=INTERRUPTED, detail="interrupted")
        failed.append(int(job_id))
    return failed


def retry_job(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    bus: ProgressBus | None = BUS,
    worker: JobWorker | None = None,
) -> Job:
    """Queue a ``failed`` job again with its original parameters (``POST /api/jobs/{id}/retry``)."""
    with conn:
        cur = conn.execute(
            "UPDATE jobs SET status = 'queued', progress = 0, error = NULL, detail = 'queued "
            "(retry)', updated_at = ? WHERE id = ? AND status = 'failed'",
            (_now(), job_id),
        )
    if cur.rowcount == 0:
        job = get_job(conn, job_id)
        if job is None:
            raise JobNotFound(f"no job {job_id}")
        raise JobNotRetryable(f"job {job_id} is {job.status}; only a failed job can be retried")
    job = get_job(conn, job_id)
    assert job is not None
    if bus is not None:
        bus.publish(job)
    if worker is not None:
        worker.notify()
    return job


# --------------------------------------------------------------- the lock
class WriterLock:
    """One job runner per library (decision on CLI vs service concurrency).

    An exclusive ``flock`` on a lock file under ``data/knowledge/``. The
    service's worker holds it for its lifetime; ``kb ingest`` runs jobs itself
    only when it can take it, and otherwise just queues them for the service.
    The lock is released when the process dies, however it dies.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Take the lock without waiting; ``True`` if this object now holds it."""
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+")  # noqa: SIM115 - held open for the lock's lifetime
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - no flock (Windows): single-process use only
            pass
        except OSError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(worker_id())
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is not None:
            self._handle.close()  # closing the descriptor releases the flock
            self._handle = None

    def __enter__(self) -> WriterLock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# ------------------------------------------------------------------- worker
class JobWorker:
    """Runs queued jobs one at a time: on a background thread, or synchronously.

    Args:
        db_path: ``knowledge.db``; the worker thread opens its own connection.
        handlers: job kind → handler.
        connect: opens a migrated connection (default :func:`open_db`).
        lock: the library's :class:`WriterLock`; the thread takes it before it
            runs anything (waiting while a ``kb ingest`` holds it).
    """

    #: longest pause after repeated failures of the loop itself
    MAX_BACKOFF_S = 30.0

    def __init__(
        self,
        db_path: Path | str,
        handlers: Mapping[str, JobHandler],
        *,
        bus: ProgressBus | None = BUS,
        poll_s: float = 2.0,
        connect: Callable[[Path | str], sqlite3.Connection] | None = None,
        lock: WriterLock | None = None,
    ) -> None:
        from hal9000.knowledge.storage import open_db

        self.db_path = db_path
        self.handlers = dict(handlers)
        self.bus = bus
        self.poll_s = poll_s
        self.lock = lock
        self._connect = connect or open_db
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._thread: threading.Thread | None = None
        self._recovered = False

    # -- the thread ------------------------------------------------------------
    def start(self) -> None:
        """Recover interrupted jobs (now, if the lock is free), then start the one thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self.lock is None or self.lock.acquire():
            conn = self._connect(self.db_path)
            try:
                recover_interrupted(conn, bus=self.bus)
                self._recovered = True
            finally:
                conn.close()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="kb-job-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 30) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def notify(self) -> None:
        """Wake the thread: a job was queued."""
        self._idle.clear()
        self._wake.set()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the thread finds the queue empty."""
        return self._idle.wait(timeout)

    def _loop(self) -> None:
        conn: sqlite3.Connection | None = None
        backoff = 0.0
        try:
            while not self._stop.is_set():
                job: Job | None = None
                try:
                    if self.lock is not None and not self.lock.acquire():
                        self._stop.wait(self.poll_s)  # a kb ingest is running jobs itself
                        continue
                    if conn is None:
                        conn = self._connect(self.db_path)
                    if not self._recovered:
                        recover_interrupted(conn, bus=self.bus)
                        self._recovered = True
                    self._wake.clear()
                    job = self._claim(conn)
                    if job is None:
                        self._idle.set()
                        self._wake.wait(self.poll_s)
                        continue
                    self._idle.clear()
                    self._run(conn, job)
                    backoff = 0.0
                except Exception:
                    # the bookkeeping itself failed (a locked or full database): never die
                    log.exception("the knowledge job worker hit an error; backing off")
                    if conn is not None:
                        self._salvage(conn, job)
                    backoff = min(max(backoff * 2, 0.5), self.MAX_BACKOFF_S)
                    self._stop.wait(backoff)
        finally:
            if conn is not None:
                conn.close()
            if self.lock is not None:
                self.lock.release()

    def _salvage(self, conn: sqlite3.Connection, job: Job | None) -> None:
        """After a loop error: roll back, and fail the job it was on if it is still running."""
        try:
            if conn.in_transaction:
                conn.rollback()
            if job is not None and job.id is not None:
                current = get_job(conn, job.id)
                if current is not None and current.status == "running":
                    update_job(
                        conn, job.id, bus=self.bus, status="failed", detail="failed",
                        error="the job worker failed while recording this job",
                    )  # fmt: skip
        except Exception:  # still failing: the next pass (or a restart) recovers it
            log.exception("could not record the failure of job %s", job.id if job else None)

    # -- one job ---------------------------------------------------------------
    def _claim(self, conn: sqlite3.Connection, job_id: int | None = None) -> Job | None:
        owner = json.dumps(worker_id())
        target = "?" if job_id is not None else (
            "(SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1)"
        )  # fmt: skip
        args: tuple = (job_id,) if job_id is not None else ()
        with conn:
            row = conn.execute(
                f"UPDATE jobs SET status = 'running', progress = 0, detail = 'starting', "
                f"error = NULL, params = json_set(params, '$._worker', json(?)), updated_at = ? "
                f"WHERE id = {target} AND status = 'queued' RETURNING id",
                (owner, _now(), *args),
            ).fetchone()
        if row is None:
            return None
        job = get_job(conn, int(row[0]))
        if job is not None and self.bus is not None:
            self.bus.publish(job)
        return job

    def _run(self, conn: sqlite3.Connection, job: Job) -> Job:
        """Run one claimed job. Bookkeeping errors propagate (the loop recovers from them)."""
        assert job.id is not None
        job_id = job.id

        def report(progress: float, detail: str) -> None:
            update_job(conn, job_id, bus=self.bus, progress=progress, detail=detail[:500])

        handler = self.handlers.get(job.kind)
        try:
            if handler is None:
                raise RuntimeError(f"no handler for {job.kind} jobs")
            detail = handler(conn, job, report)
        except Exception as error:
            if conn.in_transaction:
                conn.rollback()
            message = f"{type(error).__name__}: {error}"[:_ERROR_CHARS]
            log.warning("job %s (%s) failed: %s", job_id, job.kind, message)
            return update_job(
                conn, job_id, bus=self.bus, status="failed", error=message, detail="failed"
            )
        except BaseException as error:  # SystemExit, KeyboardInterrupt: record, then go
            try:
                if conn.in_transaction:
                    conn.rollback()
                update_job(
                    conn, job_id, bus=self.bus, status="failed", detail="failed",
                    error=f"stopped: {type(error).__name__}"[:_ERROR_CHARS],
                )  # fmt: skip
            except Exception:  # pragma: no cover - a restart's recovery marks it instead
                log.exception("could not record that job %s stopped", job_id)
            raise
        return update_job(
            conn, job_id, bus=self.bus, status="done", progress=1.0, detail=detail or "done"
        )

    # -- synchronous use (CLI, tests) ------------------------------------------
    def run_job(self, conn: sqlite3.Connection, job_id: int) -> Job:
        """Run job ``job_id`` here and now if it is queued; return its final state.

        If another worker claimed it first, wait for that worker to finish it.
        """
        job = self._claim(conn, job_id)
        if job is not None:
            return self._run(conn, job)
        while True:
            current = get_job(conn, job_id)
            if current is None:
                raise JobNotFound(f"no job {job_id}")
            if current.status in ("done", "failed"):
                return current
            if current.status == "queued":  # released and re-queued meanwhile
                job = self._claim(conn, job_id)
                if job is not None:
                    return self._run(conn, job)
            self._stop.wait(0.2)

    def run_pending(self, conn: sqlite3.Connection) -> list[Job]:
        """Run every queued job here, oldest first."""
        finished: list[Job] = []
        while (job := self._claim(conn)) is not None:
            finished.append(self._run(conn, job))
        return finished
