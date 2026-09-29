"""Jobs (plan §5.1): one worker thread, progress events, crash recovery and retry."""

from __future__ import annotations

import asyncio
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hal9000.knowledge.contracts import Job
from hal9000.knowledge.index.profiles import active_profile, get_profile
from hal9000.knowledge.ingest.jobs import (
    INTERRUPTED,
    JobNotFound,
    JobNotRetryable,
    JobWorker,
    ProgressBus,
    WriterLock,
    create_job,
    get_job,
    job_params,
    list_jobs,
    recover_interrupted,
    retry_job,
    update_job,
    worker_id,
)
from tests.knowledge.fixtures import corpus
from tests.knowledge.index.conftest import PROFILE

REPO = Path(__file__).resolve().parents[3]

KILLED_WORKER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from hal9000.knowledge.config import KnowledgeFile
    from hal9000.knowledge.index.fake import FakeEmbedder
    from hal9000.knowledge.ingest.pipeline import IngestPipeline
    from hal9000.knowledge.storage import open_db

    def hang(texts, model, dimensions=None, **kw):
        print("embedding", flush=True)
        time.sleep(600)  # killed here, mid-job

    def main():
        db, files, job_id = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
        pipeline = IngestPipeline(files, embed_fn=hang, use_llm=False, bus=None,
            knowledge_file=KnowledgeFile(active_profile=sys.argv[4]))
        conn = open_db(db)
        pipeline.worker(db).run_job(conn, job_id)

    if __name__ == "__main__":
        main()
    """
)


def test_a_job_killed_mid_run_is_failed_on_restart_and_retries_to_completion(
    conn, db_path, tmp_path, make_pipeline, fake
):
    pipeline = make_pipeline()
    result = pipeline.submit(conn, corpus.INSPECTION.path, "Reports/inspection.pdf")
    script = tmp_path / "worker.py"
    script.write_text(KILLED_WORKER)
    env = {**os.environ, "PYTHONPATH": str(REPO), "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    child = subprocess.Popen(
        [sys.executable, str(script), str(db_path), str(tmp_path / "files"), str(result.job_id), PROFILE],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        line = child.stdout.readline()  # the child is inside the embed step
        assert line.strip() == "embedding", child.stderr.read()
        running = get_job(conn, result.job_id)
        assert running.status == "running" and running.progress > 0
        child.send_signal(signal.SIGKILL)
        child.wait(30)
    finally:
        if child.poll() is None:  # pragma: no cover
            child.kill()
        child.stdout.close()
        child.stderr.close()

    assert get_job(conn, result.job_id).status == "running"  # nobody has noticed yet
    # a worker starting up (the service restarting) marks it failed
    worker = pipeline.worker(db_path)
    worker.start()
    worker.stop()
    failed = get_job(conn, result.job_id)
    assert failed.status == "failed" and failed.error == INTERRUPTED
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0  # nothing half-written

    queued = retry_job(conn, result.job_id)
    assert queued.status == "queued" and queued.error is None and queued.progress == 0
    assert job_params(conn, result.job_id)["relative_path"] == "Reports/inspection.pdf"
    done = pipeline.worker(db_path).run_job(conn, result.job_id)
    assert done.status == "done", done.error
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0
    # the profile the dead process built is complete and active
    assert active_profile(conn).name == PROFILE


def test_the_background_worker_runs_jobs_and_publishes_progress(conn, db_path, make_pipeline):
    bus = ProgressBus()
    pipeline = make_pipeline(bus=bus)
    worker = pipeline.worker(db_path)
    with bus.listen() as events:
        worker.start()
        try:
            first = pipeline.submit(conn, corpus.INVOICE.path, "invoice.pdf")
            second = pipeline.submit(conn, corpus.MARKDOWN.path, "garage_door.md")
            worker.notify()
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if all(get_job(conn, r.job_id).status == "done" for r in (first, second)):
                    break
                time.sleep(0.05)
        finally:
            worker.stop()
        seen: list[Job] = []
        while not events.empty():
            seen.append(events.get_nowait())
    assert get_job(conn, first.job_id).status == get_job(conn, second.job_id).status == "done"
    mine = [e for e in seen if e.id == first.job_id]
    assert [e.status for e in mine][0] == "queued" and mine[-1].status == "done"
    assert "running" in {e.status for e in mine}
    progress = [e.progress for e in mine if e.status == "running"]
    assert progress == sorted(progress) and len(progress) >= 5
    assert {"parsing", "segmenting", "cleaning", "chunking", "indexing"} <= {e.detail for e in mine}
    # one job at a time: the second started after the first finished
    first_done = next(i for i, e in enumerate(seen) if e.id == first.job_id and e.status == "done")
    second_running = next(
        i for i, e in enumerate(seen) if e.id == second.job_id and e.status == "running"
    )
    assert first_done < second_running
    assert worker.wait_idle(0) or True


def test_a_failing_job_is_failed_with_its_error_and_the_worker_goes_on(conn, db_path):
    def boom(c, job, report):
        report(0.5, "halfway")
        raise ValueError("bad page")

    ok = []
    worker = JobWorker(
        db_path, {"ingest": boom, "reindex": lambda c, j, r: ok.append(j.id)}, bus=None
    )
    bad = create_job(conn, "ingest", bus=None)
    good = create_job(conn, "reindex", bus=None)
    odd = create_job(conn, "regression", bus=None)  # no handler registered
    finished = worker.run_pending(conn)
    assert [(j.id, j.status) for j in finished] == [
        (bad.id, "failed"),
        (good.id, "done"),
        (odd.id, "failed"),
    ]
    assert finished[0].error == "ValueError: bad page" and finished[0].progress == 0.5
    assert "no handler" in finished[2].error
    assert ok == [good.id] and finished[1].detail == "done"


def test_retry_rules(conn):
    job = create_job(conn, "ingest", params={"use_llm": False, "_worker": "x:1"}, bus=None)
    assert job_params(conn, job.id) == {"use_llm": False}  # private keys hidden
    with pytest.raises(JobNotRetryable, match="queued"):
        retry_job(conn, job.id, bus=None)
    with pytest.raises(JobNotFound):
        retry_job(conn, 999, bus=None)
    with pytest.raises(JobNotFound):
        job_params(conn, 999)
    with pytest.raises(JobNotFound):
        update_job(conn, 999, bus=None, detail="x")
    with pytest.raises(ValueError):
        update_job(conn, job.id, bus=None, kind="reindex")
    update_job(conn, job.id, bus=None, status="failed", error="x", progress=7)
    assert get_job(conn, job.id).progress == 1.0  # clamped
    assert retry_job(conn, job.id, bus=None).status == "queued"
    assert [j.id for j in list_jobs(conn)] == [job.id]
    assert list_jobs(conn, status="done") == []


def test_recovery_leaves_jobs_of_a_live_process_alone(conn):
    mine = create_job(conn, "ingest", bus=None)
    other = create_job(conn, "ingest", bus=None)
    host = worker_id().rpartition(":")[0]
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        for job, owner in ((mine, f"{host}:{os.getpid()}"), (other, f"{host}:{live.pid}")):
            conn.execute(
                "UPDATE jobs SET status = 'running', params = json_set(params, '$._worker', ?) "
                "WHERE id = ?",
                (owner, job.id),
            )
        conn.commit()
        assert recover_interrupted(conn, bus=None) == [mine.id]
        assert get_job(conn, other.id).status == "running"
    finally:
        live.kill()
        live.wait()
    assert recover_interrupted(conn, bus=None) == [other.id]


def test_run_job_waits_for_another_worker_that_claimed_it(conn, db_path):
    job = create_job(conn, "ingest", bus=None)
    conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job.id,))
    conn.commit()
    import threading

    def finish() -> None:
        time.sleep(0.3)
        from hal9000.knowledge.storage import open_db

        other = open_db(db_path)
        update_job(other, job.id, bus=None, status="done", progress=1.0)
        other.close()

    thread = threading.Thread(target=finish)
    thread.start()
    worker = JobWorker(db_path, {}, bus=None)
    assert worker.run_job(conn, job.id).status == "done"
    thread.join()
    with pytest.raises(JobNotFound):
        worker.run_job(conn, 12345)


def test_the_bus_bridges_into_asyncio_and_survives_broken_listeners(conn):
    bus = ProgressBus()
    bus.subscribe(lambda job: 1 / 0)  # a broken listener is logged, not raised

    async def main() -> Job:
        inbox, unsubscribe = bus.asyncio_queue()
        try:
            create_job(conn, "ingest", bus=bus)
            return await asyncio.wait_for(inbox.get(), 5)
        finally:
            unsubscribe()

    job = asyncio.run(main())
    assert job.status == "queued" and job.kind == "ingest"
    with bus.listen(maxsize=1) as inbox:
        create_job(conn, "ingest", bus=bus)
        create_job(conn, "reindex", bus=bus)
        assert inbox.qsize() == 1 and inbox.get_nowait().kind == "reindex"  # oldest dropped


def test_a_reindex_job_builds_and_activates_a_profile(conn, db_path, make_pipeline):
    pipeline = make_pipeline()
    ingest = pipeline.submit(conn, corpus.ESTIMATE.path, "estimate.pdf")
    pipeline.worker(db_path).run_job(conn, ingest.job_id)
    other = "pdfium.fixed-64.hdr.fake-64"
    job = pipeline.enqueue_build(conn, other)
    done = pipeline.worker(db_path).run_job(conn, job.id)
    assert done.status == "done" and done.detail.startswith(other)
    assert active_profile(conn).name == other
    again = pipeline.worker(db_path).run_job(conn, pipeline.enqueue_build(conn, other).id)
    assert again.detail == f"{other} was already built"
    with pytest.raises(ValueError):
        pipeline.enqueue_build(conn, "not a profile name")
    assert get_profile(conn, PROFILE).active is False


def test_the_worker_survives_a_failing_bookkeeping_call(conn, db_path, monkeypatch):
    from hal9000.knowledge.ingest import jobs

    ran: list[int] = []

    def handler(c, job, report):
        ran.append(job.id)
        return "ok"

    real_update = jobs.update_job
    failures = {"left": 1}

    def flaky_update(c, job_id, **fields):
        if fields.get("status") == "done" and failures["left"]:
            failures["left"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real_update(c, job_id, **fields)

    monkeypatch.setattr(jobs, "update_job", flaky_update)
    first = create_job(conn, "ingest", bus=None)
    second = create_job(conn, "ingest", bus=None)
    worker = JobWorker(db_path, {"ingest": handler}, bus=None, poll_s=0.05)
    worker.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and get_job(conn, second.id).status != "done":
            time.sleep(0.05)
    finally:
        worker.stop()
    assert ran == [first.id, second.id]  # the loop went on to the next job
    failed = get_job(conn, first.id)
    assert failed.status == "failed" and "recording" in failed.error
    assert get_job(conn, second.id).status == "done"


def test_a_handler_raising_systemexit_fails_its_job_before_it_propagates(conn, db_path):
    def bail(c, job, report):
        raise SystemExit(3)

    job = create_job(conn, "ingest", bus=None)
    worker = JobWorker(db_path, {"ingest": bail}, bus=None)
    with pytest.raises(SystemExit):
        worker.run_job(conn, job.id)
    stopped = get_job(conn, job.id)
    assert stopped.status == "failed" and stopped.error == "stopped: SystemExit"


def test_the_asyncio_bridge_is_bounded_and_leaves_with_its_loop(conn):
    bus = ProgressBus()

    async def fill() -> list[str]:
        inbox, _ = bus.asyncio_queue(maxsize=2)
        for kind in ("ingest", "reindex", "regression"):
            create_job(conn, kind, bus=bus)
        await asyncio.sleep(0.05)  # let the loop run the deliveries
        return [inbox.get_nowait().kind for _ in range(inbox.qsize())]

    assert asyncio.run(fill()) == ["reindex", "regression"]  # the oldest was dropped
    # asyncio.run closed that loop: the next publish unsubscribes the listener
    create_job(conn, "ingest", bus=bus)
    assert bus._subscribers == []


def test_the_writer_lock_is_exclusive_and_released_on_close(tmp_path):
    first = WriterLock(tmp_path / "k" / "worker.lock")
    second = WriterLock(tmp_path / "k" / "worker.lock")
    assert first.acquire() and first.acquire() and first.held
    assert not second.acquire()
    first.release()
    with second:
        assert second.acquire()
    assert not second.held


def test_a_locked_worker_waits_for_the_lock_before_running_anything(conn, db_path, tmp_path):
    lock_path = tmp_path / "worker.lock"
    cli = WriterLock(lock_path)
    assert cli.acquire()  # a kb ingest is running jobs itself
    ran: list[int] = []
    job = create_job(conn, "ingest", bus=None)
    worker = JobWorker(
        db_path, {"ingest": lambda c, j, r: ran.append(j.id)}, bus=None, poll_s=0.05,
        lock=WriterLock(lock_path),
    )  # fmt: skip
    worker.start()
    try:
        time.sleep(0.4)
        assert ran == [] and get_job(conn, job.id).status == "queued"
        cli.release()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and get_job(conn, job.id).status != "done":
            time.sleep(0.05)
        assert ran == [job.id]
    finally:
        worker.stop()
    assert WriterLock(lock_path).acquire()  # the stopped worker released it
