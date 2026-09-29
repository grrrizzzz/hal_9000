"""The two SSE streams at the ASGI level: a client disconnect cancels the question; live jobs."""

from __future__ import annotations

import asyncio
import json
import time

from hal9000.knowledge.answer.fake import fake_deps
from hal9000.knowledge.auth import COOKIE
from hal9000.knowledge.contracts import Job
from tests.knowledge.api.conftest import Api


def _scope(method: str, path: str, token: str, body: bytes = b"") -> dict:
    headers = [(b"host", b"test"), (b"cookie", f"{COOKIE}={token}".encode())]
    if body:
        headers.append((b"content-type", b"application/json"))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }


async def _drive(app, scope, body: bytes, until, *, after=None, timeout: float = 15.0) -> bytes:
    """Run one request; disconnect once ``until(received)`` holds. Returns what was sent."""
    received = bytearray()
    gone = asyncio.Event()
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            received.extend(message.get("body", b""))
            if not gone.is_set() and until(bytes(received)):
                if after is not None:
                    after()
                    return
                gone.set()

    await asyncio.wait_for(app(scope, receive, send), timeout)
    return bytes(received)


def _token(api: Api) -> str:
    return api.sign_in().cookies.get(COOKIE)


def test_a_client_disconnect_cancels_the_question(api: Api):
    api.services.answerer.deps = fake_deps(delay_s=0.05)  # slow enough to leave mid-answer
    token = _token(api)
    body = json.dumps({"message": "How old is the furnace?"}).encode()
    sent = asyncio.run(
        _drive(
            api.client.app,
            _scope("POST", "/api/chat", token, body),
            body,
            until=lambda text: text.count(b"event: token") >= 2,
        )
    )
    assert b"event: sources" in sent and b"event: done" not in sent

    conn = api.conn()
    try:
        deadline = time.monotonic() + 10
        row = None
        while time.monotonic() < deadline:
            row = conn.execute("SELECT outcome, body FROM traces").fetchone()
            if row is not None:
                break
            time.sleep(0.05)
        assert row is not None and row["outcome"] == "cancelled"
        trace = json.loads(row["body"])
        assert trace["tokens_by_model"]["fake/chat"]["completion"] >= 1  # estimated usage
        assert "generate" in {stage["stage"] for stage in trace["stages"]}
        roles = [r[0] for r in conn.execute("SELECT role FROM messages ORDER BY id")]
        assert roles == ["user"]  # no answer was stored for the cancelled question
    finally:
        conn.close()
    chat = api.services.answerer.deps.stream_fn
    assert len(chat.calls) == 1  # generation stopped; it was not retried or drained


def test_the_job_stream_sends_recent_jobs_then_live_updates(api: Api):
    token = _token(api)
    live = Job(id=4242, kind="ingest", status="running", progress=0.5, detail="embedding")

    def publish() -> None:
        api.bus.publish(live)

    seen = {"published": False}

    def until(text: bytes) -> bool:
        if b"event: job" in text and not seen["published"]:
            seen["published"] = True
            asyncio.get_running_loop().call_later(0.05, publish)
            return False
        return b'"id":4242' in text

    sent = asyncio.run(
        _drive(api.client.app, _scope("GET", "/api/jobs/stream", token), b"", until=until)
    )
    text = sent.decode().replace("\r\n", "\n")
    events = [block for block in text.split("\n\n") if "event: job" in block]
    assert len(events) >= 4  # the three seeded ingest jobs, then the live one
    assert '"id":4242' in events[-1] and '"progress":0.5' in events[-1]
    assert api.bus._subscribers == []  # the listener went away with the client
