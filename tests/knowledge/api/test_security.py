"""The request guard (review F1–F3), sign-in under concurrency (F2), revocation, docs off."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from hal9000.knowledge.api import create_app
from hal9000.knowledge.auth import COOKIE
from tests.knowledge.api.conftest import PASSCODE, SAME_ORIGIN, Api, client_for, make_api


def _scope(method: str, path: str, headers: dict[str, str]) -> dict:
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
        "headers": [(k.encode(), v.encode()) for k, v in {"host": "test", **headers}.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }


class Trickle:
    """A client that declares a body and sends it in 16 KB pieces, forever unless stopped."""

    def __init__(self, first: bytes = b"x" * 1024, *, endless: bool = False) -> None:
        self.first = first
        self.endless = endless
        self.read = 0
        self.calls = 0

    async def receive(self):
        self.calls += 1
        if self.calls == 1:
            self.read += len(self.first)
            return {"type": "http.request", "body": self.first, "more_body": True}
        if self.endless:
            chunk = b"y" * 16384
            self.read += len(chunk)
            return {"type": "http.request", "body": chunk, "more_body": True}
        await asyncio.sleep(3600)  # a slow client that never sends the rest
        raise AssertionError("unreachable")


def _run(app, scope, receive, timeout: float = 5.0) -> tuple[int, dict]:
    sent: dict = {"body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    started = time.monotonic()
    asyncio.run(asyncio.wait_for(app(scope, receive, send), timeout))
    sent["elapsed"] = time.monotonic() - started
    return sent["status"], sent


def _cookie(api: Api) -> str:
    return f"{COOKIE}={api.sign_in().cookies.get(COOKIE)}"


# ------------------------------------------------------------------- F1 bodies
def test_an_unauthenticated_huge_body_is_refused_before_it_is_read(api: Api):
    trickle = Trickle()
    headers = {**SAME_ORIGIN, "content-type": "application/json", "content-length": "400000000"}
    status, sent = _run(api.client.app, _scope("POST", "/api/chat", headers), trickle.receive)
    assert status == 401 and sent["elapsed"] < 2
    assert trickle.calls == 0  # not one byte of the body was read


def test_an_authenticated_declared_oversize_body_is_413_at_once(api: Api):
    trickle = Trickle()
    headers = {
        **SAME_ORIGIN,
        "cookie": _cookie(api),
        "content-type": "application/json",
        "content-length": "300000000",
    }
    status, sent = _run(api.client.app, _scope("POST", "/api/chat", headers), trickle.receive)
    assert status == 413 and trickle.calls == 0 and sent["elapsed"] < 2
    uploads = {**headers, "content-type": "multipart/form-data; boundary=x"}
    status, _ = _run(api.client.app, _scope("POST", "/api/uploads", uploads), trickle.receive)
    assert status == 413 and trickle.calls == 0


def test_a_body_without_a_length_is_cut_off_while_streaming(api: Api):
    trickle = Trickle(endless=True)
    headers = {
        **SAME_ORIGIN,
        "cookie": _cookie(api),
        "content-type": "application/json",
        "transfer-encoding": "chunked",
    }
    status, sent = _run(api.client.app, _scope("POST", "/api/chat", headers), trickle.receive)
    assert status == 413 and b"larger than 65536 bytes" in sent["body"]
    assert trickle.read <= 64 * 1024 + 16384  # stopped at the limit, memory bounded


def test_long_passcodes_and_questions_are_refused(api: Api, signed):
    assert api.client.post("/api/session", json={"passcode": "x" * 257}).status_code == 422
    assert signed.post("/api/chat", json={"message": "q" * 4001}).status_code == 422


def _limited_api(tmp_path: Path, max_bytes: int) -> Api:
    config = tmp_path / "config"
    config.mkdir(parents=True)
    from tests.knowledge.answer.conftest import YAML

    (config / "knowledge.yaml").write_text(YAML + f"ingest:\n  max_upload_bytes: {max_bytes}\n")
    return make_api(tmp_path, seed=False)


def test_a_file_over_the_limit_stops_being_written_at_the_limit(tmp_path):
    api = _limited_api(tmp_path, 2000)
    try:
        client = api.sign_in()
        written: list[int] = []
        spool = api.services.settings.data_dir / "knowledge" / "tmp"

        import hal9000.knowledge.api.uploads as uploads

        real = uploads.tempfile.NamedTemporaryFile

        def watched(*args, **kw):
            handle = real(*args, **kw)
            write = handle.write

            def counting(data):
                written.append(len(data))
                return write(data)

            handle.write = counting
            return handle

        uploads.tempfile.NamedTemporaryFile = watched
        try:
            response = client.post(
                "/api/uploads",
                files=[
                    ("files", ("big.txt", b"z" * 50_000, "text/plain")),
                    ("files", ("ok.txt", b"A small synthetic note.\n", "text/plain")),
                    ("files", ("photo.jpg", b"\xff\xd8" * 5000, "image/jpeg")),
                ],
                data={"relative_path": ["Notes/big.txt", "Notes/ok.txt", "Photos/photo.jpg"]},
            )
        finally:
            uploads.tempfile.NamedTemporaryFile = real
        assert response.status_code == 200, response.text
        big, ok, photo = response.json()["results"]
        assert big["status"] == "failed" and "too large" in big["reason"]
        assert "0.1 MB" in big["reason"] or "0.0 MB" in big["reason"]
        assert ok["status"] == "added" and ok["collection"] == "Notes"
        assert photo["status"] == "skipped" and ".jpg" in photo["reason"]
        # the big file stopped at the limit and the image was never written
        assert sum(written) <= 2000 + len(b"A small synthetic note.\n")
        assert list(spool.iterdir()) == []  # nothing left behind
    finally:
        api.services.close()


# --------------------------------------------------------------------- F2
def test_concurrent_sign_ins_evaluate_at_most_ten_passcodes(api: Api, monkeypatch):
    evaluated = []
    lock = threading.Lock()
    check = api.services.auth.check_passcode

    def counted(attempt: str) -> bool:
        with lock:
            evaluated.append(attempt)
        time.sleep(0.01)  # widen any race
        return check(attempt)

    monkeypatch.setattr(api.services.auth, "check_passcode", counted)

    async def storm() -> list[int]:
        transport = httpx.ASGITransport(app=api.client.app, client=("10.0.0.9", 4000))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", headers=SAME_ORIGIN
        ) as client:
            replies = await asyncio.gather(
                *(client.post("/api/session", json={"passcode": f"wrong-{n}"}) for n in range(100))
            )
        return [reply.status_code for reply in replies]

    codes = asyncio.run(storm())
    assert len(evaluated) == 10
    assert codes.count(401) == 10 and codes.count(429) == 90


def test_a_successful_sign_in_clears_the_count(api: Api):
    for _ in range(9):
        assert api.client.post("/api/session", json={"passcode": "wrong"}).status_code == 401
    assert api.client.post("/api/session", json={"passcode": PASSCODE}).status_code == 200
    for _ in range(10):
        assert api.client.post("/api/session", json={"passcode": "wrong"}).status_code == 401
    assert api.client.post("/api/session", json={"passcode": PASSCODE}).status_code == 429


# --------------------------------------------------------------------- F3
@pytest.mark.parametrize(
    ("headers", "allowed"),
    [
        ({"origin": "http://test:8123"}, False),  # Home Assistant on the same Pi: same-site
        ({"origin": "http://test:8910"}, True),
        ({"origin": "https://test:8910"}, False),
        ({"origin": "null"}, False),
        ({}, False),
        ({"sec-fetch-site": "same-site"}, False),
        ({"sec-fetch-site": "cross-site", "origin": "http://test:8910"}, False),
        ({"sec-fetch-site": "same-origin"}, True),
        ({"sec-fetch-site": "none"}, True),
    ],
)
def test_state_changing_requests_must_come_from_the_apps_own_origin(api: Api, headers, allowed):
    from fastapi.testclient import TestClient

    raw = TestClient(api.client.app, base_url="http://test:8910")
    response = raw.post("/api/session", json={"passcode": "wrong"}, headers=headers)
    assert response.status_code == (401 if allowed else 403), response.text


def test_a_cross_site_upload_is_refused_even_with_the_cookie(api: Api):
    token = api.sign_in().cookies.get(COOKIE)
    from fastapi.testclient import TestClient

    attacker = TestClient(api.client.app, base_url="http://test:8910")
    attacker.cookies.set(COOKIE, token)
    response = attacker.post(
        "/api/uploads",
        files=[("files", ("a.txt", b"synthetic", "text/plain"))],
        data={"relative_path": "a.txt"},
        headers={"origin": "http://test:8123"},
    )
    assert response.status_code == 403
    assert attacker.get("/api/jobs").status_code == 200  # reads are not blocked


# ------------------------------------------------------------- revocation
def test_signing_out_revokes_the_cookie_on_the_server(api: Api):
    first = api.sign_in(client_for(api.client.app))
    second = api.sign_in(client_for(api.client.app))
    kept = first.cookies.get(COOKIE)
    assert first.delete("/api/session").status_code == 204
    replay = client_for(api.client.app)
    replay.cookies.set(COOKIE, kept)
    assert replay.get("/api/jobs").status_code == 401  # a copied cookie is dead too
    assert second.get("/api/jobs").status_code == 200

    third = api.sign_in(client_for(api.client.app))
    assert second.delete("/api/session", params={"everywhere": "true"}).status_code == 204
    assert third.get("/api/jobs").status_code == 401 and second.get("/api/jobs").status_code == 401


def test_sessions_survive_a_restart(api: Api, tmp_path):
    client = api.sign_in(client_for(api.client.app))
    from hal9000.knowledge.auth import SessionStore

    api.services.sessions = SessionStore(api.services.settings.db_path)  # a cold cache
    assert client.get("/api/jobs").status_code == 200


def test_the_service_publishes_no_docs(api: Api):
    served = client_for(create_app(api.services, web_dist=False))
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert served.get(path).status_code == 404
    assert create_app().openapi()["paths"]  # the exporter still gets the schema


def test_the_guard_leaves_the_contract_only_app_alone():
    client = client_for(create_app())
    assert client.get("/openapi.json").status_code == 200
    body = json.dumps({"passcode": "x"})
    assert client.post("/api/session", content=body).status_code == 503
