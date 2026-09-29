"""Request guard: origin check, session check and body limits, before any body is read (plan §14).

A pure ASGI middleware in front of every route. For ``/api/*`` requests, in order:

1. **Same origin (CSRF).** A request that can change something (anything but
   ``GET``/``HEAD``/``OPTIONS``) must come from the app's own origin. Home
   Assistant on the same Pi is *same-site* (another port), so ``SameSite=Strict``
   still sends the cookie, and a cross-site form POST needs no preflight.
   Allowed: ``Sec-Fetch-Site: same-origin`` (or ``none``: typed by the user);
   when the browser sends no ``Sec-Fetch-Site`` (it doesn't on plain-http LAN
   origins), an ``Origin`` whose scheme, host and port equal the request's.
   Anything else, including neither header, is ``403``.
2. **Session.** Every ``/api`` route except ``POST /api/session`` needs a valid
   signed cookie whose session row still exists (``401``), decided from the
   cookie alone, so an unauthenticated body is never read.
3. **Body size.** A declared ``Content-Length`` over the limit is ``413`` at
   once; otherwise the body is counted as it streams and cut off with ``413``
   at the limit. The limit is ``ingest.max_upload_bytes`` + 1 MB for
   ``/api/uploads`` (the UI sends one file per request) and 64 KB for
   everything else.

The session found is left in ``scope["state"]["kb_session"]`` for the handlers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from urllib.parse import urlsplit

from starlette.exceptions import HTTPException

__all__ = ["JSON_LIMIT", "UPLOAD_SLACK", "RequestGuard", "same_origin"]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

JSON_LIMIT = 64 * 1024
UPLOAD_SLACK = 1024 * 1024
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_DEFAULT_PORTS = {"http": "80", "https": "443"}


def _netloc(scheme: str, netloc: str) -> str:
    """``host[:port]`` lower-cased, the scheme's default port dropped; ``""`` if unreadable."""
    try:
        parts = urlsplit("//" + netloc)
        port = parts.port
    except ValueError:
        return ""
    host = (parts.hostname or "").lower()
    if port is None or str(port) == _DEFAULT_PORTS.get(scheme):
        return host
    return f"{host}:{port}"


def same_origin(scope: Scope, headers: dict[str, str]) -> bool:
    """Whether a state-changing request comes from the app's own origin (see module notes)."""
    site = headers.get("sec-fetch-site")
    if site is not None:
        return site in ("same-origin", "none")
    origin = headers.get("origin")
    host = headers.get("host")
    if not origin or origin == "null" or not host:
        return False
    parsed = urlsplit(origin)
    scheme = scope.get("scheme", "http")
    ours = _netloc(scheme, host)
    return bool(ours) and parsed.scheme == scheme and _netloc(scheme, parsed.netloc) == ours


async def _reply(send: Send, status: int, detail: str, headers: dict[str, str] | None = None):
    import json

    body = json.dumps({"detail": detail}).encode()
    raw = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    raw += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": body})


class RequestGuard:
    def __init__(self, app, *, json_limit: int = JSON_LIMIT) -> None:
        self.app = app
        self.json_limit = json_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        services = getattr(scope["app"].state, "services", None) if "app" in scope else None
        if services is None:  # the contract-only app (OpenAPI export): nothing to guard
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        path, method = scope["path"], scope["method"]
        is_api = path.startswith("/api/") or path == "/api"
        if is_api and method not in SAFE_METHODS and not same_origin(scope, headers):
            await _reply(send, 403, "Requests must come from this app's own pages.")
            return
        if is_api and not (method == "POST" and path == "/api/session"):
            session = await self._session(services, headers.get("cookie", ""))
            if session is None:
                await _reply(send, 401, "Not signed in")
                return
            scope.setdefault("state", {})["kb_session"] = session

        limit = self._limit(services, path)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                await _reply(send, 400, "Content-Length is not a number")
                return
            if size > limit:
                await _reply(send, 413, f"The request body is larger than {limit} bytes.")
                return
        await self.app(scope, _capped(receive, limit), send)

    @staticmethod
    async def _session(services, cookie_header: str):
        from starlette.concurrency import run_in_threadpool
        from starlette.requests import cookie_parser

        from hal9000.knowledge.auth import COOKIE

        session = services.auth.session(cookie_parser(cookie_header).get(COOKIE))
        if session is None:
            return None
        valid = await run_in_threadpool(services.sessions.valid, session)
        return session if valid else None

    def _limit(self, services, path: str) -> int:
        if path == "/api/uploads":
            return services.max_upload_bytes() + UPLOAD_SLACK
        return self.json_limit


def _capped(receive: Receive, limit: int) -> Receive:
    """``receive`` that raises ``413`` once more than ``limit`` body bytes have arrived."""
    seen = 0

    async def limited() -> Message:
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > limit:
                raise HTTPException(
                    status_code=413, detail=f"The request body is larger than {limit} bytes."
                )
        return message

    return limited
