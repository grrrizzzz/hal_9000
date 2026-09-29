"""The household passcode and signed session cookies (plan §10, §14).

- The passcode comes from ``KB_PASSCODE`` and is kept only as a salted scrypt
  hash in memory; attempts are compared in constant time.
- Signing in sets ``kb_session``: a random session ID signed with
  ``itsdangerous`` (timestamped, so it expires after 30 days), ``HttpOnly``,
  ``SameSite=Strict``, ``Path=/``. The signing key is a random secret kept in
  ``data/knowledge/session.key`` (created ``0600`` on first start), so
  sessions survive a restart; the signer's salt includes a fingerprint of the
  passcode, so changing the passcode signs everyone out.
- **Server-side revocation** (:class:`SessionStore`): a signed cookie is
  honoured only while its row in ``knowledge.db`` ``sessions`` exists. Signing
  out deletes the row; "sign out everywhere" deletes them all.
- **Rate limits** (sliding windows, in memory): sign-in attempts per client
  address (10 a minute), recorded atomically *before* the passcode is hashed,
  so concurrent attempts can't overshoot; a success clears the address. And
  questions per session (20 a minute, plan §7.4).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

__all__ = [
    "COOKIE",
    "MAX_AGE_S",
    "Auth",
    "RateLimiter",
    "Session",
    "SessionStore",
    "TooManyAttempts",
    "load_secret",
]

COOKIE = "kb_session"
MAX_AGE_S = 30 * 24 * 3600
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


class RateLimiter:
    """At most ``limit`` events per ``window_s`` seconds for each key."""

    def __init__(
        self, limit: int, window_s: float = 60.0, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if limit < 1 or window_s <= 0:
            raise ValueError("limit must be >= 1 and window_s > 0")
        self.limit = limit
        self.window_s = window_s
        self.clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, key: str, now: float) -> deque[float]:
        events = self._events[key]
        while events and now - events[0] >= self.window_s:
            events.popleft()
        return events

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._trim(key, self.clock())) >= self.limit

    def hit(self, key: str) -> None:
        with self._lock:
            now = self.clock()
            self._trim(key, now).append(now)

    def allow(self, key: str) -> bool:
        """Record an event if it is within the limit; ``False`` (and nothing recorded) if not."""
        with self._lock:
            now = self.clock()
            events = self._trim(key, now)
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True

    def clear(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)

    def retry_after(self, key: str) -> int:
        with self._lock:
            events = self._trim(key, self.clock())
            if len(events) < self.limit:
                return 0
            return max(1, int(self.window_s - (self.clock() - events[0])) + 1)


@dataclass(frozen=True)
class Session:
    id: str


class TooManyAttempts(RuntimeError):
    """Sign-in refused: this address used up its attempts for the minute."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("too many sign-in attempts")
        self.retry_after = retry_after


class SessionStore:
    """Session rows in ``knowledge.db`` (migration 5); a small cache of known-good IDs.

    The service is one process, and every revocation goes through it, so the
    cache is dropped exactly when a row is deleted.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = db_path
        self._known: set[str] = set()
        self._lock = threading.Lock()

    def _connect(self):
        from hal9000.knowledge.storage import connect

        return connect(self.db_path, vec=False)

    def create(self, session: Session, expires_at: datetime) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
                conn.execute(
                    "INSERT INTO sessions (id, created_at, expires_at) VALUES (?, ?, ?)",
                    (session.id, now, expires_at.isoformat(timespec="seconds")),
                )
        finally:
            conn.close()
        with self._lock:
            self._known.add(session.id)

    def valid(self, session: Session) -> bool:
        with self._lock:
            if session.id in self._known:
                return True
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND expires_at >= ?",
                (session.id, datetime.now(UTC).isoformat(timespec="seconds")),
            ).fetchone()
        finally:
            conn.close()
        if row is not None:
            with self._lock:
                self._known.add(session.id)
        return row is not None

    def revoke(self, session: Session) -> None:
        with self._lock:
            self._known.discard(session.id)
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session.id,))
        finally:
            conn.close()

    def revoke_all(self) -> int:
        with self._lock:
            self._known.clear()
        conn = self._connect()
        try:
            with conn:
                return conn.execute("DELETE FROM sessions").rowcount
        finally:
            conn.close()


def load_secret(path: Path | str) -> bytes:
    """The cookie-signing secret at ``path``, created (32 random bytes, mode 0600) if missing."""
    target = Path(path)
    if target.is_file():
        data = target.read_bytes()
        if len(data) >= 32:
            return data
    target.parent.mkdir(parents=True, exist_ok=True)
    data = secrets.token_bytes(32)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return data


class Auth:
    """Checks the passcode, issues and verifies session cookies, limits sign-in attempts."""

    def __init__(
        self,
        passcode: str,
        secret: bytes,
        *,
        max_age_s: int = MAX_AGE_S,
        sign_in_limit: int = 10,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        from itsdangerous import URLSafeTimedSerializer

        if not passcode:
            raise ValueError("a passcode is required")
        self._salt = secrets.token_bytes(16)
        self._hash = self._digest(passcode)
        fingerprint = hmac.new(secret, passcode.encode(), hashlib.sha256).hexdigest()[:16]
        self._signer = URLSafeTimedSerializer(secret, salt=f"kb-session:{fingerprint}")
        self.max_age_s = max_age_s
        self.sign_ins = RateLimiter(sign_in_limit, window_s, clock=clock)

    def _digest(self, passcode: str) -> bytes:
        return hashlib.scrypt(passcode.encode("utf-8"), salt=self._salt, **_SCRYPT)

    def check_passcode(self, attempt: str) -> bool:
        return hmac.compare_digest(self._digest(attempt), self._hash)

    def sign_in(self, client: str, attempt: str) -> bool:
        """Count the attempt for ``client`` first (atomically), then check the passcode.

        Raises :class:`TooManyAttempts` without hashing anything once the
        address has used its attempts for the window; a success clears it.
        """
        if not self.sign_ins.allow(client):
            raise TooManyAttempts(self.sign_ins.retry_after(client))
        if not self.check_passcode(attempt):
            return False
        self.sign_ins.clear(client)
        return True

    def issue(self) -> tuple[str, Session, datetime]:
        """A new session: ``(cookie value, session, expiry)``."""
        session = Session(secrets.token_urlsafe(18))
        token = self._signer.dumps({"sid": session.id})
        return token, session, datetime.now(UTC) + timedelta(seconds=self.max_age_s)

    def session(self, token: str | None) -> Session | None:
        """The session a cookie value names, or ``None`` if it is missing, forged or expired."""
        from itsdangerous import BadSignature

        if not token:
            return None
        try:
            data = self._signer.loads(token, max_age=self.max_age_s)
        except BadSignature:  # SignatureExpired is a BadSignature too
            return None
        sid = data.get("sid") if isinstance(data, dict) else None
        return Session(sid) if isinstance(sid, str) and sid else None
