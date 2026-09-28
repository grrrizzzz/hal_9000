"""Home Assistant client: long-term statistics over the websocket API, TTS over REST.

Energy sensors here are daily-resetting ``total`` sensors, so raw state
differences break at midnight and on restarts. Statistics ``change`` values are
already reset-aware, which is why every energy read goes through
``recorder/statistics_during_period``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

__all__ = ["HAClient", "HAProtocol", "HAUnavailable", "StatPoint"]


class HAUnavailable(Exception):
    """Home Assistant could not be reached or refused the request. Not recoverable."""

    error_type = "HAUnavailable"
    recoverable = False
    guidance = "Home Assistant is unreachable, so there is no production data. Stop and report."


StatPoint = tuple[datetime, float]  # (UTC start of the bucket, kWh change)


class HAProtocol(Protocol):
    def statistics(
        self, entity_ids: list[str], start: datetime, end: datetime, period: str
    ) -> dict[str, list[StatPoint]]: ...

    def speak(self, tts_entity: str, media_player: str, message: str) -> None: ...

    def ping(self) -> str: ...

    def state(self, entity_id: str) -> dict: ...


def _parse_start(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000.0, tz=UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


class HAClient:
    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        if not base_url or not token:
            raise HAUnavailable("HA_BASE_URL and HA_TOKEN must be set in .env")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _ws_url(self) -> str:
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url[len("https://") :] + "/api/websocket"
        return "ws://" + self.base_url.removeprefix("http://") + "/api/websocket"

    def statistics(
        self, entity_ids: list[str], start: datetime, end: datetime, period: str
    ) -> dict[str, list[StatPoint]]:
        """``change`` per bucket for each entity, between ``start`` and ``end`` (aware)."""
        from websockets.sync.client import connect

        request = {
            "id": 1,
            "type": "recorder/statistics_during_period",
            "start_time": start.astimezone(UTC).isoformat(),
            "end_time": end.astimezone(UTC).isoformat(),
            "statistic_ids": entity_ids,
            "period": period,
            "types": ["change"],
            "units": {"energy": "kWh"},
        }
        try:
            with connect(
                self._ws_url(), open_timeout=self.timeout, max_size=2**24, proxy=None
            ) as ws:
                hello = json.loads(ws.recv(timeout=self.timeout))
                if hello.get("type") != "auth_required":
                    raise HAUnavailable(f"Unexpected websocket greeting: {hello.get('type')}")
                ws.send(json.dumps({"type": "auth", "access_token": self.token}))
                auth = json.loads(ws.recv(timeout=self.timeout))
                if auth.get("type") != "auth_ok":
                    raise HAUnavailable("Home Assistant rejected the token (auth_invalid).")
                ws.send(json.dumps(request))
                reply = json.loads(ws.recv(timeout=self.timeout))
        except HAUnavailable:
            raise
        except Exception as error:  # network, timeout, protocol
            raise HAUnavailable(f"Home Assistant statistics request failed: {error}") from error
        if not reply.get("success"):
            raise HAUnavailable(f"Home Assistant returned an error: {reply.get('error')}")
        result = reply.get("result") or {}
        return {
            entity: [
                (_parse_start(point["start"]), float(point.get("change") or 0.0))
                for point in result.get(entity, [])
            ]
            for entity in entity_ids
        }

    def speak(self, tts_entity: str, media_player: str, message: str) -> None:
        payload = {
            "entity_id": tts_entity,
            "media_player_entity_id": media_player,
            "message": message,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/services/tts/speak",
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
                trust_env=False,  # HA is on the LAN: never route it through a proxy
            )
            response.raise_for_status()
        except Exception as error:
            raise HAUnavailable(f"TTS announcement failed: {error}") from error

    def ping(self) -> str:
        try:
            response = httpx.get(
                f"{self.base_url}/api/", headers=self._headers, timeout=10, trust_env=False
            )
            response.raise_for_status()
            return response.json().get("message", "ok")
        except Exception as error:
            raise HAUnavailable(f"Home Assistant API check failed: {error}") from error

    def state(self, entity_id: str) -> dict:
        try:
            response = httpx.get(
                f"{self.base_url}/api/states/{entity_id}",
                headers=self._headers,
                timeout=10,
                trust_env=False,
            )
            response.raise_for_status()
            return response.json()
        except Exception as error:
            raise HAUnavailable(f"Could not read {entity_id}: {error}") from error
