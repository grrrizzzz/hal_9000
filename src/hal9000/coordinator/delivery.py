"""HAL's outbound channels: the Sonos announcement and Telegram.

Agents never talk to Home Assistant or Telegram directly; they ask this service,
which applies the per-run Telegram budget and shadow mode.

- **Shadow mode** (dry run): nothing is played on Sonos. The would-be
  announcement and every alert go to the test chat, or to the main chat
  prefixed ``[DRY RUN]`` when no separate test chat is set.
- **No Telegram configured**: messages are printed and kept in the outbox.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hal9000.config import Settings
from hal9000.solar.clients.homeassistant import HAProtocol
from hal9000.solar.clients.telegram import TelegramProtocol

__all__ = ["Delivery", "OutboxItem", "TELEGRAM_PER_RUN"]

TELEGRAM_PER_RUN = 3
_LOG = logging.getLogger(__name__)
_ICON = {"info": "ℹ️", "warn": "⚠️", "critical": "🚨"}


@dataclass
class OutboxItem:
    channel: str  # "sonos" | "telegram" | "console"
    text: str
    status: str  # "sent" | "logged" | "failed" | "suppressed"


@dataclass
class Delivery:
    settings: Settings
    ha: HAProtocol | None
    telegram: TelegramProtocol | None
    shadow: bool = False
    speaker: str | None = None
    announce_enabled: bool = True
    telegram_sent: int = 0
    outbox: list[OutboxItem] = field(default_factory=list)

    @property
    def _chat(self) -> str:
        if self.shadow and self.settings.telegram_test_chat_id:
            return self.settings.telegram_test_chat_id
        return self.settings.telegram_chat_id

    @property
    def _prefix(self) -> str:
        separate_test_chat = (
            self.settings.telegram_test_chat_id
            and self.settings.telegram_test_chat_id != self.settings.telegram_chat_id
        )
        return "[DRY RUN] " if self.shadow and not separate_test_chat else ""

    def _telegram(self, text: str) -> str:
        message = self._prefix + text
        if self.telegram is None or not self.settings.telegram_configured:
            self.outbox.append(OutboxItem("console", message, "logged"))
            print(f"[telegram not configured] {message}")
            return "logged"
        try:
            self.telegram.send(self._chat, message)
            self.outbox.append(OutboxItem("telegram", message, "sent"))
            return "sent"
        except Exception as error:
            _LOG.warning("Telegram send failed: %s", error)
            self.outbox.append(OutboxItem("telegram", message, "failed"))
            return "failed"

    def send_alert(self, text: str, severity: str = "info") -> bool:
        """Send an alert or message. Returns True if it went out (or was logged)."""
        if self.telegram_sent >= TELEGRAM_PER_RUN:
            self.outbox.append(OutboxItem("telegram", text, "suppressed"))
            raise RuntimeError(
                f"The Telegram limit of {TELEGRAM_PER_RUN} messages per run has been reached."
            )
        status = self._telegram(f"{_ICON.get(severity, '')} {text}".strip())
        if status in ("sent", "logged"):
            self.telegram_sent += 1
            return True
        return False

    def announce(self, text: str) -> str:
        """Play the report on the speaker. Returns "played", "logged" or "telegram"."""
        speaker = self.speaker or self.settings.entity_speaker
        if self.shadow or not self.announce_enabled:
            self.outbox.append(OutboxItem("sonos", text, "logged"))
            if self.shadow:
                self._telegram(f"Evening report (not played): {text}")
            return "logged"
        try:
            if self.ha is None:
                raise RuntimeError("Home Assistant is not configured")
            self.ha.speak(self.settings.tts_entity, speaker, text)
            self.outbox.append(OutboxItem("sonos", text, "sent"))
            return "played"
        except Exception as error:
            _LOG.warning("Announcement failed (%s); sending to Telegram", error)
            self.outbox.append(OutboxItem("sonos", text, "failed"))
            self._telegram(f"Evening solar report (speaker unavailable): {text}")
            return "telegram"
