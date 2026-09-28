"""Telegram Bot API client: send a message to a chat."""

from __future__ import annotations

from typing import Protocol

import httpx

__all__ = ["TelegramClient", "TelegramError", "TelegramProtocol"]


class TelegramError(Exception):
    error_type = "TelegramError"
    recoverable = True
    guidance = "The Telegram message was not sent. Continue; the report is still delivered."


class TelegramProtocol(Protocol):
    def send(self, chat_id: str, text: str) -> None: ...


class TelegramClient:
    def __init__(self, bot_token: str, timeout: float = 15.0):
        self.bot_token = bot_token
        self.timeout = timeout

    def send(self, chat_id: str, text: str) -> None:
        try:
            response = httpx.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": text[:4000]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            if not response.json().get("ok", False):
                raise TelegramError(response.json().get("description", "unknown error"))
        except TelegramError:
            raise
        except Exception as error:
            # never include the token-bearing URL in the message
            raise TelegramError(f"Telegram send failed: {type(error).__name__}") from None
