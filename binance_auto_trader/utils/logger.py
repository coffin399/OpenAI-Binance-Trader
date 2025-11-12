from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import requests
from binance_auto_trader.utils.discord_notify import TradeOnlyFilter

from binance_auto_trader.config.config import Config

DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_DISCORD_MESSAGE_LENGTH = 2000


class DiscordRateLimitedHandler(logging.Handler):
    def __init__(
        self,
        bot_token: str,
        channel_ids: Iterable[str],
        max_messages: int,
        per_seconds: int,
        ansi_wrap: bool = True,
    ) -> None:
        super().__init__()
        self._session = requests.Session()
        self._bot_token = bot_token
        self._channel_ids = [str(cid) for cid in channel_ids]
        self._max_messages = max(1, max_messages)
        self._per_seconds = max(1, per_seconds)
        self._ansi_wrap = ansi_wrap
        self._queue: "queue.Queue[Optional[tuple[str, str]]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._timestamps = {cid: deque() for cid in self._channel_ids}
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if self._ansi_wrap:
                message = self._wrap_ansi(message)
            message = self._truncate(message)
            for channel_id in self._channel_ids:
                self._queue.put((channel_id, message))
        except Exception:  # noqa: BLE001 - logging handler must never raise
            self.handleError(record)

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._worker.join(timeout=5)
        self._session.close()
        super().close()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            channel_id, message = item
            try:
                self._respect_rate_limit(channel_id)
                self._send_message(channel_id, message)
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"[DiscordRateLimitedHandler] Failed to send log to channel {channel_id}: {exc}\n"
                )
                time.sleep(1)

    def _respect_rate_limit(self, channel_id: str) -> None:
        timestamps = self._timestamps.setdefault(channel_id, deque())
        now = time.monotonic()
        window_start = now - self._per_seconds
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()
        if len(timestamps) >= self._max_messages:
            sleep_for = self._per_seconds - (now - timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        timestamps.append(time.monotonic())

    def _send_message(self, channel_id: str, message: str) -> None:
        url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self._bot_token}",
            "Content-Type": "application/json",
        }
        payload = {"content": message}

        response = self._session.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 1)
            time.sleep(float(retry_after))
            self._queue.put((channel_id, message))
        elif response.status_code >= 400:
            raise RuntimeError(
                f"Discord API error {response.status_code}: {response.text}"
            )

    @staticmethod
    def _wrap_ansi(message: str) -> str:
        return f"```ansi\n{message}\n```"

    @staticmethod
    def _truncate(message: str) -> str:
        if len(message) <= MAX_DISCORD_MESSAGE_LENGTH:
            return message
        return message[: MAX_DISCORD_MESSAGE_LENGTH - 3] + "..."


def setup_logging(config: Config) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    level_name = getattr(config.logging, "level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if getattr(config.logging, "console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    log_file = getattr(config.logging, "file", None)
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = config.path.parent / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    discord_config = getattr(config, "discord", None)
    if discord_config and getattr(discord_config, "enabled", False):
        bot_token = getattr(discord_config, "bot_token", "")
        channel_ids: Optional[List[str]] = getattr(discord_config, "channel_ids", None)
        if bot_token and channel_ids:
            rate_limit_cfg = getattr(discord_config, "rate_limit", None)
            max_messages = getattr(rate_limit_cfg, "max_messages", 5) if rate_limit_cfg else 5
            per_seconds = getattr(rate_limit_cfg, "per_seconds", 5) if rate_limit_cfg else 5
            ansi_wrap = getattr(discord_config, "ansi_wrap", True)

            discord_handler = DiscordRateLimitedHandler(
                bot_token=bot_token,
                channel_ids=channel_ids,
                max_messages=max_messages,
                per_seconds=per_seconds,
                ansi_wrap=ansi_wrap,
            )
            discord_handler.setFormatter(formatter)
            discord_handler.addFilter(TradeOnlyFilter())  # 取引ログのみを通すフィルタ
            root_logger.addHandler(discord_handler)

            atexit.register(discord_handler.close)
        else:
            logging.getLogger(__name__).warning(
                "Discord logging enabled but bot token or channel IDs are missing."
            )

