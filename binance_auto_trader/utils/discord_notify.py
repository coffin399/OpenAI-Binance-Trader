from __future__ import annotations

import logging
import time
from collections import deque
from typing import Iterable, Optional, Sequence

import requests

DISCORD_API_BASE = "https://discord.com/api/v10"

logger = logging.getLogger(__name__)


class WalletDiscordNotifier:
    """ウォレット情報通知専用のDiscord通知クラス."""

    def __init__(self, config) -> None:
        discord_config = getattr(config, "discord", None)
        if not discord_config:
            self._enabled = False
            self._session = None
            self._channel_ids: Sequence[str] = ()
            return

        bot_token = getattr(discord_config, "bot_token", "")
        # ウォレット用チャンネルIDを取得（設定に追加する必要あり）
        wallet_channel_ids: Optional[Iterable[str]] = getattr(discord_config, "wallet_channel_ids", None)
        if not wallet_channel_ids:
            # ウォレット用チャンネルが設定されていない場合はメインチャンネルを使用
            wallet_channel_ids = getattr(discord_config, "channel_ids", [])
        
        if not bot_token or not wallet_channel_ids:
            self._enabled = False
            self._session = None
            self._channel_ids = ()
            return

        self._enabled = True
        self._session = requests.Session()
        self._bot_token = bot_token
        self._channel_ids = tuple(str(cid) for cid in wallet_channel_ids)

        rate_limit_cfg = getattr(discord_config, "rate_limit", None)
        max_messages = getattr(rate_limit_cfg, "max_messages", 5) if rate_limit_cfg else 5
        per_seconds = getattr(rate_limit_cfg, "per_seconds", 5) if rate_limit_cfg else 5
        self._max_messages = max(1, int(max_messages))
        self._per_seconds = max(1, int(per_seconds))
        self._timestamps = {cid: deque() for cid in self._channel_ids}

    def notify_wallet_summary(
        self,
        total_jpy: float,
        assets: list[dict[str, object]],
        hourly_change: Optional[float] = None,
        total_change: Optional[float] = None,
    ) -> None:
        """ウォレットサマリーを通知."""
        if not self._enabled:
            return
        
        # アセットリストを作成
        asset_lines = []
        for asset in assets:
            if asset["asset"] != "JPY":  # JPYは総額に含まれているので表示しない
                asset_lines.append(
                    f"• **{asset['asset']}**: `{asset['quantity']:.6f}` "
                    f"(`{asset['jpy_value']:.0f} JPY`)"
                )
        
        # 変化額をフォーマット
        change_lines = []
        if hourly_change is not None:
            direction = "▲" if hourly_change >= 0 else "▼"
            change_lines.append(f"1H Change: `{direction} {hourly_change:+.0f} JPY`")
        
        if total_change is not None:
            direction = "▲" if total_change >= 0 else "▼"
            change_lines.append(f"Total Change: `{direction} {total_change:+.0f} JPY`")
        
        message = (
            f"💰 **Wallet Summary**\n"
            f"Total Value: `{total_jpy:.0f} JPY`\n"
            f"\n"
            f"**Assets:**\n"
        )
        
        if asset_lines:
            message += "\n".join(asset_lines[:5])  # 最大5資産まで表示
            if len(asset_lines) > 5:
                message += f"\n• ... and {len(asset_lines) - 5} more"
        else:
            message += "• Only JPY balance"
        
        if change_lines:
            message += f"\n\n**Performance:**\n" + "\n".join(change_lines)
        
        self._send(message)

    def _send(self, content: str) -> None:
        if not self._enabled or not self._session:
            return
        headers = {
            "Authorization": f"Bot {self._bot_token}",
            "Content-Type": "application/json",
        }
        for channel_id in self._channel_ids:
            self._respect_rate_limit(channel_id)
            payload = {"content": content}
            try:
                response = self._session.post(
                    f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                if response.status_code == 429:
                    retry_after = response.json().get("retry_after", 1)
                    time.sleep(float(retry_after))
                    self._session.post(
                        f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                        headers=headers,
                        json=payload,
                        timeout=10,
                    )
                elif response.status_code >= 400:
                    logger.warning(
                        "Discord wallet alert failed for channel %s: %s", channel_id, response.text
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send Discord wallet alert: %s", exc)

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

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._enabled = False


class DiscordNotifier:
    """Lightweight Discord notifier dedicated to trade alerts."""

    def __init__(self, config) -> None:
        discord_config = getattr(config, "discord", None)
        if not discord_config:
            self._enabled = False
            self._session = None
            self._channel_ids: Sequence[str] = ()
            return

        bot_token = getattr(discord_config, "bot_token", "")
        channel_ids: Optional[Iterable[str]] = getattr(discord_config, "channel_ids", None)
        if not bot_token or not channel_ids:
            self._enabled = False
            self._session = None
            self._channel_ids = ()
            return

        self._enabled = True
        self._session = requests.Session()
        self._bot_token = bot_token
        self._channel_ids = tuple(str(cid) for cid in channel_ids)

        rate_limit_cfg = getattr(discord_config, "rate_limit", None)
        max_messages = getattr(rate_limit_cfg, "max_messages", 5) if rate_limit_cfg else 5
        per_seconds = getattr(rate_limit_cfg, "per_seconds", 5) if rate_limit_cfg else 5
        self._max_messages = max(1, int(max_messages))
        self._per_seconds = max(1, int(per_seconds))
        self._timestamps = {cid: deque() for cid in self._channel_ids}

    def notify_open(self, symbol: str, price: float, quantity: float, strategy: str, action: str) -> None:
        if not self._enabled:
            return
        message = (
            f"📈 **{action.upper()} {symbol}**\n"
            f"Price: `{price:.4f}`\n"
            f"Quantity: `{quantity:.6f}`\n"
            f"Strategy: `{strategy}`"
        )
        self._send(message)

    def notify_close(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl_percent: Optional[float],
        strategy: str,
    ) -> None:
        if not self._enabled:
            return
        change = pnl_percent if pnl_percent is not None else ((exit_price - entry_price) / entry_price) * 100
        direction = "▲" if change >= 0 else "▼"
        message = (
            f"✅ **Close {symbol}**\n"
            f"Entry: `{entry_price:.4f}`\n"
            f"Exit: `{exit_price:.4f}`\n"
            f"PnL: `{direction} {change:+.2f}%`\n"
            f"Strategy: `{strategy}`"
        )
        self._send(message)

    def notify_wallet_summary(
        self,
        total_jpy: float,
        assets: List[Dict[str, object]],
        hourly_change: Optional[float] = None,
        total_change: Optional[float] = None,
    ) -> None:
        """ウォレットサマリーを通知."""
        if not self._enabled:
            return
        
        # アセットリストを作成
        asset_lines = []
        for asset in assets:
            if asset["asset"] != "JPY":  # JPYは総額に含まれているので表示しない
                asset_lines.append(
                    f"• **{asset['asset']}**: `{asset['quantity']:.6f}` "
                    f"(`{asset['jpy_value']:.0f} JPY`)"
                )
        
        # 変化額をフォーマット
        change_lines = []
        if hourly_change is not None:
            direction = "▲" if hourly_change >= 0 else "▼"
            change_lines.append(f"1H Change: `{direction} {hourly_change:+.0f} JPY`")
        
        if total_change is not None:
            direction = "▲" if total_change >= 0 else "▼"
            change_lines.append(f"Total Change: `{direction} {total_change:+.0f} JPY`")
        
        message = (
            f"💰 **Wallet Summary**\n"
            f"Total Value: `{total_jpy:.0f} JPY`\n"
            f"\n"
            f"**Assets:**\n"
        )
        
        if asset_lines:
            message += "\n".join(asset_lines[:5])  # 最大5資産まで表示
            if len(asset_lines) > 5:
                message += f"\n• ... and {len(asset_lines) - 5} more"
        else:
            message += "• Only JPY balance"
        
        if change_lines:
            message += f"\n\n**Performance:**\n" + "\n".join(change_lines)
        
        self._send(message)

    def _send(self, content: str) -> None:
        if not self._enabled or not self._session:
            return
        headers = {
            "Authorization": f"Bot {self._bot_token}",
            "Content-Type": "application/json",
        }
        for channel_id in self._channel_ids:
            self._respect_rate_limit(channel_id)
            payload = {"content": content}
            try:
                response = self._session.post(
                    f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                if response.status_code == 429:
                    retry_after = response.json().get("retry_after", 1)
                    time.sleep(float(retry_after))
                    self._session.post(
                        f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                        headers=headers,
                        json=payload,
                        timeout=10,
                    )
                elif response.status_code >= 400:
                    logger.warning(
                        "Discord trade alert failed for channel %s: %s", channel_id, response.text
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send Discord trade alert: %s", exc)

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

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._enabled = False


class TradeOnlyFilter(logging.Filter):
    """BUY/SELL/CLOSE関連のログのみを通すフィルタ."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Discordに送信したいキーワードを含むログのみ通す
        message = record.getMessage().upper()
        
        # 取引関連のキーワード
        trade_keywords = [
            "BUY", "SELL", "CLOSE", "OPENED", "CLOSED",
            "POSITION OPENED", "POSITION CLOSED", 
            "EXECUTING", "ORDER", "TRADE"
        ]
        
        # 除外したいキーワード（設定読み込みなど）
        exclude_keywords = [
            "CONFIGURATION LOADED", "STARTING", "INITIALIZED",
            "RESTORING", "BALANCE", "WALLET", "DEBUG"
        ]
        
        # 除外キーワードを含む場合は拒否
        for keyword in exclude_keywords:
            if keyword in message:
                return False
        
        # 取引キーワードを含む場合は許可
        for keyword in trade_keywords:
            if keyword in message:
                return True
        
        return False
