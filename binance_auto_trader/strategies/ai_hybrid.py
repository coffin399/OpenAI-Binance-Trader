from __future__ import annotations

import logging
from statistics import mean
from typing import Dict, Optional

import pandas as pd

from binance_auto_trader.models.trade import StrategyDecision

from .base import Strategy

logger = logging.getLogger(__name__)


class AIHybridStrategy(Strategy):
    """Strategy that consults an AI provider for trading decisions."""

    def __init__(self, ai_manager, config_section) -> None:
        super().__init__()
        self.ai_manager = ai_manager
        self.provider_name = getattr(config_section, "provider", None)
        self.prompt_template = getattr(config_section, "prompt_template", "")
        if not self.provider_name:
            raise ValueError("AIHybridStrategy requires 'provider' setting")
        if not self.ai_manager or not self.ai_manager.has_provider(self.provider_name):
            raise ValueError(f"AI provider '{self.provider_name}' not available")
        self._last_timestamp: Dict[str, pd.Timestamp] = {}

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyDecision]:
        prompt_context = self._build_prompt_context(df, symbol)
        latest_timestamp: pd.Timestamp = df.iloc[-1]["timestamp"]
        if symbol in self._last_timestamp and self._last_timestamp[symbol] == latest_timestamp:
            return None
        self._last_timestamp[symbol] = latest_timestamp
        try:
            prompt = self.prompt_template.format(**prompt_context)
        except KeyError as exc:  # noqa: BLE001
            missing = exc.args[0]
            logger.warning(
                "Prompt template missing key '%s'. Falling back to basic format.",
                missing,
            )
            prompt = self.prompt_template.format(symbol=symbol)
        try:
            response = self.ai_manager.generate(self.provider_name, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("AI provider call failed for %s: %s", symbol, exc)
            return None

        action = response.strip().upper().split()[0]
        if action not in {"BUY", "SELL", "HOLD"}:
            logger.warning("AI provider returned unsupported action '%s'", action)
            return None

        if action == "HOLD":
            return None

        price = float(df.iloc[-1]["close"])
        return StrategyDecision(
            symbol=symbol,
            strategy=self.name,
            action=action,
            price=price,
            confidence=0.5,
            info=f"AI decision: {action}",
        )

    def _build_prompt_context(self, df: pd.DataFrame, symbol: str) -> Dict[str, object]:
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        timeframe = self._infer_timeframe(df)

        recent = df.tail(6)
        recent_rows = [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for _, row in recent.iterrows()
        ]
        closes = [row["close"] for row in recent_rows]
        change_pct = (
            ((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) > 1 else 0.0
        )

        fast_sma = float(df["close"].rolling(window=5).mean().iloc[-1])
        slow_sma = float(df["close"].rolling(window=20).mean().iloc[-1])
        avg_volume = mean(row["volume"] for row in recent_rows)

        latest_candle = {
            "timestamp": recent_rows[-1]["timestamp"],
            "open": recent_rows[-1]["open"],
            "high": recent_rows[-1]["high"],
            "low": recent_rows[-1]["low"],
            "close": recent_rows[-1]["close"],
            "volume": recent_rows[-1]["volume"],
            "change_pct": (
                ((recent_rows[-1]["close"] - previous["close"]) / previous["close"] * 100)
                if previous["close"]
                else 0.0
            ),
        }

        recent_text = "\n".join(
            f"{row['timestamp']}: O={row['open']:.4f} H={row['high']:.4f} "
            f"L={row['low']:.4f} C={row['close']:.4f} V={row['volume']:.2f}"
            for row in recent_rows
        )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "latest_price": float(latest["close"]),
            "latest_candle": latest_candle,
            "previous_close": float(previous["close"]),
            "recent_candles": recent_text,
            "recent_change_pct": change_pct,
            "fast_sma": fast_sma,
            "slow_sma": slow_sma,
            "volume_avg": avg_volume,
        }

    @staticmethod
    def _infer_timeframe(df: pd.DataFrame) -> str:
        if len(df) < 2:
            return "unknown"
        delta = df["timestamp"].iloc[-1] - df["timestamp"].iloc[-2]
        minutes = int(delta.total_seconds() // 60)
        if minutes <= 0:
            return "unknown"
        if minutes % 1440 == 0:
            return f"{minutes // 1440}d"
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}m"
