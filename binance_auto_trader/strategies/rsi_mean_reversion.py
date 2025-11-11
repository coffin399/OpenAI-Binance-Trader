from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from binance_auto_trader.models.trade import StrategyDecision

from .base import Strategy

logger = logging.getLogger(__name__)


class RSIMeanReversionStrategy(Strategy):
    """RSI の閾値で売買するシンプルなミーンリバージョン戦略。"""

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        minimum_history: int = 50,
    ) -> None:
        super().__init__()
        if period <= 0:
            raise ValueError("RSI period must be positive")
        if not 0 < oversold < overbought < 100:
            raise ValueError("RSI thresholds must satisfy 0 < oversold < overbought < 100")

        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.minimum_history = max(minimum_history, period + 5)

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyDecision]:
        if len(df) < self.minimum_history:
            return None

        rsi = self._relative_strength_index(df["close"], self.period)
        if rsi is None or len(rsi) < 2:
            return None

        latest_rsi = float(rsi.iloc[-1])
        previous_rsi = float(rsi.iloc[-2])
        price = float(df.iloc[-1]["close"])

        # Oversold -> BUY
        if previous_rsi < self.oversold <= latest_rsi:
            confidence = min((self.oversold - previous_rsi) / max(self.oversold, 1.0), 1.0)
            return StrategyDecision(
                symbol=symbol,
                strategy=self.name,
                action="BUY",
                price=price,
                confidence=confidence,
                info=f"RSI cross up {previous_rsi:.2f}->{latest_rsi:.2f}",
            )

        # Overbought -> SELL
        if previous_rsi > self.overbought >= latest_rsi:
            denominator = max(100 - self.overbought, 1.0)
            confidence = min((previous_rsi - self.overbought) / denominator, 1.0)
            return StrategyDecision(
                symbol=symbol,
                strategy=self.name,
                action="SELL",
                price=price,
                confidence=confidence,
                info=f"RSI cross down {previous_rsi:.2f}->{latest_rsi:.2f}",
            )

        return None

    @staticmethod
    def _relative_strength_index(series: pd.Series, period: int) -> Optional[pd.Series]:
        close = series.astype(float)
        delta = close.diff()
        if delta.isna().all():
            return None

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method="bfill")