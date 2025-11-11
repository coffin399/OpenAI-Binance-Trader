from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from binance_auto_trader.models.trade import StrategyDecision

from .base import Strategy


class MovingAverageCrossover(Strategy):
    """Classic moving-average crossover strategy."""

    def __init__(self, fast_ma: int = 9, slow_ma: int = 21) -> None:
        super().__init__()
        if fast_ma >= slow_ma:
            raise ValueError("fast_ma must be smaller than slow_ma for crossover strategy")
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.logger = logging.getLogger(__name__)

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyDecision]:
        try:
            df = df.copy()
            df["fast_ma"] = df["close"].rolling(window=self.fast_ma).mean()
            df["slow_ma"] = df["close"].rolling(window=self.slow_ma).mean()

            last = df.iloc[-1]
            fast_above = last["fast_ma"] > last["slow_ma"]
            previous = df.iloc[-2]
            crossed_up = fast_above and previous["fast_ma"] <= previous["slow_ma"]
            crossed_down = (last["fast_ma"] < last["slow_ma"]) and (
                previous["fast_ma"] >= previous["slow_ma"]
            )

            price = float(last["close"])

            if crossed_up:
                return StrategyDecision(
                    symbol=symbol,
                    strategy=self.name,
                    action="BUY",
                    price=price,
                    confidence=0.7,
                    info=f"{self.fast_ma}/{self.slow_ma} crossover bullish",
                )
            if crossed_down:
                return StrategyDecision(
                    symbol=symbol,
                    strategy=self.name,
                    action="SELL",
                    price=price,
                    confidence=0.7,
                    info=f"{self.fast_ma}/{self.slow_ma} crossover bearish",
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to evaluate MA crossover for %s: %s", symbol, exc)
        return None
