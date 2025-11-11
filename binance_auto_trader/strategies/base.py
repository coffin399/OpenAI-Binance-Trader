from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import pandas as pd

from binance_auto_trader.models.trade import StrategyDecision


class Strategy(ABC):
    name: str

    def __init__(self) -> None:
        self.name = getattr(self, "name", self.__class__.__name__.lower())

    @staticmethod
    def klines_to_dataframe(klines: Iterable) -> pd.DataFrame:
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        float_columns = ["open", "high", "low", "close", "volume"]
        for column in float_columns:
            df[column] = df[column].astype(float)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyDecision]:
        """Return a trading decision given market data."""
        raise NotImplementedError
