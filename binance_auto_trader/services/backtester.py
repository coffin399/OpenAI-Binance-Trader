from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from binance_auto_trader.config.config import Config
from binance_auto_trader.exchange.binance_client import BinanceClient
from binance_auto_trader.models.trade import BacktestResult
from binance_auto_trader.strategies import build_strategy
from binance_auto_trader.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    enabled: bool
    lookback_intervals: int
    symbols: List[str]
    timeframe: str


class Backtester:
    def __init__(
        self,
        config: Config,
        ai_manager,
        exchange: Optional[BinanceClient] = None,
    ) -> None:
        self.config = config
        self.ai_manager = ai_manager
        self.exchange = exchange or BinanceClient(config)

        backtest_section = getattr(config, "backtesting", None)
        if backtest_section:
            self.bt_config = BacktestConfig(
                enabled=bool(getattr(backtest_section, "enabled", False)),
                lookback_intervals=int(getattr(backtest_section, "lookback_intervals", 200)),
                symbols=list(getattr(backtest_section, "symbols", [])),
                timeframe=getattr(backtest_section, "timeframe", "1h"),
            )
        else:
            self.bt_config = BacktestConfig(False, 0, [], "1h")

    def run(self) -> List[BacktestResult]:
        if not self.bt_config.enabled:
            logger.info("Backtesting disabled in configuration")
            return []

        strategies = getattr(self.config.strategies, "active", [])
        if not strategies:
            logger.info("No strategies configured, skipping backtests")
            return []

        results: List[BacktestResult] = []
        for display_symbol in self.bt_config.symbols:
            normalized_symbol = display_symbol.replace("/", "")
            klines = self.exchange.get_historical_klines(
                symbol=normalized_symbol,
                interval=self.bt_config.timeframe,
                limit=max(500, self.bt_config.lookback_intervals),
            )
            if not klines:
                logger.warning("No historical data for %s, skipping backtest", display_symbol)
                continue

            dataframe = Strategy.klines_to_dataframe(klines)
            dataframe = dataframe.tail(self.bt_config.lookback_intervals).reset_index(drop=True)

            for strategy_name in strategies:
                strategy_config = None
                if (
                    hasattr(self.config.strategies, "_data")
                    and strategy_name in self.config.strategies._data
                ):
                    strategy_config = self.config.strategies.__class__(
                        self.config.strategies._data[strategy_name]
                    )
                try:
                    strategy = build_strategy(strategy_name, strategy_config, self.ai_manager)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to initialize strategy %s for backtest: %s", strategy_name, exc)
                    continue

                result = self._run_for_strategy(strategy, dataframe, display_symbol)
                if result:
                    results.append(result)
        return results

    def _run_for_strategy(
        self, strategy: Strategy, dataframe: pd.DataFrame, display_symbol: str
    ) -> Optional[BacktestResult]:
        open_trade_price: Optional[float] = None
        open_action: Optional[str] = None
        returns: List[float] = []
        trade_count = 0

        for idx in range(2, len(dataframe)):
            window = dataframe.iloc[: idx + 1]
            decision = strategy.evaluate(window, display_symbol)
            if not decision:
                continue

            price = float(window.iloc[-1]["close"])
            if decision.action == "BUY":
                if open_action == "BUY":
                    continue
                if open_action == "SELL" and open_trade_price is not None:
                    pnl = ((open_trade_price - price) / open_trade_price) * 100
                    returns.append(pnl)
                    trade_count += 1
                    open_trade_price = None
                    open_action = None
                open_trade_price = price
                open_action = "BUY"
            elif decision.action == "SELL":
                if open_action == "SELL":
                    continue
                if open_action == "BUY" and open_trade_price is not None:
                    pnl = ((price - open_trade_price) / open_trade_price) * 100
                    returns.append(pnl)
                    trade_count += 1
                    open_trade_price = None
                    open_action = None
                else:
                    open_trade_price = price
                    open_action = "SELL"

        average_return = sum(returns) / len(returns) if returns else 0.0
        max_return = max(returns) if returns else 0.0
        sortino_ratio = self._sortino_ratio(returns)

        return BacktestResult(
            strategy=strategy.name,
            symbol=display_symbol,
            trade_count=trade_count,
            average_return=average_return,
            max_return=max_return,
            sortino_ratio=sortino_ratio,
            returns=returns,
        )

    @staticmethod
    def _sortino_ratio(returns: List[float]) -> float:
        if not returns:
            return 0.0
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf")
        mean_return = sum(returns) / len(returns)
        downside_deviation = (sum((r - 0) ** 2 for r in downside) / len(downside)) ** 0.5
        if downside_deviation == 0:
            return float("inf")
        return mean_return / downside_deviation
