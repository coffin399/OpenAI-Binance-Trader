from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Sequence
import math

from binance_auto_trader.models.trade import BacktestResult, TradeRecord
from binance_auto_trader.utils.helpers import compute_sortino_ratio

COLOR_PALETTE = [
    "#6366F1",
    "#22D3EE",
    "#F97316",
    "#10B981",
    "#F59E0B",
    "#EC4899",
    "#8B5CF6",
    "#14B8A6",
    "#FACC15",
    "#EF4444",
]


class TradeTracker:
    def __init__(self, config) -> None:
        self._lock = threading.Lock()
        self.dry_run = bool(getattr(config.runtime, "dry_run", True))
        active = getattr(config.strategies, "active", []) if hasattr(config, "strategies") else []
        self.active_strategies = list(active) if isinstance(active, list) else []

        symbols_cfg: Sequence[str] = getattr(config.trading, "symbols", []) if hasattr(config, "trading") else []
        asset_config = getattr(config, "asset_management", {})
        swap_pairs_cfg: Sequence[str] = getattr(asset_config, "swap_pairs", [])
        if isinstance(symbols_cfg, list):
            self.symbol_order = list(symbols_cfg)
        else:
            self.symbol_order = [symbols_cfg] if symbols_cfg else []
        self.swap_pairs = list(swap_pairs_cfg) if swap_pairs_cfg else []

        self.open_trades: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []
        self.backtest_results: List[BacktestResult] = []
        self.price_history: Dict[str, Deque[tuple[str, float]]] = {}
        self.price_history_limit = 500
        self.symbol_palette: Dict[str, str] = {}
        self._palette_index = 0

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def append_price_point(self, symbol: str, timestamp, price: float) -> None:
        iso_timestamp = (
            timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        )
        with self._lock:
            history = self.price_history.setdefault(symbol, deque(maxlen=self.price_history_limit))
            history.append((iso_timestamp, price))
            self._ensure_color(symbol)

    def record_open_trade(
        self,
        symbol: str,
        strategy: str,
        action: str,
        quantity: float,
        price: float,
        opened_at: Optional[datetime] = None,
    ) -> None:
        opened_at = opened_at or datetime.utcnow()
        record = TradeRecord(
            symbol=symbol,
            strategy=strategy,
            action=action,
            quantity=quantity,
            entry_price=price,
            exit_price=None,
            pnl_percent=None,
            opened_at=opened_at,
            closed_at=None,
            status="OPEN",
        )
        with self._lock:
            self.open_trades[symbol] = record
            self._ensure_color(symbol)

    def record_close_trade(
        self,
        symbol: str,
        price: float,
        closed_at: Optional[datetime] = None,
    ) -> Optional[TradeRecord]:
        closed_at = closed_at or datetime.utcnow()
        with self._lock:
            record = self.open_trades.pop(symbol, None)
            if not record:
                return None

            direction = 1 if record.action.upper() == "BUY" else -1
            pnl = ((price - record.entry_price) / record.entry_price) * 100 * direction
            record.exit_price = price
            record.closed_at = closed_at
            record.pnl_percent = pnl
            record.status = "CLOSED"
            if record.quantity is None:
                record.quantity = 0.0
            self.closed_trades.append(record)
            return record

    def set_backtest_results(self, results: Iterable[BacktestResult]) -> None:
        with self._lock:
            self.backtest_results = list(results)

    def get_open_trade_count(self) -> int:
        with self._lock:
            return len(self.open_trades)

    def has_open_trade(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self.open_trades

    def get_total_open_value(self) -> float:
        with self._lock:
            total = 0.0
            for trade in self.open_trades.values():
                quantity = float(trade.quantity) if trade.quantity is not None else 0.0
                entry_price = float(trade.entry_price) if trade.entry_price is not None else 0.0
                total += abs(quantity) * entry_price
            return total

    def get_open_trade_value(self, symbol: str) -> float:
        with self._lock:
            trade = self.open_trades.get(symbol)
            if not trade:
                return 0.0
            quantity = float(trade.quantity) if trade.quantity is not None else 0.0
            entry_price = float(trade.entry_price) if trade.entry_price is not None else 0.0
            return abs(quantity) * entry_price

    # ------------------------------------------------------------------
    # Palette helpers
    # ------------------------------------------------------------------
    def _ensure_color(self, symbol: str) -> str:
        if symbol not in self.symbol_palette:
            color = COLOR_PALETTE[self._palette_index % len(COLOR_PALETTE)]
            self.symbol_palette[symbol] = color
            self._palette_index += 1
        return self.symbol_palette[symbol]

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def get_live_summary(self) -> Dict[str, float]:
        with self._lock:
            closed_returns = [
                trade.pnl_percent for trade in self.closed_trades if trade.pnl_percent is not None
            ]
            trade_count = len(self.closed_trades)
            average_profit = sum(closed_returns) / trade_count if trade_count else 0.0
            max_profit = max(closed_returns) if closed_returns else 0.0
            sortino_ratio = compute_sortino_ratio(closed_returns)

            return {
                "mode": "DRY-RUN" if self.dry_run else "LIVE",
                "trade_count": trade_count,
                "strategy_count": len(self.active_strategies),
                "average_profit": self._format_metric(average_profit),
                "max_profit": self._format_metric(max_profit),
                "sortino_ratio": self._format_metric(sortino_ratio),
            }

    def get_backtest_summary(self) -> Dict[str, float]:
        with self._lock:
            if not self.backtest_results:
                return {
                    "trade_count": 0,
                    "strategy_count": 0,
                    "average_profit": 0.0,
                    "max_profit": 0.0,
                    "sortino_ratio": 0.0,
                }

            all_returns: List[float] = []
            trade_count = 0
            strategy_names = set()
            for result in self.backtest_results:
                trade_count += result.trade_count
                strategy_names.add(result.strategy)
                all_returns.extend(result.returns)

            average_profit = sum(all_returns) / len(all_returns) if all_returns else 0.0
            max_profit = max(all_returns) if all_returns else 0.0
            sortino_ratio = compute_sortino_ratio(all_returns)

            return {
                "trade_count": trade_count,
                "strategy_count": len(strategy_names),
                "average_profit": self._format_metric(average_profit),
                "max_profit": self._format_metric(max_profit),
                "sortino_ratio": self._format_metric(sortino_ratio),
            }

    # ------------------------------------------------------------------
    # Payload helpers for the web UI
    # ------------------------------------------------------------------
    def get_summary_payload(self) -> Dict[str, object]:
        summary = self.get_live_summary()
        with self._lock:
            open_trades = [self._trade_to_dict(trade_symbol, trade) for trade_symbol, trade in self.open_trades.items()]
            recent_closed = [self._trade_to_dict(trade.symbol, trade) for trade in self.closed_trades[-10:]]
        summary.update({
            "open_trades": open_trades,
            "recent_closed_trades": recent_closed,
            "symbols": self._build_symbol_summaries(),
            "swap_pairs": self.swap_pairs,
            "wallet": self._get_wallet_summary(),
        })
        return summary

    def _get_wallet_summary(self) -> Dict[str, object]:
        """ウォレット残高サマリーを取得."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # グローバルインスタンスからexchangeを取得
            if hasattr(self, '_bot_instance') and self._bot_instance:
                account = self._bot_instance.exchange.client.get_account()
                wallet_summary = {
                    "total_jpy_value": 0.0,
                    "assets": []
                }
                
                for balance in account['balances']:
                    asset = balance['asset']
                    free_qty = float(balance['free'])
                    
                    if free_qty > 0:
                        if asset == 'JPY':
                            jpy_value = free_qty
                            price = 1.0
                        else:
                            # 現在価格を取得してJPY換算
                            try:
                                symbol = None
                                for display_symbol in self.symbol_order:
                                    if asset in display_symbol:
                                        symbol = display_symbol.replace("/", "")
                                        break
                                
                                if symbol:
                                    ticker = self._bot_instance.exchange.client.get_symbol_ticker(symbol=symbol)
                                    price = float(ticker['price'])
                                    jpy_value = free_qty * price
                                else:
                                    price = 0.0
                                    jpy_value = 0.0
                            except Exception:
                                price = 0.0
                                jpy_value = 0.0
                        
                        wallet_summary["assets"].append({
                            "asset": asset,
                            "quantity": free_qty,
                            "price": price,
                            "jpy_value": jpy_value
                        })
                        
                        wallet_summary["total_jpy_value"] += jpy_value
                
                return wallet_summary
            else:
                logger.warning("Wallet summary unavailable: bot instance not attached to TradeTracker")
                return {"total_jpy_value": 0.0, "assets": []}
        except Exception as exc:
            logger.warning("Error getting wallet summary: %s", exc)
            return {"total_jpy_value": 0.0, "assets": []}

    def get_price_history_payload(self) -> Dict[str, object]:
        with self._lock:
            series = []
            for symbol, history in self.price_history.items():
                timestamps = [row[0] for row in history]
                prices = [row[1] for row in history]
                series.append(
                    {
                        "symbol": symbol,
                        "timestamps": timestamps,
                        "prices": prices,
                        "color": self._ensure_color(symbol),
                    }
                )
        return {"series": series}

    def get_backtest_payload(self) -> Dict[str, object]:
        summary = self.get_backtest_summary()
        with self._lock:
            results = [self._backtest_to_dict(result) for result in self.backtest_results]
        return {"summary": summary, "results": results}

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _trade_to_dict(self, symbol: str, trade: TradeRecord) -> Dict[str, object]:
        color = self._ensure_color(symbol)
        return {
            "symbol": trade.symbol,
            "strategy": trade.strategy,
            "action": trade.action,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl_percent": trade.pnl_percent,
            "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
            "status": trade.status,
            "color": color,
        }

    def _backtest_to_dict(self, result: BacktestResult) -> Dict[str, object]:
        return {
            "strategy": result.strategy,
            "symbol": result.symbol,
            "trade_count": result.trade_count,
            "average_return": self._format_metric(result.average_return),
            "max_return": self._format_metric(result.max_return),
            "sortino_ratio": self._format_metric(result.sortino_ratio),
            "returns": result.returns,
            "generated_at": result.generated_at.isoformat(),
        }

    @staticmethod
    def _format_metric(value: Optional[float]) -> Optional[object]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if math.isnan(value):  # type: ignore[arg-type]
            return 0.0
        if math.isinf(value):  # type: ignore[arg-type]
            return "∞" if value > 0 else "-∞"  # type: ignore[operator]
        return round(float(value), 2)

    def _build_symbol_summaries(self) -> List[Dict[str, object]]:
        ordered_symbols: List[str] = []
        seen = set()
        for symbol in self.symbol_order:
            if symbol not in seen:
                ordered_symbols.append(symbol)
                seen.add(symbol)
        for symbol in self.price_history.keys():
            if symbol not in seen:
                ordered_symbols.append(symbol)
                seen.add(symbol)

        summaries: List[Dict[str, object]] = []
        for symbol in ordered_symbols:
            summaries.append(self._symbol_summary(symbol))
        return summaries

    def _symbol_summary(self, symbol: str) -> Dict[str, object]:
        price = self._latest_price(symbol)
        open_trade = self.open_trades.get(symbol)
        status = "OPEN" if open_trade else "FLAT"
        position = open_trade.action if open_trade else "NONE"
        quantity = open_trade.quantity if open_trade and open_trade.quantity is not None else None
        quantity_value = round(quantity, 6) if quantity is not None else 0.0
        entry_price = open_trade.entry_price if open_trade else None
        change_pct: Optional[float] = None
        if open_trade and price is not None and open_trade.entry_price:
            direction = 1 if open_trade.action.upper() == "BUY" else -1
            try:
                change_pct = (
                    (price - open_trade.entry_price) / open_trade.entry_price * 100 * direction
                )
            except ZeroDivisionError:
                change_pct = None

        balance_jpy = None
        if quantity_value and price is not None:
            balance_jpy = round(quantity_value * price, 2)

        return {
            "symbol": symbol,
            "price": round(price, 4) if price is not None else None,
            "position": position,
            "status": status,
            "change_pct": self._format_metric(change_pct) if change_pct is not None else None,
            "entry_price": round(entry_price, 4) if entry_price is not None else None,
            "quantity": quantity_value,
            "balance": quantity_value,
            "balance_jpy": balance_jpy,
            "color": self._ensure_color(symbol),
            "updated_at": self._latest_timestamp(symbol),
        }

    def _latest_price(self, symbol: str) -> Optional[float]:
        history = self.price_history.get(symbol)
        if history and len(history) > 0:
            return float(history[-1][1])
        return None

    def _latest_timestamp(self, symbol: str) -> Optional[str]:
        history = self.price_history.get(symbol)
        if history and len(history) > 0:
            return history[-1][0]
        return None
