from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from binance_auto_trader.ai.provider_manager import AIProviderManager
from binance_auto_trader.config.config import Config
from binance_auto_trader.exchange.binance_client import BinanceClient
from binance_auto_trader.models.trade import StrategyDecision
from binance_auto_trader.services.backtester import Backtester
from binance_auto_trader.services.trade_tracker import TradeTracker
from binance_auto_trader.strategies import build_strategy
from binance_auto_trader.strategies.base import Strategy
from binance_auto_trader.utils.discord_notify import DiscordNotifier
from binance_auto_trader.utils.helpers import (
    normalize_symbol,
    parse_currency_limit,
    parse_labeled_limit,
)
from binance_auto_trader.utils.logger import setup_logging
from binance_auto_trader.web.server import start_dashboard

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = BinanceClient(config)

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self.ai_manager = AIProviderManager(getattr(config, "ai", None))
        self.trade_tracker = TradeTracker(config)
        self.notifier = DiscordNotifier(config)

        symbol_list = getattr(config.trading, "symbols", None)
        if not symbol_list:
            symbol_list = [getattr(config.trading, "symbol", "BTC/USDT")]
        self.symbols_display: List[str] = symbol_list
        self.symbol_map: Dict[str, str] = {
            display: normalize_symbol(display) for display in self.symbols_display
        }

        self.timeframe = getattr(config.trading, "timeframe", "1h")
        raw_quantity = float(getattr(config.trading, "quantity", 0.0))
        # 0.0の場合はAI数量モードのためにNoneを設定せず0.0のまま保持
        self.fixed_quantity = raw_quantity if raw_quantity > 0 else 0.0
        self.max_open_trades = parse_labeled_limit(
            getattr(config.trading, "max_open_trades", "0Trades"),
            "Trades",
        )
        self.max_investment_per_trade = parse_currency_limit(
            getattr(config.trading, "max_investment_per_trade", "0JPY"),
            "JPY",
        )
        self.max_total_investment = parse_currency_limit(
            getattr(config.trading, "max_total_investment", "0JPY"),
            "JPY",
        )
        
        # 初期資金設定
        self.initial_jpy = parse_currency_limit(
            getattr(getattr(config, "initial_capital", {}), "jpy_amount", "10000JPY"),
            "JPY",
        )
        self.allow_start_from_cash = getattr(
            getattr(config, "initial_capital", {}), "allow_start_from_cash", True
        )
        self.polling_interval = int(
            getattr(config.runtime, "polling_interval_seconds", 60)
        )
        self.dry_run = bool(getattr(config.runtime, "dry_run", True))

        strategies = getattr(config.strategies, "active", [])
        self.strategies = self._initialize_strategies(strategies)

        self.positions: Dict[str, Optional[str]] = {symbol: None for symbol in self.symbols_display}

        # 起動時にオープンポジションを検知して復元
        if not self.dry_run:
            self._restore_open_positions()

        if self.dry_run:
            logger.info("Running in dry-run mode. No live orders will be sent.")

    def _initialize_strategies(self, strategy_names) -> List[Strategy]:
        if not strategy_names:
            logger.warning("No strategies configured. Defaulting to moving_average_cross.")
            strategy_names = ["moving_average_cross"]

        instances: List[Strategy] = []
        for name in strategy_names:
            config_section = None
            if hasattr(self.config.strategies, "_data") and name in self.config.strategies._data:
                value = self.config.strategies._data[name]
                if isinstance(value, dict):
                    config_section = self.config.strategies.__class__(value)
            try:
                strategy = build_strategy(name, config_section, self.ai_manager)
                instances.append(strategy)
                logger.info("Loaded strategy '%s'", name)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to initialize strategy %s: %s", name, exc)
        return instances

    def run(self) -> None:
        """Main trading loop."""
        logger.info(
            "Starting trading bot for symbols: %s", ", ".join(self.symbols_display)
        )

        while not self._stop_event.is_set():
            self._pause_event.wait()
            cycle_started = time.monotonic()
            try:
                self._process_cycle()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error in main loop: %s", exc)

            elapsed = time.monotonic() - cycle_started
            sleep_for = max(1, self.polling_interval - int(elapsed))
            time.sleep(sleep_for)

    def _process_cycle(self) -> None:
        for display_symbol, exchange_symbol in self.symbol_map.items():
            klines = self.exchange.get_historical_klines(
                symbol=exchange_symbol,
                interval=self.timeframe,
                limit=200,
            )

            if not klines:
                logger.warning(
                    "No klines data received for %s. Skipping this cycle.", display_symbol
                )
                continue

            dataframe = Strategy.klines_to_dataframe(klines)
            last_row = dataframe.iloc[-1]
            last_price = float(last_row["close"])
            live_price = self.exchange.get_symbol_price(exchange_symbol)
            price_point = live_price if live_price is not None else last_price
            self.trade_tracker.append_price_point(
                display_symbol, datetime.utcnow(), price_point
            )

            for strategy in self.strategies:
                decision = strategy.evaluate(dataframe, display_symbol)
                if decision:
                    self._handle_decision(decision, exchange_symbol, last_price)

    def _handle_decision(
        self,
        decision: StrategyDecision,
        exchange_symbol: str,
        last_price: float,
    ) -> None:
        display_symbol = decision.symbol
        current_position = self.positions.get(display_symbol)

        if decision.action == "BUY":
            if current_position == "LONG":
                return
            if current_position == "SHORT":
                self._close_position(display_symbol, exchange_symbol, "SHORT", last_price)
            self._open_position(
                display_symbol,
                exchange_symbol,
                "LONG",
                "BUY",
                decision,
                last_price,
            )

        elif decision.action == "SELL":
            if current_position == "SHORT":
                return
            if current_position == "LONG":
                self._close_position(display_symbol, exchange_symbol, "LONG", last_price)
            self._open_position(
                display_symbol,
                exchange_symbol,
                "SHORT",
                "SELL",
                decision,
                last_price,
            )

    def _should_open_position(
        self,
        display_symbol: str,
        exchange_symbol: str,
        position_label: str,
        side: str,
        decision: StrategyDecision,
        price: float,
    ) -> bool:
        open_trade_count = self.trade_tracker.get_open_trade_count()
        has_trade = self.trade_tracker.has_open_trade(display_symbol)
        if (
            self.max_open_trades > 0
            and open_trade_count >= self.max_open_trades
            and not has_trade
        ):
            logger.warning(
                "Max concurrent positions reached (%s). Skipping new position for %s.",
                self.max_open_trades,
                display_symbol,
            )
            return False
        
        # 現金残高チェック（JPYのみからエントリー開始）
        if self.allow_start_from_cash and open_trade_count == 0:
            jpy_balance = self._get_jpy_balance()
            if jpy_balance is None:
                logger.warning("Unable to get JPY balance. Skipping entry.")
                return False
            if jpy_balance < 500:  # 最低500JPY必要（少額スタート対応）
                logger.info(
                    "Insufficient JPY balance (%.0f). Need at least 500 JPY to start trading.",
                    jpy_balance
                )
                return False
            logger.info("Starting from cash position. JPY balance: %.0f", jpy_balance)

        return True

    def _open_position(
        self,
        display_symbol: str,
        exchange_symbol: str,
        position_label: str,
        side: str,
        decision: StrategyDecision,
        price: float,
    ) -> None:
        if not self._should_open_position(
            display_symbol,
            exchange_symbol,
            position_label,
            side,
            decision,
            price,
        ):
            return

        quantity = self._determine_quantity(price, decision)
        logger.info("Determined quantity: %s for %s (AI decision quantity: %s)", 
                   quantity, display_symbol, getattr(decision, 'quantity', 'None'))
        
        if quantity <= 0:
            logger.warning(
                "Calculated quantity is 0 for %s at price %.4f. Check configuration.",
                display_symbol,
                price,
            )
            return

        if self.max_total_investment > 0:
            current_total = self.trade_tracker.get_total_open_value()
            existing_value = self.trade_tracker.get_open_trade_value(display_symbol)
            projected_total = current_total - existing_value + (abs(quantity) * price)
            if projected_total > self.max_total_investment:
                logger.warning(
                    "Total capital limit reached (%.2f > %.2f). Skipping new position for %s.",
                    projected_total,
                    self.max_total_investment,
                    display_symbol,
                )
                return

        order = self._submit_order(exchange_symbol, side, quantity)
        if order:
            self.positions[display_symbol] = position_label
            self.trade_tracker.record_open_trade(
                symbol=display_symbol,
                strategy=decision.strategy,
                action=decision.action,
                quantity=quantity,
                price=price,
            )
            logger.info(
                "TRADE OPEN %s %s qty=%.6f price=%.4f strategy=%s",
                display_symbol,
                decision.action,
                quantity,
                price,
                decision.strategy,
            )
            try:
                self.notifier.notify_open(
                    symbol=display_symbol,
                    price=price,
                    quantity=quantity,
                    strategy=decision.strategy,
                    action=decision.action,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to send Discord open notification")

    def _close_position(
        self,
        display_symbol: str,
        exchange_symbol: str,
        position_type: str,
        price: float,
    ) -> None:
        side = "SELL" if position_type == "LONG" else "BUY"
        open_record = self.trade_tracker.open_trades.get(display_symbol)
        if open_record and open_record.quantity:
            quantity = open_record.quantity
        else:
            quantity = self._determine_quantity(price, None)

        if quantity <= 0:
            logger.warning(
                "Close position skipped for %s. Unable to resolve quantity.",
                display_symbol,
            )
            return

        order = self._submit_order(exchange_symbol, side, quantity)
        if order:
            self.positions[display_symbol] = None
            closed = self.trade_tracker.record_close_trade(display_symbol, price)
            logger.info(
                "TRADE CLOSE %s %s qty=%.6f price=%.4f record=%s",
                display_symbol,
                side,
                quantity,
                price,
                closed,
            )
            if closed:
                try:
                    self.notifier.notify_close(
                        symbol=display_symbol,
                        entry_price=closed.entry_price,
                        exit_price=price,
                        pnl_percent=closed.pnl_percent,
                        strategy=closed.strategy,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to send Discord close notification")

    def _submit_order(
        self, exchange_symbol: str, side: str, quantity: float
    ) -> Optional[dict]:
        order_details = {
            "symbol": exchange_symbol,
            "side": side,
            "quantity": quantity,
        }

        if self.dry_run:
            logger.info(
                "[DRY-RUN] Prepared order %s %.6f for %s",
                side,
                quantity,
                exchange_symbol,
            )
            return {
                "status": "DRY_RUN",
                "details": order_details,
            }

        logger.debug("Submitting order: %s", order_details)
        return self.exchange.create_order(**order_details)

    def _determine_quantity(self, last_price: float, strategy_decision: Optional[StrategyDecision] = None) -> float:
        logger.info("CONFIG CHECK - fixed_quantity: %s, max_investment: %s, strategy_decision: %s", 
                   self.fixed_quantity, self.max_investment_per_trade, strategy_decision is not None)
        
        # AI数量を絶対優先（fixed_quantityが0.0の場合）
        if self.fixed_quantity == 0.0 and strategy_decision and hasattr(strategy_decision, 'quantity'):
            ai_quantity = getattr(strategy_decision, 'quantity', None)
            logger.info("AI QUANTITY PRIORITY - ai_quantity: %s, ai_quantity > 0: %s", 
                       ai_quantity, ai_quantity and ai_quantity > 0)
            if ai_quantity and ai_quantity > 0:
                filtered_quantity = self._apply_lot_size_filter(strategy_decision.symbol if strategy_decision else "BTC/JPY", ai_quantity)
                logger.info("✅ USING AI QUANTITY: %s -> %s", ai_quantity, filtered_quantity)
                return filtered_quantity
        
        # ここには到達しないはず（AI数量優先のため）
        logger.warning("⚠️  FALLBACK - Using investment-based (AI quantity failed)")
        if self.max_investment_per_trade > 0 and last_price > 0:
            quantity = self.max_investment_per_trade / last_price
            logger.info("Using investment-based quantity: %s (JPY: %s / price: %s)", 
                       quantity, self.max_investment_per_trade, last_price)
            return self._apply_lot_size_filter("BTC/JPY", quantity)
        
        # 最終フォールバック
        logger.warning("Using default minimum quantity 0.001 - no valid quantity method found")
        return self._apply_lot_size_filter("BTC/JPY", 0.001)
    
    def _get_jpy_balance(self) -> Optional[float]:
        """JPY残高を取得."""
        try:
            return self.exchange.get_account_balance("JPY")
        except Exception as exc:
            logger.error("Error getting JPY balance: %s", exc)
            return None
    
    def _apply_lot_size_filter(self, symbol: str, quantity: float) -> float:
        """BinanceのLOT_SIZEフィルターを適用して数量を調整."""
        try:
            # シンボル情報を取得
            exchange_symbol = symbol.replace("/", "")
            symbol_info = None
            
            # 利用可能なシンボル情報を検索
            for s in self.exchange.client.futures_exchange_info()['symbols'] if hasattr(self.exchange.client, 'futures_exchange_info') else []:
                if s['symbol'] == exchange_symbol:
                    symbol_info = s
                    break
            
            # スポット取引のシンボル情報を取得
            if not symbol_info:
                for s in self.exchange.client.get_exchange_info()['symbols']:
                    if s['symbol'] == exchange_symbol:
                        symbol_info = s
                        break
            
            if symbol_info:
                # LOT_SIZEフィルターを取得
                for filter_item in symbol_info['filters']:
                    if filter_item['filterType'] == 'LOT_SIZE':
                        min_qty = float(filter_item['minQty'])
                        max_qty = float(filter_item['maxQty'])
                        step_size = float(filter_item['stepSize'])
                        
                        # ステップサイズに合わせて調整
                        adjusted_quantity = (quantity // step_size) * step_size
                        
                        # 最小/最大数量チェック
                        if adjusted_quantity < min_qty:
                            adjusted_quantity = min_qty
                        elif adjusted_quantity > max_qty:
                            adjusted_quantity = max_qty
                        
                        logger.debug("Applied LOT_SIZE filter for %s: %s -> %s (min: %s, max: %s, step: %s)", 
                                   symbol, quantity, adjusted_quantity, min_qty, max_qty, step_size)
                        return round(adjusted_quantity, 8)
            
            # フィルター情報が取得できない場合は丸めるのみ
            logger.warning("Could not get LOT_SIZE filter for %s, using rounded quantity", symbol)
            return round(quantity, 8)
            
        except Exception as exc:
            logger.warning("Error applying LOT_SIZE filter for %s: %s", symbol, exc)
            return round(quantity, 8)
    
    def _restore_open_positions(self) -> None:
        """Binanceからオープンポジションを検知してtrade_trackerに復元."""
        try:
            logger.info("Restoring open positions from exchange...")
            
            # スポットの資産状況を取得
            account = self.exchange.client.get_account()
            restored_count = 0
            
            for balance in account['balances']:
                asset = balance['asset']
                free_qty = float(balance['free'])
                
                # JPY以外で保有量が0より大きい場合、オープンポジションとみなす
                if asset != 'JPY' and free_qty > 0:
                    # 対応するシンボルを検索
                    symbol = None
                    for display_symbol in self.symbols_display:
                        if asset in display_symbol:
                            symbol = display_symbol
                            break
                    
                    if symbol:
                        # 現在価格を取得
                        try:
                            ticker = self.exchange.client.get_symbol_ticker(symbol=symbol.replace("/", ""))
                            current_price = float(ticker['price'])
                            
                            # trade_trackerにポジションを復元
                            from binance_auto_trader.models.trade import TradeRecord
                            from datetime import datetime
                            
                            trade_record = TradeRecord(
                                symbol=symbol,
                                strategy="restored",  # 復元されたポジション
                                action="BUY",  # スポットはBUYのみ
                                quantity=free_qty,
                                entry_price=current_price,  # 現在価格で仮設定
                                exit_price=None,
                                pnl_percent=None,
                                status="OPEN",
                                entry_time=datetime.utcnow(),
                                exit_time=None
                            )
                            
                            self.trade_tracker.open_trades[symbol] = trade_record
                            self.positions[symbol] = "LONG"
                            restored_count += 1
                            
                            logger.info("Restored position: %s qty=%s price=%s", 
                                       symbol, free_qty, current_price)
                            
                        except Exception as exc:
                            logger.warning("Could not get price for restored position %s: %s", symbol, exc)
            
            logger.info("Position restoration completed. Restored %d positions.", restored_count)
            
        except Exception as exc:
            logger.error("Error restoring open positions: %s", exc)

    # ------------------------------------------------------------------
    # External control (Discord commands / orchestration)
    # ------------------------------------------------------------------
    def pause_trading(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()
            logger.info("Trading bot paused.")

    def resume_trading(self) -> None:
        if not self._pause_event.is_set():
            self._pause_event.set()
            logger.info("Trading bot resumed.")

    def stop_trading(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        logger.info("Trading bot stop requested.")

    def close_all_positions(self) -> None:
        logger.info("Closing all open positions on request.")
        for display_symbol, position in list(self.positions.items()):
            if position is None:
                continue
            exchange_symbol = self.symbol_map.get(display_symbol, display_symbol)
            history = self.trade_tracker.price_history.get(display_symbol)
            price = 0.0
            if history:
                price = float(history[-1][1])
            if price <= 0:
                klines = self.exchange.get_historical_klines(
                    exchange_symbol, self.timeframe, limit=1
                )
                if klines:
                    price = float(klines[-1][4])
            if price <= 0:
                logger.warning(
                    "Skipping close for %s — unable to resolve last price.",
                    display_symbol,
                )
                continue
            self._close_position(display_symbol, exchange_symbol, position, price)


def main() -> None:
    config = Config()
    setup_logging(config)
    logger.info("Configuration loaded from %s", config.path)

    bot = TradingBot(config)

    backtester = Backtester(config, bot.ai_manager, bot.exchange)
    backtest_results = backtester.run()
    if backtest_results:
        bot.trade_tracker.set_backtest_results(backtest_results)
        logger.info("Loaded %s backtest result sets", len(backtest_results))

    start_dashboard(bot.trade_tracker, config)

    trading_thread = threading.Thread(target=bot.run, name="TradingLoop", daemon=True)
    trading_thread.start()

    try:
        from binance_auto_trader.discord_bot import start_discord_bot
    except ImportError:  # pragma: no cover - optional dependency
        start_discord_bot = None

    discord_runner = None
    if start_discord_bot is not None:
        discord_runner = start_discord_bot(config, bot)

    try:
        while trading_thread.is_alive():
            trading_thread.join(timeout=1)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
        bot.stop_trading()
    except Exception:  # noqa: BLE001
        logger.exception("Fatal error during execution")
        bot.stop_trading()

    if discord_runner is not None:
        discord_runner.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception:  # noqa: BLE001
        logger.exception("Fatal error during execution")
