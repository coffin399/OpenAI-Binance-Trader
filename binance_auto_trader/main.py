from __future__ import annotations

import logging
import math
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
from binance_auto_trader.utils.discord_notify import DiscordNotifier, WalletDiscordNotifier
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
        self.trade_tracker._bot_instance = self  # ウォレット情報取得用
        self.notifier = DiscordNotifier(config)
        self.wallet_notifier = WalletDiscordNotifier(config)  # ウォレット通知用

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
        
        # 保有資産活用設定
        asset_config = getattr(config, "asset_management", {})
        self.use_held_assets = getattr(asset_config, "use_held_assets", False)
        self.max_asset_utilization = getattr(asset_config, "max_asset_utilization", 0.8)
        self.prefer_profitable_assets = getattr(asset_config, "prefer_profitable_assets", True)
        self.direct_swap = getattr(asset_config, "direct_swap", False)
        self.swap_pairs = getattr(asset_config, "swap_pairs", [])
        self.min_swap_ratio = getattr(asset_config, "min_swap_ratio", 0.1)
        self.auto_convert_to_jpy = getattr(asset_config, "auto_convert_to_jpy", False)
        self.jpy_convert_threshold = getattr(asset_config, "jpy_convert_threshold", 0.5)
        self.max_convert_positions = getattr(asset_config, "max_convert_positions", 2)
        self.polling_interval = int(
            getattr(config.runtime, "polling_interval_seconds", 60)
        )
        self.dry_run = bool(getattr(config.runtime, "dry_run", True))

        strategies = getattr(config.strategies, "active", [])
        self.strategies = self._initialize_strategies(strategies)

        self.positions: Dict[str, Optional[str]] = {symbol: None for symbol in self.symbols_display}
        self._symbol_info_cache: Dict[str, Dict[str, object]] = {}
        
        # ウォレット通知用
        self._last_wallet_notification = time.time()
        self._last_wallet_value = 0.0  # 前回のウォレット価値
        self._initial_wallet_value = 0.0  # 初期ウォレット価値
        self._wallet_notification_interval = 3600  # 1時間（秒）

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
        # デバッグ：定期的に残高確認（1分に1回程度）
        import time
        current_time = time.time()
        if not hasattr(self, '_last_balance_check'):
            self._last_balance_check = 0
        
        if current_time - self._last_balance_check > 60:  # 60秒ごと
            self._debug_account_balances()
            self._last_balance_check = current_time
        
        # ウォレット通知（1時間ごと）
        self._notify_wallet_summary()
        
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
                # 保有資産を活用する場合
                if self.use_held_assets:
                    # 資産間直接交換を試みる
                    if self.direct_swap and self._can_direct_swap(exchange_symbol):
                        logger.info("Direct swap available for %s", display_symbol)
                        return True
                    
                    # 従来通りJPYに変換する方法
                    available_jpy = self._get_available_jpy_from_assets()
                    total_available = jpy_balance + available_jpy
                    if total_available >= 500:
                        logger.info("Using held assets. Available: %.0f JPY (cash: %.0f + assets: %.0f)", 
                                   total_available, jpy_balance, available_jpy)
                        
                        # 実際に資産を売却してJPYを確保
                        needed_jpy = 500 - jpy_balance
                        if needed_jpy > 0 and not self.dry_run:
                            if self._sell_asset_for_jpy(needed_jpy):
                                logger.info("Successfully sold assets to secure %.0f JPY", needed_jpy)
                            else:
                                logger.warning("Failed to sell assets, proceeding with available balance")
                    else:
                        logger.info("Insufficient total funds (%.0f). Need at least 500 JPY.", total_available)
                        return False
                else:
                    logger.info(
                        "Insufficient JPY balance (%.0f). Need at least 500 JPY to start trading.",
                        jpy_balance
                    )
                    return False
            else:
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

        quantity = self._determine_quantity(price, decision, display_symbol)
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

        # 直接交換を試みる
        order = None
        if self.direct_swap and side == "BUY":
            jpy_balance = self._get_jpy_balance()
            if jpy_balance and jpy_balance < 500:  # JPY不足時のみ直接交換
                logger.info("Attempting direct swap for %s (JPY balance: %.0f)", display_symbol, jpy_balance)
                actual_quantity = self._execute_direct_swap(exchange_symbol, quantity)
                if actual_quantity:
                    # 直接交換成功の場合、ポジションを記録
                    self.positions[display_symbol] = position_label
                    
                    # 正しい戦略情報を取得
                    strategy_name = getattr(decision, 'strategy_name', 'ai_hybrid')
                    confidence = getattr(decision, 'confidence', 0.5)
                    reasoning = getattr(decision, 'reasoning', 'Direct swap executed')
                    
                    # デバッグ情報
                    logger.info("Recording direct swap position: symbol=%s, strategy=%s, action=%s, qty=%s, price=%s", 
                               display_symbol, strategy_name, side, actual_quantity, price)
                    
                    self.trade_tracker.record_open_trade(
                        symbol=display_symbol,
                        strategy=strategy_name,
                        action=side,
                        quantity=actual_quantity,  # 実際の購入数量を使用
                        price=price,
                    )
                    
                    # 確認ログ
                    if display_symbol in self.trade_tracker.open_trades:
                        record = self.trade_tracker.open_trades[display_symbol]
                        logger.info("✅ Position recorded in tracker: %s qty=%s price=%s", 
                                   record.symbol, record.quantity, record.entry_price)
                    else:
                        logger.warning("❌ Position NOT found in tracker after recording")
                    
                    logger.info("Position opened via direct swap: %s %s at %.4f (qty: %s, strategy: %s)", 
                               side, display_symbol, price, actual_quantity, strategy_name)
                    return
        
        # 通常のオーダー実行
        if not order:
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

    def _determine_quantity(self, last_price: float, strategy_decision: Optional[StrategyDecision] = None, symbol: str = "BTC/JPY") -> float:
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
            jpy_balance = self.exchange.get_account_balance("JPY")
            logger.info("Current JPY balance: %s", jpy_balance)
            return jpy_balance
        except Exception as exc:
            logger.error("Error getting JPY balance: %s", exc)
            return None
    
    def _debug_account_balances(self) -> None:
        """デバッグ用：全資産残高を表示."""
        try:
            account = self.exchange.client.get_account()
            logger.info("=== Account Balances ===")
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                if free > 0 or locked > 0:
                    logger.info("Asset: %s | Free: %s | Locked: %s", asset, free, locked)
            
            logger.info("========================")
            
        except Exception as exc:
            logger.error("Error getting account balances: %s", exc)
    
    def _notify_wallet_summary(self) -> None:
        """1時間ごとにウォレットサマリーを通知."""
        try:
            current_time = time.time()
            
            # 通知間隔をチェック
            if current_time - self._last_wallet_notification < self._wallet_notification_interval:
                return
            
            # ウォレット情報を取得
            wallet_summary = self.trade_tracker._get_wallet_summary()
            current_total = wallet_summary.get("total_jpy_value", 0.0)
            assets = wallet_summary.get("assets", [])
            
            # 初回起動時の値を記録
            if self._initial_wallet_value == 0.0:
                self._initial_wallet_value = current_total
                self._last_wallet_value = current_total
                self._last_wallet_notification = current_time
                
                # 初回通知
                self.wallet_notifier.notify_wallet_summary(
                    total_jpy=current_total,
                    assets=assets,
                    hourly_change=0.0,
                    total_change=0.0
                )
                logger.info("Initial wallet summary sent: %.0f JPY", current_total)
                return
            
            # 変化額を計算
            hourly_change = current_total - self._last_wallet_value
            total_change = current_total - self._initial_wallet_value
            
            # ウォレット通知を送信
            self.wallet_notifier.notify_wallet_summary(
                total_jpy=current_total,
                assets=assets,
                hourly_change=hourly_change,
                total_change=total_change
            )
            
            # 状態を更新
            self._last_wallet_value = current_total
            self._last_wallet_notification = current_time
            
            logger.info("Wallet summary sent: %.0f JPY (1H: %+.0f, Total: %+.0f)", 
                       current_total, hourly_change, total_change)
            
        except Exception as exc:
            logger.error("Error sending wallet summary: %s", exc)
    
    def _get_available_jpy_from_assets(self) -> float:
        """保有資産を売却して得られるJPY額を計算."""
        try:
            account = self.exchange.client.get_account()
            total_jpy_value = 0.0
            
            for balance in account['balances']:
                asset = balance['asset']
                free_qty = float(balance['free'])
                
                if asset != 'JPY' and free_qty > 0:
                    # 現在価格を取得
                    try:
                        # 対応するシンボルを検索
                        symbol = None
                        for display_symbol in self.symbols_display:
                            if asset in display_symbol:
                                symbol = display_symbol.replace("/", "")
                                break
                        
                        if symbol:
                            ticker = self.exchange.client.get_symbol_ticker(symbol=symbol)
                            current_price = float(ticker['price'])
                            jpy_value = free_qty * current_price
                            
                            # 利用率を適用
                            usable_jpy = jpy_value * self.max_asset_utilization
                            total_jpy_value += usable_jpy
                            
                            logger.debug("Asset %s: qty=%s price=%s JPY_value=%.0f usable=%.0f", 
                                       asset, free_qty, current_price, jpy_value, usable_jpy)
                    
                    except Exception as exc:
                        logger.warning("Could not get price for asset %s: %s", asset, exc)
            
            logger.info("Total available JPY from assets: %.0f", total_jpy_value)
            return total_jpy_value
            
        except Exception as exc:
            logger.error("Error calculating available JPY from assets: %s", exc)
            return 0.0
    
    def _sell_asset_for_jpy(self, target_jpy_amount: float) -> bool:
        """保有資産を売却してJPYを確保."""
        try:
            logger.info("Selling assets to secure %.0f JPY", target_jpy_amount)
            
            account = self.exchange.client.get_account()
            secured_jpy = 0.0
            
            # 資産をJPY価値でソート（大きいものを優先）
            assets_to_sell = []
            
            for balance in account['balances']:
                asset = balance['asset']
                free_qty = float(balance['free'])
                
                if asset != 'JPY' and free_qty > 0:
                    # 対応するシンボルを検索
                    symbol = None
                    for display_symbol in self.symbols_display:
                        if asset in display_symbol:
                            symbol = display_symbol.replace("/", "")
                            break
                    
                    if symbol:
                        try:
                            ticker = self.exchange.client.get_symbol_ticker(symbol=symbol)
                            current_price = float(ticker['price'])
                            jpy_value = free_qty * current_price
                            
                            assets_to_sell.append({
                                'asset': asset,
                                'symbol': symbol,
                                'quantity': free_qty,
                                'jpy_value': jpy_value
                            })
                            
                        except Exception as exc:
                            logger.warning("Could not evaluate asset %s: %s", asset, exc)
            
            # JPY価値の大きい順にソート
            assets_to_sell.sort(key=lambda x: x['jpy_value'], reverse=True)
            
            # 必要なJPY額になるまで資産を売却
            for asset_info in assets_to_sell:
                if secured_jpy >= target_jpy_amount:
                    break
                
                sell_quantity = asset_info['quantity']
                symbol = asset_info['symbol']
                
                # 売却実行
                order = self.exchange.create_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=sell_quantity
                )
                
                if order:
                    secured_jpy += asset_info['jpy_value'] * self.max_asset_utilization
                    logger.info("Sold %s %s for %.0f JPY", 
                               asset_info['asset'], sell_quantity, 
                               asset_info['jpy_value'] * self.max_asset_utilization)
                    
                    # trade_trackerからポジションを削除
                    for display_symbol in self.symbols_display:
                        if asset_info['asset'] in display_symbol:
                            if display_symbol in self.trade_tracker.open_trades:
                                del self.trade_tracker.open_trades[display_symbol]
                                self.positions[display_symbol] = None
                            break
            
            logger.info("Asset sale completed. Secured %.0f JPY", secured_jpy)
            return secured_jpy >= target_jpy_amount
            
        except Exception as exc:
            logger.error("Error selling assets for JPY: %s", exc)
            return False
    
    def _can_direct_swap(self, target_symbol: str) -> bool:
        """指定されたシンボルに直接交換できるか判定."""
        try:
            target_asset = target_symbol.replace("JPY", "")
            account = self.exchange.client.get_account()

            for balance in account["balances"]:
                asset = balance["asset"]
                free_qty = float(balance["free"])

                if asset != "JPY" and free_qty > 0:
                    swap_pair = f"{asset}/{target_asset}"
                    reverse_pair = f"{target_asset}/{asset}"

                    if swap_pair in self.swap_pairs or reverse_pair in self.swap_pairs:
                        sell_symbol = f"{asset}JPY"
                        buy_symbol = target_symbol

                        if not self._symbol_exists(sell_symbol):
                            logger.debug(
                                "Direct swap skipped: sell symbol %s not available on exchange",
                                sell_symbol,
                            )
                            continue

                        if not self._symbol_exists(buy_symbol):
                            logger.debug(
                                "Direct swap skipped: buy symbol %s not available on exchange",
                                buy_symbol,
                            )
                            continue

                        logger.info(
                            "Found direct swap pair: %s (asset: %s, qty: %s)",
                            swap_pair,
                            asset,
                            free_qty,
                        )
                        return True

            return False

        except Exception as exc:
            logger.error("Error checking direct swap availability: %s", exc)
            return False
    
    def _execute_direct_swap(self, target_symbol: str, target_quantity: float) -> Optional[float]:
        """資産間直接交換を実行. 成功した場合は実際の購入数量を返す."""
        try:
            # ターゲットの資産名を取得
            target_asset = target_symbol.replace("JPY", "")
            
            # 保有資産を取得
            account = self.exchange.client.get_account()
            
            for balance in account['balances']:
                asset = balance['asset']
                free_qty = float(balance['free'])
                
                if asset != 'JPY' and free_qty > 0:
                    # 交換ペアをチェック
                    swap_pair = f"{asset}/{target_asset}"
                    reverse_pair = f"{target_asset}/{asset}"
                    
                    if swap_pair in self.swap_pairs or reverse_pair in self.swap_pairs:
                        # 交換する数量を計算
                        if swap_pair in self.swap_pairs:
                            # asset -> target_asset の直接売却
                            sell_symbol = f"{asset}JPY"
                            buy_symbol = target_symbol
                            sell_quantity = free_qty * self.max_asset_utilization
                        else:
                            # target_asset -> asset の場合は一旦JPY経由
                            continue
                        
                        # LOT_SIZEフィルターを適用
                        sell_quantity = self._apply_lot_size_filter(sell_symbol, sell_quantity)

                        # NOTIONALフィルター対応：最小取引金額チェック
                        min_notional = self._get_min_notional(sell_symbol)
                        estimated_jpy_value = self._estimate_jpy_value(sell_symbol, sell_quantity)

                        if estimated_jpy_value is None:
                            logger.warning(
                                "Could not estimate notional for %s, skipping direct swap",
                                sell_symbol,
                            )
                            continue

                        if estimated_jpy_value < min_notional:
                            logger.warning(
                                "Direct swap amount too small: %.0f JPY < %.0f JPY (NOTIONAL filter)",
                                estimated_jpy_value,
                                min_notional,
                            )
                            continue

                        # 売却実行
                        logger.info(
                            "Executing direct swap: %s -> %s (qty: %s, value: %.0f JPY)",
                            asset,
                            target_asset,
                            sell_quantity,
                            estimated_jpy_value,
                        )
                        
                        # まずassetをJPYに売却
                        sell_order = self.exchange.create_order(
                            symbol=sell_symbol,
                            side="SELL",
                            quantity=sell_quantity
                        )
                        
                        if sell_order:
                            # 購入数量にもLOT_SIZEフィルターを適用
                            buy_quantity = self._apply_lot_size_filter(buy_symbol, target_quantity)
                            
                            # すぐにtarget_assetを購入
                            buy_order = self.exchange.create_order(
                                symbol=buy_symbol,
                                side="BUY",
                                quantity=buy_quantity
                            )
                            
                            if buy_order:
                                # trade_trackerを更新
                                # 旧ポジションを削除
                                for display_symbol in self.symbols_display:
                                    if asset in display_symbol:
                                        if display_symbol in self.trade_tracker.open_trades:
                                            del self.trade_tracker.open_trades[display_symbol]
                                            self.positions[display_symbol] = None
                                        break
                                
                                logger.info("Direct swap completed: %s -> %s (sold: %s, bought: %s)", 
                                           asset, target_asset, sell_quantity, buy_quantity)
                                return buy_quantity  # 実際の購入数量を返す
                            else:
                                logger.warning("Buy order failed in direct swap")
                        else:
                            logger.warning("Sell order failed in direct swap")
            
            return False
            
        except Exception as exc:
            logger.error("Error executing direct swap: %s", exc)
            return False
    
    def _auto_convert_profitable_to_jpy(self) -> None:
        """利益の出ているポジションを自動でJPYに変換."""
        if not self.auto_convert_to_jpy or self.dry_run:
            return
        
        try:
            logger.info("Checking for profitable positions to convert to JPY...")
            
            profitable_positions = []
            current_prices = {}
            
            # 現在価格を取得
            for symbol in self.symbols_display:
                try:
                    ticker = self.exchange.client.get_symbol_ticker(symbol=symbol.replace("/", ""))
                    current_prices[symbol] = float(ticker['price'])
                except Exception as exc:
                    logger.warning("Could not get price for %s: %s", symbol, exc)
                    continue
            
            # 利益ポジションを特定
            for symbol, trade_record in self.trade_tracker.open_trades.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                entry_price = trade_record.entry_price
                
                if current_price > 0 and entry_price > 0:
                    profit_ratio = (current_price - entry_price) / entry_price
                    
                    if profit_ratio >= self.jpy_convert_threshold:
                        profitable_positions.append({
                            'symbol': symbol,
                            'trade_record': trade_record,
                            'profit_ratio': profit_ratio,
                            'current_price': current_price
                        })
            
            # 利益率の高い順にソート
            profitable_positions.sort(key=lambda x: x['profit_ratio'], reverse=True)
            
            # 最大変換数まで実行
            converted_count = 0
            for pos in profitable_positions[:self.max_convert_positions]:
                symbol = pos['symbol']
                trade_record = pos['trade_record']
                current_price = pos['current_price']
                
                logger.info("Converting profitable position to JPY: %s (profit: %.1f%%)", 
                           symbol, pos['profit_ratio'] * 100)
                
                # 売却実行
                sell_quantity = trade_record.quantity
                sell_quantity = self._apply_lot_size_filter(symbol.replace("/", ""), sell_quantity)
                
                order = self.exchange.create_order(
                    symbol=symbol.replace("/", ""),
                    side="SELL",
                    quantity=sell_quantity
                )
                
                if order:
                    # ポジションをクローズ
                    self.trade_tracker.record_close_trade(
                        symbol=symbol,
                        exit_price=current_price,
                        close_time=datetime.utcnow()
                    )
                    
                    self.positions[symbol] = None
                    
                    # 利益を計算
                    profit_jpy = sell_quantity * (current_price - trade_record.entry_price)
                    
                    logger.info("✅ Converted %s to JPY: profit %.0f JPY (%.1f%%)", 
                               symbol, profit_jpy, pos['profit_ratio'] * 100)
                    
                    converted_count += 1
                else:
                    logger.warning("Failed to convert %s to JPY", symbol)
            
            if converted_count > 0:
                logger.info("Auto-conversion completed: %d positions converted to JPY", converted_count)
            else:
                logger.info("No profitable positions met conversion threshold (%.1f%%)", 
                           self.jpy_convert_threshold * 100)
                
        except Exception as exc:
            logger.error("Error in auto JPY conversion: %s", exc)
    
    def _apply_lot_size_filter(self, symbol: str, quantity: float) -> float:
        """BinanceのLOT_SIZEフィルターを適用して数量を調整."""
        try:
            exchange_symbol = symbol.replace("/", "")
            symbol_info = self._get_symbol_info(exchange_symbol)

            if symbol_info:
                lot_filter = self._get_filter(symbol_info, "LOT_SIZE")
                if lot_filter:
                    min_qty = float(lot_filter["minQty"])
                    max_qty = float(lot_filter["maxQty"])
                    step_size = float(lot_filter["stepSize"])

                    if step_size <= 0:
                        logger.warning("Invalid step size for %s: %s", symbol, step_size)
                        return round(quantity, 8)

                    steps = math.floor(quantity / step_size)
                    adjusted_quantity = steps * step_size

                    if adjusted_quantity < min_qty:
                        adjusted_quantity = min_qty
                    elif adjusted_quantity > max_qty:
                        adjusted_quantity = max_qty

                    logger.debug(
                        "Applied LOT_SIZE filter for %s: %s -> %s (min: %s, max: %s, step: %s)",
                        symbol,
                        quantity,
                        adjusted_quantity,
                        min_qty,
                        max_qty,
                        step_size,
                    )
                    return round(adjusted_quantity, 8)

            logger.warning("Could not get LOT_SIZE filter for %s, using rounded quantity", symbol)
            return round(quantity, 8)

        except Exception as exc:
            logger.warning("Error applying LOT_SIZE filter for %s: %s", symbol, exc)
            return round(quantity, 8)

    def _get_symbol_info(self, exchange_symbol: str) -> Optional[Dict[str, object]]:
        """Binanceのシンボル情報をキャッシュ付きで取得."""
        if exchange_symbol in self._symbol_info_cache:
            return self._symbol_info_cache[exchange_symbol]

        try:
            exchange_info = self.exchange.client.get_exchange_info()
            for info in exchange_info.get("symbols", []):
                self._symbol_info_cache[info["symbol"]] = info
            return self._symbol_info_cache.get(exchange_symbol)
        except Exception as exc:
            logger.warning("Could not fetch exchange info for %s: %s", exchange_symbol, exc)
            return None

    @staticmethod
    def _get_filter(symbol_info: Dict[str, object], filter_type: str) -> Optional[Dict[str, object]]:
        for filter_item in symbol_info.get("filters", []):
            if filter_item.get("filterType") == filter_type:
                return filter_item
        return None

    def _get_min_notional(self, symbol: str) -> float:
        """MIN_NOTIONALフィルタから最小取引金額を取得."""
        exchange_symbol = symbol.replace("/", "")
        symbol_info = self._get_symbol_info(exchange_symbol)
        if not symbol_info:
            return 1000.0

        notional_filter = self._get_filter(symbol_info, "MIN_NOTIONAL")
        if notional_filter:
            try:
                return float(notional_filter.get("minNotional", 1000.0))
            except (TypeError, ValueError):
                pass

        return 1000.0

    def _estimate_jpy_value(self, symbol: str, quantity: float) -> Optional[float]:
        """数量をJPY換算した概算値を計算."""
        exchange_symbol = symbol.replace("/", "")
        try:
            ticker = self.exchange.client.get_symbol_ticker(symbol=exchange_symbol)
            price = float(ticker["price"])
            return quantity * price
        except Exception as exc:
            logger.warning("Could not estimate JPY value for %s: %s", symbol, exc)
            return None
    
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
                                opened_at=datetime.utcnow(),
                                closed_at=None
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
