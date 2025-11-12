from __future__ import annotations

import json
import logging
from statistics import mean
from typing import Dict, Optional, List

import pandas as pd

from binance_auto_trader.models.trade import StrategyDecision

from .base import Strategy

logger = logging.getLogger(__name__)


class Coffin299Strategy(Strategy):
    """Coffin299's AI-powered trading strategy with enhanced technical analysis.
    
    Features:
    - Dynamic confidence calculation based on multiple technical indicators
    - Signal reinforcement using RSI, momentum, and volume trends
    - Enhanced AI response parsing (supports JSON format)
    - Trend detection and volatility-based risk management
    - Aggressive trading mode for maximum opportunities
    """

    supports_backtesting = False

    def __init__(self, ai_manager, config_section) -> None:
        super().__init__()
        self.ai_manager = ai_manager
        self.provider_name = getattr(config_section, "provider", None)
        self.prompt_template = getattr(config_section, "prompt_template", "")
        
        # Enhanced configuration for aggressive trading
        self.min_confidence = getattr(config_section, "min_confidence", 0.35)
        self.aggressive_mode = getattr(config_section, "aggressive_mode", True)
        self.use_technical_signals = getattr(config_section, "use_technical_signals", True)
        self.dynamic_threshold = getattr(config_section, "dynamic_threshold", True)
        self.opportunity_detection = getattr(config_section, "opportunity_detection", True)
        
        # Position management for better sell decisions
        self.track_positions = getattr(config_section, "track_positions", True)
        self.take_profit_pct = getattr(config_section, "take_profit_pct", 2.0)  # 2%利益確定
        self.stop_loss_pct = getattr(config_section, "stop_loss_pct", 1.5)  # 1.5%損切り
        
        if not self.provider_name:
            raise ValueError("Coffin299Strategy requires 'provider' setting")
        if not self.ai_manager or not self.ai_manager.has_provider(self.provider_name):
            raise ValueError(f"AI provider '{self.provider_name}' not available")
        self._last_timestamp: Dict[str, pd.Timestamp] = {}
        self._positions: Dict[str, Dict] = {}  # {symbol: {"entry_price": float, "quantity": float}}

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

        # Parse AI response (supports both simple text and JSON format)
        ai_action, ai_confidence, ai_reasoning = self._parse_ai_response(response)
        
        if ai_action not in {"BUY", "SELL", "HOLD"}:
            logger.warning("AI provider returned unsupported action '%s'", ai_action)
            return None

        # Calculate dynamic confidence based on technical indicators
        technical_signals = self._calculate_technical_signals(df, prompt_context)
        final_confidence = self._calculate_confidence(
            ai_action, ai_confidence, technical_signals, prompt_context
        )
        
        # Dynamic confidence threshold based on market conditions
        effective_threshold = self._get_dynamic_threshold(technical_signals, prompt_context)
        
        # Detect high-probability opportunities
        is_strong_opportunity = self._detect_opportunity(ai_action, technical_signals, prompt_context)
        
        # Apply minimum confidence threshold (relaxed for strong opportunities)
        if is_strong_opportunity:
            effective_threshold *= 0.7  # Lower threshold for strong opportunities
            logger.debug("%s: Strong opportunity detected, threshold: %.2f", symbol, effective_threshold)
        
        if final_confidence < effective_threshold:
            logger.info(
                "%s: Confidence %.2f below threshold %.2f, skipping trade",
                symbol, final_confidence, effective_threshold
            )
            return None

        # Get current price early for position exit checks
        price = float(df.iloc[-1]["close"])
        
        # Check if technical signals reinforce AI decision (relaxed in aggressive mode)
        if self.use_technical_signals and not self.aggressive_mode:
            signal_agreement = self._check_signal_agreement(ai_action, technical_signals)
            if not signal_agreement:
                logger.info(
                    "%s: Technical signals do not support %s, skipping (conservative mode)",
                    symbol, ai_action
                )
                return None
        
        # Check position-based sell signals (profit taking / stop loss)
        position_action = self._check_position_exit(symbol, price, technical_signals)
        if position_action:
            ai_action = position_action
            final_confidence = max(final_confidence, 0.7)  # High confidence for position exits
            logger.info("%s: Position exit signal: %s", symbol, position_action)
        
        # In aggressive mode, consider HOLD as potential trade opportunity
        if ai_action == "HOLD":
            if self.aggressive_mode and is_strong_opportunity:
                # Override HOLD with technical signal direction (more balanced)
                if technical_signals["combined_signal"] > 0.4:
                    ai_action = "BUY"
                    logger.info("%s: Overriding HOLD -> BUY (strong bullish signals)", symbol)
                elif technical_signals["combined_signal"] < -0.4:
                    ai_action = "SELL"
                    logger.info("%s: Overriding HOLD -> SELL (strong bearish signals)", symbol)
                else:
                    return None
            else:
                return None
        
        # Update position tracking
        if ai_action == "BUY" and self.track_positions:
            # Will be updated after trade execution
            pass
        elif ai_action == "SELL" and self.track_positions and symbol in self._positions:
            entry_price = self._positions[symbol]["entry_price"]
            profit_pct = ((price - entry_price) / entry_price) * 100
            logger.info("%s: Selling position - Entry: %.4f, Current: %.4f, P/L: %+.2f%%",
                       symbol, entry_price, price, profit_pct)
        
        info_parts = [f"AI: {ai_action} (conf: {ai_confidence:.2f})", f"Final: {final_confidence:.2f}"]
        if ai_reasoning:
            info_parts.append(f"Reason: {ai_reasoning[:50]}")
        info_parts.append(f"RSI: {technical_signals['rsi']:.1f}")
        info_parts.append(f"Trend: {technical_signals['trend']}")
        info_parts.append(f"Signal: {technical_signals['combined_signal']:+.2f}")
        
        # AIに数量を決定させる（aggressive_modeの場合）
        ai_quantity = None
        if self.aggressive_mode and ai_action != "HOLD":
            logger.debug("Calculating AI quantity for %s action %s", symbol, ai_action)
            ai_quantity = self._calculate_ai_quantity(price, technical_signals, prompt_context)
            logger.info("AI calculated quantity: %s for %s", ai_quantity, symbol)
            info_parts.append(f"Qty: {ai_quantity:.6f}")
        else:
            logger.debug("Skipping AI quantity calculation - aggressive_mode: %s, action: %s", 
                        self.aggressive_mode, ai_action)
        
        return StrategyDecision(
            symbol=symbol,
            strategy=self.name,
            action=ai_action,
            price=price,
            confidence=final_confidence,
            info=" | ".join(info_parts),
            quantity=ai_quantity,
        )

    def _build_prompt_context(self, df: pd.DataFrame, symbol: str) -> Dict[str, object]:
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        timeframe = self._infer_timeframe(df)

        recent = df.tail(6)
        extended = df.tail(24)
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
            ((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) > 1 and closes[0] else 0.0
        )

        fast_sma = float(df["close"].rolling(window=5).mean().iloc[-1])
        slow_sma = float(df["close"].rolling(window=20).mean().iloc[-1])
        avg_volume = mean(row["volume"] for row in recent_rows)
        latest_volume = float(latest["volume"])
        volume_trend = (latest_volume / avg_volume) if avg_volume else 1.0
        momentum = fast_sma - slow_sma

        range_high = float(extended["high"].max()) if not extended.empty else float(latest["high"])
        range_low = float(extended["low"].min()) if not extended.empty else float(latest["low"])
        volatility_pct = (
            ((range_high - range_low) / range_low * 100) if range_low else 0.0
        )

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0)).abs()
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        latest_rsi = rsi_series.iloc[-1] if len(rsi_series) else None
        if pd.isna(latest_rsi):
            latest_rsi = 50.0

        recent_changes: List[float] = []
        closes_series = df["close"].tail(5).tolist()
        for idx in range(1, len(closes_series)):
            prev_close = closes_series[idx - 1]
            current_close = closes_series[idx]
            if prev_close:
                recent_changes.append((current_close - prev_close) / prev_close * 100)
        recent_close_changes = ", ".join(f"{value:+.2f}%" for value in recent_changes[-3:]) or "N/A"

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
            "volume_trend": volume_trend,
            "range_high": range_high,
            "range_low": range_low,
            "volatility_pct": volatility_pct,
            "rsi": float(latest_rsi) if latest_rsi is not None else 50.0,
            "momentum": momentum,
            "recent_close_changes": recent_close_changes,
        }

    def _parse_ai_response(self, response: str) -> tuple[str, float, str]:
        """Parse AI response, supporting both simple text and JSON format.
        
        Returns:
            tuple: (action, confidence, reasoning)
        """
        response = response.strip()
        
        # Try to parse as JSON first
        try:
            data = json.loads(response)
            action = data.get("action", "").upper()
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            return action, confidence, reasoning
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Fallback to simple text parsing
        action = response.upper().split()[0] if response else "HOLD"
        confidence = 0.5
        reasoning = ""
        
        # Try to extract confidence from text (e.g., "BUY 0.8" or "BUY (80%)")
        parts = response.split()
        if len(parts) > 1:
            try:
                conf_str = parts[1].strip("()%")
                confidence = float(conf_str)
                if confidence > 1.0:
                    confidence /= 100.0
            except ValueError:
                pass
        
        return action, confidence, reasoning
    
    def _calculate_technical_signals(self, df: pd.DataFrame, context: Dict) -> Dict[str, float]:
        """Calculate technical trading signals.
        
        Returns:
            dict: Technical indicators and signals
        """
        rsi = context["rsi"]
        momentum = context["momentum"]
        volume_trend = context["volume_trend"]
        volatility = context["volatility_pct"]
        fast_sma = context["fast_sma"]
        slow_sma = context["slow_sma"]
        latest_price = context["latest_price"]
        
        # RSI signals (oversold/overbought) - Optimized thresholds for balanced trading
        rsi_signal = 0.0
        if rsi < 30:
            rsi_signal = 1.0  # Strong buy signal (oversold)
        elif rsi < 40:
            rsi_signal = 0.5  # Moderate buy signal
        elif rsi > 70:
            rsi_signal = -1.0  # Strong sell signal (overbought)
        elif rsi > 60:
            rsi_signal = -0.5  # Moderate sell signal
        elif rsi > 50:
            rsi_signal = -0.2  # Weak sell bias
        
        # Momentum/Trend signals
        momentum_signal = 0.0
        if momentum > 0:
            momentum_signal = min(momentum / latest_price * 100, 1.0)  # Bullish
        else:
            momentum_signal = max(momentum / latest_price * 100, -1.0)  # Bearish
        
        # Volume trend signal
        volume_signal = 0.0
        if volume_trend > 1.5:
            volume_signal = 0.5  # High volume confirms trend
        elif volume_trend < 0.7:
            volume_signal = -0.3  # Low volume weakens signal
        
        # Trend detection
        trend = "neutral"
        if fast_sma > slow_sma * 1.01:
            trend = "bullish"
        elif fast_sma < slow_sma * 0.99:
            trend = "bearish"
        
        # Combined signal strength (-1 to 1)
        combined_signal = (rsi_signal * 0.4 + momentum_signal * 0.4 + volume_signal * 0.2)
        
        return {
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "momentum_signal": momentum_signal,
            "volume_signal": volume_signal,
            "combined_signal": combined_signal,
            "trend": trend,
            "volatility": volatility,
        }
    
    def _calculate_confidence(
        self,
        action: str,
        ai_confidence: float,
        technical_signals: Dict,
        context: Dict
    ) -> float:
        """Calculate final confidence score based on AI and technical analysis.
        
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        combined_signal = technical_signals["combined_signal"]
        trend = technical_signals["trend"]
        volatility = technical_signals["volatility"]
        
        # Start with AI confidence
        confidence = ai_confidence
        
        # Adjust based on technical signal agreement
        if action == "BUY":
            if combined_signal > 0.3:
                confidence += 0.15  # Strong technical support
            elif combined_signal > 0:
                confidence += 0.08  # Moderate technical support
            elif combined_signal < -0.3:
                confidence -= 0.2  # Technical contradiction
            
            # Trend alignment bonus
            if trend == "bullish":
                confidence += 0.1
            elif trend == "bearish":
                confidence -= 0.1
                
        elif action == "SELL":
            if combined_signal < -0.3:
                confidence += 0.15  # Strong technical support
            elif combined_signal < 0:
                confidence += 0.08  # Moderate technical support
            elif combined_signal > 0.3:
                confidence -= 0.2  # Technical contradiction
            
            # Trend alignment bonus
            if trend == "bearish":
                confidence += 0.1
            elif trend == "bullish":
                confidence -= 0.1
        
        # Volatility adjustment - Use volatility as opportunity indicator
        # High volatility = more trading opportunities in aggressive mode
        if volatility > 8.0:
            confidence += 0.05  # High volatility = more opportunities
        elif volatility > 5.0:
            confidence += 0.03  # Moderate volatility = some opportunities
        elif volatility < 2.0:
            confidence -= 0.05  # Low volatility = fewer opportunities
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))
    
    def _check_signal_agreement(
        self,
        action: str,
        technical_signals: Dict
    ) -> bool:
        """Check if technical signals agree with AI decision (relaxed thresholds).
        
        Returns:
            bool: True if signals agree or are neutral
        """
        combined_signal = technical_signals["combined_signal"]
        
        if action == "BUY":
            return combined_signal > -0.5  # Very relaxed - allow weak contrary signals
        elif action == "SELL":
            return combined_signal < 0.5  # Very relaxed - allow weak contrary signals
        else:
            return True  # HOLD is always acceptable
    
    def _get_dynamic_threshold(self, technical_signals: Dict, context: Dict) -> float:
        """Calculate dynamic confidence threshold based on market conditions.
        
        Returns:
            float: Adjusted confidence threshold
        """
        if not self.dynamic_threshold:
            return self.min_confidence
        
        threshold = self.min_confidence
        trend = technical_signals["trend"]
        volatility = technical_signals["volatility"]
        combined_signal = technical_signals["combined_signal"]
        
        # Lower threshold in strong trending markets
        if trend in ["bullish", "bearish"] and abs(combined_signal) > 0.4:
            threshold *= 0.85  # 15% lower threshold in strong trends
        
        # Lower threshold in high volatility (more opportunities)
        if volatility > 6.0:
            threshold *= 0.9  # 10% lower threshold in volatile markets
        
        return max(0.25, threshold)  # Never go below 0.25
    
    def _detect_opportunity(self, action: str, technical_signals: Dict, context: Dict) -> bool:
        """Detect strong trading opportunities.
        
        Returns:
            bool: True if strong opportunity detected
        """
        if not self.opportunity_detection:
            return False
        
        rsi = technical_signals["rsi"]
        combined_signal = technical_signals["combined_signal"]
        trend = technical_signals["trend"]
        volume_signal = technical_signals["volume_signal"]
        momentum_signal = technical_signals["momentum_signal"]
        
        # Strong buy opportunities
        if action == "BUY":
            # Oversold with volume confirmation
            if rsi < 35 and volume_signal > 0.3:
                return True
            # Strong bullish momentum with trend alignment
            if momentum_signal > 0.6 and trend == "bullish":
                return True
            # Combined strong signal
            if combined_signal > 0.6:
                return True
        
        # Strong sell opportunities (strengthened)
        elif action == "SELL":
            # Overbought with volume confirmation
            if rsi > 65 and volume_signal > 0.3:
                return True
            # Strong bearish momentum with trend alignment
            if momentum_signal < -0.6 and trend == "bearish":
                return True
            # Combined strong signal
            if combined_signal < -0.6:
                return True
        
        return False
    
    def _calculate_ai_quantity(self, price: float, technical_signals: Dict, context: Dict) -> float:
        """AIが市場状況に基づいて最適な数量を計算（4000円少額スタート対応）.
        
        Returns:
            float: 計算された数量
        """
        # 基本数量（4000円少額スタート向けに調整）
        symbol = context["symbol"]
        if "BTC" in symbol:
            base_quantity = 0.0001  # BTC: 約1000円相当（0.0001 BTC）
        elif "ETH" in symbol:
            base_quantity = 0.001   # ETH: 約300円相当（0.001 ETH）
        else:
            base_quantity = 50.0    # DOGEなど: 約50円相当
        
        # 信頼度に基づく数量調整（少額なので控えめに）
        confidence_multiplier = 1.0
        rsi = technical_signals["rsi"]
        volatility = technical_signals["volatility"]
        combined_signal = technical_signals["combined_signal"]
        
        # RSIが極端な場合、数量を少し増やす
        if rsi < 30 or rsi > 70:
            confidence_multiplier += 0.2
        
        # 強いシグナルの場合、数量を少し増やす
        if abs(combined_signal) > 0.6:
            confidence_multiplier += 0.15
        
        # ボラティリティが高い場合、数量を減らす（リスク管理）
        if volatility > 8.0:
            confidence_multiplier -= 0.2
        
        # 最終数量を計算
        final_quantity = base_quantity * confidence_multiplier
        
        # 最小/最大数量の制限（4000円前提）
        if "BTC" in symbol:
            final_quantity = max(0.00005, min(final_quantity, 0.0005))  # 50円〜500円相当
        elif "ETH" in symbol:
            final_quantity = max(0.0005, min(final_quantity, 0.005))   # 15円〜150円相当
        else:
            final_quantity = max(10.0, min(final_quantity, 200.0))      # 10円〜200円相当
        
        return round(final_quantity, 6)
    
    def _check_position_exit(self, symbol: str, current_price: float, technical_signals: Dict) -> Optional[str]:
        """Check if we should exit current position (profit taking or stop loss).
        
        Returns:
            Optional[str]: "SELL" if should exit, None otherwise
        """
        if not self.track_positions or symbol not in self._positions:
            return None
        
        position = self._positions[symbol]
        entry_price = position["entry_price"]
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Take profit condition
        if profit_pct >= self.take_profit_pct:
            logger.info("%s: Take profit triggered at +%.2f%%", symbol, profit_pct)
            return "SELL"
        
        # Stop loss condition
        if profit_pct <= -self.stop_loss_pct:
            logger.info("%s: Stop loss triggered at %.2f%%", symbol, profit_pct)
            return "SELL"
        
        # Technical-based exit: Strong bearish signals while in profit
        if profit_pct > 0.5 and technical_signals["combined_signal"] < -0.5:
            logger.info("%s: Technical exit: profit %.2f%%, bearish signal %.2f",
                       symbol, profit_pct, technical_signals["combined_signal"])
            return "SELL"
        
        # RSI overbought exit while in profit
        if profit_pct > 0.5 and technical_signals["rsi"] > 70:
            logger.info("%s: RSI overbought exit: profit %.2f%%, RSI %.1f",
                       symbol, profit_pct, technical_signals["rsi"])
            return "SELL"
        
        return None
    
    def update_position(self, symbol: str, action: str, price: float, quantity: float) -> None:
        """Update position tracking after trade execution.
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            price: Execution price
            quantity: Trade quantity
        """
        if not self.track_positions:
            return
        
        if action == "BUY":
            if symbol in self._positions:
                # Average down/up
                old_qty = self._positions[symbol]["quantity"]
                old_price = self._positions[symbol]["entry_price"]
                new_qty = old_qty + quantity
                new_avg_price = ((old_price * old_qty) + (price * quantity)) / new_qty
                self._positions[symbol] = {"entry_price": new_avg_price, "quantity": new_qty}
                logger.info("%s: Position averaged - New entry: %.4f, Qty: %.6f",
                           symbol, new_avg_price, new_qty)
            else:
                self._positions[symbol] = {"entry_price": price, "quantity": quantity}
                logger.info("%s: New position opened - Entry: %.4f, Qty: %.6f",
                           symbol, price, quantity)
        
        elif action == "SELL" and symbol in self._positions:
            old_qty = self._positions[symbol]["quantity"]
            if quantity >= old_qty:
                # Full exit
                entry_price = self._positions[symbol]["entry_price"]
                profit_pct = ((price - entry_price) / entry_price) * 100
                logger.info("%s: Position closed - P/L: %+.2f%%", symbol, profit_pct)
                del self._positions[symbol]
            else:
                # Partial exit
                self._positions[symbol]["quantity"] -= quantity
                logger.info("%s: Partial exit - Remaining qty: %.6f",
                           symbol, self._positions[symbol]["quantity"])
    
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
