from __future__ import annotations

import logging
from typing import Optional

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

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[StrategyDecision]:
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
