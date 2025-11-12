from __future__ import annotations

from typing import Dict, Optional

from binance_auto_trader.ai.provider_manager import AIProviderManager

from .coffin299 import Coffin299Strategy
from .moving_average_cross import MovingAverageCrossover
from .rsi_mean_reversion import RSIMeanReversionStrategy
STRATEGY_REGISTRY = {
    "moving_average_cross": MovingAverageCrossover,
    "coffin299": Coffin299Strategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
}


def build_strategy(
    name: str,
    config_section,
    ai_manager: Optional[AIProviderManager] = None,
):
    strategy_cls = STRATEGY_REGISTRY.get(name)
    if not strategy_cls:
        raise ValueError(f"Strategy '{name}' is not registered")

    if name == "coffin299":
        if ai_manager is None:
            raise ValueError("Coffin299Strategy requires an AIProviderManager instance")
        return strategy_cls(ai_manager, config_section)

    if config_section:
        return strategy_cls(**config_section.to_dict())
    return strategy_cls()