from __future__ import annotations

from typing import Dict, Optional

from binance_auto_trader.ai.provider_manager import AIProviderManager

from .ai_hybrid import AIHybridStrategy
from .moving_average_cross import MovingAverageCrossover

STRATEGY_REGISTRY = {
    "moving_average_cross": MovingAverageCrossover,
    "ai_hybrid": AIHybridStrategy,
}


def build_strategy(
    name: str,
    config_section,
    ai_manager: Optional[AIProviderManager] = None,
):
    strategy_cls = STRATEGY_REGISTRY.get(name)
    if not strategy_cls:
        raise ValueError(f"Strategy '{name}' is not registered")

    if name == "ai_hybrid":
        if ai_manager is None:
            raise ValueError("AIHybridStrategy requires an AIProviderManager instance")
        return strategy_cls(ai_manager, config_section)

    if config_section:
        return strategy_cls(**config_section.to_dict())
    return strategy_cls()
