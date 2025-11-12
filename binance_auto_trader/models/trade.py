from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class StrategyDecision:
    symbol: str
    strategy: str
    action: str
    price: float
    confidence: float = 0.0
    info: str = ""
    quantity: Optional[float] = None  # AIが決定した数量（オプション）
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeRecord:
    symbol: str
    strategy: str
    action: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    pnl_percent: Optional[float]
    opened_at: datetime
    closed_at: Optional[datetime]
    status: str = "OPEN"


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    trade_count: int
    average_return: float
    max_return: float
    sortino_ratio: float
    returns: List[float] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
