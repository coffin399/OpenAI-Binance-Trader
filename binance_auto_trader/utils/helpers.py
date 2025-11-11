from __future__ import annotations

from statistics import mean
from typing import Iterable, List, Sequence


def normalize_symbol(symbol: str) -> str:
    """Convert symbols like DOGE/USDT to Binance format DOGEUSDT."""
    return symbol.replace("/", "").upper()


def compute_sortino_ratio(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    downside = [r for r in returns if r < 0]
    if not downside:
        return float("inf")
    mean_return = mean(returns)
    downside_deviation = (sum((r - 0) ** 2 for r in downside) / len(downside)) ** 0.5
    if downside_deviation == 0:
        return float("inf")
    return mean_return / downside_deviation


def parse_labeled_limit(value, suffix: str) -> int:
    """Parse values like '5Trades' into integers. Returns 0 for unlimited."""
    if value is None:
        return 0
    if isinstance(value, int):
        return max(value, 0)
    text = str(value).strip()
    if not text:
        return 0
    if text.lower().endswith(suffix.lower()):
        text = text[: -len(suffix)].strip()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid limit format '{value}'. Expected e.g. '5{suffix}'.") from exc
    return max(parsed, 0)


def parse_currency_limit(value, suffix: str = "JPY") -> float:
    """Parse values like '10000JPY' into floats. Returns 0.0 for unlimited."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    text = str(value).strip()
    if not text:
        return 0.0
    if text.lower().endswith(suffix.lower()):
        text = text[: -len(suffix)].strip()
    text = text.replace(",", "")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid currency limit '{value}'. Expected e.g. '10000{suffix}'.") from exc
    return max(parsed, 0.0)
