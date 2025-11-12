from __future__ import annotations

import logging
from typing import Optional

from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET

from ..config.config import Config


class BinanceClient:
    def __init__(self, config: Config, testnet: Optional[bool] = None):
        self.config = config
        self.testnet = (
            config.binance.testnet if testnet is None else bool(testnet)
        )
        self.client = self._initialize_client()

    def _initialize_client(self) -> Client:
        """Initialize the Binance client with API credentials."""

        api_key = getattr(self.config.binance, "api_key", "")
        api_secret = getattr(self.config.binance, "api_secret", "")

        if not api_key or not api_secret:
            raise ValueError(
                "Binance API key and secret must be provided in config or environment."
            )

        if self.testnet:
            return Client(api_key, api_secret, testnet=True)
        return Client(api_key, api_secret)

    def get_account_balance(self, asset: str = "JPY") -> Optional[float]:
        """Get account balance for a specific asset."""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance["free"])
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error(
                "Error getting balance for %s: %s", asset, exc
            )
            return None

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_str: Optional[str] = None,
        end_str: Optional[str] = None,
        limit: int = 500,
    ):
        """Get historical klines (candlestick data)."""
        try:
            return self.client.get_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error(
                "Error getting historical klines: %s", exc
            )
            return None

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = ORDER_TYPE_MARKET,
        **kwargs,
    ):
        """Create a new order."""
        try:
            # Binance APIは数量を文字列形式で要求するため、適切にフォーマット
            # 科学的記法を避け、末尾のゼロを削除
            quantity_str = f"{quantity:.8f}".rstrip('0').rstrip('.')
            
            logging.getLogger(__name__).info(
                "Creating order: %s %s %s (quantity: %s)", 
                side, symbol, quantity_str, type(quantity_str)
            )
            
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity_str,
                **kwargs,
            )
            logging.getLogger(__name__).info("Order created: %s", order)
            return order
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error(
                "Error creating order for %s: %s", symbol, exc
            )
            return None

    def get_symbol_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error(
                "Error fetching price ticker for %s: %s", symbol, exc
            )
            return None
