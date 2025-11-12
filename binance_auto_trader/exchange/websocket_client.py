"""Binance WebSocket client for real-time price updates.

This module provides WebSocket streaming for live market data,
significantly reducing API calls and avoiding rate limits.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import websocket

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """Binance WebSocket client for real-time market data streaming.
    
    Features:
    - Multi-symbol support
    - Automatic reconnection
    - Thread-safe price updates
    - Fallback to REST API on failure
    """
    
    def __init__(self, symbols: List[str], testnet: bool = True):
        """Initialize WebSocket client.
        
        Args:
            symbols: List of trading symbols (e.g., ["BTCJPY", "ETHJPY"])
            testnet: Use testnet endpoint if True
        """
        self.symbols = [s.lower() for s in symbols]  # WebSocket uses lowercase
        self.testnet = testnet
        
        # WebSocket endpoints
        if testnet:
            self.ws_url = "wss://testnet.binance.vision/stream"
        else:
            self.ws_url = "wss://stream.binance.com:9443/stream"
        
        # Price cache (thread-safe)
        self._prices: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # WebSocket connection
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_delay = 5  # seconds
        
        # Callbacks
        self._on_price_update: Optional[Callable[[str, float], None]] = None
        
        logger.info("BinanceWebSocketClient initialized for symbols: %s", symbols)
    
    def start(self) -> None:
        """Start WebSocket connection in background thread."""
        if self._running:
            logger.warning("WebSocket already running")
            return
        
        self._running = True
        self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self._ws_thread.start()
        logger.info("WebSocket client started")
    
    def stop(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)
        logger.info("WebSocket client stopped")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol (thread-safe).
        
        Args:
            symbol: Trading symbol (e.g., "BTCJPY")
        
        Returns:
            Latest price or None if not available
        """
        with self._lock:
            return self._prices.get(symbol.lower())
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all cached prices (thread-safe).
        
        Returns:
            Dictionary of symbol -> price
        """
        with self._lock:
            return self._prices.copy()
    
    def set_price_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for price updates.
        
        Args:
            callback: Function called with (symbol, price) on each update
        """
        self._on_price_update = callback
    
    def _run_websocket(self) -> None:
        """Run WebSocket connection with auto-reconnect."""
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                if self._running:
                    logger.info("Reconnecting in %d seconds...", self._reconnect_delay)
                    time.sleep(self._reconnect_delay)
    
    def _connect(self) -> None:
        """Establish WebSocket connection."""
        # Build stream names for combined stream
        streams = [f"{symbol}@ticker" for symbol in self.symbols]
        stream_path = "/".join(streams)
        
        # For combined streams, use /stream?streams=
        url = f"{self.ws_url}?streams={stream_path}"
        
        logger.info("Connecting to WebSocket: %s", url)
        
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        
        # Run forever (blocking)
        self._ws.run_forever()
    
    def _on_open(self, ws) -> None:
        """Handle WebSocket connection opened."""
        logger.info("WebSocket connected")
    
    def _on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket message.
        
        Args:
            ws: WebSocket instance
            message: JSON message string
        """
        try:
            data = json.loads(message)
            
            # Combined stream format: {"stream": "btcjpy@ticker", "data": {...}}
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                ticker_data = data["data"]
                
                # Extract symbol from stream name (e.g., "btcjpy@ticker" -> "btcjpy")
                symbol = stream_name.split("@")[0]
                
                # Get current price (last traded price)
                price = float(ticker_data.get("c", 0))  # 'c' = close/current price
                
                if price > 0:
                    # Update cache
                    with self._lock:
                        self._prices[symbol] = price
                    
                    # Call callback if set
                    if self._on_price_update:
                        try:
                            self._on_price_update(symbol.upper(), price)
                        except Exception as e:
                            logger.error("Error in price callback: %s", e)
                    
                    logger.debug("Price update: %s = %.4f", symbol.upper(), price)
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse WebSocket message: %s", e)
        except Exception as e:
            logger.error("Error processing WebSocket message: %s", e)
    
    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error.
        
        Args:
            ws: WebSocket instance
            error: Error object
        """
        logger.error("WebSocket error: %s", error)
    
    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket connection closed.
        
        Args:
            ws: WebSocket instance
            close_status_code: Close status code
            close_msg: Close message
        """
        logger.warning("WebSocket closed: %s %s", close_status_code, close_msg)


class WebSocketPriceProvider:
    """Price provider that uses WebSocket with REST API fallback.
    
    This class provides a unified interface for getting prices,
    automatically falling back to REST API if WebSocket is unavailable.
    """
    
    def __init__(self, exchange_client, symbols: List[str], testnet: bool = True):
        """Initialize price provider.
        
        Args:
            exchange_client: BinanceClient instance for REST API fallback
            symbols: List of trading symbols
            testnet: Use testnet if True
        """
        self.exchange_client = exchange_client
        self.symbols = symbols
        self.testnet = testnet
        
        # WebSocket client
        self.ws_client: Optional[BinanceWebSocketClient] = None
        self.ws_enabled = True
        
        # Price cache with timestamp
        self._price_cache: Dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)
        self._cache_ttl = 60.0  # Cache REST API prices for 60 seconds
        
        logger.info("WebSocketPriceProvider initialized")
    
    def start(self) -> None:
        """Start WebSocket connection."""
        if not self.ws_enabled:
            logger.info("WebSocket disabled, using REST API only")
            return
        
        try:
            # Convert display symbols to exchange symbols (e.g., "BTC/JPY" -> "BTCJPY")
            ws_symbols = [s.replace("/", "") for s in self.symbols]
            
            self.ws_client = BinanceWebSocketClient(ws_symbols, self.testnet)
            self.ws_client.start()
            
            # Wait a bit for initial connection
            time.sleep(2)
            
            logger.info("WebSocket price provider started")
        except Exception as e:
            logger.error("Failed to start WebSocket: %s", e)
            logger.info("Falling back to REST API")
            self.ws_enabled = False
    
    def stop(self) -> None:
        """Stop WebSocket connection."""
        if self.ws_client:
            self.ws_client.stop()
            logger.info("WebSocket price provider stopped")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/JPY" or "BTCJPY")
        
        Returns:
            Latest price or None
        """
        # Normalize symbol
        exchange_symbol = symbol.replace("/", "")
        
        # Try WebSocket first
        if self.ws_client:
            price = self.ws_client.get_price(exchange_symbol)
            if price:
                logger.debug("Price from WebSocket: %s = %.4f", symbol, price)
                return price
        
        # Fallback to REST API with cache
        return self._get_price_rest(exchange_symbol)
    
    def _get_price_rest(self, exchange_symbol: str) -> Optional[float]:
        """Get price from REST API with caching.
        
        Args:
            exchange_symbol: Exchange symbol (e.g., "BTCJPY")
        
        Returns:
            Price or None
        """
        current_time = time.time()
        
        # Check cache
        if exchange_symbol in self._price_cache:
            cached_price, cached_time = self._price_cache[exchange_symbol]
            if current_time - cached_time < self._cache_ttl:
                logger.debug("Price from cache: %s = %.4f", exchange_symbol, cached_price)
                return cached_price
        
        # Fetch from REST API
        try:
            ticker = self.exchange_client.client.get_symbol_ticker(symbol=exchange_symbol)
            price = float(ticker["price"])
            
            # Update cache
            self._price_cache[exchange_symbol] = (price, current_time)
            
            logger.debug("Price from REST API: %s = %.4f", exchange_symbol, price)
            return price
        except Exception as e:
            logger.error("Failed to get price from REST API for %s: %s", exchange_symbol, e)
            return None
