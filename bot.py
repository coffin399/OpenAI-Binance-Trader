import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('binance_auto_trader')

class BinanceAutoTrader:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Binance client
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.testnet = os.getenv('TESTNET', 'True').lower() == 'true'
        
        if not self.api_key or not self.api_secret:
            logger.error("API key or secret not found in .env file")
            raise ValueError("API key and secret are required")
        
        # Initialize Binance client
        self.client = Client(
            self.api_key, 
            self.api_secret,
            testnet=self.testnet
        )
        
        # Trading parameters
        self.symbol = os.getenv('TRADING_PAIR', 'BTCUSDT')
        self.quantity = float(os.getenv('TRADE_QUANTITY', 0.001))
        
        # Strategy parameters
        self.fast_ma = 10
        self.slow_ma = 30
        self.timeframe = Client.KLINE_INTERVAL_15MINUTE
        
        # Track current position
        self.in_position = False
        
        logger.info(f"Initialized Binance Auto Trader for {self.symbol}")
        if self.testnet:
            logger.info("Running in TESTNET mode")
        else:
            logger.warning("RUNNING IN LIVE TRADING MODE")

    def get_historical_data(self, limit=100):
        """Get historical klines data"""
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Calculate moving averages
            df['fast_ma'] = df['close'].rolling(window=self.fast_ma).mean()
            df['slow_ma'] = df['close'].rolling(window=self.slow_ma).mean()
            
            # Generate signals
            df['signal'] = 0
            df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Buy signal
            df.loc[df['fast_ma'] <= df['slow_ma'], 'signal'] = -1  # Sell signal
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    def execute_trade(self, signal):
        """Execute buy or sell order"""
        try:
            if signal == 1 and not self.in_position:
                # Buy signal
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=self.quantity
                )
                self.in_position = True
                logger.info(f"BUY order executed: {order}")
                return order
                
            elif signal == -1 and self.in_position:
                # Sell signal
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=self.quantity
                )
                self.in_position = False
                logger.info(f"SELL order executed: {order}")
                return order
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    def check_balance(self):
        """Check account balance"""
        try:
            account = self.client.get_account()
            balances = {}
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:  # Only show assets with balance
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            return balances
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return None

    def run(self):
        """Main trading loop"""
        logger.info("Starting trading bot...")
        
        try:
            while True:
                try:
                    # Get historical data
                    df = self.get_historical_data(limit=100)
                    if df is None:
                        time.sleep(60)  # Wait before retrying
                        continue
                    
                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    if df is None:
                        time.sleep(60)
                        continue
                    
                    # Get the latest signal
                    current_signal = df['signal'].iloc[-1]
                    previous_signal = df['signal'].iloc[-2] if len(df) > 1 else 0
                    
                    logger.info(f"Current price: {df['close'].iloc[-1]:.2f} | "
                              f"Fast MA: {df['fast_ma'].iloc[-1]:.2f} | "
                              f"Slow MA: {df['slow_ma'].iloc[-1]:.2f} | "
                              f"Signal: {current_signal} | "
                              f"In position: {self.in_position}")
                    
                    # Execute trade if signal changed
                    if current_signal != previous_signal:
                        self.execute_trade(current_signal)
                    
                    # Check balance every 15 minutes
                    if datetime.now().minute % 15 == 0:
                        balance = self.check_balance()
                        if balance:
                            logger.info(f"Account balance: {balance}")
                    
                    # Wait for the next candle
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            
        finally:
            logger.info("Trading bot stopped")

if __name__ == "__main__":
    # Create and run the bot
    bot = BinanceAutoTrader()
    bot.run()
