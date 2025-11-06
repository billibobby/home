"""
Multi-Asset Time Series Data Fetcher for QQQ Trading Bot

Fetches 1-minute bars for QQQ, SPY, and VXX using Alpaca API,
merges data, and adds technical indicators for CNN-LSTM model training.

Note: This class is used for Alpaca-based minute data fetching.
StockDataFetcher now also supports multi-source via yfinance.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
from pathlib import Path
import json

# Suppress pandas warnings
warnings.filterwarnings('ignore')

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

try:
    from trading_bot.utils.exceptions import DataError
    from trading_bot.utils.helpers import retry_on_failure, ensure_dir
    from trading_bot.utils.paths import get_writable_app_dir
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Fallback exception class
    class DataError(Exception):
        pass


class MultiAssetDataFetcher:
    """
    Multi-asset data fetcher using Alpaca API for QQQ trading bot.
    
    Fetches minute-level bars for QQQ, SPY, and VXX, merges them,
    and adds technical indicators.
    
    Note: This class is used for Alpaca-specific functionality.
    StockDataFetcher now also supports multi-source via yfinance.
    """
    
    def __init__(self, config=None, logger=None):
        """
        Initialize the multi-asset data fetcher.
        
        Args:
            config: Optional configuration object
            logger: Optional logger instance for logging
        
        Raises:
            DataError: If Alpaca credentials are not found
            ImportError: If required libraries are not installed
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is not installed. Install with: pip install alpaca-py"
            )
        
        if not PANDAS_TA_AVAILABLE:
            raise ImportError(
                "pandas_ta is not installed. Install with: pip install pandas-ta"
            )
        
        self.config = config
        self.logger = logger
        
        # Get API keys from config or environment
        if config:
            try:
                self.api_key = config.get_env('ALPACA_API_KEY') or config.get_env('APCA_API_KEY_ID')
                self.api_secret = config.get_env('ALPACA_API_SECRET') or config.get_env('APCA_API_SECRET_KEY')
            except AttributeError:
                # Fallback if config doesn't have get_env method
                self.api_key = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
                self.api_secret = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        else:
            # Fallback to environment variables
            self.api_key = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
            self.api_secret = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise DataError(
                "Alpaca API credentials not found. "
                "Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables, "
                "or configure via config object."
            )
        
        # Initialize Alpaca client
        self.client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )
        
        # Get symbols from config or use defaults (VXX instead of VIXY)
        if config:
            try:
                self.symbols = config.get('data.multi_source.symbols', ['QQQ', 'SPY', 'VXX'])
            except AttributeError:
                self.symbols = ['QQQ', 'SPY', 'VXX']
        else:
            self.symbols = ['QQQ', 'SPY', 'VXX']
        
        # Cache configuration
        self.cache_enabled = False
        self.cache_dir = None
        if config:
            try:
                self.cache_enabled = config.get('data.cache_historical_data', False)
                if self.cache_enabled:
                    historical_data_path = config.get('data.historical_data_path', 'data/historical/')
                    if Path(historical_data_path).is_absolute():
                        self.cache_dir = str(Path(historical_data_path))
                    else:
                        if UTILS_AVAILABLE:
                            subdir = Path(historical_data_path).parts[-1] if Path(historical_data_path).parts else 'historical'
                            self.cache_dir = get_writable_app_dir(subdir)
                            ensure_dir(self.cache_dir)
            except (AttributeError, Exception):
                pass
        
        if self.logger:
            self.logger.info(f"MultiAssetDataFetcher initialized with Alpaca API (symbols: {', '.join(self.symbols)})")
    
    def _log(self, message: str, level: str = 'info'):
        """Log message if logger is available."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def fetch_historical_data(self, years: int = 5, interval: str = '1m') -> pd.DataFrame:
        """
        Fetch minute-level bars for QQQ, SPY, and VXX for the last N years.
        
        Args:
            years: Number of years of historical data to fetch (default: 5)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
        
        Returns:
            Consolidated DataFrame with merged data from all symbols
        """
        # Map interval string to TimeFrame
        interval_map = {
            '1m': TimeFrame(1, TimeFrameUnit.Minute),
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '30m': TimeFrame(30, TimeFrameUnit.Minute),
            '1h': TimeFrame(1, TimeFrameUnit.Hour),
            '1d': TimeFrame(1, TimeFrameUnit.Day)
        }
        
        if interval not in interval_map:
            raise DataError(
                f"Unsupported interval: {interval}. Supported: {list(interval_map.keys())}",
                timeframe=interval
            )
        
        timeframe = interval_map[interval]
        
        # Calculate start date (N years ago from today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        self._log(
            f"Fetching {interval} bars from {start_date.date()} to {end_date.date()} "
            f"for symbols: {', '.join(self.symbols)}"
        )
        
        # Check cache first
        if self.cache_enabled and self.cache_dir:
            cached_df = self._load_cached_data(interval)
            if cached_df is not None:
                self._log(f"Using cached data ({len(cached_df)} rows)")
                return cached_df
        
        dataframes = {}
        
        # Fetch data for each symbol
        for symbol in self.symbols:
            try:
                df = self._fetch_symbol_data(symbol, start_date, end_date, timeframe)
                if df is not None and not df.empty:
                    dataframes[symbol] = df
                    self._log(f"Successfully fetched {len(df)} rows for {symbol}")
                else:
                    self._log(f"No data retrieved for {symbol}", 'warning')
            except Exception as e:
                self._log(f"Error fetching data for {symbol}: {str(e)}", 'error')
                continue
        
        if not dataframes:
            raise DataError("No data was successfully fetched for any symbol")
        
        # Merge all DataFrames
        master_df = self._merge_dataframes(dataframes)
        
        # Cache the data
        if self.cache_enabled and self.cache_dir:
            self._cache_data(master_df, interval)
        
        self._log(f"Created master DataFrame with {len(master_df)} rows and {len(master_df.columns)} columns")
        
        return master_df
    
    def _fetch_symbol_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, timeframe: TimeFrame = TimeFrame.Minute) -> Optional[pd.DataFrame]:
        """
        Fetch minute-level bars for a single symbol.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start datetime
            end_date: End datetime
            timeframe: TimeFrame object for interval
        
        Returns:
            DataFrame with OHLCV data and timestamp index
        """
        try:
            # Create request for bars
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            # Fetch data
            bars = self.client.get_stock_bars(request_params)
            
            if not bars or symbol not in bars:
                return None
            
            # Convert to DataFrame
            symbol_bars = bars[symbol]
            
            if not symbol_bars:
                return None
            
            # Extract data
            data = {
                'timestamp': [bar.timestamp for bar in symbol_bars],
                'open': [float(bar.open) for bar in symbol_bars],
                'high': [float(bar.high) for bar in symbol_bars],
                'low': [float(bar.low) for bar in symbol_bars],
                'close': [float(bar.close) for bar in symbol_bars],
                'volume': [int(bar.volume) for bar in symbol_bars]
            }
            
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Rename columns with symbol prefix
            df.columns = [f"{symbol}_{col.upper()}" for col in df.columns]
            
            return df
            
        except Exception as e:
            self._log(f"Error fetching {symbol}: {str(e)}", 'error')
            return None
    
    def _merge_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames using inner join on timestamp index.
        
        Args:
            dataframes: Dictionary of {symbol: DataFrame} with timestamp index
        
        Returns:
            Merged DataFrame with all columns
        """
        if not dataframes:
            raise DataError("No dataframes to merge")
        
        # Start with the first DataFrame
        master_df = dataframes[list(dataframes.keys())[0]].copy()
        
        # Merge remaining DataFrames using inner join
        for symbol, df in list(dataframes.items())[1:]:
            master_df = master_df.join(df, how='inner')
        
        # Sort by timestamp
        master_df.sort_index(inplace=True)
        
        # Remove any duplicate timestamps
        master_df = master_df[~master_df.index.duplicated(keep='first')]
        
        self._log(f"Merged {len(dataframes)} DataFrames with inner join")
        
        return master_df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the master DataFrame.
        
        Calculates RSI (14-period) and MACD (12, 26, 9) on QQQ_Close,
        and adds lagged features (1, 2, 5 periods).
        
        Args:
            df: Master DataFrame with merged data
        
        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()
        
        # Ensure QQQ_CLOSE exists
        if 'QQQ_CLOSE' not in df.columns:
            raise ValueError("QQQ_CLOSE column not found in DataFrame")
        
        self._log("Adding technical indicators...")
        
        # Calculate RSI (14-period) on QQQ_Close
        if PANDAS_TA_AVAILABLE:
            rsi = ta.rsi(df['QQQ_CLOSE'], length=14)
            df['QQQ_RSI_14'] = rsi
            self._log("Added RSI(14) indicator")
        else:
            # Fallback: manual RSI calculation
            df['QQQ_RSI_14'] = self._calculate_rsi(df['QQQ_CLOSE'], period=14)
            self._log("Added RSI(14) indicator (manual calculation)")
        
        # Calculate MACD (12, 26, 9) on QQQ_Close
        if PANDAS_TA_AVAILABLE:
            macd = ta.macd(df['QQQ_CLOSE'], fast=12, slow=26, signal=9)
            if isinstance(macd, pd.DataFrame):
                df['QQQ_MACD'] = macd[f'MACD_12_26_9']
                df['QQQ_MACD_Signal'] = macd[f'MACDs_12_26_9']
                df['QQQ_MACD_Histogram'] = macd[f'MACDh_12_26_9']
            else:
                df['QQQ_MACD'] = macd
            self._log("Added MACD(12, 26, 9) indicators")
        else:
            # Fallback: manual MACD calculation
            macd_result = self._calculate_macd(df['QQQ_CLOSE'], fast=12, slow=26, signal=9)
            df['QQQ_MACD'] = macd_result['MACD']
            df['QQQ_MACD_Signal'] = macd_result['Signal']
            df['QQQ_MACD_Histogram'] = macd_result['Histogram']
            self._log("Added MACD(12, 26, 9) indicators (manual calculation)")
        
        # Add lagged features for QQQ_Close (1, 2, 5 periods)
        df['QQQ_Close_Lag_1'] = df['QQQ_CLOSE'].shift(1)
        df['QQQ_Close_Lag_2'] = df['QQQ_CLOSE'].shift(2)
        df['QQQ_Close_Lag_5'] = df['QQQ_CLOSE'].shift(5)
        
        self._log("Added lagged features (1, 2, 5 periods)")
        
        # Remove rows with NaN values (from indicators and lagged features)
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            self._log(f"Removed {removed_rows} rows with NaN values after feature engineering")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index manually.
        
        Args:
            prices: Price series
            period: RSI period (default: 14)
        
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD manually.
        
        Args:
            prices: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
        
        Returns:
            Dictionary with MACD, Signal, and Histogram
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def get_processed_data(self, years: int = 5) -> pd.DataFrame:
        """
        Complete pipeline: fetch data, merge, and add technical indicators.
        
        Args:
            years: Number of years of historical data to fetch (default: 5)
        
        Returns:
            Final processed DataFrame ready for CNN-LSTM model training
        """
        self._log("Starting data processing pipeline...")
        
        # Step 1: Fetch historical data
        df = self.fetch_historical_data(years=years)
        
        # Step 2: Add technical indicators
        df = self.add_technical_indicators(df)
        
        self._log(
            f"Data processing complete. Final DataFrame: {len(df)} rows, "
            f"{len(df.columns)} columns"
        )
        
        return df
    
    def _cache_data(self, dataframe: pd.DataFrame, interval: str) -> None:
        """
        Cache merged data to local file.
        
        Args:
            dataframe: DataFrame to cache
            interval: Data interval
        """
        if not self.cache_enabled or not self.cache_dir:
            return
        
        try:
            cache_key = '_'.join(sorted(self.symbols))
            cache_file = Path(self.cache_dir) / f"{cache_key}_{interval}.csv"
            
            # Convert index to Date column if needed
            df_cache = dataframe.copy()
            if isinstance(df_cache.index, pd.DatetimeIndex):
                df_cache.reset_index(inplace=True)
                df_cache = df_cache.rename(columns={df_cache.columns[0]: 'Date'})
            
            df_cache.to_csv(cache_file, index=False)
            
            # Save metadata
            metadata = {
                'symbols': self.symbols,
                'interval': interval,
                'cached_at': datetime.now().isoformat(),
                'rows': len(df_cache),
                'source': 'alpaca'
            }
            if 'Date' in df_cache.columns:
                metadata['start_date'] = str(df_cache['Date'].min())
                metadata['end_date'] = str(df_cache['Date'].max())
            
            metadata_file = Path(self.cache_dir) / f"{cache_key}_{interval}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._log(f"Cached data to {cache_file}")
            
        except Exception as e:
            self._log(f"Failed to cache data: {str(e)}", 'warning')
    
    def _load_cached_data(self, interval: str) -> Optional[pd.DataFrame]:
        """
        Load cached data if available and fresh.
        
        Args:
            interval: Data interval
            
        Returns:
            Cached DataFrame or None
        """
        if not self.cache_enabled or not self.cache_dir:
            return None
        
        try:
            cache_key = '_'.join(sorted(self.symbols))
            cache_file = Path(self.cache_dir) / f"{cache_key}_{interval}.csv"
            metadata_file = Path(self.cache_dir) / f"{cache_key}_{interval}_meta.json"
            
            if not cache_file.exists() or not metadata_file.exists():
                return None
            
            # Check cache age (1 day max)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cached_at = datetime.fromisoformat(metadata.get('cached_at', ''))
            age = datetime.now() - cached_at
            
            if age.days > 1:
                self._log(f"Cache is stale ({age.days} days old)", 'debug')
                return None
            
            # Load cached data
            df = pd.read_csv(cache_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            self._log(f"Loaded cached data ({len(df)} rows)", 'debug')
            return df
            
        except Exception as e:
            self._log(f"Failed to load cached data: {str(e)}", 'warning')
            return None


def main():
    """
    Example usage of MultiAssetDataFetcher.
    """
    # Simple logger for demonstration
    class SimpleLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def debug(self, msg): print(f"[DEBUG] {msg}")
    
    logger = SimpleLogger()
    
    try:
        # Initialize fetcher
        fetcher = MultiAssetDataFetcher(logger=logger)
        
        # Get processed data (5 years of 1-minute bars)
        df = fetcher.get_processed_data(years=5)
        
        # Display summary
        print("\n" + "=" * 60)
        print("Data Processing Complete")
        print("=" * 60)
        print(f"\nDataFrame Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nDate Range: {df.index.min()} to {df.index.max()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nDataFrame Info:")
        print(df.info())
        
        # Save to CSV (optional)
        output_file = "qqq_multi_asset_data.csv"
        df.to_csv(output_file)
        print(f"\nData saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

