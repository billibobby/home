"""
Stock Data Fetcher Module

Fetches historical and real-time stock market data using yfinance.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import json

from trading_bot.utils.helpers import retry_on_failure, ensure_dir
from trading_bot.utils.exceptions import DataError
from trading_bot.utils.paths import get_writable_app_dir


class StockDataFetcher:
    """
    Fetches stock market data using yfinance.
    
    Handles data caching, validation, and error recovery.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the stock data fetcher.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Get cache configuration
        self.cache_enabled = config.get('data.cache_historical_data', True)
        historical_data_path = config.get('data.historical_data_path', 'data/historical/')
        
        # Cache strategy configuration
        self.cache_strategy = config.get('data.cache_strategy', 'full')
        self.cache_window_days = config.get('data.cache_window_days', 60)
        
        # Multi-source configuration
        self.multi_source_enabled = config.get('data.multi_source.enabled', False)
        self.multi_source_symbols = config.get('data.multi_source.symbols', ['QQQ', 'SPY', 'VXX'])
        
        # Supported intervals
        self.supported_intervals = ['1m', '5m', '15m', '30m', '1h', '1d']
        
        # Setup cache directory
        if self.cache_enabled:
            # Check if path is absolute
            if Path(historical_data_path).is_absolute():
                # Use absolute path as-is
                self.cache_dir = str(Path(historical_data_path))
            else:
                # Use user-writable app directory with subdir name
                # Extract just the subdirectory name (e.g., 'historical' from 'data/historical/')
                subdir = Path(historical_data_path).parts[-1] if Path(historical_data_path).parts else 'historical'
                self.cache_dir = get_writable_app_dir(subdir)
            
            ensure_dir(self.cache_dir)
            self.logger.info(f"Stock data cache enabled at: {self.cache_dir} (strategy: {self.cache_strategy})")
        else:
            self.cache_dir = None
            self.logger.info("Stock data cache disabled")
    
    @retry_on_failure(max_attempts=3, delay=2.0, backoff=2.0)
    def fetch_historical_data(self, symbol: str, start_date: str, 
                            end_date: str = None, interval: str = '1d') -> pd.DataFrame:
        """
        Download historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Date
            
        Raises:
            DataError: If data fetching fails
        """
        try:
            import yfinance as yf
            
            # Validate interval
            if interval not in self.supported_intervals:
                raise DataError(
                    f"Unsupported interval: {interval}. Supported: {self.supported_intervals}",
                    symbol=symbol,
                    timeframe=interval
                )
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Check cache first
            if self.cache_enabled and self.cache_strategy != 'disabled':
                cached_df = self._load_cached_data(symbol, interval, max_age_days=1, start_date=start_date)
                if cached_df is not None:
                    # Apply rolling window if needed
                    if self.cache_strategy == 'rolling_window':
                        cached_df = self._apply_rolling_window(cached_df, start_date)
                    # Check if cached data covers the requested range
                    if len(cached_df) > 0:
                        cached_start = cached_df['Date'].min()
                        cached_end = cached_df['Date'].max()
                        if (pd.to_datetime(start_date) >= cached_start and 
                            pd.to_datetime(end_date) <= cached_end):
                            self.logger.info(f"Using cached data for {symbol} ({len(cached_df)} rows)")
                            return cached_df
            
            self.logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date} (interval: {interval})")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise DataError(
                    f"No data retrieved for symbol {symbol}",
                    symbol=symbol,
                    details={'start_date': start_date, 'end_date': end_date, 'interval': interval}
                )
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Normalize datetime column name to 'Date'
            df = self._normalize_date_column(df)
            
            # Validate data
            self.validate_data(df)
            
            # Cache the data
            if self.cache_enabled and self.cache_strategy != 'disabled':
                self._cache_data(symbol, df, interval)
                # Prune cache after caching if using rolling_window strategy
                if self.cache_strategy == 'rolling_window':
                    self._prune_cache()
            
            self.logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except ImportError:
            raise DataError(
                "yfinance library not installed. Install with: pip install yfinance",
                symbol=symbol
            )
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            raise DataError(
                f"Data fetch failed: {str(e)}",
                symbol=symbol,
                details={'error': str(e)}
            )
    
    def fetch_latest_data(self, symbol: str, period: str = '1d', interval: str = '1d') -> pd.DataFrame:
        """
        Get recent data for real-time predictions.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, etc.)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            
        Returns:
            DataFrame with recent OHLCV data
            
        Raises:
            DataError: If data fetching fails
        """
        try:
            import yfinance as yf
            
            # Validate interval
            if interval not in self.supported_intervals:
                raise DataError(
                    f"Unsupported interval: {interval}. Supported: {self.supported_intervals}",
                    symbol=symbol,
                    timeframe=interval
                )
            
            self.logger.debug(f"Fetching latest data for {symbol} (period: {period}, interval: {interval})")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise DataError(
                    f"No recent data available for {symbol}",
                    symbol=symbol
                )
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Normalize datetime column name to 'Date'
            df = self._normalize_date_column(df)
            
            self.validate_data(df)
            
            self.logger.debug(f"Fetched {len(df)} recent rows for {symbol}")
            return df
            
        except ImportError:
            raise DataError(
                "yfinance library not installed",
                symbol=symbol
            )
        except Exception as e:
            self.logger.error(f"Failed to fetch latest data for {symbol}: {str(e)}")
            raise DataError(
                f"Latest data fetch failed: {str(e)}",
                symbol=symbol,
                details={'error': str(e)}
            )
    
    def fetch_multi_source_data(self, symbols: List[str], start_date: str, 
                                end_date: str = None, interval: str = '5m') -> pd.DataFrame:
        """
        Fetch historical data for multiple symbols and merge them with timestamp alignment.
        
        Args:
            symbols: List of stock ticker symbols (e.g., ['QQQ', 'SPY', 'VXX'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            
        Returns:
            Merged DataFrame with symbol-prefixed columns (e.g., QQQ_Close, SPY_Volume)
            
        Raises:
            DataError: If data fetching fails
        """
        if not symbols:
            raise DataError("No symbols provided for multi-source fetching")
        
        # Validate interval
        if interval not in self.supported_intervals:
            raise DataError(
                f"Unsupported interval: {interval}. Supported: {self.supported_intervals}",
                timeframe=interval
            )
        
        # Check cache for multi-source data
        if self.cache_enabled and self.cache_strategy != 'disabled':
            cache_key = '_'.join(sorted(symbols))
            cached_df = self._load_cached_multi_source_data(cache_key, interval, start_date=start_date, end_date=end_date)
            if cached_df is not None:
                # Apply rolling window if needed
                if self.cache_strategy == 'rolling_window':
                    cached_df = self._apply_rolling_window(cached_df, start_date)
                # Verify cached data covers requested date range
                if len(cached_df) > 0:
                    cached_start = pd.to_datetime(cached_df['Date'].min())
                    cached_end = pd.to_datetime(cached_df['Date'].max())
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date) if end_date else pd.to_datetime(datetime.now())
                    if cached_start <= start_dt and cached_end >= end_dt:
                        self.logger.info(f"Using cached multi-source data ({len(cached_df)} rows)")
                        return cached_df
                    else:
                        self.logger.debug(f"Cached multi-source data does not cover requested range: {start_date} to {end_date}")
                else:
                    self.logger.debug(f"Cached multi-source data is empty")
        
        self.logger.info(f"Fetching multi-source data for {', '.join(symbols)} from {start_date} to {end_date or 'today'}")
        
        dataframes = {}
        
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                df = self.fetch_historical_data(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    # Prefix columns with symbol
                    df = df.copy()
                    rename_map = {col: f"{symbol}_{col}" for col in df.columns if col != 'Date'}
                    df = df.rename(columns=rename_map)
                    dataframes[symbol] = df
                    self.logger.debug(f"Fetched {len(df)} rows for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        if not dataframes:
            raise DataError("No data was successfully fetched for any symbol")
        
        # Synchronize and merge dataframes
        merged_df = self._synchronize_timestamps(dataframes)
        
        # Cache merged data
        if self.cache_enabled and self.cache_strategy != 'disabled':
            cache_key = '_'.join(sorted(symbols))
            self._cache_multi_source_data(cache_key, merged_df, interval, symbols)
            # Prune cache after caching if using rolling_window strategy
            if self.cache_strategy == 'rolling_window':
                self._prune_cache()
        
        self.logger.info(f"Multi-source data merge complete: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        return merged_df
    
    def fetch_latest_multi_source(self, symbols: List[str], period: str = '1d', 
                                  interval: str = '5m') -> pd.DataFrame:
        """
        Fetch latest data for multiple symbols and merge them.
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period (1d, 5d, 1mo, etc.)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            
        Returns:
            Merged DataFrame with symbol-prefixed columns
        """
        if not symbols:
            raise DataError("No symbols provided for multi-source fetching")
        
        # Validate interval
        if interval not in self.supported_intervals:
            raise DataError(
                f"Unsupported interval: {interval}. Supported: {self.supported_intervals}",
                timeframe=interval
            )
        
        self.logger.debug(f"Fetching latest multi-source data for {', '.join(symbols)}")
        
        dataframes = {}
        
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                df = self.fetch_latest_data(symbol, period, interval)
                if df is not None and not df.empty:
                    # Prefix columns with symbol
                    df = df.copy()
                    rename_map = {col: f"{symbol}_{col}" for col in df.columns if col != 'Date'}
                    df = df.rename(columns=rename_map)
                    dataframes[symbol] = df
            except Exception as e:
                self.logger.warning(f"Failed to fetch latest data for {symbol}: {str(e)}")
                continue
        
        if not dataframes:
            raise DataError("No data was successfully fetched for any symbol")
        
        # Synchronize and merge dataframes
        merged_df = self._synchronize_timestamps(dataframes)
        
        self.logger.debug(f"Latest multi-source data merge complete: {len(merged_df)} rows")
        return merged_df
    
    def _synchronize_timestamps(self, dataframes: dict) -> pd.DataFrame:
        """
        Synchronize timestamps across multiple DataFrames using inner join.
        
        Args:
            dataframes: Dictionary of {symbol: DataFrame} with 'Date' column
            
        Returns:
            Merged DataFrame with synchronized timestamps
        """
        if not dataframes:
            raise DataError("No dataframes to synchronize")
        
        # Ensure all Date columns are datetime and timezone-normalized
        for symbol, df in dataframes.items():
            if 'Date' not in df.columns:
                raise DataError(f"Date column missing in {symbol} DataFrame")
            df['Date'] = pd.to_datetime(df['Date'])
            # Normalize timezone: convert to UTC if naive, then ensure UTC
            date_series = df['Date']
            if date_series.dt.tz is None:
                df['Date'] = date_series.dt.tz_localize('UTC')
            else:
                df['Date'] = date_series.dt.tz_convert('UTC')
            df.set_index('Date', inplace=True)
        
        # Start with the first DataFrame
        merged_df = dataframes[list(dataframes.keys())[0]].copy()
        
        # Merge remaining DataFrames using inner join (ensures timestamp alignment)
        for symbol, df in list(dataframes.items())[1:]:
            merged_df = merged_df.join(df, how='inner', rsuffix=f'_{symbol}')
        
        # Remove any duplicate timestamps
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        merged_df.sort_index(inplace=True)
        
        # Reset index to make Date a column
        merged_df.reset_index(inplace=True)
        merged_df = self._normalize_date_column(merged_df)
        
        # Validate minimum overlap
        if len(merged_df) == 0:
            raise DataError("No overlapping timestamps found between data sources")
        
        self.logger.debug(f"Synchronized {len(dataframes)} dataframes: {len(merged_df)} overlapping timestamps")
        return merged_df
    
    def _apply_rolling_window(self, df: pd.DataFrame, start_date: str) -> pd.DataFrame:
        """
        Apply rolling window filter to DataFrame based on cache_window_days.
        
        Args:
            df: DataFrame with Date column
            start_date: Start date to filter from
            
        Returns:
            Filtered DataFrame
        """
        if 'Date' not in df.columns:
            return df
        
        df['Date'] = pd.to_datetime(df['Date'])
        start_datetime = pd.to_datetime(start_date)
        cutoff_date = start_datetime - timedelta(days=self.cache_window_days)
        
        # Keep only data within the window
        filtered_df = df[df['Date'] >= cutoff_date].copy()
        
        if len(filtered_df) < len(df):
            self.logger.debug(f"Applied rolling window: kept {len(filtered_df)}/{len(df)} rows")
        
        return filtered_df
    
    def _prune_cache(self) -> None:
        """
        Prune old cache files when using rolling_window strategy.
        """
        if not self.cache_enabled or self.cache_dir is None:
            return
        
        if self.cache_strategy != 'rolling_window':
            return
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.cache_window_days)
            
            for cache_file in Path(self.cache_dir).glob("*.csv"):
                # Try to determine file age from metadata
                metadata_file = cache_file.parent / f"{cache_file.stem}_meta.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        cached_at = datetime.fromisoformat(metadata.get('cached_at', ''))
                        if cached_at < cutoff_date:
                            cache_file.unlink()
                            metadata_file.unlink()
                            self.logger.debug(f"Pruned old cache file: {cache_file.name}")
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"Failed to prune cache: {str(e)}")
    
    def _cache_data(self, symbol: str, dataframe: pd.DataFrame, interval: str = '1d') -> None:
        """
        Save data to local cache.
        
        Args:
            symbol: Stock ticker symbol
            dataframe: Data to cache
            interval: Data interval
        """
        if not self.cache_enabled or self.cache_dir is None or self.cache_strategy == 'disabled':
            return
        
        try:
            # Apply rolling window if needed before caching
            if self.cache_strategy == 'rolling_window' and 'Date' in dataframe.columns:
                # Use max date from dataframe as reference
                reference_date = str(dataframe['Date'].max())
                dataframe = self._apply_rolling_window(dataframe.copy(), reference_date)
            
            cache_file = Path(self.cache_dir) / f"{symbol}_{interval}.csv"
            dataframe.to_csv(cache_file, index=False)
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'interval': interval,
                'cached_at': datetime.now().isoformat(),
                'rows': len(dataframe),
                'start_date': str(dataframe['Date'].min()),
                'end_date': str(dataframe['Date'].max()),
                'cache_strategy': self.cache_strategy
            }
            metadata_file = Path(self.cache_dir) / f"{symbol}_{interval}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Cached data for {symbol} to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache data for {symbol}: {str(e)}")
    
    def _cache_multi_source_data(self, cache_key: str, dataframe: pd.DataFrame, 
                                 interval: str, symbols: List[str]) -> None:
        """
        Cache multi-source merged data.
        
        Args:
            cache_key: Cache key (e.g., 'QQQ_SPY_VXX')
            dataframe: Merged DataFrame to cache
            interval: Data interval
            symbols: List of symbols in the merged data
        """
        if not self.cache_enabled or self.cache_dir is None or self.cache_strategy == 'disabled':
            return
        
        try:
            # Apply rolling window if needed
            if self.cache_strategy == 'rolling_window' and 'Date' in dataframe.columns:
                # Use max date from dataframe as reference
                reference_date = str(dataframe['Date'].max())
                dataframe = self._apply_rolling_window(dataframe.copy(), reference_date)
            
            cache_file = Path(self.cache_dir) / f"{cache_key}_{interval}.csv"
            dataframe.to_csv(cache_file, index=False)
            
            # Save metadata
            metadata = {
                'symbols': symbols,
                'cache_key': cache_key,
                'interval': interval,
                'cached_at': datetime.now().isoformat(),
                'rows': len(dataframe),
                'start_date': str(dataframe['Date'].min()),
                'end_date': str(dataframe['Date'].max()),
                'cache_strategy': self.cache_strategy
            }
            metadata_file = Path(self.cache_dir) / f"{cache_key}_{interval}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Cached multi-source data to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache multi-source data: {str(e)}")
    
    def _load_cached_multi_source_data(self, cache_key: str, interval: str,
                                      max_age_days: int = 1, start_date: str = None,
                                      end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load cached multi-source data.
        
        Args:
            cache_key: Cache key (e.g., 'QQQ_SPY_VXX')
            interval: Data interval
            max_age_days: Maximum age of cache in days
            
        Returns:
            Cached DataFrame or None
        """
        if not self.cache_enabled or self.cache_dir is None:
            return None
        
        try:
            cache_file = Path(self.cache_dir) / f"{cache_key}_{interval}.csv"
            metadata_file = Path(self.cache_dir) / f"{cache_key}_{interval}_meta.json"
            
            if not cache_file.exists() or not metadata_file.exists():
                return None
            
            # Check cache age
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cached_at = datetime.fromisoformat(metadata['cached_at'])
            age = datetime.now() - cached_at
            
            if age.days > max_age_days:
                self.logger.debug(f"Cache for {cache_key} is stale ({age.days} days old)")
                return None
            
            # Load cached data
            df = pd.read_csv(cache_file)
            df = self._normalize_date_column(df)
            
            # Apply rolling window if needed
            if self.cache_strategy == 'rolling_window' and start_date:
                reference_date = start_date
            else:
                reference_date = datetime.now().strftime('%Y-%m-%d')
            if self.cache_strategy == 'rolling_window':
                df = self._apply_rolling_window(df, reference_date)
            
            self.logger.debug(f"Loaded cached multi-source data ({len(df)} rows)")
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached multi-source data: {str(e)}")
            return None
    
    def _load_cached_data(self, symbol: str, interval: str = '1d', 
                         max_age_days: int = 1, start_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and fresh.
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval
            max_age_days: Maximum age of cache in days
            
        Returns:
            Cached DataFrame or None if not available/stale
        """
        if not self.cache_enabled or self.cache_dir is None or self.cache_strategy == 'disabled':
            return None
        
        try:
            cache_file = Path(self.cache_dir) / f"{symbol}_{interval}.csv"
            metadata_file = Path(self.cache_dir) / f"{symbol}_{interval}_meta.json"
            
            if not cache_file.exists() or not metadata_file.exists():
                return None
            
            # Check cache age
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cached_at = datetime.fromisoformat(metadata['cached_at'])
            age = datetime.now() - cached_at
            
            if age.days > max_age_days:
                self.logger.debug(f"Cache for {symbol} is stale ({age.days} days old)")
                return None
            
            # Load cached data
            df = pd.read_csv(cache_file)
            # Normalize Date column
            df = self._normalize_date_column(df)
            
            # Apply rolling window if needed
            if self.cache_strategy == 'rolling_window':
                # Use requested start_date or current date as reference
                if start_date:
                    reference_date = start_date
                else:
                    reference_date = datetime.now().strftime('%Y-%m-%d')
                df = self._apply_rolling_window(df, reference_date)
            
            self.logger.debug(f"Loaded cached data for {symbol} ({len(df)} rows)")
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached data for {symbol}: {str(e)}")
            return None
    
    def _normalize_date_column(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the datetime column to 'Date' if it exists.
        
        Args:
            dataframe: DataFrame with datetime index reset
            
        Returns:
            DataFrame with normalized 'Date' column
        """
        # Check if 'Date' column already exists
        if 'Date' in dataframe.columns:
            # Ensure it's datetime type
            dataframe['Date'] = pd.to_datetime(dataframe['Date'])
            return dataframe
        
        # Find first datetime-like column
        for col in dataframe.columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                # Rename to 'Date'
                dataframe = dataframe.rename(columns={col: 'Date'})
                return dataframe
        
        # If no datetime column found, create one from index if datetime
        if pd.api.types.is_datetime64_any_dtype(dataframe.index):
            dataframe['Date'] = dataframe.index
            dataframe = dataframe.reset_index(drop=True)
            return dataframe
        
        # If still no Date column, log warning
        self.logger.warning("No datetime column found after reset_index, Date column may be missing")
        return dataframe
    
    def validate_data(self, dataframe: pd.DataFrame) -> None:
        """
        Check data quality and completeness.
        
        Args:
            dataframe: DataFrame to validate
            
        Raises:
            DataError: If validation fails
        """
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        
        if missing_columns:
            raise DataError(
                f"Missing required columns: {missing_columns}",
                details={'columns': list(dataframe.columns)}
            )
        
        # Check for Date column
        if 'Date' not in dataframe.columns:
            raise DataError(
                "Missing 'Date' column. Ensure datetime column is normalized.",
                details={'columns': list(dataframe.columns)}
            )
        
        # Ensure Date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(dataframe['Date']):
            try:
                dataframe['Date'] = pd.to_datetime(dataframe['Date'])
            except Exception as e:
                raise DataError(
                    f"Date column is not datetime-compatible: {str(e)}",
                    details={'date_column': dataframe['Date'].dtype}
                )
        
        # Check for empty dataframe
        if dataframe.empty:
            raise DataError("DataFrame is empty")
        
        # Check for missing values
        missing_counts = dataframe[required_columns].isnull().sum()
        if missing_counts.any():
            self.logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check for invalid values
        if (dataframe[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            self.logger.warning("Invalid price values (<=0) detected in data")
        
        self.logger.debug(f"Data validation passed: {len(dataframe)} rows, {len(dataframe.columns)} columns")

