"""
Stock Data Fetcher Module

Fetches historical and real-time stock market data using yfinance.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
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
            self.logger.info(f"Stock data cache enabled at: {self.cache_dir}")
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
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Date
            
        Raises:
            DataError: If data fetching fails
        """
        try:
            import yfinance as yf
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise DataError(
                    f"No data retrieved for symbol {symbol}",
                    symbol=symbol,
                    details={'start_date': start_date, 'end_date': end_date}
                )
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Normalize datetime column name to 'Date'
            df = self._normalize_date_column(df)
            
            # Validate data
            self.validate_data(df)
            
            # Cache the data
            if self.cache_enabled:
                self._cache_data(symbol, df, interval)
            
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
    
    def fetch_latest_data(self, symbol: str, period: str = '1d') -> pd.DataFrame:
        """
        Get recent data for real-time predictions.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, etc.)
            
        Returns:
            DataFrame with recent OHLCV data
            
        Raises:
            DataError: If data fetching fails
        """
        try:
            import yfinance as yf
            
            self.logger.debug(f"Fetching latest data for {symbol} (period: {period})")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
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
    
    def _cache_data(self, symbol: str, dataframe: pd.DataFrame, interval: str = '1d') -> None:
        """
        Save data to local cache.
        
        Args:
            symbol: Stock ticker symbol
            dataframe: Data to cache
            interval: Data interval
        """
        if not self.cache_enabled or self.cache_dir is None:
            return
        
        try:
            cache_file = Path(self.cache_dir) / f"{symbol}_{interval}.csv"
            dataframe.to_csv(cache_file, index=False)
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'interval': interval,
                'cached_at': datetime.now().isoformat(),
                'rows': len(dataframe),
                'start_date': str(dataframe['Date'].min()),
                'end_date': str(dataframe['Date'].max())
            }
            metadata_file = Path(self.cache_dir) / f"{symbol}_{interval}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Cached data for {symbol} to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache data for {symbol}: {str(e)}")
    
    def _load_cached_data(self, symbol: str, interval: str = '1d', 
                         max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and fresh.
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval
            max_age_days: Maximum age of cache in days
            
        Returns:
            Cached DataFrame or None if not available/stale
        """
        if not self.cache_enabled or self.cache_dir is None:
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

