"""
Feature Engineering Module

Transforms raw OHLCV data into ML-ready features with technical indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import json
from pathlib import Path

from trading_bot.utils.exceptions import DataError


class FeatureEngineer:
    """
    Creates features from raw stock data for ML models.
    
    Handles lagged features, technical indicators, and volatility metrics.
    Critical: Feature engineering must be identical between training and inference.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Load feature configuration
        self.lagged_prices = config.get('models.features.lagged_prices', [1, 5, 10, 20])
        self.technical_indicators = config.get(
            'models.features.technical_indicators_list',
            ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower']
        )
        self.volatility_window = config.get('models.features.volatility_window', 20)
        self.lookback_days = config.get('models.xgboost.lookback_days', 60)
        
        self.logger.info(f"Feature engineer initialized with {len(self.technical_indicators)} indicators")
    
    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw OHLCV data.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            DataError: If insufficient data or feature creation fails
        """
        try:
            df = dataframe.copy()
            
            # Validate input data
            if len(df) < self.lookback_days:
                raise DataError(
                    f"Insufficient data: need {self.lookback_days} days, got {len(df)}",
                    details={'required': self.lookback_days, 'available': len(df)}
                )
            
            self.logger.info(f"Creating features for {len(df)} rows of data")
            
            # Add price change features
            df = self._add_price_changes(df)
            
            # Add lagged features
            df = self._add_lagged_features(df, self.lagged_prices)
            
            # Add technical indicators
            df = self._add_technical_indicators(df, self.technical_indicators)
            
            # Add volatility features
            df = self._add_volatility_features(df, self.volatility_window)
            
            # Handle missing values
            df = self.handle_missing_values(df, method='forward_fill')
            
            self.logger.info(f"Feature engineering complete: {len(df.columns)} features created")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise DataError(
                f"Feature creation failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _add_price_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price change and return features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with price change features
        """
        # Daily returns
        df['return_1d'] = df['Close'].pct_change()
        
        # Price changes
        df['price_change_1d'] = df['Close'].diff()
        
        # Percentage change from open to close
        df['intraday_change'] = (df['Close'] - df['Open']) / df['Open']
        
        # High-low range
        df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Add lagged price features.
        
        Args:
            df: DataFrame with price data
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        for lag in lags:
            # Lagged close prices
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            
            # Lagged returns
            df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, 
                                  indicators: List[str]) -> pd.DataFrame:
        """
        Compute technical indicators using ta library.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicator names
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            import ta
            
            for indicator in indicators:
                if indicator.startswith('SMA_'):
                    # Simple Moving Average
                    window = int(indicator.split('_')[1])
                    df[indicator] = ta.trend.sma_indicator(df['Close'], window=window)
                
                elif indicator.startswith('EMA_'):
                    # Exponential Moving Average
                    window = int(indicator.split('_')[1])
                    df[indicator] = ta.trend.ema_indicator(df['Close'], window=window)
                
                elif indicator.startswith('RSI_'):
                    # Relative Strength Index
                    window = int(indicator.split('_')[1])
                    df[indicator] = ta.momentum.rsi(df['Close'], window=window)
                
                elif indicator == 'MACD':
                    # MACD
                    macd = ta.trend.MACD(df['Close'])
                    df['MACD'] = macd.macd()
                    df['MACD_signal'] = macd.macd_signal()
                    df['MACD_diff'] = macd.macd_diff()
                
                elif indicator.startswith('BB_'):
                    # Bollinger Bands
                    bb = ta.volatility.BollingerBands(df['Close'])
                    if indicator == 'BB_upper':
                        df['BB_upper'] = bb.bollinger_hband()
                    elif indicator == 'BB_lower':
                        df['BB_lower'] = bb.bollinger_lband()
                    elif indicator == 'BB_middle':
                        df['BB_middle'] = bb.bollinger_mavg()
                
                elif indicator == 'ATR':
                    # Average True Range
                    df['ATR'] = ta.volatility.average_true_range(
                        df['High'], df['Low'], df['Close']
                    )
            
            return df
            
        except ImportError:
            self.logger.warning("ta library not installed, skipping technical indicators")
            return df
        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {str(e)}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Add volatility metrics.
        
        Args:
            df: DataFrame with price data
            window: Rolling window size
            
        Returns:
            DataFrame with volatility features
        """
        # Rolling standard deviation of returns
        df[f'volatility_{window}d'] = df['return_1d'].rolling(window=window).std()
        
        # Rolling standard deviation of close prices
        df[f'price_volatility_{window}d'] = df['Close'].rolling(window=window).std()
        
        # Average True Range (if not already added)
        if 'ATR' not in df.columns:
            try:
                import ta
                df['ATR'] = ta.volatility.average_true_range(
                    df['High'], df['Low'], df['Close'], window=window
                )
            except ImportError:
                pass
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle NaN values in features.
        
        Args:
            df: DataFrame with features
            method: Imputation method ('forward_fill', 'drop', 'zero')
            
        Returns:
            DataFrame with NaN values handled
        """
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            self.logger.debug(f"Handling {missing_before} missing values using {method}")
        
        if method == 'forward_fill':
            # Only forward fill to avoid data leakage in time-series
            # Backward fill removed to prevent using future information
            df = df.fillna(method='ffill')
        elif method == 'drop':
            df = df.dropna()
        elif method == 'zero':
            df = df.fillna(0)
        
        missing_after = df.isnull().sum().sum()
        if missing_after > 0:
            self.logger.warning(f"{missing_after} missing values remain after imputation")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature column names.
        
        Returns:
            List of feature names
        """
        # This should be updated based on actual features created
        feature_names = [
            'return_1d', 'price_change_1d', 'intraday_change', 'high_low_range'
        ]
        
        # Add lagged features
        for lag in self.lagged_prices:
            feature_names.append(f'close_lag_{lag}')
            feature_names.append(f'return_lag_{lag}')
        
        # Add technical indicators
        feature_names.extend(self.technical_indicators)
        
        # Add MACD components if MACD is included
        if 'MACD' in self.technical_indicators:
            feature_names.extend(['MACD_signal', 'MACD_diff'])
        
        # Add volatility features
        feature_names.append(f'volatility_{self.volatility_window}d')
        feature_names.append(f'price_volatility_{self.volatility_window}d')
        
        return feature_names
    
    def save_feature_config(self, filepath: str) -> None:
        """
        Export feature configuration for training/inference consistency.
        
        Args:
            filepath: Path to save configuration JSON
        """
        config = {
            'lagged_prices': self.lagged_prices,
            'technical_indicators': self.technical_indicators,
            'volatility_window': self.volatility_window,
            'lookback_days': self.lookback_days,
            'feature_names': self.get_feature_names()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Feature configuration saved to {filepath}")
    
    def load_feature_config(self, filepath: str) -> Dict:
        """
        Load feature configuration from JSON.
        
        Args:
            filepath: Path to configuration JSON
            
        Returns:
            Feature configuration dictionary
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"Feature configuration loaded from {filepath}")
        return config

