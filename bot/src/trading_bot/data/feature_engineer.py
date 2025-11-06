"""
Feature Engineering Module

Transforms raw OHLCV data into ML-ready features with technical indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path

from trading_bot.utils.exceptions import DataError
from trading_bot.data.alpha_features import AlphaFeatures


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
        
        # Multi-source and sentiment configuration
        self.multi_source_enabled = config.get('models.features.multi_source_enabled', False)
        self.sentiment_enabled = config.get('models.features.sentiment_enabled', False)
        self.sequence_length = config.get('models.features.sequence_length', 60)
        self.cross_asset_features = config.get('models.features.cross_asset_features', 
                                              ['price_ratio', 'correlation', 'relative_strength'])
        self.sentiment_features = config.get('models.features.sentiment_features',
                                           ['sentiment_score', 'sentiment_momentum', 'sentiment_divergence'])
        
        # Alpha features configuration
        self.use_alpha_features = config.get('models.features.use_alpha_features', False)
        self.alpha_feature_groups = config.get('models.features.alpha_feature_groups', [])
        self.market_benchmark = config.get('models.features.market_benchmark', 'SPY')
        self.sector_etfs = config.get('models.features.sector_etfs', {})
        
        # Conditionally import and instantiate AlphaFeatures
        if self.use_alpha_features:
            try:
                from trading_bot.data.alpha_features import AlphaFeatures
                self.alpha_features = AlphaFeatures(config, logger)
                self.logger.info(f"Alpha features enabled with groups: {self.alpha_feature_groups}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AlphaFeatures: {str(e)}")
                self.use_alpha_features = False
                self.alpha_features = None
        else:
            self.alpha_features = None
        
        self.logger.info(f"Feature engineer initialized with {len(self.technical_indicators)} indicators")
    
    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw OHLCV data.
        
        Automatically detects if input is multi-source (has symbol-prefixed columns)
        and routes to appropriate feature creation method.
        
        Args:
            dataframe: DataFrame with OHLCV data (single or multi-source)
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            DataError: If insufficient data or feature creation fails
        """
        try:
            df = dataframe.copy()
            
            # Detect if multi-source data
            is_multi_source = self._is_multi_source_data(df)
            
            if is_multi_source and self.multi_source_enabled:
                # Extract source symbols from column names
                source_symbols = self._extract_source_symbols(df)
                self.logger.info(f"Detected multi-source data with symbols: {source_symbols}")
                return self.create_multi_source_features(df, source_symbols)
            
            # Single-source feature creation
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
            
            # Add alpha features if enabled
            if self.use_alpha_features:
                df = self.create_alpha_features(df, market_data=None)
            
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
    
    def _is_multi_source_data(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame contains multi-source data (symbol-prefixed columns).
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if multi-source data detected
        """
        # Check for symbol-prefixed columns (e.g., QQQ_Close, SPY_Volume)
        for col in df.columns:
            if col != 'Date' and '_' in col:
                parts = col.split('_')
                if len(parts) >= 2 and parts[0].isupper() and len(parts[0]) <= 5:
                    # Likely symbol prefix (e.g., QQQ, SPY)
                    return True
        return False
    
    def _extract_source_symbols(self, df: pd.DataFrame) -> List[str]:
        """
        Extract source symbols from column names.
        
        Args:
            df: DataFrame with symbol-prefixed columns
            
        Returns:
            List of unique source symbols
        """
        symbols = set()
        for col in df.columns:
            if col != 'Date' and '_' in col:
                parts = col.split('_')
                if len(parts) >= 2 and parts[0].isupper() and len(parts[0]) <= 5:
                    symbols.add(parts[0])
        return sorted(list(symbols))
    
    def create_multi_source_features(self, dataframe: pd.DataFrame, 
                                    source_symbols: List[str]) -> pd.DataFrame:
        """
        Create features for multi-source data.
        
        Args:
            dataframe: DataFrame with symbol-prefixed columns
            source_symbols: List of source symbols (e.g., ['QQQ', 'SPY', 'VXX'])
            
        Returns:
            DataFrame with multi-source features
        """
        df = dataframe.copy()
        
        # Validate input
        if len(df) < self.lookback_days:
            raise DataError(
                f"Insufficient data: need {self.lookback_days} periods, got {len(df)}",
                details={'required': self.lookback_days, 'available': len(df)}
            )
        
        self.logger.info(f"Creating multi-source features for {len(df)} rows with symbols: {source_symbols}")
        
        # Create per-symbol technical indicators
        for symbol in source_symbols:
            close_col = f"{symbol}_Close"
            if close_col in df.columns:
                # Add technical indicators for this symbol
                df = self._add_symbol_technical_indicators(df, symbol, close_col)
        
        # Add cross-asset features
        if 'price_ratio' in self.cross_asset_features:
            df = self._add_price_ratio_features(df, source_symbols)
        
        if 'correlation' in self.cross_asset_features:
            df = self._add_correlation_features(df, source_symbols)
        
        if 'relative_strength' in self.cross_asset_features:
            df = self._add_relative_strength_features(df, source_symbols)
        
        # Handle missing values
        df = self.handle_missing_values(df, method='forward_fill')
        
        self.logger.info(f"Multi-source feature engineering complete: {len(df.columns)} features created")
        return df
    
    def _add_symbol_technical_indicators(self, df: pd.DataFrame, symbol: str, 
                                        close_col: str) -> pd.DataFrame:
        """
        Add technical indicators for a specific symbol.
        
        Args:
            df: DataFrame with symbol-prefixed columns
            symbol: Symbol name (e.g., 'QQQ')
            close_col: Close price column name (e.g., 'QQQ_Close')
            
        Returns:
            DataFrame with added indicators
        """
        try:
            import ta
            
            for indicator in self.technical_indicators:
                if indicator.startswith('SMA_'):
                    window = int(indicator.split('_')[1])
                    df[f"{symbol}_{indicator}"] = ta.trend.sma_indicator(df[close_col], window=window)
                
                elif indicator.startswith('EMA_'):
                    window = int(indicator.split('_')[1])
                    df[f"{symbol}_{indicator}"] = ta.trend.ema_indicator(df[close_col], window=window)
                
                elif indicator.startswith('RSI_'):
                    window = int(indicator.split('_')[1])
                    df[f"{symbol}_{indicator}"] = ta.momentum.rsi(df[close_col], window=window)
                
                elif indicator == 'MACD':
                    macd = ta.trend.MACD(df[close_col])
                    df[f"{symbol}_MACD"] = macd.macd()
                    df[f"{symbol}_MACD_signal"] = macd.macd_signal()
                    df[f"{symbol}_MACD_diff"] = macd.macd_diff()
                
                elif indicator.startswith('BB_'):
                    bb = ta.volatility.BollingerBands(df[close_col])
                    if indicator == 'BB_upper':
                        df[f"{symbol}_BB_upper"] = bb.bollinger_hband()
                    elif indicator == 'BB_lower':
                        df[f"{symbol}_BB_lower"] = bb.bollinger_lband()
                    elif indicator == 'BB_middle':
                        df[f"{symbol}_BB_middle"] = bb.bollinger_mavg()
            
            return df
            
        except ImportError:
            self.logger.warning("ta library not installed, skipping technical indicators")
            return df
        except Exception as e:
            self.logger.warning(f"Error adding technical indicators for {symbol}: {str(e)}")
            return df
    
    def _add_price_ratio_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Add price ratio features between symbols (e.g., QQQ/SPY).
        
        Args:
            df: DataFrame with symbol-prefixed columns
            symbols: List of symbols
            
        Returns:
            DataFrame with price ratio features
        """
        if len(symbols) < 2:
            return df
        
        # Calculate ratios between pairs
        primary_symbol = symbols[0]  # Typically QQQ
        primary_close = f"{primary_symbol}_Close"
        
        if primary_close not in df.columns:
            return df
        
        for symbol in symbols[1:]:
            other_close = f"{symbol}_Close"
            if other_close in df.columns:
                ratio_col = f"{primary_symbol}_{symbol}_ratio"
                df[ratio_col] = df[primary_close] / df[other_close]
                # Add ratio change
                df[f"{ratio_col}_change"] = df[ratio_col].pct_change()
        
        return df
    
    def _add_correlation_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Add rolling correlation features between symbols.
        
        Args:
            df: DataFrame with symbol-prefixed columns
            symbols: List of symbols
            
        Returns:
            DataFrame with correlation features
        """
        if len(symbols) < 2:
            return df
        
        window = min(20, len(df) // 4)  # Use smaller window for correlations
        if window < 5:
            return df
        
        # Calculate returns for each symbol
        returns_cols = []
        for symbol in symbols:
            close_col = f"{symbol}_Close"
            if close_col in df.columns:
                returns_col = f"{symbol}_returns"
                df[returns_col] = df[close_col].pct_change()
                returns_cols.append(returns_col)
        
        # Calculate rolling correlations between pairs
        if len(returns_cols) >= 2:
            primary_returns = returns_cols[0]
            for returns_col in returns_cols[1:]:
                corr_col = f"{returns_cols[0].replace('_returns', '')}_{returns_col.replace('_returns', '')}_corr"
                df[corr_col] = df[primary_returns].rolling(window=window).corr(df[returns_col])
        
        return df
    
    def _add_relative_strength_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Add relative strength features.
        
        Args:
            df: DataFrame with symbol-prefixed columns
            symbols: List of symbols
            
        Returns:
            DataFrame with relative strength features
        """
        if len(symbols) < 2:
            return df
        
        primary_symbol = symbols[0]
        primary_close = f"{primary_symbol}_Close"
        
        if primary_close not in df.columns:
            return df
        
        window = min(20, len(df) // 4)
        if window < 5:
            return df
        
        # Calculate relative strength vs other symbols
        for symbol in symbols[1:]:
            other_close = f"{symbol}_Close"
            if other_close in df.columns:
                # Relative strength (RS) = (Primary Price / Other Price) / SMA(Primary Price / Other Price)
                ratio = df[primary_close] / df[other_close]
                rs_sma = ratio.rolling(window=window).mean()
                df[f"{primary_symbol}_{symbol}_relative_strength"] = ratio / rs_sma
        
        return df
    
    def add_sentiment_features(self, dataframe: pd.DataFrame, sentiment_data: Dict) -> pd.DataFrame:
        """
        Add sentiment features to the feature DataFrame.
        
        Args:
            dataframe: DataFrame with features
            sentiment_data: Dictionary with sentiment data (from SentimentDataFetcher)
            
        Returns:
            DataFrame with sentiment features added
        """
        df = dataframe.copy()
        
        if not sentiment_data:
            self.logger.warning("No sentiment data provided, skipping sentiment features")
            return df
        
        # Ensure Date column exists and is datetime
        if 'Date' not in df.columns:
            raise DataError("Date column required for sentiment feature integration")
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check if multi-source data
        is_multi_source = self._is_multi_source_data(df)
        
        # Determine close column for divergence calculation
        if is_multi_source:
            # Use first symbol's close column
            source_symbols = self._extract_source_symbols(df)
            if source_symbols:
                close_col = f"{source_symbols[0]}_Close"
            else:
                close_col = None
        else:
            close_col = 'Close'
        
        # Handle time alignment with sentiment timestamp if present
        sentiment_timestamp = sentiment_data.get('timestamp')
        if sentiment_timestamp:
            if isinstance(sentiment_timestamp, str):
                sentiment_timestamp = pd.to_datetime(sentiment_timestamp)
            elif isinstance(sentiment_timestamp, datetime):
                sentiment_timestamp = pd.to_datetime(sentiment_timestamp)
            
            # Initialize sentiment columns with NaN before timestamp
            sentiment_score_series = pd.Series(index=df.index, dtype=float)
            # Set sentiment score starting from timestamp, NaN before
            sentiment_score_series.loc[df['Date'] >= sentiment_timestamp] = sentiment_data.get('sentiment_score', 0.0)
            # Forward fill after timestamp (carry sentiment forward in time)
            sentiment_score_series = sentiment_score_series.fillna(method='ffill')
            # Fill remaining NaN (before first timestamp) with 0
            sentiment_score_series = sentiment_score_series.fillna(0.0)
        else:
            # No timestamp, use constant value
            sentiment_score_series = pd.Series(sentiment_data.get('sentiment_score', 0.0), index=df.index)
        
        # Add raw sentiment score
        if 'sentiment_score' in self.sentiment_features:
            df['sentiment_score'] = sentiment_score_series.values
        
        # Add sentiment momentum (change over time)
        if 'sentiment_momentum' in self.sentiment_features:
            if 'sentiment_score' in df.columns:
                df['sentiment_momentum'] = df['sentiment_score'].diff()
            else:
                df['sentiment_momentum'] = 0.0
        
        # Add sentiment divergence (sentiment vs price direction)
        if 'sentiment_divergence' in self.sentiment_features:
            if 'sentiment_score' in df.columns and close_col and close_col in df.columns:
                # Calculate price direction
                price_direction = df[close_col].pct_change().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                # Sentiment divergence = sentiment_score * price_direction (negative = divergence)
                df['sentiment_divergence'] = df['sentiment_score'] * price_direction
            else:
                df['sentiment_divergence'] = 0.0
        
        # Forward fill missing sentiment values (since sentiment is updated less frequently)
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        if sentiment_cols:
            df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill')
            df[sentiment_cols] = df[sentiment_cols].fillna(0)  # Fill remaining NaN with 0
        
        self.logger.info(f"Added sentiment features: {sentiment_cols}")
        return df
    
    def prepare_sequences(self, dataframe: pd.DataFrame, sequence_length: int = None,
                         return_labels: bool = False, target_column: str = 'Close') -> np.ndarray:
        """
        Prepare sequences for LSTM/CNN-LSTM input.
        
        Args:
            dataframe: DataFrame with features
            sequence_length: Length of each sequence (default: from config)
            return_labels: Whether to return labels along with sequences
            target_column: Column to use as target (for labels)
            
        Returns:
            3D numpy array of shape (n_samples, sequence_length, n_features) or
            tuple of (sequences, labels) if return_labels=True
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Remove Date column and target column if present
        feature_df = dataframe.copy()
        exclude_cols = ['Date']
        if target_column in feature_df.columns:
            if return_labels:
                labels = feature_df[target_column].values
            exclude_cols.append(target_column)
        else:
            labels = None
        
        # Get feature columns
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        feature_data = feature_df[feature_cols].values
        
        # Create sequences
        sequences = []
        sequence_labels = []
        
        for i in range(len(feature_data) - sequence_length + 1):
            seq = feature_data[i:i + sequence_length]
            sequences.append(seq)
            if return_labels and labels is not None:
                # Use next value as label
                sequence_labels.append(labels[i + sequence_length - 1])
        
        sequences_array = np.array(sequences)
        
        self.logger.info(f"Prepared {len(sequences)} sequences of length {sequence_length}")
        
        if return_labels and labels is not None:
            return sequences_array, np.array(sequence_labels)
        
        return sequences_array
    
    def scale_sequences(self, sequences: np.ndarray, scaler) -> np.ndarray:
        """
        Scale sequences using a fitted scaler.
        
        Args:
            sequences: 3D array of shape (n_samples, sequence_length, n_features)
            scaler: Fitted scaler (e.g., StandardScaler, MinMaxScaler)
            
        Returns:
            Scaled sequences with same shape
        """
        original_shape = sequences.shape
        n_samples, sequence_length, n_features = original_shape
        
        # Reshape to 2D for scaling
        sequences_2d = sequences.reshape(-1, n_features)
        
        # Scale
        sequences_scaled = scaler.transform(sequences_2d)
        
        # Reshape back to 3D
        sequences_scaled = sequences_scaled.reshape(original_shape)
        
        self.logger.debug(f"Scaled sequences: shape {original_shape}")
        return sequences_scaled
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature column names.
        
        Returns:
            List of feature names
        """
        # Base feature names
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
        
        # Add sentiment features if enabled
        if self.sentiment_enabled:
            feature_names.extend(self.sentiment_features)
        
        # Add alpha feature names if enabled
        if self.use_alpha_features and self.alpha_features is not None:
            alpha_feature_names = self.alpha_features.get_feature_names(self.alpha_feature_groups)
            feature_names.extend(alpha_feature_names)
        
        return feature_names
    
    def create_alpha_features(self, df: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create alpha features and merge with existing features.
        
        Args:
            df: DataFrame with existing features
            market_data: Optional DataFrame with market data (SPY, sector ETFs)
        
        Returns:
            DataFrame with alpha features added
        """
        if not self.use_alpha_features:
            return df
        
        if self.alpha_features is None:
            self.logger.warning("Alpha features enabled but not initialized, skipping")
            return df
        
        try:
            # Create alpha features
            alpha_features_df = self.alpha_features.create_all_features(
                df, market_data, self.alpha_feature_groups
            )
            
            # Merge: alpha features may have overlapping columns, so we need to handle that
            # Get only the new alpha feature columns
            existing_cols = set(df.columns)
            new_alpha_cols = [col for col in alpha_features_df.columns if col not in existing_cols]
            
            if new_alpha_cols:
                # Merge new alpha features
                for col in new_alpha_cols:
                    if col in alpha_features_df.columns:
                        df[col] = alpha_features_df[col]
                
                self.logger.info(f"Added {len(new_alpha_cols)} alpha features")
            else:
                self.logger.debug("No new alpha features to add")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating alpha features: {str(e)}")
            # Return original df on error (graceful degradation)
            return df
    
    def create_features_with_market_data(self, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features with market data for relative strength features.
        
        This method should be used when SPY/sector ETF data is available
        to enable relative strength alpha features.
        
        Args:
            df: DataFrame with OHLCV data
            market_data: DataFrame with market data (SPY, sector ETFs)
        
        Returns:
            DataFrame with engineered features including alpha features
        """
        try:
            # Start with standard feature creation
            df = self.create_features(df)
            
            # Add alpha features with market data
            if self.use_alpha_features:
                df = self.create_alpha_features(df, market_data)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature creation with market data failed: {str(e)}")
            raise DataError(f"Feature creation with market data failed: {str(e)}")
    
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
            'multi_source_enabled': self.multi_source_enabled,
            'sentiment_enabled': self.sentiment_enabled,
            'sequence_length': self.sequence_length,
            'cross_asset_features': self.cross_asset_features,
            'sentiment_features': self.sentiment_features,
            'use_alpha_features': self.use_alpha_features,
            'alpha_feature_groups': self.alpha_feature_groups,
            'market_benchmark': self.market_benchmark,
            'sector_etfs': self.sector_etfs,
            'feature_names': self.get_feature_names()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Feature configuration saved to {filepath}")

