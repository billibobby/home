"""
Alpha Features Module

Creates advanced alpha-generating features for stock price prediction.
Includes microstructure, regime, momentum, volume, relative strength, and time-based features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import functools
from datetime import datetime, timedelta
import time

from trading_bot.utils.exceptions import DataError


class AlphaFeatures:
    """
    Creates advanced alpha-generating features from OHLCV data.
    
    Feature groups:
    - Microstructure: Spread, VWAP, price impact, order flow
    - Regime: Volatility regime, trend strength, Hurst exponent
    - Momentum: Multi-timeframe momentum and acceleration
    - Volume: OBV, volume-price trend, volume divergence
    - Relative Strength: Market-relative and sector-relative performance
    - Time-Based: Day of week, month-end effects
    """
    
    def __init__(self, config, logger):
        """
        Initialize the alpha features generator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Load configuration
        self.market_benchmark = config.get('models.features.market_benchmark', 'SPY')
        self.sector_etfs = config.get('models.features.sector_etfs', {})
        
        # Initialize cache for expensive calculations
        self._hurst_cache = {}
        # Enable profiling (can be disabled for production)
        self.enable_profiling = config.get('models.features.enable_profiling', False)
        
        self.logger.info("AlphaFeatures initialized")
    
    def create_all_features(self, df: pd.DataFrame, market_data: Optional[pd.DataFrame] = None,
                           feature_groups: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create all alpha features based on enabled feature groups.
        
        Args:
            df: DataFrame with OHLCV data
            market_data: Optional DataFrame with market data (SPY, sector ETFs)
            feature_groups: List of feature groups to create. If None, creates all groups.
                           Options: ['microstructure', 'regime', 'momentum', 'volume', 
                                   'relative_strength', 'time_based']
        
        Returns:
            DataFrame with alpha features added
        """
        if feature_groups is None:
            feature_groups = ['microstructure', 'regime', 'momentum', 'volume', 
                            'relative_strength', 'time_based']
        
        result_df = df.copy()
        all_skipped_features = []
        
        self.logger.info(f"Creating alpha features for groups: {feature_groups}")
        
        total_start_time = time.time() if self.enable_profiling else None
        
        try:
            # Create each feature group with optional profiling
            if 'microstructure' in feature_groups:
                group_start = time.time() if self.enable_profiling else None
                microstructure_features = self.create_microstructure_features(result_df)
                result_df = pd.concat([result_df, microstructure_features], axis=1)
                if self.enable_profiling and group_start:
                    self.logger.debug(f"Microstructure features: {time.time() - group_start:.3f}s")
                self.logger.debug(f"Added {len(microstructure_features.columns)} microstructure features")
            
            if 'regime' in feature_groups:
                group_start = time.time() if self.enable_profiling else None
                regime_features = self.create_regime_features(result_df)
                result_df = pd.concat([result_df, regime_features], axis=1)
                if self.enable_profiling and group_start:
                    self.logger.debug(f"Regime features: {time.time() - group_start:.3f}s")
                self.logger.debug(f"Added {len(regime_features.columns)} regime features")
            
            if 'momentum' in feature_groups:
                group_start = time.time() if self.enable_profiling else None
                momentum_features = self.create_momentum_features(result_df)
                result_df = pd.concat([result_df, momentum_features], axis=1)
                if self.enable_profiling and group_start:
                    self.logger.debug(f"Momentum features: {time.time() - group_start:.3f}s")
                self.logger.debug(f"Added {len(momentum_features.columns)} momentum features")
            
            if 'volume' in feature_groups:
                group_start = time.time() if self.enable_profiling else None
                volume_features = self.create_volume_features(result_df)
                result_df = pd.concat([result_df, volume_features], axis=1)
                if self.enable_profiling and group_start:
                    self.logger.debug(f"Volume features: {time.time() - group_start:.3f}s")
                self.logger.debug(f"Added {len(volume_features.columns)} volume features")
            
            if 'relative_strength' in feature_groups:
                group_start = time.time() if self.enable_profiling else None
                relative_features = self.create_relative_strength_features(result_df, market_data)
                result_df = pd.concat([result_df, relative_features], axis=1)
                # Check for skipped features (NaN columns)
                skipped_in_relative = [col for col in relative_features.columns 
                                      if relative_features[col].isna().all()]
                all_skipped_features.extend(skipped_in_relative)
                if self.enable_profiling and group_start:
                    self.logger.debug(f"Relative strength features: {time.time() - group_start:.3f}s")
                self.logger.debug(f"Added {len(relative_features.columns)} relative strength features")
            
            if 'time_based' in feature_groups:
                group_start = time.time() if self.enable_profiling else None
                time_features = self.create_time_features(result_df)
                result_df = pd.concat([result_df, time_features], axis=1)
                if self.enable_profiling and group_start:
                    self.logger.debug(f"Time-based features: {time.time() - group_start:.3f}s")
                self.logger.debug(f"Added {len(time_features.columns)} time-based features")
            
            # Handle missing values
            result_df = self._handle_missing_data(result_df)
            
            # Report skipped features summary and performance
            if self.enable_profiling and total_start_time:
                total_elapsed = time.time() - total_start_time
                self.logger.info(
                    f"Alpha feature creation complete in {total_elapsed:.3f}s: "
                    f"{len(result_df.columns)} total columns"
                )
            
            if all_skipped_features:
                self.logger.info(
                    f"Alpha feature creation complete with {len(all_skipped_features)} skipped features: "
                    f"{', '.join(all_skipped_features)}"
                )
            else:
                if not self.enable_profiling:
                    self.logger.info(f"Alpha feature creation complete: {len(result_df.columns)} total columns")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error creating alpha features: {str(e)}")
            # Return original df on error (graceful degradation)
            return df
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create microstructure features.
        
        Features:
        - spread_proxy: (High - Low) / Close
        - vwap: Volume-weighted average price
        - price_impact: |Return| / (Volume / AvgVolume)
        - order_flow_imbalance: (Close - Open) / (High - Low)
        - relative_volume: Volume / SMA(Volume, 20)
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=df.index)
        
        # Spread proxy
        features['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # VWAP (Volume-Weighted Average Price)
        features['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Price impact
        volume_ma = df['Volume'].rolling(20).mean()
        price_impact_denom = df['Volume'] / volume_ma
        price_impact_denom = price_impact_denom.replace(0, np.nan)  # Avoid division by zero
        features['price_impact'] = abs(df['Close'].pct_change()) / price_impact_denom
        
        # Order flow imbalance
        high_low_range = df['High'] - df['Low']
        high_low_range = high_low_range.replace(0, np.nan)  # Handle division by zero
        features['order_flow_imbalance'] = (df['Close'] - df['Open']) / high_low_range
        
        # Relative volume
        features['relative_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        return features
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create regime detection features.
        
        Features:
        - volatility_regime: Binary (0=low, 1=high) based on rolling volatility vs median
        - trend_strength: (SMA_20 / SMA_60) - 1
        - z_score: (Close - SMA_20) / StdDev_20
        - hurst_exponent: R/S analysis over 30-day window
        - adx: Average Directional Index
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=df.index)
        
        # Volatility regime (binary classification)
        rolling_vol = df['Close'].pct_change().rolling(20).std()
        vol_median = rolling_vol.median()
        features['volatility_regime'] = (rolling_vol > vol_median).astype(int)
        
        # Trend strength
        sma_20 = df['Close'].rolling(20).mean()
        sma_60 = df['Close'].rolling(60).mean()
        features['trend_strength'] = (sma_20 / sma_60) - 1
        
        # Z-score (mean reversion signal)
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        std_20 = std_20.replace(0, np.nan)  # Avoid division by zero
        features['z_score'] = (df['Close'] - sma_20) / std_20
        
        # Hurst exponent (cached calculation with optimized rolling)
        start_time = time.time() if self.enable_profiling else None
        hurst_values = []
        
        # Use vectorized approach where possible
        # For rolling windows, we still need to calculate per window due to R/S analysis complexity
        # But we can optimize by memoizing window segments
        close_values = df['Close'].values
        
        for i in range(len(df)):
            if i < 30:
                hurst_values.append(np.nan)
            else:
                # Extract window (30 days)
                window_start = i - 29
                window_end = i + 1
                window_data = close_values[window_start:window_end]
                
                # Create cache key from window data hash
                # Use tuple of first/last few values + length for faster hashing
                cache_key = (tuple(window_data[:3]), tuple(window_data[-3:]), len(window_data))
                
                if cache_key in self._hurst_cache:
                    hurst = self._hurst_cache[cache_key]
                else:
                    hurst = self._calculate_hurst_exponent(tuple(window_data))
                    # Limit cache size to prevent memory issues
                    if len(self._hurst_cache) < 1000:
                        self._hurst_cache[cache_key] = hurst
                
                hurst_values.append(hurst)
        
        features['hurst_exponent'] = hurst_values
        
        if self.enable_profiling and start_time:
            elapsed = time.time() - start_time
            self.logger.debug(f"Hurst exponent calculation took {elapsed:.3f}s for {len(df)} rows")
        
        # ADX (Average Directional Index)
        try:
            import ta
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            features['adx'] = adx.adx()
        except (ImportError, Exception) as e:
            self.logger.warning(f"Could not calculate ADX: {str(e)}")
            features['adx'] = np.nan
        
        return features
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum features.
        
        Features:
        - momentum_Nd: Close.pct_change(N) for N in [5, 10, 20, 60]
        - momentum_rank_Nd: Percentile rank over 252-day window
        - acceleration: momentum_10d.diff()
        - rsi_divergence: Detect when price makes new high but RSI doesn't
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)
        
        # Multi-timeframe momentum
        for period in [5, 10, 20, 60]:
            momentum_col = f'momentum_{period}d'
            features[momentum_col] = df['Close'].pct_change(period)
            
            # Momentum rank (percentile rank over 252-day window)
            rank_col = f'momentum_rank_{period}d'
            if len(df) >= 252:
                features[rank_col] = features[momentum_col].rolling(252).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 252 else np.nan,
                    raw=False
                )
            else:
                features[rank_col] = np.nan
        
        # Acceleration (second derivative of price)
        if 'momentum_10d' in features.columns:
            features['acceleration'] = features['momentum_10d'].diff()
        
        # RSI divergence detection
        try:
            import ta
            rsi = ta.momentum.RSIIndicator(df['Close'], window=14)
            rsi_values = rsi.rsi()
            
            # Detect bearish divergence: price makes new high but RSI doesn't
            price_highs = df['Close'].rolling(20).max()
            rsi_highs = rsi_values.rolling(20).max()
            
            # Check if current price is near high but RSI is not
            price_near_high = (df['Close'] >= price_highs * 0.99)
            rsi_not_high = (rsi_values < rsi_highs * 0.95)
            features['rsi_divergence'] = (price_near_high & rsi_not_high).astype(int)
        except (ImportError, Exception) as e:
            self.logger.warning(f"Could not calculate RSI divergence: {str(e)}")
            features['rsi_divergence'] = np.nan
        
        return features
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Features:
        - obv: On-Balance Volume (cumulative sum of Volume * sign(Close.diff()))
        - volume_price_trend: Cumulative sum of Volume * Close.pct_change()
        - volume_momentum: Volume.pct_change(5)
        - volume_divergence: Detect price up but volume down
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)
        
        # On-Balance Volume (OBV)
        price_change_sign = np.sign(df['Close'].diff())
        price_change_sign = price_change_sign.replace(0, 1)  # Treat zero change as positive
        features['obv'] = (df['Volume'] * price_change_sign).cumsum()
        
        # Volume-Price Trend (VPT)
        features['volume_price_trend'] = (df['Volume'] * df['Close'].pct_change()).cumsum()
        
        # Volume momentum
        features['volume_momentum'] = df['Volume'].pct_change(5)
        
        # Volume divergence (price up but volume down)
        price_up = df['Close'].pct_change() > 0
        volume_down = df['Volume'].pct_change() < 0
        features['volume_divergence'] = (price_up & volume_down).astype(int)
        
        # Alternative: rolling correlation between price and volume
        price_returns = df['Close'].pct_change()
        volume_returns = df['Volume'].pct_change()
        features['price_volume_correlation'] = price_returns.rolling(20).corr(volume_returns)
        
        return features
    
    def create_relative_strength_features(self, df: pd.DataFrame, 
                                         market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create relative strength features vs market and sector.
        
        Features:
        - relative_to_spy: Stock return - SPY return
        - relative_to_sector: Stock return - Sector ETF return
        - beta_rolling: Rolling 60-day beta with SPY
        - correlation_to_market: Rolling 60-day correlation with SPY
        
        Args:
            df: DataFrame with OHLCV data
            market_data: Optional DataFrame with market data (SPY, sector ETFs)
        
        Returns:
            DataFrame with relative strength features (may contain NaN if data missing)
        """
        features = pd.DataFrame(index=df.index)
        skipped_features = []
        
        # Validate market data
        if market_data is None:
            skipped_features = ['relative_to_spy', 'relative_to_sector', 'beta_rolling', 'correlation_to_market']
            self.logger.warning(
                f"No market data provided, skipping relative strength features: {', '.join(skipped_features)}"
            )
            # Initialize with NaN
            for feat in skipped_features:
                features[feat] = np.nan
            return features
        
        # Try to find SPY data
        spy_close_col = None
        for col in market_data.columns:
            if self.market_benchmark.upper() in col.upper() and 'CLOSE' in col.upper():
                spy_close_col = col
                break
        
        if spy_close_col is None:
            # Also try direct column name match
            if f'{self.market_benchmark}_Close' in market_data.columns:
                spy_close_col = f'{self.market_benchmark}_Close'
            elif 'Close' in market_data.columns and len(market_data.columns) == 1:
                # Single column DataFrame - assume it's the benchmark
                spy_close_col = 'Close'
        
        if spy_close_col is None:
            skipped_features = ['relative_to_spy', 'relative_to_sector', 'beta_rolling', 'correlation_to_market']
            self.logger.warning(
                f"Market benchmark '{self.market_benchmark}' data not found in market_data, "
                f"skipping relative strength features: {', '.join(skipped_features)}"
            )
            # Initialize with NaN
            for feat in skipped_features:
                features[feat] = np.nan
            return features
        
        # Stock returns
        stock_returns = df['Close'].pct_change()
        
        # SPY returns
        spy_returns = market_data[spy_close_col].pct_change()
        
        # Align indices
        if len(stock_returns) != len(spy_returns):
            # Try to align by Date if available
            if 'Date' in df.columns and 'Date' in market_data.columns:
                merged = pd.merge(df[['Date', 'Close']], market_data[['Date', spy_close_col]], 
                                 on='Date', how='inner')
                stock_returns = merged['Close'].pct_change()
                spy_returns = merged[spy_close_col].pct_change()
            else:
                # Take minimum length
                min_len = min(len(stock_returns), len(spy_returns))
                stock_returns = stock_returns.iloc[:min_len]
                spy_returns = spy_returns.iloc[:min_len]
        
        # Relative to SPY
        try:
            features['relative_to_spy'] = stock_returns - spy_returns
        except Exception as e:
            self.logger.warning(f"Failed to compute relative_to_spy: {str(e)}")
            features['relative_to_spy'] = np.nan
            skipped_features.append('relative_to_spy')
        
        # Rolling beta (60-day)
        try:
            if len(stock_returns) >= 60:
                window = 60
                # Calculate rolling beta: beta = cov(stock, spy) / var(spy)
                beta_values = []
                for i in range(len(stock_returns)):
                    if i < window - 1:
                        beta_values.append(np.nan)
                    else:
                        stock_window = stock_returns.iloc[i-window+1:i+1]
                        spy_window = spy_returns.iloc[i-window+1:i+1]
                        if len(stock_window) == window and len(spy_window) == window:
                            covariance = np.cov(stock_window, spy_window)[0, 1]
                            spy_var = np.var(spy_window)
                            if spy_var > 0:
                                beta_values.append(covariance / spy_var)
                            else:
                                beta_values.append(np.nan)
                        else:
                            beta_values.append(np.nan)
                features['beta_rolling'] = beta_values
            else:
                features['beta_rolling'] = np.nan
                skipped_features.append('beta_rolling')
        except Exception as e:
            self.logger.warning(f"Failed to compute beta_rolling: {str(e)}")
            features['beta_rolling'] = np.nan
            skipped_features.append('beta_rolling')
        
        # Rolling correlation with market
        try:
            if len(stock_returns) >= 60:
                aligned_df = pd.DataFrame({'stock': stock_returns, 'spy': spy_returns})
                features['correlation_to_market'] = aligned_df['stock'].rolling(60).corr(aligned_df['spy'])
            else:
                features['correlation_to_market'] = np.nan
                skipped_features.append('correlation_to_market')
        except Exception as e:
            self.logger.warning(f"Failed to compute correlation_to_market: {str(e)}")
            features['correlation_to_market'] = np.nan
            skipped_features.append('correlation_to_market')
        
        # Relative to sector (if sector ETF data available)
        # Try to find sector ETF data
        sector_close_col = None
        for col in market_data.columns:
            for sector_etf in self.sector_etfs.values():
                if sector_etf.upper() in col.upper() and 'CLOSE' in col.upper():
                    sector_close_col = col
                    break
            if sector_close_col:
                break
        
        if sector_close_col:
            try:
                sector_returns = market_data[sector_close_col].pct_change()
                
                # Align indices
                if len(stock_returns) != len(sector_returns):
                    if 'Date' in df.columns and 'Date' in market_data.columns:
                        merged = pd.merge(df[['Date', 'Close']], market_data[['Date', sector_close_col]], 
                                         on='Date', how='inner')
                        stock_returns_aligned = merged['Close'].pct_change()
                        sector_returns_aligned = merged[sector_close_col].pct_change()
                        features['relative_to_sector'] = stock_returns_aligned - sector_returns_aligned
                    else:
                        min_len = min(len(stock_returns), len(sector_returns))
                        features['relative_to_sector'] = (stock_returns.iloc[:min_len] - 
                                                         sector_returns.iloc[:min_len])
                else:
                    features['relative_to_sector'] = stock_returns - sector_returns
            except Exception as e:
                self.logger.warning(f"Failed to compute relative_to_sector: {str(e)}")
                features['relative_to_sector'] = np.nan
                skipped_features.append('relative_to_sector')
        else:
            self.logger.debug("Sector ETF data not found, skipping relative_to_sector feature")
            features['relative_to_sector'] = np.nan
            skipped_features.append('relative_to_sector')
        
        # Log summary of skipped features
        if skipped_features:
            self.logger.info(f"Skipped relative strength features: {', '.join(skipped_features)}")
        
        return features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Features:
        - day_of_week: 0=Monday, 4=Friday
        - hour_of_day: Hour of day (if intraday data)
        - is_month_end: Binary flag for last 3 trading days of month
        - days_to_earnings: Days until next earnings (skipped if unavailable)
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with time-based features
        """
        features = pd.DataFrame(index=df.index)
        
        # Day of week
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
            features['day_of_week'] = dates.dt.dayofweek  # 0=Monday, 6=Sunday
        else:
            features['day_of_week'] = np.nan
        
        # Hour of day (for intraday data)
        if 'Hour' in df.columns:
            features['hour_of_day'] = df['Hour']
        elif 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
            features['hour_of_day'] = dates.dt.hour
        else:
            features['hour_of_day'] = np.nan
        
        # Month-end flag (last 3 trading days of month)
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
            # Get last trading day of each month
            month_ends = dates + pd.offsets.MonthEnd(0)
            # Check if within last 3 trading days
            is_month_end = (month_ends - dates).dt.days <= 3
            features['is_month_end'] = is_month_end.astype(int)
        else:
            features['is_month_end'] = np.nan
        
        # Days to earnings (skipped if earnings data unavailable)
        # This would require earnings calendar data, so we skip it for now
        self.logger.debug("Earnings calendar data not available, skipping days_to_earnings feature")
        features['days_to_earnings'] = np.nan
        
        return features
    
    @functools.lru_cache(maxsize=100)
    def _calculate_hurst_exponent(self, series_tuple: tuple) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        Args:
            series_tuple: Tuple of series values (for caching)
        
        Returns:
            Hurst exponent value
        """
        series = np.array(series_tuple)
        
        if len(series) < 30:
            return np.nan
        
        # R/S analysis
        lags = range(2, len(series) // 2)
        tau = []
        
        for lag in lags:
            # Divide series into blocks
            n_blocks = len(series) // lag
            if n_blocks < 2:
                continue
            
            rs_values = []
            for i in range(n_blocks):
                block = series[i * lag:(i + 1) * lag]
                if len(block) < 2:
                    continue
                
                # Calculate mean-adjusted cumulative sum
                mean_block = np.mean(block)
                cumsum = np.cumsum(block - mean_block)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(block)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.mean(rs_values))
            else:
                tau.append(np.nan)
        
        if len(tau) < 2 or all(np.isnan(tau)):
            return np.nan
        
        # Remove NaN values
        valid_tau = [(np.log(lags[i]), np.log(tau[i])) for i in range(len(tau)) 
                     if not np.isnan(tau[i])]
        
        if len(valid_tau) < 2:
            return np.nan
        
        # Linear regression to find Hurst exponent
        x = [v[0] for v in valid_tau]
        y = [v[1] for v in valid_tau]
        
        if len(x) < 2:
            return np.nan
        
        # Simple linear regression
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return np.nan
        
        hurst = numerator / denominator
        return hurst
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in alpha features.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with missing values handled
        """
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            self.logger.debug(f"Handling {missing_before} missing values in alpha features")
            # Forward fill to avoid data leakage
            df = df.fillna(method='ffill')
            # Fill remaining with 0 for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        missing_after = df.isnull().sum().sum()
        if missing_after > 0:
            self.logger.warning(f"{missing_after} missing values remain after imputation")
        
        return df
    
    def get_feature_names(self, feature_groups: Optional[List[str]] = None) -> List[str]:
        """
        Get list of alpha feature names.
        
        Args:
            feature_groups: List of feature groups. If None, returns all feature names.
        
        Returns:
            List of feature names
        """
        if feature_groups is None:
            feature_groups = ['microstructure', 'regime', 'momentum', 'volume', 
                            'relative_strength', 'time_based']
        
        feature_names = []
        
        if 'microstructure' in feature_groups:
            feature_names.extend(['spread_proxy', 'vwap', 'price_impact', 
                                'order_flow_imbalance', 'relative_volume'])
        
        if 'regime' in feature_groups:
            feature_names.extend(['volatility_regime', 'trend_strength', 'z_score', 
                                'hurst_exponent', 'adx'])
        
        if 'momentum' in feature_groups:
            feature_names.extend([f'momentum_{p}d' for p in [5, 10, 20, 60]])
            feature_names.extend([f'momentum_rank_{p}d' for p in [5, 10, 20, 60]])
            feature_names.extend(['acceleration', 'rsi_divergence'])
        
        if 'volume' in feature_groups:
            feature_names.extend(['obv', 'volume_price_trend', 'volume_momentum', 
                                'volume_divergence', 'price_volume_correlation'])
        
        if 'relative_strength' in feature_groups:
            feature_names.extend(['relative_to_spy', 'relative_to_sector', 
                                'beta_rolling', 'correlation_to_market'])
        
        if 'time_based' in feature_groups:
            feature_names.extend(['day_of_week', 'hour_of_day', 'is_month_end', 
                                'days_to_earnings'])
        
        return feature_names

