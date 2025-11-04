"""
Unit Tests for Data Collection and Processing Modules

Tests stock data fetching, feature engineering, and preprocessing.
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.data import StockDataFetcher, FeatureEngineer, DataPreprocessor
from trading_bot.utils.exceptions import DataError, ValidationError


class TestStockDataFetcher(unittest.TestCase):
    """Test cases for StockDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'data.cache_historical_data': True,
            'data.historical_data_path': 'data/historical/',
            'data.stock_data_source': 'yfinance'
        }.get(key, default))
        
        self.mock_logger = Mock()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('trading_bot.data.stock_fetcher.get_writable_app_dir')
    def test_initialization(self, mock_get_dir):
        """Test StockDataFetcher initialization."""
        mock_get_dir.return_value = self.temp_dir
        
        fetcher = StockDataFetcher(self.mock_config, self.mock_logger)
        
        self.assertIsNotNone(fetcher)
        self.assertEqual(fetcher.cache_enabled, True)
    
    @patch('trading_bot.data.stock_fetcher.yf')
    @patch('trading_bot.data.stock_fetcher.get_writable_app_dir')
    def test_fetch_historical_data_success(self, mock_get_dir, mock_yf):
        """Test successful historical data fetching."""
        mock_get_dir.return_value = self.temp_dir
        
        # Create mock data
        mock_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [104, 105, 106, 107, 108],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        mock_df.set_index('Date', inplace=True)
        
        mock_ticker = Mock()
        mock_ticker.history = Mock(return_value=mock_df)
        mock_yf.Ticker = Mock(return_value=mock_ticker)
        
        fetcher = StockDataFetcher(self.mock_config, self.mock_logger)
        result = fetcher.fetch_historical_data('AAPL', '2024-01-01', '2024-01-05')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        self.assertIn('Close', result.columns)
    
    @patch('trading_bot.data.stock_fetcher.get_writable_app_dir')
    def test_validate_data_success(self, mock_get_dir):
        """Test data validation with valid data."""
        mock_get_dir.return_value = self.temp_dir
        
        valid_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
        
        fetcher = StockDataFetcher(self.mock_config, self.mock_logger)
        # Should not raise exception
        fetcher.validate_data(valid_df)
    
    @patch('trading_bot.data.stock_fetcher.get_writable_app_dir')
    def test_validate_data_missing_columns(self, mock_get_dir):
        """Test data validation with missing columns."""
        mock_get_dir.return_value = self.temp_dir
        
        invalid_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [104, 105, 106]
        })
        
        fetcher = StockDataFetcher(self.mock_config, self.mock_logger)
        
        with self.assertRaises(DataError):
            fetcher.validate_data(invalid_df)


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'models.features.lagged_prices': [1, 5],
            'models.features.technical_indicators_list': ['SMA_20'],
            'models.features.volatility_window': 20,
            'models.xgboost.lookback_days': 30
        }.get(key, default))
        
        self.mock_logger = Mock()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.uniform(1000, 2000, 100)
        })
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(self.mock_config, self.mock_logger)
        
        self.assertIsNotNone(engineer)
        self.assertEqual(engineer.lagged_prices, [1, 5])
        self.assertEqual(engineer.volatility_window, 20)
    
    def test_create_features(self):
        """Test feature creation."""
        engineer = FeatureEngineer(self.mock_config, self.mock_logger)
        
        result = engineer.create_features(self.sample_data)
        
        self.assertIsNotNone(result)
        # Should have more columns than input
        self.assertGreater(len(result.columns), len(self.sample_data.columns))
        
        # Check for expected features
        self.assertIn('return_1d', result.columns)
        self.assertIn('close_lag_1', result.columns)
    
    def test_add_lagged_features(self):
        """Test lagged feature creation."""
        engineer = FeatureEngineer(self.mock_config, self.mock_logger)
        
        df = self.sample_data.copy()
        df['return_1d'] = df['Close'].pct_change()
        
        result = engineer._add_lagged_features(df, [1, 5])
        
        self.assertIn('close_lag_1', result.columns)
        self.assertIn('close_lag_5', result.columns)
        self.assertIn('return_lag_1', result.columns)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        engineer = FeatureEngineer(self.mock_config, self.mock_logger)
        
        df = self.sample_data.copy()
        df.loc[5, 'Close'] = np.nan
        
        result = engineer.handle_missing_values(df, method='forward_fill')
        
        # Should have no NaN values (or very few at boundaries)
        self.assertLessEqual(result.isnull().sum().sum(), 0)
    
    def test_insufficient_data_error(self):
        """Test error with insufficient data."""
        engineer = FeatureEngineer(self.mock_config, self.mock_logger)
        
        # Create data with too few rows
        small_data = self.sample_data.head(10)
        
        with self.assertRaises(DataError):
            engineer.create_features(small_data)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'models.training.test_size': 0.2,
            'models.training.validation_size': 0.1,
            'models.training.random_state': 42,
            'models.features.scaling_method': 'standard',
            'models.xgboost.target_type': 'regression'
        }.get(key, default))
        
        self.mock_logger = Mock()
        
        # Create sample data
        self.sample_features = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 100),
            'feature2': np.random.uniform(0, 1, 100),
            'Close': np.random.uniform(100, 110, 100)
        })
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(self.mock_config, self.mock_logger)
        
        self.assertIsNotNone(preprocessor)
        self.assertEqual(preprocessor.test_size, 0.2)
        self.assertEqual(preprocessor.scaling_method, 'standard')
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        preprocessor = DataPreprocessor(self.mock_config, self.mock_logger)
        
        X, y = preprocessor.prepare_training_data(self.sample_features, 'Close')
        
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertNotIn('Close', X.columns)
        self.assertEqual(len(X), len(y))
    
    def test_train_test_split(self):
        """Test time-series aware train/test split."""
        preprocessor = DataPreprocessor(self.mock_config, self.mock_logger)
        
        X = self.sample_features.drop('Close', axis=1)
        y = self.sample_features['Close']
        
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
        
        # Check sizes
        self.assertAlmostEqual(len(X_test) / len(X), 0.2, delta=0.05)
        
        # Check no shuffle (last rows should be test set)
        self.assertEqual(len(X), len(X_train) + len(X_test))
    
    def test_fit_and_transform_scaler(self):
        """Test scaler fitting and transformation."""
        preprocessor = DataPreprocessor(self.mock_config, self.mock_logger)
        
        X = self.sample_features.drop('Close', axis=1)
        
        # Fit scaler
        preprocessor.fit_scaler(X)
        self.assertIsNotNone(preprocessor.scaler)
        
        # Transform
        X_scaled = preprocessor.transform(X)
        self.assertEqual(X_scaled.shape, X.shape)
    
    def test_scaler_persistence(self):
        """Test scaler save and load."""
        preprocessor = DataPreprocessor(self.mock_config, self.mock_logger)
        
        X = self.sample_features.drop('Close', axis=1)
        preprocessor.fit_scaler(X)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            scaler_path = f.name
        
        try:
            preprocessor.save_scaler(scaler_path)
            
            # Create new preprocessor and load
            preprocessor2 = DataPreprocessor(self.mock_config, self.mock_logger)
            preprocessor2.load_scaler(scaler_path)
            
            # Verify they produce same output
            X_scaled1 = preprocessor.transform(X)
            X_scaled2 = preprocessor2.transform(X)
            
            np.testing.assert_array_almost_equal(X_scaled1, X_scaled2)
        finally:
            import os
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
    
    def test_create_regression_target(self):
        """Test regression target creation."""
        preprocessor = DataPreprocessor(self.mock_config, self.mock_logger)
        
        df = self.sample_features.copy()
        target = preprocessor.create_regression_target(df, 'Close', shift=-1)
        
        self.assertEqual(len(target), len(df))
        # Last value should be NaN (no next day price)
        self.assertTrue(pd.isna(target.iloc[-1]))
    
    def test_create_classification_target(self):
        """Test classification target creation."""
        preprocessor = DataPreprocessor(self.mock_config, self.mock_logger)
        
        df = self.sample_features.copy()
        target = preprocessor.create_classification_target(df, 'Close')
        
        self.assertEqual(len(target), len(df))
        # Should contain only 0 and 1 (and possibly NaN)
        unique_values = target.dropna().unique()
        self.assertTrue(all(v in [0, 1] for v in unique_values))


if __name__ == '__main__':
    unittest.main()

