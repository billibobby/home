"""
Unit Tests for Backtesting Framework
"""

import unittest
from unittest.mock import Mock, patch
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.backtesting.walk_forward import WalkForwardBacktest
from trading_bot.backtesting.results import BacktestResults


def create_synthetic_data(n_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='D')
    
    # Generate random walk price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_rows)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, n_rows)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_rows))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_rows))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_rows)
    })
    
    return data


class TestWalkForwardBacktest(unittest.TestCase):
    """Test cases for WalkForwardBacktest."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'backtesting.walk_forward.train_period_days': 252,
            'backtesting.walk_forward.test_period_days': 21,
            'backtesting.walk_forward.step_size_days': 21,
            'backtesting.walk_forward.retrain_frequency_days': 21,
            'backtesting.walk_forward.min_data_points': 252,
            'backtesting.transaction_costs.commission_pct': 0.001,
            'backtesting.transaction_costs.slippage_bps': 5,
            'backtesting.transaction_costs.spread_bps': 2,
            'backtesting.transaction_costs.market_impact_enabled': True,
            'backtesting.transaction_costs.market_impact_coeff': 0.01,
            'backtesting.risk_free_rate': 0.02,
            'models.xgboost.target_type': 'regression',
            'models.features.lagged_prices': [1, 5, 10],
            'models.features.technical_indicators_list': ['SMA_20', 'RSI_14'],
            'models.features.volatility_window': 20,
            'models.xgboost.lookback_days': 60,
            'models.prediction.buy_threshold': 0.6,
            'models.prediction.sell_threshold': 0.6,
            'models.prediction.min_confidence': 0.5,
            'trading.position_size_percentage': 10,
            'models.training.test_size': 0.2,
            'models.training.validation_size': 0.1,
            'models.training.random_state': 42,
            'models.training.cross_validation_folds': 5,
            'models.features.scaling_method': 'standard',
        }.get(key, default))
        
        self.mock_logger = Mock()
        
        # Create mock feature engineer and signal generator
        self.mock_feature_engineer = Mock()
        self.mock_signal_generator = Mock()
        
        # Mock model class
        self.mock_model_class = Mock()
    
    def test_create_windows(self):
        """Test window creation."""
        data = create_synthetic_data(500)
        
        with patch('trading_bot.backtesting.walk_forward.FeatureEngineer'), \
             patch('trading_bot.backtesting.walk_forward.SignalGenerator'), \
             patch('trading_bot.backtesting.walk_forward.TransactionCostModel'), \
             patch('trading_bot.backtesting.walk_forward.PerformanceMetrics'), \
             patch('trading_bot.backtesting.walk_forward.DataPreprocessor'):
            
            backtest = WalkForwardBacktest(
                data=data,
                config=self.mock_config,
                logger=self.mock_logger,
                feature_engineer=self.mock_feature_engineer,
                signal_generator=self.mock_signal_generator,
                model_class=self.mock_model_class
            )
            
            windows = backtest._create_windows()
            self.assertGreater(len(windows), 0)
            
            # Verify no look-ahead bias (test should start at or after train ends)
            for window in windows:
                self.assertGreaterEqual(window['test_start_idx'], window['train_end_idx'])


class TestBacktestResults(unittest.TestCase):
    """Test cases for BacktestResults."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        results = BacktestResults(
            total_return=10.0,
            sharpe_ratio=1.5,
            symbol='AAPL'
        )
        
        result_dict = results.to_dict()
        self.assertIn('total_return', result_dict)
        self.assertIn('sharpe_ratio', result_dict)
        self.assertIn('symbol', result_dict)
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        results = BacktestResults(
            total_return=10.0,
            sharpe_ratio=1.5
        )
        
        df = results.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


if __name__ == '__main__':
    unittest.main()

