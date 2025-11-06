"""
Unit Tests for Performance Metrics
"""

import unittest
from unittest.mock import Mock
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.backtesting.metrics import PerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'backtesting.risk_free_rate': 0.02,
        }.get(key, default))
        
        self.mock_logger = Mock()
        self.metrics = PerformanceMetrics(self.mock_config, self.mock_logger)
    
    def test_sharpe_ratio_known_values(self):
        """Test Sharpe ratio calculation."""
        # Create returns with known properties
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01] * 50)
        
        sharpe = self.metrics.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, 0)
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = pd.Series([0.01] * 100)
        sharpe = self.metrics.calculate_sharpe_ratio(returns)
        self.assertEqual(sharpe, 0.0)
    
    def test_max_drawdown_simple(self):
        """Test max drawdown calculation."""
        equity = pd.Series([100, 110, 105, 115, 120])
        drawdown_dict = self.metrics.calculate_max_drawdown(equity)
        
        self.assertIn('max_drawdown_pct', drawdown_dict)
        self.assertLess(drawdown_dict['max_drawdown_pct'], 0)
    
    def test_calculate_trade_statistics(self):
        """Test trade statistics calculation."""
        trades = [
            {'pnl': 100, 'entry_time': '2020-01-01', 'exit_time': '2020-01-02'},
            {'pnl': -50, 'entry_time': '2020-01-03', 'exit_time': '2020-01-04'},
            {'pnl': 150, 'entry_time': '2020-01-05', 'exit_time': '2020-01-06'},
        ]
        
        stats = self.metrics.calculate_trade_statistics(trades)
        
        self.assertEqual(stats['total_trades'], 3)
        self.assertEqual(stats['winning_trades'], 2)
        self.assertEqual(stats['losing_trades'], 1)
        self.assertAlmostEqual(stats['win_rate_pct'], 66.67, places=1)
    
    def test_empty_trades(self):
        """Test metrics with no trades."""
        trades = []
        equity = pd.Series([10000])
        returns = pd.Series([0.0])
        
        metrics = self.metrics.calculate_all_metrics(trades, equity, returns)
        self.assertEqual(metrics['total_trades'], 0)


if __name__ == '__main__':
    unittest.main()



