"""
Unit Tests for Trading Signal Generation and Strategy

Tests signal generator and XGBoost trading strategy.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.trading import SignalGenerator, SignalType, XGBoostStrategy


class TestSignalGenerator(unittest.TestCase):
    """Test cases for SignalGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'models.prediction.buy_threshold': 0.6,
            'models.prediction.sell_threshold': 0.6,
            'models.prediction.min_confidence': 0.5,
            'models.xgboost.target_type': 'regression'
        }.get(key, default))
        
        self.mock_logger = Mock()
    
    def test_initialization(self):
        """Test SignalGenerator initialization."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        self.assertIsNotNone(generator)
        self.assertEqual(generator.buy_threshold, 0.6)
        self.assertEqual(generator.sell_threshold, 0.6)
        self.assertEqual(generator.min_confidence, 0.5)
    
    def test_generate_signal_buy(self):
        """Test buy signal generation."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        # Predicted price significantly higher than current
        predicted_price = 120.0
        current_price = 100.0
        confidence = 0.8
        
        signal = generator.generate_signal(
            predicted_price, confidence, current_price, 'AAPL'
        )
        
        self.assertEqual(signal['type'], SignalType.BUY.value)
        self.assertEqual(signal['confidence'], confidence)
        self.assertEqual(signal['symbol'], 'AAPL')
    
    def test_generate_signal_sell(self):
        """Test sell signal generation."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        # Predicted price significantly lower than current
        predicted_price = 80.0
        current_price = 100.0
        confidence = 0.75
        
        signal = generator.generate_signal(
            predicted_price, confidence, current_price, 'AAPL'
        )
        
        self.assertEqual(signal['type'], SignalType.SELL.value)
    
    def test_generate_signal_hold(self):
        """Test hold signal generation."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        # Predicted price close to current
        predicted_price = 101.0
        current_price = 100.0
        confidence = 0.7
        
        signal = generator.generate_signal(
            predicted_price, confidence, current_price, 'AAPL'
        )
        
        self.assertEqual(signal['type'], SignalType.HOLD.value)
    
    def test_generate_signal_low_confidence(self):
        """Test signal with low confidence."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        # High predicted return but low confidence
        predicted_price = 150.0
        current_price = 100.0
        confidence = 0.3  # Below min_confidence
        
        signal = generator.generate_signal(
            predicted_price, confidence, current_price, 'AAPL'
        )
        
        # Should be HOLD due to low confidence
        self.assertEqual(signal['type'], SignalType.HOLD.value)
    
    def test_validate_signal_success(self):
        """Test signal validation with valid signal."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        valid_signal = {
            'type': SignalType.BUY.value,
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat()
        }
        
        self.assertTrue(generator.validate_signal(valid_signal))
    
    def test_validate_signal_missing_field(self):
        """Test signal validation with missing field."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        invalid_signal = {
            'type': SignalType.BUY.value,
            # Missing confidence and timestamp
        }
        
        self.assertFalse(generator.validate_signal(invalid_signal))
    
    def test_get_signal_strength(self):
        """Test signal strength calculation."""
        generator = SignalGenerator(self.mock_config, self.mock_logger)
        
        # Strong buy should have high strength
        strength = generator.get_signal_strength(SignalType.STRONG_BUY, 0.9)
        self.assertGreater(strength, 0.8)
        
        # Hold should have low strength
        strength = generator.get_signal_strength(SignalType.HOLD, 0.9)
        self.assertEqual(strength, 0.0)


class TestXGBoostStrategy(unittest.TestCase):
    """Test cases for XGBoostStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'trading.position_size_percentage': 10,
            'trading.risk_percentage': 2,
            'trading.max_positions': 5,
            'trading.stop_loss_percentage': 2,
            'trading.take_profit_percentage': 5,
            'models.xgboost.lookback_days': 60
        }.get(key, default))
        
        self.mock_logger = Mock()
        self.mock_predictor = Mock()
        self.mock_signal_generator = Mock()
    
    def test_initialization(self):
        """Test XGBoostStrategy initialization."""
        strategy = XGBoostStrategy(
            self.mock_config,
            self.mock_logger,
            self.mock_predictor,
            self.mock_signal_generator
        )
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.position_size_pct, 10)
        self.assertEqual(strategy.max_positions, 5)
    
    @patch('trading_bot.trading.strategy.StockDataFetcher')
    @patch('trading_bot.trading.strategy.FeatureEngineer')
    def test_analyze_symbol(self, mock_engineer_class, mock_fetcher_class):
        """Test symbol analysis."""
        # Setup mocks
        mock_fetcher = Mock()
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'Close': np.random.uniform(100, 110, 100)
        })
        mock_fetcher.fetch_latest_data.return_value = mock_data
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_engineer = Mock()
        mock_features = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 100),
            'feature2': np.random.uniform(0, 1, 100)
        })
        mock_engineer.create_features.return_value = mock_features
        mock_engineer_class.return_value = mock_engineer
        
        self.mock_predictor.predict.return_value = np.array([105.0])
        self.mock_predictor.get_confidence.return_value = 0.8
        
        mock_signal = {
            'type': SignalType.BUY.value,
            'confidence': 0.8,
            'strength': 0.75,
            'timestamp': datetime.now().isoformat()
        }
        self.mock_signal_generator.generate_signal.return_value = mock_signal
        self.mock_signal_generator.should_execute_signal.return_value = True
        
        strategy = XGBoostStrategy(
            self.mock_config,
            self.mock_logger,
            self.mock_predictor,
            self.mock_signal_generator
        )
        
        decision = strategy.analyze('AAPL', account_balance=10000)
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision['symbol'], 'AAPL')
        self.assertIn('signal', decision)
        self.assertIn('position_size', decision)
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        mock_signal = {
            'strength': 0.8,
            'confidence': 0.75
        }
        
        strategy = XGBoostStrategy(
            self.mock_config,
            self.mock_logger,
            self.mock_predictor,
            self.mock_signal_generator
        )
        
        position_size = strategy._calculate_position_size(mock_signal, 10000)
        
        # Should be between 0 and account balance
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 10000)
    
    def test_should_execute_trade_max_positions(self):
        """Test trade execution with max positions reached."""
        mock_signal = {
            'type': SignalType.BUY.value,
            'confidence': 0.8,
            'symbol': 'AAPL'
        }
        
        self.mock_signal_generator.should_execute_signal.return_value = True
        
        strategy = XGBoostStrategy(
            self.mock_config,
            self.mock_logger,
            self.mock_predictor,
            self.mock_signal_generator
        )
        
        # Fill up active positions
        for i in range(5):
            strategy.active_positions[f'STOCK{i}'] = {'size': 100}
        
        # Should not execute due to max positions
        self.assertFalse(strategy.should_execute_trade(mock_signal))
    
    def test_update_and_close_position(self):
        """Test position tracking."""
        strategy = XGBoostStrategy(
            self.mock_config,
            self.mock_logger,
            self.mock_predictor,
            self.mock_signal_generator
        )
        
        # Update position
        strategy.update_position('AAPL', 100.0, 1000.0)
        self.assertIn('AAPL', strategy.active_positions)
        
        # Close position
        strategy.close_position('AAPL')
        self.assertNotIn('AAPL', strategy.active_positions)
    
    def test_get_signal_history(self):
        """Test signal history retrieval."""
        strategy = XGBoostStrategy(
            self.mock_config,
            self.mock_logger,
            self.mock_predictor,
            self.mock_signal_generator
        )
        
        # Add signals
        for i in range(10):
            strategy.signal_history.append({'id': i})
        
        # Get last 5
        history = strategy.get_signal_history(limit=5)
        self.assertEqual(len(history), 5)
        
        # Get all
        history = strategy.get_signal_history()
        self.assertEqual(len(history), 10)


if __name__ == '__main__':
    unittest.main()

