"""
Unit Tests for Paper Trading Engine

Tests paper trading simulation including order execution, position management,
stop-loss/take-profit, and performance metrics.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.trading.paper_trading import PaperTradingEngine
from trading_bot.data import DatabaseManager
from trading_bot.utils.exceptions import DatabaseError


class TestPaperTradingEngine(unittest.TestCase):
    """Test cases for PaperTradingEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'paper_trading.enabled': True,
            'paper_trading.initial_balance': 10000.0,
            'paper_trading.commission': 0.1,
            'paper_trading.slippage.enabled': True,
            'paper_trading.slippage.min_percentage': 0.01,
            'paper_trading.slippage.max_percentage': 0.1,
            'paper_trading.slippage.market_order_slippage': 0.2,
            'paper_trading.execution_delay_ms': 0,  # Disable delay for tests
            'paper_trading.partial_fills': False,
            'paper_trading.reset_on_startup': False,
            'trading.stop_loss_percentage': 2,
            'trading.take_profit_percentage': 5,
            'database.type': 'sqlite',
            'database.path': 'test_trading_bot.db',
            'database.backup_enabled': True,
            'database.backup_interval_hours': 24,
            'database.connection_pool_size': 5,
            'database.timeout_seconds': 30,
            'database.enable_wal_mode': True,
            'data.timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'data.retention_days': 365
        }.get(key, default))
        
        self.mock_logger = Mock()
        
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_trading_bot.db')
        
        # Patch get_writable_app_dir to return temp directory
        self.patcher = patch('trading_bot.data.database_manager.get_writable_app_dir')
        self.mock_get_dir = self.patcher.start()
        self.mock_get_dir.return_value = self.temp_dir
        
        # Initialize database manager
        self.db_manager = DatabaseManager(self.mock_config, self.mock_logger)
        
        # Initialize paper trading engine
        self.engine = PaperTradingEngine(self.mock_config, self.mock_logger, self.db_manager)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        # Clean up database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    # Portfolio Initialization Tests
    
    def test_initialize_fresh_portfolio(self):
        """Test fresh portfolio initialization."""
        self.assertEqual(self.engine.cash_balance, 10000.0)
        self.assertEqual(self.engine.initial_balance, 10000.0)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(self.engine.total_equity, 10000.0)
    
    def test_load_existing_portfolio(self):
        """Test loading existing portfolio from database."""
        # Create a snapshot and position
        self.db_manager.insert_portfolio_snapshot(
            total_equity=10500.0,
            cash_balance=5000.0,
            positions_value=5500.0,
            num_positions=1
        )
        
        self.db_manager.insert_position(
            symbol='AAPL',
            side='BUY',
            entry_price=100.0,
            quantity=55.0,
            stop_loss=98.0,
            take_profit=105.0
        )
        
        # Create new engine instance
        new_engine = PaperTradingEngine(self.mock_config, self.mock_logger, self.db_manager)
        
        # Verify state restored
        self.assertEqual(new_engine.cash_balance, 5000.0)
        self.assertEqual(len(new_engine.positions), 1)
        self.assertIn('AAPL', new_engine.positions)
    
    # Buy Order Tests
    
    def test_execute_buy_order_success(self):
        """Test successful buy order execution."""
        result = self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['quantity'], 10.0)
        self.assertLess(self.engine.cash_balance, 10000.0)  # Cash deducted
        self.assertIn('AAPL', self.engine.positions)
        self.assertEqual(self.engine.positions['AAPL']['quantity'], 10.0)
    
    def test_execute_buy_order_insufficient_funds(self):
        """Test buy order with insufficient funds."""
        # Try to buy more than available
        result = self.engine.execute_buy_order('AAPL', 1000.0, 100.0)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Insufficient funds')
        self.assertEqual(self.engine.cash_balance, 10000.0)  # No change
        self.assertEqual(len(self.engine.positions), 0)
    
    def test_execute_buy_order_with_slippage(self):
        """Test slippage is applied to buy orders."""
        # Disable random slippage for predictable test
        self.engine.slippage_enabled = True
        self.engine.slippage_min = 0.01
        self.engine.slippage_max = 0.01  # Fixed slippage
        
        result = self.engine.execute_buy_order('AAPL', 10.0, 100.0, order_type='limit')
        
        self.assertTrue(result['success'])
        # Execution price should be higher due to slippage
        self.assertGreaterEqual(result['execution_price'], 100.0)
        self.assertGreater(result['slippage'], 0)
    
    def test_execute_buy_order_with_commission(self):
        """Test commission is calculated and deducted."""
        result = self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        self.assertTrue(result['success'])
        self.assertGreater(result['commission'], 0)
        # Total cost should include commission
        expected_cost = 100.0 * 10.0 + result['commission']
        self.assertAlmostEqual(result['total_cost'], expected_cost, places=2)
    
    def test_execute_buy_order_existing_position(self):
        """Test buying more of existing position updates average price."""
        # First buy
        result1 = self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.assertTrue(result1['success'])
        entry_price1 = result1['execution_price']
        
        # Second buy at different price
        result2 = self.engine.execute_buy_order('AAPL', 10.0, 110.0)
        self.assertTrue(result2['success'])
        
        # Verify average price
        position = self.engine.positions['AAPL']
        self.assertEqual(position['quantity'], 20.0)
        # Average should be between 100 and 110
        self.assertGreater(position['entry_price'], entry_price1)
        self.assertLess(position['entry_price'], 110.0)
    
    # Sell Order Tests
    
    def test_execute_sell_order_success(self):
        """Test successful sell order execution."""
        # First buy
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        initial_cash = self.engine.cash_balance
        
        # Then sell
        result = self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        self.assertTrue(result['success'])
        self.assertGreater(self.engine.cash_balance, initial_cash)
        self.assertNotIn('AAPL', self.engine.positions)  # Position closed
        self.assertGreater(result['pnl'], 0)  # Profit
    
    def test_execute_sell_order_no_position(self):
        """Test sell order without position."""
        result = self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        self.assertFalse(result['success'])
        self.assertIn('No position found', result['error'])
    
    def test_execute_sell_order_insufficient_quantity(self):
        """Test sell order with insufficient quantity."""
        # Buy small position
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        # Try to sell more
        result = self.engine.execute_sell_order('AAPL', 20.0, 110.0)
        
        self.assertFalse(result['success'])
        self.assertIn('Insufficient quantity', result['error'])
    
    def test_execute_sell_order_partial(self):
        """Test partial sell order."""
        # Buy position
        self.engine.execute_buy_order('AAPL', 20.0, 100.0)
        
        # Partial sell
        result = self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        self.assertTrue(result['success'])
        self.assertFalse(result['position_closed'])
        self.assertEqual(self.engine.positions['AAPL']['quantity'], 10.0)
    
    def test_execute_sell_order_profit(self):
        """Test sell order with profit."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        result = self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        self.assertTrue(result['success'])
        self.assertGreater(result['pnl'], 0)
        self.assertGreater(result['pnl_percentage'], 0)
    
    def test_execute_sell_order_loss(self):
        """Test sell order with loss."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        result = self.engine.execute_sell_order('AAPL', 10.0, 90.0)
        
        self.assertTrue(result['success'])
        self.assertLess(result['pnl'], 0)
        self.assertLess(result['pnl_percentage'], 0)
    
    # Position Management Tests
    
    def test_get_position(self):
        """Test getting position."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        position = self.engine.get_position('AAPL')
        self.assertIsNotNone(position)
        self.assertEqual(position['symbol'], 'AAPL')
        self.assertEqual(position['quantity'], 10.0)
    
    def test_get_all_positions(self):
        """Test getting all positions."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.execute_buy_order('GOOGL', 5.0, 200.0)
        
        positions = self.engine.get_all_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn('AAPL', positions)
        self.assertIn('GOOGL', positions)
    
    def test_update_position_prices(self):
        """Test updating position prices."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        # Update price
        market_data = {'AAPL': 110.0}
        self.engine.update_position_prices(market_data)
        
        position = self.engine.positions['AAPL']
        self.assertEqual(position['current_price'], 110.0)
        self.assertGreater(position['unrealized_pnl'], 0)
    
    def test_close_position(self):
        """Test manually closing position."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.close_position('AAPL', 110.0, reason='manual')
        
        self.assertNotIn('AAPL', self.engine.positions)
        # Verify trade recorded
        trades = self.db_manager.get_all_trades()
        self.assertGreater(len(trades), 0)
    
    # Stop Loss / Take Profit Tests
    
    def test_stop_loss_triggered(self):
        """Test stop loss triggers position closure."""
        # Buy with stop loss
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        # Update price below stop loss
        position = self.engine.positions['AAPL']
        stop_loss = position['stop_loss']
        market_data = {'AAPL': stop_loss - 1.0}
        self.engine.update_position_prices(market_data)
        
        # Check stop loss
        self.engine.check_stop_loss_take_profit()
        
        # Position should be closed
        self.assertNotIn('AAPL', self.engine.positions)
    
    def test_take_profit_triggered(self):
        """Test take profit triggers position closure."""
        # Buy with take profit
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        # Update price above take profit
        position = self.engine.positions['AAPL']
        take_profit = position['take_profit']
        market_data = {'AAPL': take_profit + 1.0}
        self.engine.update_position_prices(market_data)
        
        # Check take profit
        self.engine.check_stop_loss_take_profit()
        
        # Position should be closed
        self.assertNotIn('AAPL', self.engine.positions)
    
    def test_no_trigger(self):
        """Test no trigger when price is within range."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        # Update price within range
        market_data = {'AAPL': 102.0}
        self.engine.update_position_prices(market_data)
        
        # Check triggers
        self.engine.check_stop_loss_take_profit()
        
        # Position should remain open
        self.assertIn('AAPL', self.engine.positions)
    
    # Performance Metrics Tests
    
    def test_calculate_performance_metrics_no_trades(self):
        """Test metrics with no trades."""
        metrics = self.engine.calculate_performance_metrics()
        
        self.assertEqual(metrics['total_trades'], 0)
        self.assertEqual(metrics['win_rate'], 0.0)
        self.assertEqual(metrics['total_pnl'], 0.0)
    
    def test_calculate_performance_metrics_with_trades(self):
        """Test metrics calculation with trades."""
        # Execute winning trade
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        # Execute losing trade
        self.engine.execute_buy_order('GOOGL', 5.0, 200.0)
        self.engine.execute_sell_order('GOOGL', 5.0, 190.0)
        
        metrics = self.engine.calculate_performance_metrics()
        
        self.assertEqual(metrics['total_trades'], 2)
        self.assertEqual(metrics['winning_trades'], 1)
        self.assertEqual(metrics['losing_trades'], 1)
        self.assertEqual(metrics['win_rate'], 50.0)
        self.assertGreater(metrics['avg_win'], 0)
        self.assertLess(metrics['avg_loss'], 0)
    
    def test_get_daily_pnl(self):
        """Test daily PnL calculation."""
        # Execute trade
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        daily_pnl = self.engine.get_daily_pnl()
        # Should be positive after profitable trade
        self.assertGreater(daily_pnl, 0)
    
    def test_get_total_pnl(self):
        """Test total PnL calculation."""
        self.assertEqual(self.engine.get_total_pnl(), 0.0)
        
        # Execute trade
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        total_pnl = self.engine.get_total_pnl()
        self.assertGreater(total_pnl, 0)
    
    def test_get_pnl_percentage(self):
        """Test PnL percentage calculation."""
        self.assertEqual(self.engine.get_pnl_percentage(), 0.0)
        
        # Execute trade
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        pnl_pct = self.engine.get_pnl_percentage()
        self.assertGreater(pnl_pct, 0)
    
    # Portfolio Value Tests
    
    def test_get_portfolio_value_cash_only(self):
        """Test portfolio value with cash only."""
        value = self.engine.get_portfolio_value()
        self.assertEqual(value, 10000.0)
    
    def test_get_portfolio_value_with_positions(self):
        """Test portfolio value with positions."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        value = self.engine.get_portfolio_value()
        # Should be approximately initial balance (minus commission)
        self.assertGreater(value, 9000.0)
        self.assertLess(value, 10000.0)
    
    def test_get_account_summary(self):
        """Test account summary."""
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        summary = self.engine.get_account_summary()
        
        self.assertIn('cash_balance', summary)
        self.assertIn('total_equity', summary)
        self.assertIn('positions_value', summary)
        self.assertIn('num_positions', summary)
        self.assertEqual(summary['num_positions'], 1)
    
    # Signal Integration Tests
    
    def test_execute_signal_buy(self):
        """Test executing BUY signal."""
        signal = {
            'type': 'BUY',
            'symbol': 'AAPL',
            'confidence': 0.8,
            'strength': 0.7
        }
        
        result = self.engine.execute_signal(signal, 100.0)
        
        self.assertTrue(result['success'])
        self.assertIn('AAPL', self.engine.positions)
    
    def test_execute_signal_sell(self):
        """Test executing SELL signal."""
        # First buy
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        
        signal = {
            'type': 'SELL',
            'symbol': 'AAPL',
            'confidence': 0.8,
            'strength': 0.7
        }
        
        result = self.engine.execute_signal(signal, 110.0)
        
        self.assertTrue(result['success'])
        self.assertNotIn('AAPL', self.engine.positions)
    
    def test_execute_signal_hold(self):
        """Test HOLD signal does nothing."""
        signal = {
            'type': 'HOLD',
            'symbol': 'AAPL',
            'confidence': 0.5
        }
        
        result = self.engine.execute_signal(signal, 100.0)
        
        self.assertFalse(result['success'])
        self.assertIn('HOLD signal', result['error'])
    
    # Validation Tests
    
    def test_validate_order_params_invalid_quantity(self):
        """Test validation with invalid quantity."""
        with self.assertRaises(ValueError):
            self.engine._validate_order_params('AAPL', -10.0, 100.0)
        
        with self.assertRaises(ValueError):
            self.engine._validate_order_params('AAPL', 0.0, 100.0)
    
    def test_validate_order_params_invalid_price(self):
        """Test validation with invalid price."""
        with self.assertRaises(ValueError):
            self.engine._validate_order_params('AAPL', 10.0, -100.0)
        
        with self.assertRaises(ValueError):
            self.engine._validate_order_params('AAPL', 10.0, 0.0)
    
    def test_can_execute_order_checks(self):
        """Test can_execute_order validation."""
        # Buy order check - sufficient funds
        can_execute, reason = self.engine.can_execute_order('AAPL', 10.0, 100.0, 'buy')
        self.assertTrue(can_execute)
        
        # Buy order check - insufficient funds
        can_execute, reason = self.engine.can_execute_order('AAPL', 10000.0, 100.0, 'buy')
        self.assertFalse(can_execute)
        
        # Sell order check - no position
        can_execute, reason = self.engine.can_execute_order('AAPL', 10.0, 100.0, 'sell')
        self.assertFalse(can_execute)
    
    # Reset Tests
    
    def test_reset_portfolio(self):
        """Test portfolio reset."""
        # Create positions and trades
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        # Reset
        self.engine.reset_portfolio()
        
        # Verify reset
        self.assertEqual(self.engine.cash_balance, 10000.0)
        self.assertEqual(len(self.engine.positions), 0)
    
    # Trade History Tests
    
    def test_get_trade_history(self):
        """Test getting trade history."""
        # Execute trades
        self.engine.execute_buy_order('AAPL', 10.0, 100.0)
        self.engine.execute_sell_order('AAPL', 10.0, 110.0)
        
        history = self.engine.get_trade_history()
        self.assertGreater(len(history), 0)
        
        # Filter by symbol
        aapl_history = self.engine.get_trade_history(symbol='AAPL')
        self.assertEqual(len(aapl_history), 1)


if __name__ == '__main__':
    unittest.main()





