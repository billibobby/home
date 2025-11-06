"""
Unit Tests for Database Manager Module

Tests database operations for trades, positions, portfolio snapshots, and performance metrics.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.data import DatabaseManager
from trading_bot.utils.exceptions import DatabaseError


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
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
        self.db = DatabaseManager(self.mock_config, self.mock_logger)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'db'):
            self.db.close()
        self.patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    # Initialization Tests
    
    def test_database_manager_init(self):
        """Test DatabaseManager initialization."""
        self.assertIsNotNone(self.db)
        self.assertEqual(self.db.db_path, os.path.join(self.temp_dir, 'test_trading_bot.db'))
    
    def test_schema_creation(self):
        """Test that all tables are created."""
        self.assertTrue(self.db.validate_schema())
    
    def test_indexes_creation(self):
        """Test that indexes exist."""
        with self.db.get_connection() as conn:
            indexes = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_%'
            """).fetchall()
            
            index_names = [idx[0] for idx in indexes]
            self.assertIn('idx_trades_symbol', index_names)
            self.assertIn('idx_positions_symbol', index_names)
            self.assertIn('idx_snapshots_timestamp', index_names)
    
    # Trades CRUD Tests
    
    def test_insert_trade(self):
        """Test inserting a trade."""
        entry_time = datetime.now().isoformat()
        exit_time = (datetime.now() + timedelta(days=1)).isoformat()
        
        trade_id = self.db.insert_trade(
            symbol='AAPL',
            side='BUY',
            entry_price=150.0,
            exit_price=155.0,
            quantity=10,
            entry_time=entry_time,
            exit_time=exit_time,
            timeframe='1d',
            strategy='XGBoost'
        )
        
        self.assertIsNotNone(trade_id)
        self.assertGreater(trade_id, 0)
        
        # Verify trade was inserted
        trade = self.db.get_trade_by_id(trade_id)
        self.assertIsNotNone(trade)
        self.assertEqual(trade['symbol'], 'AAPL')
        self.assertEqual(trade['pnl'], 50.0)  # (155 - 150) * 10
    
    def test_get_trade_by_id(self):
        """Test fetching trade by ID."""
        entry_time = datetime.now().isoformat()
        exit_time = (datetime.now() + timedelta(days=1)).isoformat()
        
        trade_id = self.db.insert_trade(
            symbol='MSFT',
            side='BUY',
            entry_price=300.0,
            exit_price=310.0,
            quantity=5,
            entry_time=entry_time,
            exit_time=exit_time
        )
        
        trade = self.db.get_trade_by_id(trade_id)
        self.assertIsNotNone(trade)
        self.assertEqual(trade['symbol'], 'MSFT')
        self.assertEqual(trade['entry_price'], 300.0)
    
    def test_get_trades_with_filters(self):
        """Test filtering trades."""
        now = datetime.now()
        entry_time = now.isoformat()
        exit_time = (now + timedelta(days=1)).isoformat()
        
        # Insert multiple trades
        self.db.insert_trade('AAPL', 'BUY', 150.0, 155.0, 10, entry_time, exit_time, timeframe='1d')
        self.db.insert_trade('MSFT', 'BUY', 300.0, 310.0, 5, entry_time, exit_time, timeframe='1h')
        self.db.insert_trade('GOOGL', 'BUY', 100.0, 105.0, 20, entry_time, exit_time, timeframe='1d')
        
        # Filter by symbol
        trades = self.db.get_trades(symbol='AAPL')
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['symbol'], 'AAPL')
        
        # Filter by timeframe
        trades = self.db.get_trades(timeframe='1d')
        self.assertEqual(len(trades), 2)
    
    def test_update_trade(self):
        """Test updating a trade."""
        entry_time = datetime.now().isoformat()
        exit_time = (datetime.now() + timedelta(days=1)).isoformat()
        
        trade_id = self.db.insert_trade(
            'AAPL', 'BUY', 150.0, 155.0, 10, entry_time, exit_time
        )
        
        # Update trade
        self.db.update_trade(trade_id, notes='Updated notes')
        
        trade = self.db.get_trade_by_id(trade_id)
        self.assertEqual(trade['notes'], 'Updated notes')
    
    def test_delete_trade(self):
        """Test deleting a trade."""
        entry_time = datetime.now().isoformat()
        exit_time = (datetime.now() + timedelta(days=1)).isoformat()
        
        trade_id = self.db.insert_trade(
            'AAPL', 'BUY', 150.0, 155.0, 10, entry_time, exit_time
        )
        
        # Delete trade
        self.db.delete_trade(trade_id)
        
        trade = self.db.get_trade_by_id(trade_id)
        self.assertIsNone(trade)
    
    def test_get_trade_statistics(self):
        """Test trade statistics calculation."""
        now = datetime.now()
        entry_time = now.isoformat()
        exit_time = (now + timedelta(days=1)).isoformat()
        
        # Insert winning and losing trades
        self.db.insert_trade('AAPL', 'BUY', 150.0, 155.0, 10, entry_time, exit_time)  # +50
        self.db.insert_trade('MSFT', 'BUY', 300.0, 295.0, 5, entry_time, exit_time)   # -25
        
        stats = self.db.get_trade_statistics()
        
        self.assertEqual(stats['total_trades'], 2)
        self.assertEqual(stats['winning_trades'], 1)
        self.assertEqual(stats['losing_trades'], 1)
        self.assertEqual(stats['win_rate'], 50.0)
        self.assertAlmostEqual(stats['total_pnl'], 25.0, places=2)
    
    # Positions CRUD Tests
    
    def test_insert_position(self):
        """Test creating a position."""
        entry_time = datetime.now().isoformat()
        
        position_id = self.db.insert_position(
            symbol='AAPL',
            side='BUY',
            entry_price=150.0,
            quantity=10,
            stop_loss=145.0,
            take_profit=160.0,
            entry_time=entry_time,
            timeframe='1d'
        )
        
        self.assertIsNotNone(position_id)
        
        position = self.db.get_position_by_symbol('AAPL')
        self.assertIsNotNone(position)
        self.assertEqual(position['entry_price'], 150.0)
        self.assertEqual(position['current_price'], 150.0)
        self.assertEqual(position['unrealized_pnl'], 0.0)
    
    def test_get_position_by_symbol(self):
        """Test fetching position by symbol."""
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
        
        position = self.db.get_position_by_symbol('AAPL')
        self.assertIsNotNone(position)
        self.assertEqual(position['symbol'], 'AAPL')
    
    def test_update_position_price(self):
        """Test updating position price and PnL calculation."""
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
        
        # Update price
        self.db.update_position_price('AAPL', 155.0)
        
        position = self.db.get_position_by_symbol('AAPL')
        self.assertEqual(position['current_price'], 155.0)
        self.assertEqual(position['unrealized_pnl'], 50.0)  # (155 - 150) * 10
    
    def test_close_position(self):
        """Test closing position and creating trade."""
        entry_time = datetime.now().isoformat()
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=entry_time)
        
        # Close position
        exit_time = (datetime.now() + timedelta(days=1)).isoformat()
        trade_id = self.db.close_position('AAPL', 155.0, exit_time)
        
        self.assertIsNotNone(trade_id)
        
        # Verify position is closed
        position = self.db.get_position_by_symbol('AAPL')
        self.assertIsNone(position)
        
        # Verify trade was created
        trade = self.db.get_trade_by_id(trade_id)
        self.assertIsNotNone(trade)
        self.assertEqual(trade['symbol'], 'AAPL')
        self.assertEqual(trade['pnl'], 50.0)
    
    def test_duplicate_position_prevention(self):
        """Test that duplicate positions are prevented based on (symbol, timeframe, strategy)."""
        # Insert position without timeframe/strategy
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
        
        # Try to insert duplicate (same symbol, timeframe, strategy = all None)
        with self.assertRaises(DatabaseError):
            self.db.insert_position('AAPL', 'BUY', 160.0, 5, entry_time=datetime.now().isoformat())
        
        # But should allow same symbol with different timeframe
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat(), timeframe='1d')
        
        # Or same symbol and timeframe with different strategy
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat(), timeframe='1d', strategy='XGBoost')
        
        # But duplicate (symbol, timeframe, strategy) should fail
        with self.assertRaises(DatabaseError):
            self.db.insert_position('AAPL', 'BUY', 160.0, 5, entry_time=datetime.now().isoformat(), timeframe='1d', strategy='XGBoost')
    
    def test_get_position_count(self):
        """Test position count."""
        self.assertEqual(self.db.get_position_count(), 0)
        
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
        self.db.insert_position('MSFT', 'BUY', 300.0, 5, entry_time=datetime.now().isoformat())
        
        self.assertEqual(self.db.get_position_count(), 2)
    
    # Portfolio Snapshots Tests
    
    def test_insert_snapshot(self):
        """Test creating a portfolio snapshot."""
        snapshot_id = self.db.insert_portfolio_snapshot(
            total_equity=10000.0,
            cash_balance=5000.0,
            positions_value=5000.0,
            num_positions=2,
            daily_pnl=100.0,
            total_pnl=500.0
        )
        
        self.assertIsNotNone(snapshot_id)
        
        snapshot = self.db.get_snapshot_by_id(snapshot_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot['total_equity'], 10000.0)
    
    def test_get_latest_snapshot(self):
        """Test fetching most recent snapshot."""
        self.db.insert_portfolio_snapshot(10000.0, 5000.0, 5000.0, 2)
        
        # Wait a bit and insert another
        import time
        time.sleep(0.1)
        self.db.insert_portfolio_snapshot(10100.0, 5000.0, 5100.0, 2)
        
        latest = self.db.get_latest_snapshot()
        self.assertIsNotNone(latest)
        self.assertEqual(latest['total_equity'], 10100.0)
    
    def test_get_equity_curve(self):
        """Test equity curve data."""
        self.db.insert_portfolio_snapshot(10000.0, 5000.0, 5000.0, 2)
        self.db.insert_portfolio_snapshot(10100.0, 5000.0, 5100.0, 2)
        
        curve = self.db.get_equity_curve()
        self.assertEqual(len(curve), 2)
        self.assertEqual(curve[0][1], 10000.0)
        self.assertEqual(curve[1][1], 10100.0)
    
    # Performance Metrics Tests
    
    def test_insert_metrics(self):
        """Test storing performance metrics."""
        metrics_id = self.db.insert_performance_metrics(
            period='daily',
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            win_rate=0.6,
            total_trades=100,
            winning_trades=60
        )
        
        self.assertIsNotNone(metrics_id)
        
        metrics = self.db.get_latest_metrics('daily')
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['sharpe_ratio'], 1.5)
        self.assertEqual(metrics['win_rate'], 0.6)
    
    def test_get_latest_metrics(self):
        """Test fetching latest metrics by period."""
        self.db.insert_performance_metrics('daily', win_rate=0.6, total_trades=100)
        self.db.insert_performance_metrics('weekly', win_rate=0.65, total_trades=500)
        
        daily_metrics = self.db.get_latest_metrics('daily')
        self.assertIsNotNone(daily_metrics)
        self.assertEqual(daily_metrics['period'], 'daily')
    
    def test_metrics_history(self):
        """Test historical metrics retrieval."""
        for i in range(5):
            self.db.insert_performance_metrics('daily', win_rate=0.6 + i*0.01, total_trades=100+i)
        
        history = self.db.get_metrics_history('daily', limit=5)
        self.assertEqual(len(history), 5)
    
    # Transaction Tests
    
    def test_transaction_commit(self):
        """Test transaction commit."""
        with self.db.transaction():
            self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
            self.db.insert_position('MSFT', 'BUY', 300.0, 5, entry_time=datetime.now().isoformat())
        
        # Verify both positions were committed
        self.assertEqual(self.db.get_position_count(), 2)
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        try:
            with self.db.transaction():
                self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify position was not committed
        self.assertEqual(self.db.get_position_count(), 0)
    
    def test_atomic_position_close(self):
        """Test that position close is atomic."""
        entry_time = datetime.now().isoformat()
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=entry_time)
        
        # Close position
        trade_id = self.db.close_position('AAPL', 155.0)
        
        # Verify both operations succeeded
        self.assertIsNotNone(trade_id)
        self.assertIsNone(self.db.get_position_by_symbol('AAPL'))
        
        trade = self.db.get_trade_by_id(trade_id)
        self.assertIsNotNone(trade)
    
    # Maintenance Tests
    
    def test_create_backup(self):
        """Test backup creation."""
        # Insert some data
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
        
        # Create backup
        backup_path = self.db.create_backup()
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(os.path.exists(backup_path))
    
    def test_cleanup_old_data(self):
        """Test data retention cleanup."""
        # Insert old trade
        old_date = (datetime.now() - timedelta(days=400)).isoformat()
        self.db.insert_trade('AAPL', 'BUY', 150.0, 155.0, 10, old_date, old_date)
        
        # Cleanup
        self.db.cleanup_old_data(retention_days=365)
        
        # Verify old trade was deleted
        trades = self.db.get_all_trades()
        self.assertEqual(len(trades), 0)
    
    def test_database_integrity(self):
        """Test integrity check."""
        integrity_ok = self.db.check_integrity()
        self.assertTrue(integrity_ok)
    
    # Error Handling Tests
    
    def test_database_error_on_invalid_query(self):
        """Test that DatabaseError is raised on invalid queries."""
        with self.assertRaises(DatabaseError):
            self.db._execute_query("SELECT * FROM nonexistent_table", fetch=True)
    
    def test_get_database_stats(self):
        """Test database statistics."""
        # Insert some data
        self.db.insert_position('AAPL', 'BUY', 150.0, 10, entry_time=datetime.now().isoformat())
        entry_time = datetime.now().isoformat()
        exit_time = (datetime.now() + timedelta(days=1)).isoformat()
        self.db.insert_trade('MSFT', 'BUY', 300.0, 310.0, 5, entry_time, exit_time)
        
        stats = self.db.get_database_stats()
        
        self.assertEqual(stats['total_trades'], 1)
        self.assertEqual(stats['total_positions'], 1)
        self.assertGreater(stats['database_size'], 0)
    
    # Multi-Timeframe Tests
    
    def test_trades_by_timeframe(self):
        """Test filtering trades by timeframe."""
        now = datetime.now()
        entry_time = now.isoformat()
        exit_time = (now + timedelta(days=1)).isoformat()
        
        self.db.insert_trade('AAPL', 'BUY', 150.0, 155.0, 10, entry_time, exit_time, timeframe='1d')
        self.db.insert_trade('MSFT', 'BUY', 300.0, 310.0, 5, entry_time, exit_time, timeframe='1h')
        
        trades = self.db.get_trades_by_timeframe('1d')
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['timeframe'], '1d')
    
    def test_performance_by_timeframe(self):
        """Test performance aggregation by timeframe."""
        now = datetime.now()
        entry_time = now.isoformat()
        exit_time = (now + timedelta(days=1)).isoformat()
        
        self.db.insert_trade('AAPL', 'BUY', 150.0, 155.0, 10, entry_time, exit_time, timeframe='1d')
        
        perf = self.db.get_performance_by_timeframe('1d')
        self.assertEqual(perf['timeframe'], '1d')
        self.assertEqual(perf['total_trades'], 1)


if __name__ == '__main__':
    unittest.main()

