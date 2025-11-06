"""
Unit Tests for Alpaca Integration

Tests all Alpaca exchange functionality with mocked API responses.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.exchanges.alpaca_client import AlpacaClient
from trading_bot.exchanges.alpaca_exchange import AlpacaExchange
from trading_bot.exchanges.alpaca_stream import AlpacaStream, ConnectionState
from trading_bot.exchanges.factory import create_exchange
from trading_bot.exchanges.paper_exchange import PaperTradingExchange


class TestAlpacaClient(unittest.TestCase):
    """Test cases for AlpacaClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.api_key = 'test_api_key'
        self.api_secret = 'test_api_secret'
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get = Mock(return_value=None)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_client_init_paper_mode(self, mock_data_client, mock_trading_client):
        """Test client initialization in paper mode."""
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        
        self.assertEqual(client.paper_mode, True)
        mock_trading_client.assert_called_once_with(self.api_key, self.api_secret, paper=True)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_client_init_live_mode(self, mock_data_client, mock_trading_client):
        """Test client initialization in live mode."""
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=False, logger=self.mock_logger)
        
        self.assertEqual(client.paper_mode, False)
        mock_trading_client.assert_called_once_with(self.api_key, self.api_secret, paper=False)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_get_account_success(self, mock_data_client, mock_trading_client):
        """Test successful account retrieval."""
        # Mock account response
        mock_account = Mock()
        mock_account.cash = 10000.0
        mock_account.portfolio_value = 12000.0
        mock_account.buying_power = 20000.0
        mock_account.equity = 12000.0
        mock_account.currency = 'USD'
        
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.get_account()
        
        self.assertNotIn('error', result)
        self.assertEqual(result['cash'], 10000.0)
        self.assertEqual(result['equity'], 12000.0)
        self.assertEqual(result['buying_power'], 20000.0)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_get_position_exists(self, mock_data_client, mock_trading_client):
        """Test getting an existing position."""
        # Mock position response
        mock_position = Mock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = 10.0
        mock_position.avg_entry_price = 150.0
        mock_position.current_price = 155.0
        mock_position.market_value = 1550.0
        mock_position.unrealized_pl = 50.0
        mock_position.unrealized_plpc = 3.33
        
        mock_client_instance = Mock()
        mock_client_instance.get_open_position.return_value = mock_position
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.get_position('AAPL')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['qty'], 10.0)
        self.assertEqual(result['avg_entry_price'], 150.0)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_get_position_not_found(self, mock_data_client, mock_trading_client):
        """Test getting a non-existent position."""
        from alpaca.common.exceptions import APIError
        
        mock_client_instance = Mock()
        error = APIError('Position not found')
        error.status_code = 404
        mock_client_instance.get_open_position.side_effect = error
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.get_position('AAPL')
        
        self.assertIsNone(result)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_place_market_order_buy(self, mock_data_client, mock_trading_client):
        """Test placing a market buy order."""
        # Mock order response
        mock_order = Mock()
        mock_order.id = 'order_123'
        mock_order.filled_avg_price = 155.5
        mock_order.status = Mock()
        mock_order.status.value = 'filled'
        
        mock_client_instance = Mock()
        mock_client_instance.submit_order.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.place_order('AAPL', 10, 'buy', 'market', 'day', None)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], 'order_123')
        self.assertEqual(result['filled_avg_price'], 155.5)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_is_market_open_true(self, mock_data_client, mock_trading_client):
        """Test market open check when market is open."""
        mock_clock = Mock()
        mock_clock.is_open = True
        
        mock_client_instance = Mock()
        mock_client_instance.get_clock.return_value = mock_clock
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.is_market_open()
        
        self.assertTrue(result)


class TestAlpacaExchange(unittest.TestCase):
    """Test cases for AlpacaExchange."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'exchanges.alpaca.paper_trading': True,
            'api.exchanges.alpaca.paper_trading': True
        }.get(key, default))
    
    @patch('trading_bot.exchanges.alpaca_exchange.get_api_key')
    @patch('trading_bot.exchanges.alpaca_exchange.AlpacaClient')
    @patch('trading_bot.exchanges.alpaca_exchange.AlpacaStream')
    def test_exchange_get_account(self, mock_stream, mock_client_class, mock_get_key):
        """Test exchange get_account method."""
        mock_get_key.side_effect = lambda exchange, key: f'test_{key}'
        
        # Mock client methods
        mock_client = Mock()
        mock_client.get_account.return_value = {
            'cash': 10000.0,
            'equity': 12000.0,
            'buying_power': 20000.0
        }
        mock_client_class.return_value = mock_client
        
        exchange = AlpacaExchange(self.mock_config, self.mock_logger)
        result = exchange.get_account()
        
        self.assertEqual(result['cash_balance'], 10000.0)
        self.assertEqual(result['total_equity'], 12000.0)
        self.assertEqual(result['buying_power'], 20000.0)
    
    @patch('trading_bot.exchanges.alpaca_exchange.get_api_key')
    @patch('trading_bot.exchanges.alpaca_exchange.AlpacaClient')
    @patch('trading_bot.exchanges.alpaca_exchange.AlpacaStream')
    def test_exchange_get_exchange_name(self, mock_stream, mock_client_class, mock_get_key):
        """Test exchange name retrieval."""
        mock_get_key.side_effect = lambda exchange, key: f'test_{key}'
        mock_client_class.return_value = Mock()
        
        exchange = AlpacaExchange(self.mock_config, self.mock_logger)
        name = exchange.get_exchange_name()
        
        self.assertEqual(name, 'alpaca_paper')


class TestExchangeFactory(unittest.TestCase):
    """Test cases for exchange factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.mock_config = Mock()
        self.mock_db_manager = Mock()
    
    @patch('trading_bot.exchanges.factory.PaperTradingExchange')
    def test_create_paper_exchange(self, mock_paper_class):
        """Test creating paper exchange."""
        self.mock_config.get = Mock(return_value=False)
        mock_paper_class.return_value = Mock()
        
        exchange = create_exchange(self.mock_config, self.mock_logger, self.mock_db_manager)
        
        mock_paper_class.assert_called_once()
        self.assertIsNotNone(exchange)
    
    @patch('trading_bot.exchanges.factory.AlpacaExchange')
    @patch('trading_bot.exchanges.factory.get_api_key')
    def test_create_alpaca_exchange(self, mock_get_key, mock_alpaca_class):
        """Test creating Alpaca exchange."""
        mock_get_key.side_effect = lambda exchange, key: f'test_{key}'
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'exchanges.alpaca.enabled': True,
            'exchanges.alpaca.paper_trading': True
        }.get(key, default))
        mock_alpaca_class.return_value = Mock()
        
        exchange = create_exchange(self.mock_config, self.mock_logger, self.mock_db_manager, exchange_type='alpaca')
        
        mock_alpaca_class.assert_called_once()
        self.assertIsNotNone(exchange)
    
    @patch('trading_bot.exchanges.factory.PaperTradingExchange')
    def test_create_exchange_auto_detect_paper(self, mock_paper_class):
        """Test auto-detection of paper exchange."""
        self.mock_config.get = Mock(return_value=False)
        mock_paper_class.return_value = Mock()
        
        exchange = create_exchange(self.mock_config, self.mock_logger, self.mock_db_manager)
        
        mock_paper_class.assert_called_once()


    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_get_order_status_with_remaining_qty(self, mock_data_client, mock_trading_client):
        """Test get_order_status includes remaining_qty."""
        mock_order = Mock()
        mock_order.id = 'order_123'
        mock_order.qty = 10.0
        mock_order.filled_qty = 7.0
        mock_order.status = Mock()
        mock_order.status.value = 'partially_filled'
        mock_order.filled_avg_price = 155.5
        
        mock_client_instance = Mock()
        mock_client_instance.get_order_by_id.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.get_order_status('order_123')
        
        self.assertEqual(result['order_id'], 'order_123')
        self.assertEqual(result['filled_qty'], 7.0)
        self.assertEqual(result['remaining_qty'], 3.0)
        self.assertEqual(result['filled_avg_price'], 155.5)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_retry_with_backoff_429_error(self, mock_data_client, mock_trading_client):
        """Test retry logic for 429 rate limit errors."""
        from alpaca.common.exceptions import APIError
        
        mock_client_instance = Mock()
        error_429 = APIError('Rate limit exceeded')
        error_429.status_code = 429
        
        # First two calls fail, third succeeds
        mock_client_instance.get_account.side_effect = [error_429, error_429, Mock(cash=10000.0, equity=12000.0, buying_power=20000.0, portfolio_value=12000.0, currency='USD')]
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.get_account()
        
        # Should eventually succeed after retries
        self.assertNotIn('error', result)
        self.assertEqual(result['cash'], 10000.0)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_retry_with_backoff_no_retry_on_400(self, mock_data_client, mock_trading_client):
        """Test that 400 errors are not retried."""
        from alpaca.common.exceptions import APIError
        
        mock_client_instance = Mock()
        error_400 = APIError('Bad request')
        error_400.status_code = 400
        
        mock_client_instance.get_account.side_effect = error_400
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.get_account()
        
        # Should not retry 400 errors
        self.assertIn('error', result)
        self.assertEqual(mock_client_instance.get_account.call_count, 1)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_get_bars_timeframe_conversion(self, mock_data_client, mock_trading_client):
        """Test timeframe conversion for get_bars."""
        import pandas as pd
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        
        mock_bars_data = Mock()
        mock_bars_data.data = {'AAPL': []}
        
        mock_data_instance = Mock()
        mock_data_instance.get_stock_bars.return_value = mock_bars_data
        mock_data_client.return_value = mock_data_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        
        # Test different timeframes
        result = client.get_bars('AAPL', '1Min', limit=10)
        self.assertIsInstance(result, pd.DataFrame)
        
        result = client.get_bars('AAPL', '5Min', limit=10)
        self.assertIsInstance(result, pd.DataFrame)
        
        result = client.get_bars('AAPL', '1Hour', limit=10)
        self.assertIsInstance(result, pd.DataFrame)
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_place_order_error_mapping_insufficient_funds(self, mock_data_client, mock_trading_client):
        """Test error mapping for insufficient funds."""
        from alpaca.common.exceptions import APIError
        
        mock_client_instance = Mock()
        error = APIError('Insufficient funds')
        error.status_code = 403
        mock_client_instance.submit_order.side_effect = error
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.place_order('AAPL', 1000, 'buy', 'market', 'day', None)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'INSUFFICIENT_FUNDS')
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_place_order_error_mapping_invalid_symbol(self, mock_data_client, mock_trading_client):
        """Test error mapping for invalid symbol."""
        from alpaca.common.exceptions import APIError
        
        mock_client_instance = Mock()
        error = APIError('Symbol not found')
        error.status_code = 404
        mock_client_instance.submit_order.side_effect = error
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.place_order('INVALID', 10, 'buy', 'market', 'day', None)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'INVALID_SYMBOL')
    
    @patch('trading_bot.exchanges.alpaca_client.TradingClient')
    @patch('trading_bot.exchanges.alpaca_client.StockHistoricalDataClient')
    def test_place_order_error_mapping_market_closed(self, mock_data_client, mock_trading_client):
        """Test error mapping for market closed."""
        from alpaca.common.exceptions import APIError
        
        mock_client_instance = Mock()
        error = APIError('Market is closed')
        error.status_code = 422
        mock_client_instance.submit_order.side_effect = error
        mock_trading_client.return_value = mock_client_instance
        
        client = AlpacaClient(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        result = client.place_order('AAPL', 10, 'buy', 'market', 'day', None)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'MARKET_CLOSED')
    
    @patch('trading_bot.exchanges.alpaca_exchange.get_api_key')
    @patch('trading_bot.exchanges.alpaca_exchange.AlpacaClient')
    @patch('trading_bot.exchanges.alpaca_exchange.AlpacaStream')
    def test_place_order_respects_market_hours_config(self, mock_stream, mock_client_class, mock_get_key):
        """Test that place_order respects market hours config."""
        mock_get_key.side_effect = lambda exchange, key: f'test_{key}'
        
        mock_client = Mock()
        mock_client.is_market_open.return_value = False
        mock_client.place_order.return_value = {'success': True}
        mock_client_class.return_value = mock_client
        
        # Test with respect_market_hours=True
        config_with_respect = Mock()
        config_with_respect.get = Mock(side_effect=lambda key, default=None: {
            'exchanges.alpaca.paper_trading': True,
            'exchanges.alpaca.respect_market_hours': True,
            'exchanges.alpaca.extended_hours': False
        }.get(key, default))
        
        exchange = AlpacaExchange(config_with_respect, Mock())
        result = exchange.place_order('AAPL', 10, 'buy', 'market', 'day', None)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'MARKET_CLOSED')
        
        # Test with respect_market_hours=False
        config_no_respect = Mock()
        config_no_respect.get = Mock(side_effect=lambda key, default=None: {
            'exchanges.alpaca.paper_trading': True,
            'exchanges.alpaca.respect_market_hours': False,
            'exchanges.alpaca.extended_hours': False
        }.get(key, default))
        
        exchange2 = AlpacaExchange(config_no_respect, Mock())
        result2 = exchange2.place_order('AAPL', 10, 'buy', 'market', 'day', None)
        
        # Should allow order even when market is closed
        self.assertTrue(result2['success'])


class TestAlpacaStream(unittest.TestCase):
    """Test cases for AlpacaStream."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.api_key = 'test_api_key'
        self.api_secret = 'test_api_secret'
    
    @patch('trading_bot.exchanges.alpaca_stream.StockDataStream')
    def test_stream_init_with_config(self, mock_stream_class):
        """Test stream initialization with config."""
        mock_config = Mock()
        mock_config.get = Mock(side_effect=lambda key, default=None: {
            'exchanges.alpaca.streaming.auto_reconnect': True,
            'exchanges.alpaca.streaming.reconnect_delay_seconds': 10
        }.get(key, default))
        
        mock_stream_instance = Mock()
        mock_stream_class.return_value = mock_stream_instance
        
        stream = AlpacaStream(self.api_key, self.api_secret, paper_mode=True, 
                             logger=self.mock_logger, config=mock_config)
        
        self.assertTrue(stream.auto_reconnect)
        self.assertEqual(stream.reconnect_delay, 10)
    
    @patch('trading_bot.exchanges.alpaca_stream.StockDataStream')
    def test_subscribe_bars_with_timeframe(self, mock_stream_class):
        """Test subscribe_bars accepts timeframe parameter."""
        mock_stream_instance = Mock()
        mock_stream_class.return_value = mock_stream_instance
        
        stream = AlpacaStream(self.api_key, self.api_secret, paper_mode=True, logger=self.mock_logger)
        stream.state = ConnectionState.CONNECTED
        
        callback = Mock()
        stream.subscribe_bars(['AAPL'], callback, timeframe='5Min')
        
        # Verify callback was registered
        self.assertIn('AAPL', stream.callbacks['bars'])
        self.assertEqual(stream.callbacks['bars']['AAPL']['timeframe'], '5Min')
    
    @patch('trading_bot.exchanges.alpaca_stream.StockDataStream')
    def test_reconnect_scheduling(self, mock_stream_class):
        """Test reconnect scheduling on disconnect."""
        mock_stream_instance = Mock()
        mock_stream_class.return_value = mock_stream_instance
        
        stream = AlpacaStream(self.api_key, self.api_secret, paper_mode=True, 
                             logger=self.mock_logger, auto_reconnect=True)
        stream.state = ConnectionState.CONNECTED
        stream._stop_event.clear()
        
        # Trigger disconnect
        stream._on_disconnect()
        
        # Should schedule reconnect
        self.assertEqual(stream.state, ConnectionState.DISCONNECTED)
        self.assertEqual(stream.reconnect_attempts, 1)
    
    @patch('trading_bot.exchanges.alpaca_stream.StockDataStream')
    def test_reconnect_backoff(self, mock_stream_class):
        """Test exponential backoff for reconnection."""
        mock_stream_instance = Mock()
        mock_stream_class.return_value = mock_stream_instance
        
        stream = AlpacaStream(self.api_key, self.api_secret, paper_mode=True, 
                             logger=self.mock_logger, auto_reconnect=True)
        stream.reconnect_delay = 1
        stream.max_reconnect_delay = 10
        stream._stop_event.clear()
        
        # Simulate multiple reconnect attempts
        stream._schedule_reconnect()
        attempts_1 = stream.reconnect_attempts
        
        stream._schedule_reconnect()
        attempts_2 = stream.reconnect_attempts
        
        self.assertEqual(attempts_2, attempts_1 + 1)


if __name__ == '__main__':
    unittest.main()

