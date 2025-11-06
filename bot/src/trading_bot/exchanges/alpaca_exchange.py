"""
Alpaca Exchange Adapter

Wraps AlpacaClient and AlpacaStream to provide a unified exchange interface for the strategy.
"""

import threading
import time
from typing import Dict, Optional, List

from trading_bot.exchanges.exchange_interface import ExchangeInterface
from trading_bot.exchanges.alpaca_client import AlpacaClient
from trading_bot.exchanges.alpaca_stream import AlpacaStream
from trading_bot.utils.secrets_store import get_api_key


class AlpacaExchange(ExchangeInterface):
    """
    Alpaca exchange adapter implementing ExchangeInterface.
    
    Wraps AlpacaClient and AlpacaStream to provide a unified exchange interface
    for the strategy.
    """
    
    def __init__(self, config, logger, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the Alpaca exchange adapter.
        
        Args:
            config: Configuration object
            logger: Logger instance
            api_key: Optional API key override (otherwise loaded from secrets)
            api_secret: Optional API secret override (otherwise loaded from secrets)
        """
        self.config = config
        self.logger = logger
        
        # Load API credentials from secrets store if not provided
        if api_key is None:
            api_key = get_api_key('alpaca', 'api_key')
        if api_secret is None:
            api_secret = get_api_key('alpaca', 'api_secret')
        
        if not api_key or not api_secret:
            raise ValueError(
                "Alpaca API credentials not found. "
                "Set them using secrets_store.store_api_key('alpaca', 'api_key', 'YOUR_KEY')"
            )
        
        # Get paper_mode from config
        paper_mode = config.get('exchanges.alpaca.paper_trading', True)
        if paper_mode is None:
            # Fallback to deprecated config location
            paper_mode = config.get('api.exchanges.alpaca.paper_trading', True)
        
        # Get config endpoints
        base_url = config.get('exchanges.alpaca.base_url')
        data_url = config.get('exchanges.alpaca.data_url')
        stream_url = config.get('exchanges.alpaca.stream_url')
        
        # Create Alpaca client with optional endpoints
        self.client = AlpacaClient(
            api_key, api_secret, paper_mode=paper_mode, logger=logger,
            base_url=base_url, data_url=data_url
        )
        
        # Create Alpaca stream with config and optional stream_url
        self.stream = AlpacaStream(
            api_key, api_secret, paper_mode=paper_mode, logger=logger,
            config=config, stream_url=stream_url
        )
        
        # Position cache
        self.position_cache: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()
        self.cache_timestamps = {
            'account': 0,
            'positions': 0,
            'market_open': 0
        }
        
        # Cache TTLs (in seconds)
        self.cache_ttl = {
            'account': 1,
            'positions': 5,
            'market_open': 60
        }
        
        # Set up streaming callback for position updates
        self.stream.subscribe_trades(
            list(self.position_cache.keys()),
            self._on_trade_update
        )
        
        mode_str = "Paper" if paper_mode else "Live"
        self.logger.info(f"AlpacaExchange initialized in {mode_str} mode")
    
    def get_account(self) -> Dict:
        """Get account information."""
        try:
            # Check cache
            now = time.time()
            if now - self.cache_timestamps['account'] < self.cache_ttl['account']:
                # Return cached account data if available
                if hasattr(self, '_cached_account'):
                    return self._cached_account
            
            # Fetch from API
            account_data = self.client.get_account()
            
            if 'error' in account_data:
                return account_data
            
            # Map to interface format
            result = {
                'cash_balance': account_data.get('cash', 0.0),
                'total_equity': account_data.get('equity', 0.0),
                'buying_power': account_data.get('buying_power', 0.0),
                'positions_value': account_data.get('equity', 0.0) - account_data.get('cash', 0.0)
            }
            
            # Cache result
            self._cached_account = result
            self.cache_timestamps['account'] = now
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to get account: {str(e)}")
            return {
                'cash_balance': 0.0,
                'total_equity': 0.0,
                'buying_power': 0.0,
                'positions_value': 0.0
            }
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a symbol."""
        try:
            # Check cache first
            with self.cache_lock:
                if symbol in self.position_cache:
                    now = time.time()
                    if now - self.cache_timestamps['positions'] < self.cache_ttl['positions']:
                        return self.position_cache[symbol].copy()
            
            # Fetch from API
            position = self.client.get_position(symbol)
            
            if position is None:
                # Remove from cache if exists
                with self.cache_lock:
                    self.position_cache.pop(symbol, None)
                return None
            
            # Map to interface format
            result = {
                'symbol': position['symbol'],
                'quantity': position['qty'],
                'entry_price': position['avg_entry_price'],
                'current_price': position['current_price'],
                'unrealized_pnl': position['unrealized_pl'],
                'stop_loss': None,  # Managed separately
                'take_profit': None  # Managed separately
            }
            
            # Update cache
            with self.cache_lock:
                self.position_cache[symbol] = result
                self.cache_timestamps['positions'] = time.time()
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to get position for {symbol}: {str(e)}")
            return None
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all open positions."""
        try:
            # Check cache
            now = time.time()
            if now - self.cache_timestamps['positions'] < self.cache_ttl['positions']:
                with self.cache_lock:
                    if self.position_cache:
                        return {k: v.copy() for k, v in self.position_cache.items()}
            
            # Fetch from API
            positions = self.client.get_all_positions()
            
            # Convert list to dict and map to interface format
            result = {}
            with self.cache_lock:
                self.position_cache.clear()
                
                for pos in positions:
                    mapped_pos = {
                        'symbol': pos['symbol'],
                        'quantity': pos['qty'],
                        'entry_price': pos['avg_entry_price'],
                        'current_price': pos['current_price'],
                        'unrealized_pnl': pos['unrealized_pl'],
                        'stop_loss': None,
                        'take_profit': None
                    }
                    result[pos['symbol']] = mapped_pos
                    self.position_cache[pos['symbol']] = mapped_pos
                
                self.cache_timestamps['positions'] = time.time()
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to get all positions: {str(e)}")
            return {}
    
    def place_order(self, symbol: str, qty: float, side: str,
                   order_type: str = 'market', time_in_force: str = 'day',
                   limit_price: Optional[float] = None) -> Dict:
        """Place an order."""
        try:
            # Check if market hours should be respected
            respect_market_hours = self.config.get('exchanges.alpaca.respect_market_hours', True)
            extended_hours = self.config.get('exchanges.alpaca.extended_hours', False)
            
            # Validate market is open only if respect_market_hours is enabled
            if respect_market_hours:
                if not self.is_market_open():
                    # If extended hours is enabled, allow orders outside regular hours
                    # Note: This is a simplified check - actual extended hours support
                    # would require checking specific extended hours time windows
                    if not extended_hours:
                        return {
                            'success': False,
                            'order_id': None,
                            'execution_price': None,
                            'error': 'MARKET_CLOSED',
                            'message': 'Market is currently closed and extended hours trading is disabled'
                        }
                    # extended_hours is enabled - allow order to proceed
                    # Note: Alpaca SDK may handle extended_hours parameter in order request
                    # For now, we just allow the order to proceed
            
            # Place order
            result = self.client.place_order(
                symbol, qty, side, order_type, time_in_force, limit_price
            )
            
            # Map response to interface format
            if result.get('success'):
                # Invalidate position cache
                with self.cache_lock:
                    self.position_cache.pop(symbol, None)
                    self.cache_timestamps['positions'] = 0
                    self.cache_timestamps['account'] = 0
                
                return {
                    'success': True,
                    'order_id': result.get('order_id'),
                    'execution_price': result.get('filled_avg_price'),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'order_id': None,
                    'execution_price': None,
                    'error': result.get('error', 'UNKNOWN_ERROR'),
                    'message': result.get('message', 'Order placement failed')
                }
        except Exception as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            return {
                'success': False,
                'order_id': None,
                'execution_price': None,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        try:
            result = self.client.cancel_order(order_id)
            return result
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return {
                'success': False,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        try:
            result = self.client.get_order_status(order_id)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            return {
                'order_id': order_id,
                'status': 'error',
                'filled_qty': 0.0,
                'remaining_qty': 0.0,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def close_position(self, symbol: str, current_price: Optional[float] = None,
                      reason: str = 'manual') -> Dict:
        """Close a position."""
        try:
            result = self.client.close_position(symbol)
            
            # Invalidate position cache
            with self.cache_lock:
                self.position_cache.pop(symbol, None)
                self.cache_timestamps['positions'] = 0
                self.cache_timestamps['account'] = 0
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {str(e)}")
            return {
                'success': False,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def update_position_prices(self, market_data: Dict[str, float]) -> None:
        """Update position prices with current market data."""
        try:
            with self.cache_lock:
                for symbol, price in market_data.items():
                    if symbol in self.position_cache:
                        position = self.position_cache[symbol]
                        position['current_price'] = price
                        
                        # Recalculate unrealized PnL
                        entry_price = position['entry_price']
                        quantity = position['quantity']
                        position['unrealized_pnl'] = (price - entry_price) * quantity
                        
                        if self.logger:
                            self.logger.debug(f"Updated price for {symbol}: ${price:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to update position prices: {str(e)}")
    
    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary."""
        try:
            account = self.get_account()
            positions = self.get_all_positions()
            
            # Calculate additional metrics
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
            
            return {
                'cash_balance': account.get('cash_balance', 0.0),
                'total_equity': account.get('total_equity', 0.0),
                'buying_power': account.get('buying_power', 0.0),
                'positions_value': account.get('positions_value', 0.0),
                'num_positions': len(positions),
                'available_buying_power': account.get('buying_power', 0.0),
                'total_pnl': total_pnl
            }
        except Exception as e:
            self.logger.error(f"Failed to get account summary: {str(e)}")
            return {}
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        try:
            # Check cache
            now = time.time()
            if now - self.cache_timestamps['market_open'] < self.cache_ttl['market_open']:
                if hasattr(self, '_cached_market_open'):
                    return self._cached_market_open
            
            # Fetch from API
            is_open = self.client.is_market_open()
            
            # Cache result
            self._cached_market_open = is_open
            self.cache_timestamps['market_open'] = now
            
            return is_open
        except Exception as e:
            self.logger.error(f"Failed to check market status: {str(e)}")
            return False
    
    def get_exchange_name(self) -> str:
        """Get exchange identifier name."""
        paper_mode = self.config.get('exchanges.alpaca.paper_trading', True)
        if paper_mode is None:
            paper_mode = self.config.get('api.exchanges.alpaca.paper_trading', True)
        
        return 'alpaca_paper' if paper_mode else 'alpaca_live'
    
    def start_streaming(self, symbols: List[str]) -> None:
        """
        Start streaming real-time data for symbols.
        
        Args:
            symbols: List of symbols to stream
        """
        try:
            # Subscribe to trades for real-time price updates
            self.stream.subscribe_trades(symbols, self._on_trade_update)
            
            # Start stream if not already running
            if not self.stream.is_connected():
                self.stream.start()
            
            self.logger.info(f"Started streaming for {len(symbols)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {str(e)}")
    
    def stop_streaming(self) -> None:
        """Stop streaming."""
        try:
            self.stream.stop()
            self.logger.info("Stopped streaming")
        except Exception as e:
            self.logger.error(f"Failed to stop streaming: {str(e)}")
    
    def _on_trade_update(self, trade_data: Dict) -> None:
        """
        Handle trade update from stream.
        
        Args:
            trade_data: Trade data dictionary
        """
        try:
            symbol = trade_data['symbol']
            price = trade_data['price']
            
            # Update position cache
            with self.cache_lock:
                if symbol in self.position_cache:
                    position = self.position_cache[symbol]
                    position['current_price'] = price
                    
                    # Recalculate unrealized PnL
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    position['unrealized_pnl'] = (price - entry_price) * quantity
        except Exception as e:
            self.logger.error(f"Error processing trade update: {str(e)}")
    
    def refresh_positions(self) -> None:
        """Force refresh position cache."""
        with self.cache_lock:
            self.cache_timestamps['positions'] = 0
            self.position_cache.clear()
    
    def get_client(self) -> AlpacaClient:
        """Return underlying client for advanced usage."""
        return self.client
    
    def get_stream(self) -> AlpacaStream:
        """Return stream instance."""
        return self.stream

