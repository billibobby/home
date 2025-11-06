"""
Alpaca REST API Client

Provides a clean interface to Alpaca's REST API for account info, orders, positions, and market data.
"""

import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
    )
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    TradingClient = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    StopOrderRequest = None
    StopLimitOrderRequest = None
    OrderSide = None
    OrderType = None
    TimeInForce = None
    StockHistoricalDataClient = None
    StockBarsRequest = None
    TimeFrame = None
    TimeFrameUnit = None
    APIError = Exception


class AlpacaClient:
    """
    REST API client for Alpaca using the alpaca-py SDK.
    
    Provides methods for account operations, order management, position tracking,
    and market data retrieval.
    """
    
    def __init__(self, api_key: str, api_secret: str, paper_mode: bool = True, logger=None,
                 base_url: Optional[str] = None, data_url: Optional[str] = None):
        """
        Initialize the Alpaca client.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper_mode: Use paper trading API (True) or live trading (False)
            logger: Logger instance
            base_url: Optional base URL override for TradingClient
            data_url: Optional data URL override for StockHistoricalDataClient
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is not installed. Install with: pip install alpaca-py>=0.20.0"
            )
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_mode = paper_mode
        self.logger = logger
        
        # Create trading client with optional base_url
        # Note: Check if TradingClient supports base_url parameter
        try:
            if base_url:
                self.trading_client = TradingClient(api_key, api_secret, paper=paper_mode, base_url=base_url)
            else:
                self.trading_client = TradingClient(api_key, api_secret, paper=paper_mode)
        except TypeError:
            # base_url not supported, use default
            self.trading_client = TradingClient(api_key, api_secret, paper=paper_mode)
            if self.logger and base_url:
                self.logger.warning(f"Base URL override not supported by SDK, using default URL")
        
        # Create data client with optional data_url
        # Note: Check if StockHistoricalDataClient supports data_url parameter
        try:
            if data_url:
                self.data_client = StockHistoricalDataClient(api_key, api_secret, url_override=data_url)
            else:
                self.data_client = StockHistoricalDataClient(api_key, api_secret)
        except TypeError:
            # data_url not supported, use default
            self.data_client = StockHistoricalDataClient(api_key, api_secret)
            if self.logger and data_url:
                self.logger.warning(f"Data URL override not supported by SDK, using default URL")
        
        # Rate limiting state
        self.request_timestamps = []
        self.rate_limit_requests_per_minute = 200
        self.rate_limit_requests_per_second = 10
        
        mode_str = "Paper Trading" if paper_mode else "Live Trading"
        if self.logger:
            self.logger.info(f"AlpacaClient initialized in {mode_str} mode")
    
    def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Dictionary with account data or error dict
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug("API call: get_account")
            
            self._check_rate_limit()
            account = self._retry_with_backoff(lambda: self.trading_client.get_account())
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: get_account - cash={account.cash}, equity={account.equity}")
            
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'currency': account.currency
            }
        except APIError as e:
            return self._handle_api_error(e)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get account: {str(e)}")
            return {
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position dictionary or None if not found
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API call: get_position - symbol={symbol}")
            
            self._check_rate_limit()
            position = self._retry_with_backoff(lambda: self.trading_client.get_open_position(symbol))
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: get_position - symbol={symbol}, qty={position.qty}")
            
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc)
            }
        except APIError as e:
            if e.status_code == 404:
                # Position doesn't exist
                return None
            return self._handle_api_error(e)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get position for {symbol}: {str(e)}")
            return None
    
    def get_all_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug("API call: get_all_positions")
            
            self._check_rate_limit()
            positions = self._retry_with_backoff(lambda: self.trading_client.get_all_positions())
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: get_all_positions - count={len(positions)}")
            
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }
                for pos in positions
            ]
        except APIError as e:
            if self.logger:
                self.logger.error(f"Failed to get all positions: {str(e)}")
            return []
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get all positions: {str(e)}")
            return []
    
    def place_order(self, symbol: str, qty: float, side: str,
                   order_type: str = 'market', time_in_force: str = 'day',
                   limit_price: Optional[float] = None) -> Dict:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price (required for limit orders)
            
        Returns:
            Order result dictionary
        """
        try:
            self._check_rate_limit()
            
            # Convert side
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Convert time in force
            tif_map = {
                'day': TimeInForce.DAY,
                'gtc': TimeInForce.GTC,
                'ioc': TimeInForce.IOC,
                'fok': TimeInForce.FOK
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
            
            # Create appropriate order request
            if order_type.lower() == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            elif order_type.lower() == 'limit':
                if limit_price is None:
                    return {
                        'success': False,
                        'error': 'INVALID_SYMBOL',
                        'message': 'limit_price required for limit orders'
                    }
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price
                )
            elif order_type.lower() == 'stop':
                if limit_price is None:
                    return {
                        'success': False,
                        'error': 'INVALID_SYMBOL',
                        'message': 'stop_price required for stop orders'
                    }
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    stop_price=limit_price
                )
            elif order_type.lower() == 'stop_limit':
                if limit_price is None:
                    return {
                        'success': False,
                        'error': 'INVALID_SYMBOL',
                        'message': 'limit_price and stop_price required for stop_limit orders'
                    }
                # For stop_limit, we'd need both stop and limit prices
                # This is simplified - in practice you'd need separate parameters
                order_request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                    stop_price=limit_price  # This should be separate in real implementation
                )
            else:
                return {
                    'success': False,
                    'error': 'INVALID_SYMBOL',
                    'message': f'Unsupported order type: {order_type}'
                }
            
            # Submit order
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API call: place_order - symbol={symbol}, qty={qty}, side={side}, type={order_type}")
            
            order = self._retry_with_backoff(lambda: self.trading_client.submit_order(order_request))
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: place_order - order_id={order.id}, status={order.status}")
            
            return {
                'success': True,
                'order_id': str(order.id),
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'status': order.status.value if hasattr(order.status, 'value') else str(order.status)
            }
        except APIError as e:
            error_dict = self._handle_api_error(e)
            return {
                'success': False,
                'order_id': None,
                'filled_avg_price': None,
                'status': None,
                **error_dict
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to place order: {str(e)}")
            return {
                'success': False,
                'order_id': None,
                'filled_avg_price': None,
                'status': None,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Success/error dictionary
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API call: cancel_order - order_id={order_id}")
            
            self._check_rate_limit()
            self._retry_with_backoff(lambda: self.trading_client.cancel_order_by_id(order_id))
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: cancel_order - order_id={order_id} cancelled")
            return {
                'success': True,
                'error': None
            }
        except APIError as e:
            return self._handle_api_error(e)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return {
                'success': False,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary with keys: order_id, status, filled_qty, remaining_qty
            (and optionally filled_avg_price in extra fields)
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API call: get_order_status - order_id={order_id}")
            
            self._check_rate_limit()
            order = self._retry_with_backoff(lambda: self.trading_client.get_order_by_id(order_id))
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: get_order_status - order_id={order_id}, status={order.status}")
            
            # Calculate remaining_qty from order.qty - order.filled_qty
            order_qty = float(order.qty) if order.qty else 0.0
            filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
            remaining_qty = order_qty - filled_qty
            
            result = {
                'order_id': str(order.id),
                'status': order.status.value if hasattr(order.status, 'value') else str(order.status),
                'filled_qty': filled_qty,
                'remaining_qty': remaining_qty
            }
            
            # Add filled_avg_price as an optional extra field
            if order.filled_avg_price:
                result['filled_avg_price'] = float(order.filled_avg_price)
            
            return result
        except APIError as e:
            error_dict = self._handle_api_error(e)
            return {
                'order_id': order_id,
                'status': 'error',
                'filled_qty': 0.0,
                'remaining_qty': 0.0,
                **error_dict
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            return {
                'order_id': order_id,
                'status': 'error',
                'filled_qty': 0.0,
                'remaining_qty': 0.0,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def get_bars(self, symbol: str, timeframe: str, limit: int = 100,
                 start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical bars data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string ('1Min', '5Min', '1Hour', '1Day')
            limit: Maximum number of bars
            start: Start datetime
            end: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._check_rate_limit()
            
            # Convert timeframe string to TimeFrame enum
            tf = self._convert_timeframe(timeframe)
            
            # Create request
            request_params = {
                'symbol_or_symbols': symbol,
                'timeframe': tf,
                'limit': limit
            }
            
            if start:
                request_params['start'] = start
            if end:
                request_params['end'] = end
            
            request = StockBarsRequest(**request_params)
            
            # Get bars
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API call: get_bars - symbol={symbol}, timeframe={timeframe}, limit={limit}")
            
            bars = self._retry_with_backoff(lambda: self.data_client.get_stock_bars(request))
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                bar_count = len(bars.data.get(symbol, [])) if bars.data else 0
                self.logger.debug(f"API response: get_bars - symbol={symbol}, bars_returned={bar_count}")
            
            # Convert to DataFrame
            data = []
            for bar in bars.data.get(symbol, []):
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get bars for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def close_position(self, symbol: str) -> Dict:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Success/error dictionary
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API call: close_position - symbol={symbol}")
            
            self._check_rate_limit()
            self._retry_with_backoff(lambda: self.trading_client.close_position(symbol))
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: close_position - symbol={symbol} closed")
            return {
                'success': True,
                'error': None
            }
        except APIError as e:
            return self._handle_api_error(e)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to close position {symbol}: {str(e)}")
            return {
                'success': False,
                'error': 'NETWORK_ERROR',
                'message': str(e)
            }
    
    def is_market_open(self) -> bool:
        """
        Check if market is open.
        
        Returns:
            True if market is open, False otherwise
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug("API call: is_market_open")
            
            self._check_rate_limit()
            clock = self._retry_with_backoff(lambda: self.trading_client.get_clock())
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: is_market_open - is_open={clock.is_open}")
            
            return clock.is_open
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to check market status: {str(e)}")
            return False
    
    def get_market_hours(self) -> Dict:
        """
        Get market hours information.
        
        Returns:
            Dictionary with next_open and next_close timestamps
        """
        try:
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug("API call: get_market_hours")
            
            self._check_rate_limit()
            clock = self._retry_with_backoff(lambda: self.trading_client.get_clock())
            
            if self.logger and self.logger.isEnabledFor(10):  # DEBUG level
                self.logger.debug(f"API response: get_market_hours - is_open={clock.is_open}")
            
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat() if clock.next_open else None,
                'next_close': clock.next_close.isoformat() if clock.next_close else None
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get market hours: {str(e)}")
            return {}
    
    def _convert_timeframe(self, timeframe_str: str) -> TimeFrame:
        """
        Convert timeframe string to TimeFrame enum.
        
        Args:
            timeframe_str: Timeframe string ('1Min', '5Min', '15Min', '1Hour', '1Day')
            
        Returns:
            TimeFrame enum value
        """
        if not ALPACA_AVAILABLE or TimeFrameUnit is None:
            raise ImportError("alpaca-py is not installed or TimeFrameUnit not available")
        
        tf_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day
        }
        return tf_map.get(timeframe_str, TimeFrame.Minute)
    
    def _handle_api_error(self, exception: Exception) -> Dict:
        """
        Handle Alpaca API exceptions and map to standard error codes.
        
        Args:
            exception: API exception
            
        Returns:
            Error dictionary with code and message
        """
        if not isinstance(exception, APIError):
            return {
                'error': 'NETWORK_ERROR',
                'message': str(exception)
            }
        
        status_code = getattr(exception, 'status_code', None)
        message = str(exception)
        
        # Map status codes to error types
        if status_code == 429:
            return {
                'error': 'RATE_LIMIT',
                'message': 'Rate limit exceeded'
            }
        elif status_code == 403:
            return {
                'error': 'INSUFFICIENT_FUNDS',
                'message': message or 'Insufficient funds or permissions'
            }
        elif status_code == 404:
            return {
                'error': 'INVALID_SYMBOL',
                'message': message or 'Symbol not found'
            }
        elif 'market is closed' in message.lower() or 'market closed' in message.lower():
            return {
                'error': 'MARKET_CLOSED',
                'message': 'Market is currently closed'
            }
        elif 'insufficient' in message.lower() or 'funds' in message.lower():
            return {
                'error': 'INSUFFICIENT_FUNDS',
                'message': message
            }
        else:
            return {
                'error': 'NETWORK_ERROR',
                'message': message
            }
    
    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if now - ts < 60
        ]
        
        # Check per-minute limit
        if len(self.request_timestamps) >= self.rate_limit_requests_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                if self.logger:
                    self.logger.warning(f"Rate limit approaching, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Check per-second limit
        recent_requests = [
            ts for ts in self.request_timestamps
            if now - ts < 1
        ]
        if len(recent_requests) >= self.rate_limit_requests_per_second:
            sleep_time = 1 - (now - recent_requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Record this request
        self.request_timestamps.append(now)
    
    def _retry_with_backoff(self, func, max_attempts: int = 3):
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            max_attempts: Maximum number of attempts
            
        Returns:
            Function result
        """
        for attempt in range(max_attempts):
            try:
                return func()
            except APIError as e:
                if e.status_code in [400, 403, 404]:
                    # Don't retry client errors
                    raise
                
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    if self.logger:
                        self.logger.warning(f"API error, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    if self.logger:
                        self.logger.warning(f"Error, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    raise

