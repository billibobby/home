"""
Alpaca WebSocket Streaming Client

Streams live trades, quotes, and bars from Alpaca's WebSocket API for real-time market data.
"""

import threading
import time
import random
from typing import Callable, List, Set, Dict, Optional
from enum import Enum

try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.models import Trade, Quote, Bar
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    StockDataStream = None
    Trade = None
    Quote = None
    Bar = None


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class AlpacaStream:
    """
    WebSocket streaming client for real-time Alpaca data.
    
    Provides methods to subscribe to trades, quotes, and bars with callback support.
    """
    
    def __init__(self, api_key: str, api_secret: str, paper_mode: bool = True, logger=None, 
                 config=None, stream_url: Optional[str] = None, auto_reconnect: bool = True):
        """
        Initialize the Alpaca stream client.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper_mode: Use paper trading stream (True) or live stream (False)
            logger: Logger instance
            config: Optional config object for reading reconnect settings
            stream_url: Optional stream URL override (if supported by SDK)
            auto_reconnect: Enable automatic reconnection with backoff
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is not installed. Install with: pip install alpaca-py>=0.20.0"
            )
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_mode = paper_mode
        self.logger = logger
        self.config = config
        self.auto_reconnect = auto_reconnect
        
        # Read reconnect settings from config if available
        if config:
            self.auto_reconnect = config.get('exchanges.alpaca.streaming.auto_reconnect', auto_reconnect)
            self.reconnect_delay = config.get('exchanges.alpaca.streaming.reconnect_delay_seconds', 5)
            self.max_reconnect_delay = config.get('exchanges.alpaca.streaming.max_reconnect_delay_seconds', 60)
        else:
            self.reconnect_delay = 5
            self.max_reconnect_delay = 60
        
        # Create stream client with optional URL override
        # Note: Check if StockDataStream supports url_override parameter
        # If not supported, this will need to be handled via SDK configuration
        try:
            if stream_url:
                # Try to pass url_override if SDK supports it
                # This may need adjustment based on actual SDK API
                self.stream = StockDataStream(api_key, api_secret, url_override=stream_url)
            else:
                self.stream = StockDataStream(api_key, api_secret)
        except TypeError:
            # URL override not supported, use default
            self.stream = StockDataStream(api_key, api_secret)
            if self.logger and stream_url:
                self.logger.warning(f"Stream URL override not supported by SDK, using default URL")
        
        # Subscription tracking
        self.subscribed_symbols: Set[str] = set()
        
        # Callback registry
        self.callbacks = {
            'trades': {},
            'quotes': {},
            'bars': {}
        }
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        
        # Reconnection state
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10  # Max attempts before giving up
        
        # Threading
        self.stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reconnect_timer: Optional[threading.Timer] = None
        
        if self.logger:
            mode_str = "Paper Trading" if paper_mode else "Live Trading"
            self.logger.info(f"AlpacaStream initialized in {mode_str} mode (auto_reconnect={self.auto_reconnect})")
    
    def subscribe_trades(self, symbols: List[str], callback: Callable) -> None:
        """
        Subscribe to trade updates.
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Callback function that receives trade data dict
        """
        for symbol in symbols:
            self.subscribed_symbols.add(symbol)
            self.callbacks['trades'][symbol] = callback
            
            # If connected, subscribe immediately
            if self.state == ConnectionState.CONNECTED:
                try:
                    wrapped_callback = self._wrap_trade_callback(callback)
                    self.stream.subscribe_trades(wrapped_callback, symbol)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to subscribe to trades for {symbol}: {str(e)}")
    
    def subscribe_quotes(self, symbols: List[str], callback: Callable) -> None:
        """
        Subscribe to quote updates (bid/ask).
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Callback function that receives quote data dict
        """
        for symbol in symbols:
            self.subscribed_symbols.add(symbol)
            self.callbacks['quotes'][symbol] = callback
            
            # If connected, subscribe immediately
            if self.state == ConnectionState.CONNECTED:
                try:
                    wrapped_callback = self._wrap_quote_callback(callback)
                    self.stream.subscribe_quotes(wrapped_callback, symbol)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to subscribe to quotes for {symbol}: {str(e)}")
    
    def subscribe_bars(self, symbols: List[str], callback: Callable, timeframe: str = '1Min') -> None:
        """
        Subscribe to bar updates (OHLCV).
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Callback function that receives bar data dict
            timeframe: Timeframe for bars (e.g., '1Min', '5Min', '15Min', '1Hour', '1Day')
                      Note: Alpaca streaming API typically only supports minute-based bars.
                      Non-minute timeframes may require client-side aggregation or REST API usage.
        """
        # Validate timeframe - Alpaca streaming typically only supports minute bars
        if not timeframe.endswith('Min') and timeframe not in ['1Min']:
            if self.logger:
                self.logger.warning(
                    f"Timeframe {timeframe} may not be supported by Alpaca streaming. "
                    f"Only minute-based bars are typically supported. "
                    f"Consider using AlpacaClient.get_bars() for non-minute timeframes."
                )
        
        for symbol in symbols:
            self.subscribed_symbols.add(symbol)
            # Store callback with timeframe info
            self.callbacks['bars'][symbol] = {
                'callback': callback,
                'timeframe': timeframe
            }
            
            # If connected, subscribe immediately
            if self.state == ConnectionState.CONNECTED:
                try:
                    wrapped_callback = self._wrap_bar_callback(callback)
                    self.stream.subscribe_bars(wrapped_callback, symbol)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to subscribe to bars for {symbol}: {str(e)}")
    
    def unsubscribe(self, symbols: List[str]) -> None:
        """
        Unsubscribe from updates for symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
            
            # Remove callbacks
            self.callbacks['trades'].pop(symbol, None)
            self.callbacks['quotes'].pop(symbol, None)
            self.callbacks['bars'].pop(symbol, None)
            
            # If connected, unsubscribe immediately
            if self.state == ConnectionState.CONNECTED:
                try:
                    self.stream.unsubscribe_trades(symbol)
                    self.stream.unsubscribe_quotes(symbol)
                    self.stream.unsubscribe_bars(symbol)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to unsubscribe {symbol}: {str(e)}")
    
    def start(self) -> None:
        """Start the WebSocket stream."""
        if self.state == ConnectionState.CONNECTED:
            if self.logger:
                self.logger.warning("Stream is already connected")
            return
        
        self.state = ConnectionState.CONNECTING
        self._stop_event.clear()
        
        # Start stream in background thread
        self.stream_thread = threading.Thread(target=self._run_stream, daemon=True)
        self.stream_thread.start()
        
        # Wait a bit for connection
        import time
        time.sleep(1)
        
        if self.state == ConnectionState.CONNECTED:
            if self.logger:
                self.logger.info("Alpaca stream connected successfully")
        else:
            if self.logger:
                self.logger.warning("Stream connection status unclear")
    
    def stop(self) -> None:
        """Stop the WebSocket stream."""
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        # Disable auto-reconnect
        self._stop_event.set()
        self.state = ConnectionState.DISCONNECTED
        
        # Cancel any pending reconnect
        if self._reconnect_timer:
            self._reconnect_timer.cancel()
        
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error stopping stream: {str(e)}")
        
        # Clear subscriptions
        self.subscribed_symbols.clear()
        self.callbacks = {'trades': {}, 'quotes': {}, 'bars': {}}
        self.reconnect_attempts = 0
        
        if self.logger:
            self.logger.info("Alpaca stream stopped")
    
    def _run_stream(self):
        """Run the stream in background thread."""
        try:
            # Note: Using private attributes for connection callbacks as public APIs
            # may not be available in the SDK. If the SDK provides public subscription
            # methods for connection events, those should be used instead.
            # This is a known risk and may break with SDK updates.
            if hasattr(self.stream, '_on_connect'):
                self.stream._on_connect = self._on_connect
            if hasattr(self.stream, '_on_disconnect'):
                self.stream._on_disconnect = self._on_disconnect
            if hasattr(self.stream, '_on_error'):
                self.stream._on_error = self._on_error
            
            # Run the stream (blocking call)
            self.stream.run()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Stream error: {str(e)}")
            self.state = ConnectionState.DISCONNECTED
            
            # Trigger reconnection if auto_reconnect is enabled
            if self.auto_reconnect and not self._stop_event.is_set():
                self._schedule_reconnect()
    
    def _on_connect(self):
        """Called when WebSocket connects."""
        self.state = ConnectionState.CONNECTED
        self.reconnect_attempts = 0  # Reset on successful connection
        if self.logger:
            self.logger.info("Alpaca WebSocket connected")
        
        # Resubscribe to all symbols
        self._resubscribe_all()
    
    def _on_disconnect(self):
        """Called when WebSocket disconnects."""
        if self.logger:
            self.logger.warning("Alpaca WebSocket disconnected")
        self.state = ConnectionState.DISCONNECTED
        
        # Trigger reconnection if auto_reconnect is enabled
        if self.auto_reconnect and not self._stop_event.is_set():
            self._schedule_reconnect()
    
    def _on_error(self, error):
        """Called when WebSocket error occurs."""
        if self.logger:
            self.logger.error(f"Alpaca WebSocket error: {str(error)}")
        
        # Trigger reconnection if auto_reconnect is enabled
        if self.auto_reconnect and not self._stop_event.is_set():
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Schedule a reconnection attempt with exponential backoff."""
        if self._stop_event.is_set():
            return
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            if self.logger:
                self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping reconnection.")
            return
        
        # Calculate backoff with exponential increase and jitter
        base_delay = self.reconnect_delay
        exponential_delay = min(base_delay * (2 ** self.reconnect_attempts), self.max_reconnect_delay)
        jitter = random.uniform(0, exponential_delay * 0.1)  # 10% jitter
        delay = exponential_delay + jitter
        
        self.reconnect_attempts += 1
        self.state = ConnectionState.RECONNECTING
        
        if self.logger:
            self.logger.info(f"Scheduling reconnection attempt {self.reconnect_attempts} in {delay:.2f}s")
        
        # Schedule reconnect in a separate thread to avoid blocking
        def reconnect():
            if self._stop_event.is_set():
                return
            
            time.sleep(delay)
            
            if self._stop_event.is_set():
                return
            
            try:
                if self.logger:
                    self.logger.info(f"Attempting reconnection ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                
                # Recreate stream if needed
                if not hasattr(self, 'stream') or self.stream is None:
                    self.stream = StockDataStream(self.api_key, self.api_secret)
                
                # Restart stream
                self.start()
                
                # Reset reconnect attempts on successful connection
                if self.state == ConnectionState.CONNECTED:
                    self.reconnect_attempts = 0
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Reconnection attempt failed: {str(e)}")
                # Schedule another reconnect attempt
                if not self._stop_event.is_set():
                    self._schedule_reconnect()
        
        reconnect_thread = threading.Thread(target=reconnect, daemon=True)
        reconnect_thread.start()
    
    def _resubscribe_all(self):
        """Resubscribe to all symbols after reconnection."""
        # Resubscribe trades
        for symbol, callback in self.callbacks['trades'].items():
            try:
                wrapped_callback = self._wrap_trade_callback(callback)
                self.stream.subscribe_trades(wrapped_callback, symbol)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to resubscribe trades for {symbol}: {str(e)}")
        
        # Resubscribe quotes
        for symbol, callback in self.callbacks['quotes'].items():
            try:
                wrapped_callback = self._wrap_quote_callback(callback)
                self.stream.subscribe_quotes(wrapped_callback, symbol)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to resubscribe quotes for {symbol}: {str(e)}")
        
        # Resubscribe bars
        for symbol, callback_data in self.callbacks['bars'].items():
            try:
                # Handle both old format (direct callback) and new format (dict with callback)
                if isinstance(callback_data, dict):
                    callback = callback_data['callback']
                else:
                    callback = callback_data
                wrapped_callback = self._wrap_bar_callback(callback)
                self.stream.subscribe_bars(wrapped_callback, symbol)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to resubscribe bars for {symbol}: {str(e)}")
    
    def _wrap_trade_callback(self, user_callback: Callable) -> Callable:
        """
        Wrap user callback to convert Trade object to dict.
        
        Args:
            user_callback: User's callback function
            
        Returns:
            Wrapped callback function
        """
        def wrapped(trade: Trade):
            try:
                trade_data = {
                    'symbol': trade.symbol,
                    'price': float(trade.price),
                    'size': int(trade.size),
                    'timestamp': trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                    'exchange': trade.exchange if hasattr(trade, 'exchange') else None
                }
                user_callback(trade_data)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in trade callback: {str(e)}")
        
        return wrapped
    
    def _wrap_quote_callback(self, user_callback: Callable) -> Callable:
        """
        Wrap user callback to convert Quote object to dict.
        
        Args:
            user_callback: User's callback function
            
        Returns:
            Wrapped callback function
        """
        def wrapped(quote: Quote):
            try:
                quote_data = {
                    'symbol': quote.symbol,
                    'bid_price': float(quote.bid_price),
                    'bid_size': int(quote.bid_size),
                    'ask_price': float(quote.ask_price),
                    'ask_size': int(quote.ask_size),
                    'timestamp': quote.timestamp.isoformat() if hasattr(quote.timestamp, 'isoformat') else str(quote.timestamp)
                }
                user_callback(quote_data)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in quote callback: {str(e)}")
        
        return wrapped
    
    def _wrap_bar_callback(self, user_callback: Callable) -> Callable:
        """
        Wrap user callback to convert Bar object to dict.
        
        Args:
            user_callback: User's callback function
            
        Returns:
            Wrapped callback function
        """
        def wrapped(bar: Bar):
            try:
                bar_data = {
                    'symbol': bar.symbol,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'timestamp': bar.timestamp.isoformat() if hasattr(bar.timestamp, 'isoformat') else str(bar.timestamp)
                }
                user_callback(bar_data)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in bar callback: {str(e)}")
        
        return wrapped
    
    def is_connected(self) -> bool:
        """Check if stream is connected."""
        return self.state == ConnectionState.CONNECTED
    
    def get_subscribed_symbols(self) -> Set[str]:
        """Get set of currently subscribed symbols."""
        return self.subscribed_symbols.copy()
    
    def get_connection_state(self) -> str:
        """Get connection state string."""
        return self.state.value

