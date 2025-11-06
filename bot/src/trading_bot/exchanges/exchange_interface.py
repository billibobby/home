"""
Exchange Interface Abstract Base Class

Defines the common contract that all exchange implementations must follow,
enabling strategy code to work with any exchange.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class ExchangeInterface(ABC):
    """
    Abstract base class defining the exchange interface.
    
    All exchange implementations (paper trading, Alpaca, etc.) must implement
    these methods to ensure compatibility with strategy code.
    """
    
    @abstractmethod
    def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Dictionary with keys:
                - cash_balance: float - Available cash
                - total_equity: float - Total account value
                - buying_power: float - Available buying power
                - positions_value: float - Value of all positions
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with keys:
                - symbol: str
                - quantity: float
                - entry_price: float
                - current_price: float
                - unrealized_pnl: float
                - stop_loss: Optional[float]
                - take_profit: Optional[float]
            Or None if position doesn't exist
        """
        pass
    
    @abstractmethod
    def get_all_positions(self) -> Dict[str, Dict]:
        """
        Get all open positions.
        
        Returns:
            Dictionary mapping symbol to position data (same format as get_position)
        """
        pass
    
    @abstractmethod
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
            Dictionary with keys:
                - success: bool
                - order_id: Optional[str] - Order ID if successful
                - execution_price: Optional[float] - Execution price if filled
                - error: Optional[str] - Error message if failed
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dictionary with keys:
                - success: bool
                - error: Optional[str] - Error message if failed
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dictionary with keys:
                - order_id: str
                - status: str - Order status (pending, filled, cancelled, etc.)
                - filled_qty: float - Filled quantity
                - remaining_qty: float - Remaining quantity
        """
        pass
    
    @abstractmethod
    def close_position(self, symbol: str, current_price: Optional[float] = None,
                      reason: str = 'manual') -> Dict:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price (optional)
            reason: Reason for closure
            
        Returns:
            Dictionary with keys:
                - success: bool
                - error: Optional[str] - Error message if failed
        """
        pass
    
    @abstractmethod
    def update_position_prices(self, market_data: Dict[str, float]) -> None:
        """
        Update position prices with current market data.
        
        Args:
            market_data: Dictionary mapping symbol to current price
        """
        pass
    
    @abstractmethod
    def get_account_summary(self) -> Dict:
        """
        Get comprehensive account summary.
        
        Returns:
            Dictionary with comprehensive account information including
            positions, balances, PnL, etc.
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is open for trading.
        
        Returns:
            True if market is open, False otherwise
        """
        pass
    
    @abstractmethod
    def get_exchange_name(self) -> str:
        """
        Get exchange identifier name.
        
        Returns:
            Exchange name string (e.g., 'paper', 'alpaca_paper', 'alpaca_live')
        """
        pass




