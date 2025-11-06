"""
Paper Trading Exchange Adapter

Wraps the existing PaperTradingEngine to implement ExchangeInterface,
maintaining backward compatibility.
"""

from typing import Dict, Optional

from trading_bot.exchanges.exchange_interface import ExchangeInterface
from trading_bot.trading.paper_trading import PaperTradingEngine


class PaperTradingExchange(ExchangeInterface):
    """
    Adapter wrapping PaperTradingEngine to implement ExchangeInterface.
    
    Maintains backward compatibility by wrapping the existing paper trading
    engine behind the new interface.
    """
    
    def __init__(self, config, logger, db_manager):
        """
        Initialize the paper trading exchange adapter.
        
        Args:
            config: Configuration object
            logger: Logger instance
            db_manager: DatabaseManager instance
        """
        self.config = config
        self.logger = logger
        self.db_manager = db_manager
        
        # Instantiate the wrapped engine
        self.engine = PaperTradingEngine(config, logger, db_manager)
        
        self.logger.info("PaperTradingExchange initialized")
    
    def get_account(self) -> Dict:
        """Get account information."""
        try:
            summary = self.engine.get_account_summary()
            return {
                'cash_balance': summary.get('cash_balance', 0.0),
                'total_equity': summary.get('total_equity', 0.0),
                'buying_power': summary.get('available_buying_power', summary.get('cash_balance', 0.0)),
                'positions_value': summary.get('positions_value', 0.0)
            }
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
            return self.engine.get_position(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get position for {symbol}: {str(e)}")
            return None
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all open positions."""
        try:
            return self.engine.get_all_positions()
        except Exception as e:
            self.logger.error(f"Failed to get all positions: {str(e)}")
            return {}
    
    def place_order(self, symbol: str, qty: float, side: str,
                   order_type: str = 'market', time_in_force: str = 'day',
                   limit_price: Optional[float] = None) -> Dict:
        """Place an order."""
        try:
            # Get current price for paper trading simulation
            # For market orders, use limit_price if provided (strategy can pass current_price here)
            # Otherwise, try to get from existing position for sell orders
            current_price = limit_price
            
            if current_price is None or current_price <= 0:
                if side.lower() == 'sell':
                    # For sell orders, try to get price from existing position
                    position = self.engine.get_position(symbol)
                    if position:
                        current_price = position.get('current_price', position.get('entry_price', 0.0))
                    else:
                        return {
                            'success': False,
                            'order_id': None,
                            'execution_price': None,
                            'error': f'No position found for {symbol} and no price provided'
                        }
                else:
                    # For buy orders, we need a price
                    return {
                        'success': False,
                        'order_id': None,
                        'execution_price': None,
                        'error': 'Price required for paper trading buy orders (pass as limit_price parameter)'
                    }
            
            if side.lower() == 'buy':
                result = self.engine.execute_buy_order(
                    symbol, qty, current_price, order_type=order_type
                )
            elif side.lower() == 'sell':
                result = self.engine.execute_sell_order(
                    symbol, qty, current_price, order_type=order_type
                )
            else:
                return {
                    'success': False,
                    'error': f'Invalid side: {side}'
                }
            
            # Map result to interface format
            if result.get('success'):
                return {
                    'success': True,
                    'order_id': result.get('symbol', '') + '_' + str(result.get('timestamp', '')),
                    'execution_price': result.get('execution_price'),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'order_id': None,
                    'execution_price': None,
                    'error': result.get('error', 'Unknown error')
                }
        except Exception as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            return {
                'success': False,
                'order_id': None,
                'execution_price': None,
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order (not supported in paper trading)."""
        return {
            'success': False,
            'error': 'Order cancellation not supported in paper trading (orders execute immediately)'
        }
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status (not supported in paper trading)."""
        return {
            'order_id': order_id,
            'status': 'not_supported',
            'filled_qty': 0.0,
            'remaining_qty': 0.0
        }
    
    def close_position(self, symbol: str, current_price: Optional[float] = None,
                      reason: str = 'manual') -> Dict:
        """Close a position."""
        try:
            if current_price is None:
                # Try to get current price from position
                position = self.engine.get_position(symbol)
                if position:
                    current_price = position.get('current_price', position.get('entry_price', 0.0))
                else:
                    return {
                        'success': False,
                        'error': f'No position found for {symbol} and no current_price provided'
                    }
            
            self.engine.close_position(symbol, current_price, reason=reason)
            return {
                'success': True,
                'error': None
            }
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_position_prices(self, market_data: Dict[str, float]) -> None:
        """Update position prices with current market data."""
        try:
            self.engine.update_position_prices(market_data)
        except Exception as e:
            self.logger.error(f"Failed to update position prices: {str(e)}")
    
    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary."""
        try:
            return self.engine.get_account_summary()
        except Exception as e:
            self.logger.error(f"Failed to get account summary: {str(e)}")
            return {}
    
    def is_market_open(self) -> bool:
        """Check if market is open (paper trading always available)."""
        return True
    
    def get_exchange_name(self) -> str:
        """Get exchange identifier name."""
        return 'paper'
    
    # Additional methods for direct access to paper engine
    def get_paper_engine(self) -> PaperTradingEngine:
        """Return wrapped engine for direct access if needed."""
        return self.engine
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.engine.reset_portfolio()
    
    def get_trade_history(self, limit: Optional[int] = None, symbol: Optional[str] = None):
        """Get trade history."""
        return self.engine.get_trade_history(limit=limit, symbol=symbol)
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        return self.engine.calculate_performance_metrics()

