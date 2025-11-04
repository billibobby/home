"""
Paper Trading Engine Module

Simulates realistic order execution and portfolio management for risk-free strategy testing.
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from trading_bot.utils.exceptions import DatabaseError


class PaperTradingEngine:
    """
    Paper trading engine that simulates realistic order execution.
    
    Maintains virtual portfolio state, simulates slippage and commissions,
    and tracks positions with stop-loss/take-profit levels.
    """
    
    def __init__(self, config, logger, database_manager):
        """
        Initialize the paper trading engine.
        
        Args:
            config: Configuration object
            logger: Logger instance
            database_manager: DatabaseManager instance
        """
        self.config = config
        self.logger = logger
        self.db = database_manager
        
        # Load paper trading configuration using dot-notation
        self.enabled = config.get('paper_trading.enabled', True)
        self.initial_balance = config.get('paper_trading.initial_balance', 10000.0)
        self.commission_rate = config.get('paper_trading.commission', 0.1) / 100.0  # Convert to decimal
        
        # Slippage configuration
        self.slippage_enabled = config.get('paper_trading.slippage.enabled', True)
        self.slippage_min = config.get('paper_trading.slippage.min_percentage', 0.01) / 100.0
        self.slippage_max = config.get('paper_trading.slippage.max_percentage', 0.1) / 100.0
        self.market_order_slippage = config.get('paper_trading.slippage.market_order_slippage', 0.2) / 100.0
        
        # Execution settings
        self.execution_delay_ms = config.get('paper_trading.execution_delay_ms', 100)
        self.partial_fills = config.get('paper_trading.partial_fills', False)
        self.reset_on_startup = config.get('paper_trading.reset_on_startup', False)
        
        # Load trading parameters
        self.stop_loss_pct = config.get('trading.stop_loss_percentage', 2) / 100.0
        self.take_profit_pct = config.get('trading.take_profit_percentage', 5) / 100.0
        
        # Portfolio state
        self.cash_balance = 0.0
        self.positions = {}  # {symbol: position_dict}
        self.total_equity = 0.0
        
        # Load or initialize portfolio state
        self._load_portfolio_state()
        
        self.logger.info(
            f"Paper trading engine initialized (enabled: {self.enabled}, "
            f"initial_balance: ${self.initial_balance:.2f})"
        )
    
    def _load_portfolio_state(self):
        """Load portfolio state from database or initialize fresh."""
        try:
            if self.reset_on_startup:
                self.logger.info("Resetting portfolio on startup")
                self._reset_portfolio_internal()
                return
            
            # Try to get latest snapshot
            latest_snapshot = self.db.get_latest_snapshot()
            
            if latest_snapshot:
                # Restore from snapshot
                self.cash_balance = latest_snapshot.get('cash_balance', self.initial_balance)
                self.total_equity = latest_snapshot.get('total_equity', self.initial_balance)
                
                # Load open positions
                db_positions = self.db.get_all_positions()
                for pos in db_positions:
                    symbol = pos['symbol']
                    self.positions[symbol] = {
                        'id': pos['id'],
                        'symbol': symbol,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'quantity': pos['quantity'],
                        'current_price': pos['current_price'],
                        'unrealized_pnl': pos.get('unrealized_pnl', 0.0),
                        'stop_loss': pos.get('stop_loss'),
                        'take_profit': pos.get('take_profit'),
                        'entry_time': pos['entry_time']
                    }
                
                self.logger.info(
                    f"Portfolio state restored: cash=${self.cash_balance:.2f}, "
                    f"positions={len(self.positions)}"
                )
            else:
                # Initialize fresh
                self.cash_balance = self.initial_balance
                self.total_equity = self.initial_balance
                self.positions = {}
                self.logger.info(f"Initialized fresh portfolio: ${self.initial_balance:.2f}")
        
        except Exception as e:
            self.logger.error(f"Failed to load portfolio state: {str(e)}")
            # Fallback to fresh initialization
            self.cash_balance = self.initial_balance
            self.total_equity = self.initial_balance
            self.positions = {}
    
    def _save_portfolio_snapshot(self):
        """Save current portfolio state to database."""
        try:
            positions_value = sum(
                pos['quantity'] * pos.get('current_price', pos['entry_price'])
                for pos in self.positions.values()
            )
            
            total_pnl = self.total_equity - self.initial_balance
            
            # Calculate daily PnL (compare with previous snapshot)
            previous_snapshot = self.db.get_latest_snapshot()
            daily_pnl = None
            if previous_snapshot:
                previous_equity = previous_snapshot.get('total_equity', self.initial_balance)
                daily_pnl = self.total_equity - previous_equity
            
            self.db.insert_portfolio_snapshot(
                total_equity=self.total_equity,
                cash_balance=self.cash_balance,
                positions_value=positions_value,
                num_positions=len(self.positions),
                daily_pnl=daily_pnl,
                total_pnl=total_pnl
            )
        
        except Exception as e:
            self.logger.error(f"Failed to save portfolio snapshot: {str(e)}")
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self._reset_portfolio_internal()
        self.logger.info("Portfolio reset completed")
    
    def _reset_portfolio_internal(self):
        """Internal method to reset portfolio."""
        # Close all positions in database
        for symbol in list(self.positions.keys()):
            try:
                self.db.delete_position(symbol, commit=False)
            except Exception as e:
                self.logger.warning(f"Failed to delete position {symbol}: {str(e)}")
        
        # Reset state
        self.cash_balance = self.initial_balance
        self.total_equity = self.initial_balance
        self.positions = {}
        
        # Save snapshot
        self._save_portfolio_snapshot()
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio equity."""
        positions_value = sum(
            pos['quantity'] * pos.get('current_price', pos['entry_price'])
            for pos in self.positions.values()
        )
        self.total_equity = self.cash_balance + positions_value
        return self.total_equity
    
    def execute_buy_order(self, symbol: str, quantity: float, price: float,
                         order_type: str = 'market', signal: Optional[Dict] = None) -> Dict:
        """
        Execute a buy order.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares/units to buy
            price: Expected price per unit
            order_type: Order type ('market', 'limit', etc.)
            signal: Optional signal dictionary for metadata
            
        Returns:
            Execution result dictionary
        """
        try:
            # Validate order parameters
            self._validate_order_params(symbol, quantity, price)
            
            # Check if partial fills are enabled
            if self.partial_fills:
                result = self._execute_partial_fills(symbol, quantity, price, order_type, is_sell=False, signal=signal)
                if result['success']:
                    # Update portfolio value and save snapshot
                    self.get_portfolio_value()
                    self._save_portfolio_snapshot()
                return result
            
            # Simulate execution delay
            if self.execution_delay_ms > 0:
                time.sleep(self.execution_delay_ms / 1000.0)
            
            # Apply slippage
            execution_price = self._simulate_slippage(price, order_type)
            slippage_pct = ((execution_price - price) / price) * 100.0 if price > 0 else 0.0
            
            # Calculate costs
            total_cost = execution_price * quantity
            commission = self._calculate_commission(execution_price, quantity)
            total_required = total_cost + commission
            
            # Check sufficient balance
            if not self.can_execute_order(symbol, quantity, execution_price, 'buy')[0]:
                return {
                    'success': False,
                    'error': 'Insufficient funds',
                    'required': total_required,
                    'available': self.cash_balance
                }
            
            # Deduct cash
            self.cash_balance -= total_required
            
            # Update or create position
            if symbol in self.positions:
                # Update existing position (average price)
                existing = self.positions[symbol]
                total_quantity = existing['quantity'] + quantity
                total_cost_existing = existing['entry_price'] * existing['quantity']
                avg_entry_price = (total_cost_existing + total_cost) / total_quantity
                
                existing['quantity'] = total_quantity
                existing['entry_price'] = avg_entry_price
                existing['current_price'] = execution_price
                
                # Update in database
                self.db.update_position(
                    symbol,
                    quantity=total_quantity,
                    entry_price=avg_entry_price,
                    current_price=execution_price
                )
                
                self.logger.info(
                    f"Position updated: {symbol} +{quantity} @ ${execution_price:.2f} "
                    f"(avg: ${avg_entry_price:.2f}, total: {total_quantity})"
                )
            else:
                # Create new position
                stop_loss = execution_price * (1 - self.stop_loss_pct)
                take_profit = execution_price * (1 + self.take_profit_pct)
                
                position = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'entry_price': execution_price,
                    'quantity': quantity,
                    'current_price': execution_price,
                    'unrealized_pnl': 0.0,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now().isoformat()
                }
                
                self.positions[symbol] = position
                
                # Insert into database
                position_id = self.db.insert_position(
                    symbol=symbol,
                    side='BUY',
                    entry_price=execution_price,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                position['id'] = position_id
                
                self.logger.info(
                    f"Buy order executed: {symbol} {quantity} @ ${execution_price:.2f} "
                    f"(slippage: {slippage_pct:.2f}%, commission: ${commission:.2f})"
                )
            
            # Update portfolio value
            self.get_portfolio_value()
            
            # Save snapshot
            self._save_portfolio_snapshot()
            
            return {
                'success': True,
                'symbol': symbol,
                'quantity': quantity,
                'execution_price': execution_price,
                'slippage': slippage_pct,
                'commission': commission,
                'total_cost': total_required,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Buy order execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_sell_order(self, symbol: str, quantity: float, price: float,
                          order_type: str = 'market', signal: Optional[Dict] = None) -> Dict:
        """
        Execute a sell order.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares/units to sell
            price: Expected price per unit
            order_type: Order type ('market', 'limit', etc.)
            signal: Optional signal dictionary for metadata
            
        Returns:
            Execution result dictionary
        """
        try:
            # Validate order parameters
            self._validate_order_params(symbol, quantity, price)
            
            # Check position exists
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f'No position found for {symbol}'
                }
            
            position = self.positions[symbol]
            
            # Check sufficient quantity
            if quantity > position['quantity']:
                return {
                    'success': False,
                    'error': f'Insufficient quantity: requested {quantity}, have {position["quantity"]}'
                }
            
            # Check if partial fills are enabled
            if self.partial_fills:
                result = self._execute_partial_fills(symbol, quantity, price, order_type, is_sell=True, signal=signal)
                if result['success']:
                    # Update portfolio value and save snapshot
                    self.get_portfolio_value()
                    self._save_portfolio_snapshot()
                return result
            
            # Simulate execution delay
            if self.execution_delay_ms > 0:
                time.sleep(self.execution_delay_ms / 1000.0)
            
            # Apply slippage (negative for sells)
            execution_price = self._simulate_slippage(price, order_type, is_sell=True)
            slippage_pct = ((price - execution_price) / price) * 100.0 if price > 0 else 0.0
            
            # Calculate proceeds
            proceeds = execution_price * quantity
            commission = self._calculate_commission(execution_price, quantity)
            net_proceeds = proceeds - commission
            
            # Calculate PnL
            entry_price = position['entry_price']
            pnl = (execution_price - entry_price) * quantity - commission
            pnl_percentage = ((execution_price - entry_price) / entry_price) * 100.0
            
            # Add proceeds to cash
            self.cash_balance += net_proceeds
            
            # Update position
            remaining_quantity = position['quantity'] - quantity
            
            if remaining_quantity <= 0.0001:  # Close entire position (with tolerance for floating point)
                # Close position
                entry_time = position.get('entry_time', datetime.now().isoformat())
                
                # Move to trades table (pass commission to ensure PnL is correct)
                trade_id = self.db.close_position(symbol, execution_price, commission=commission)
                
                # Remove from in-memory positions
                del self.positions[symbol]
                
                self.logger.info(
                    f"Sell order executed (CLOSED): {symbol} {quantity} @ ${execution_price:.2f} "
                    f"(PnL: ${pnl:.2f}, {pnl_percentage:.2f}%)"
                )
            else:
                # Partial sell - update position
                position['quantity'] = remaining_quantity
                position['current_price'] = execution_price
                
                # Update in database
                self.db.update_position(
                    symbol,
                    quantity=remaining_quantity,
                    current_price=execution_price
                )
                
                self.logger.info(
                    f"Sell order executed (PARTIAL): {symbol} {quantity} @ ${execution_price:.2f} "
                    f"(remaining: {remaining_quantity}, PnL: ${pnl:.2f})"
                )
            
            # Update portfolio value
            self.get_portfolio_value()
            
            # Save snapshot
            self._save_portfolio_snapshot()
            
            return {
                'success': True,
                'symbol': symbol,
                'quantity': quantity,
                'execution_price': execution_price,
                'slippage': slippage_pct,
                'commission': commission,
                'net_proceeds': net_proceeds,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'position_closed': remaining_quantity <= 0.0001,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Sell order execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute_partial_fills(self, symbol: str, total_quantity: float, price: float,
                               order_type: str, is_sell: bool, signal: Optional[Dict] = None) -> Dict:
        """
        Execute an order with partial fills simulation.
        
        Args:
            symbol: Trading symbol
            total_quantity: Total quantity to fill
            price: Expected price
            order_type: Order type
            is_sell: Whether this is a sell order
            signal: Optional signal dictionary
            
        Returns:
            Aggregated execution result
        """
        # Determine number of fills (2-4 random fills)
        num_fills = random.randint(2, 4)
        fill_quantities = []
        
        # Distribute quantity across fills (with some randomness)
        remaining = total_quantity
        for i in range(num_fills - 1):
            # Random portion between 20-40% of remaining
            portion = random.uniform(0.2, 0.4)
            fill_qty = remaining * portion
            fill_quantities.append(fill_qty)
            remaining -= fill_qty
        fill_quantities.append(remaining)  # Last fill gets the remainder
        
        # Calculate base slippage range
        if order_type == 'market':
            base_slippage = self.market_order_slippage
        else:
            base_slippage = random.uniform(self.slippage_min, self.slippage_max)
        
        # For sell orders, capture entry info before fills
        entry_info = None
        if is_sell and symbol in self.positions:
            pos = self.positions[symbol]
            entry_info = {
                'entry_price': pos['entry_price'],
                'entry_time': pos.get('entry_time', datetime.now().isoformat()),
                'timeframe': pos.get('timeframe'),
                'strategy': pos.get('strategy')
            }
        
        total_executed = 0.0
        total_cost = 0.0
        total_commission = 0.0
        execution_prices = []
        fill_quantities_executed = []
        position_closed = False
        
        # Execute each fill
        for i, fill_qty in enumerate(fill_quantities):
            # Check if position was closed in previous fill
            if is_sell and (symbol not in self.positions or position_closed):
                break
            
            # Small delay between fills (10-50ms)
            if i > 0:
                delay_ms = random.uniform(10, 50)
                time.sleep(delay_ms / 1000.0)
            
            # Vary execution price within slippage bounds
            price_variation = random.uniform(-base_slippage * 0.5, base_slippage * 0.5)
            fill_price = price * (1 + price_variation) if not is_sell else price * (1 - price_variation)
            
            # Execute this fill
            if is_sell:
                fill_result = self._execute_single_sell_fill(symbol, fill_qty, fill_price, order_type, signal)
                if fill_result.get('position_closed', False):
                    position_closed = True
            else:
                fill_result = self._execute_single_buy_fill(symbol, fill_qty, fill_price, order_type, signal)
            
            if not fill_result['success']:
                # If a fill fails, stop and return partial result
                break
            
            executed_qty = fill_result['quantity']
            total_executed += executed_qty
            total_cost += fill_result.get('total_cost', 0) if not is_sell else -fill_result.get('net_proceeds', 0)
            total_commission += fill_result.get('commission', 0)
            execution_prices.append(fill_price)
            fill_quantities_executed.append(executed_qty)
        
        # If position was closed during fills, update trade record with average execution price
        if is_sell and position_closed and entry_info and total_executed > 0:
            # Get the most recent trade (should be the one we just created)
            trades = self.db.get_trades_by_symbol(symbol, limit=1)
            if trades:
                trade = trades[0]
                # Calculate PnL with average execution price
                avg_execution_price = sum(p * q for p, q in zip(execution_prices, fill_quantities_executed)) / total_executed
                pnl = (avg_execution_price - entry_info['entry_price']) * total_executed - total_commission
                pnl_percentage = ((avg_execution_price - entry_info['entry_price']) / entry_info['entry_price']) * 100.0
                
                # Update trade with correct PnL and average execution price
                self.db.update_trade(
                    trade['id'],
                    exit_price=avg_execution_price,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    notes=f"Partial fills: {len(execution_prices)} fills; Commission: ${total_commission:.2f}"
                )
        
        # Calculate average execution price
        avg_execution_price = sum(p * q for p, q in zip(execution_prices, fill_quantities_executed)) / total_executed if total_executed > 0 else price
        avg_slippage = ((avg_execution_price - price) / price) * 100.0 if price > 0 and not is_sell else ((price - avg_execution_price) / price) * 100.0 if price > 0 else 0.0
        
        return {
            'success': total_executed > 0,
            'symbol': symbol,
            'quantity': total_executed,
            'execution_price': avg_execution_price,
            'slippage': avg_slippage,
            'commission': total_commission,
            'total_cost': abs(total_cost) + total_commission if not is_sell else None,
            'net_proceeds': abs(total_cost) - total_commission if is_sell else None,
            'timestamp': datetime.now().isoformat(),
            'num_fills': len(execution_prices)
        }
    
    def _execute_single_buy_fill(self, symbol: str, quantity: float, execution_price: float,
                                 order_type: str, signal: Optional[Dict] = None) -> Dict:
        """Execute a single buy fill (internal helper for partial fills)."""
        # Calculate costs
        total_cost = execution_price * quantity
        commission = self._calculate_commission(execution_price, quantity)
        total_required = total_cost + commission
        
        # Check sufficient balance
        if total_required > self.cash_balance:
            return {'success': False, 'error': 'Insufficient funds'}
        
        # Deduct cash
        self.cash_balance -= total_required
        
        # Update or create position
        if symbol in self.positions:
            existing = self.positions[symbol]
            total_quantity = existing['quantity'] + quantity
            total_cost_existing = existing['entry_price'] * existing['quantity']
            avg_entry_price = (total_cost_existing + total_cost) / total_quantity
            
            existing['quantity'] = total_quantity
            existing['entry_price'] = avg_entry_price
            existing['current_price'] = execution_price
            
            self.db.update_position(
                symbol,
                quantity=total_quantity,
                entry_price=avg_entry_price,
                current_price=execution_price
            )
        else:
            stop_loss = execution_price * (1 - self.stop_loss_pct)
            take_profit = execution_price * (1 + self.take_profit_pct)
            
            position = {
                'symbol': symbol,
                'side': 'BUY',
                'entry_price': execution_price,
                'quantity': quantity,
                'current_price': execution_price,
                'unrealized_pnl': 0.0,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now().isoformat()
            }
            
            self.positions[symbol] = position
            
            position_id = self.db.insert_position(
                symbol=symbol,
                side='BUY',
                entry_price=execution_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            position['id'] = position_id
        
        return {
            'success': True,
            'quantity': quantity,
            'total_cost': total_required,
            'commission': commission
        }
    
    def _execute_single_sell_fill(self, symbol: str, quantity: float, execution_price: float,
                                  order_type: str, signal: Optional[Dict] = None) -> Dict:
        """Execute a single sell fill (internal helper for partial fills)."""
        if symbol not in self.positions:
            return {'success': False, 'error': 'No position found'}
        
        position = self.positions[symbol]
        
        if quantity > position['quantity']:
            return {'success': False, 'error': 'Insufficient quantity'}
        
        # Calculate proceeds
        proceeds = execution_price * quantity
        commission = self._calculate_commission(execution_price, quantity)
        net_proceeds = proceeds - commission
        
        # Add proceeds to cash
        self.cash_balance += net_proceeds
        
        # Update position
        remaining_quantity = position['quantity'] - quantity
        position_closed = False
        
        if remaining_quantity <= 0.0001:
            # Close position (will be updated with average price later if partial fills)
            trade_id = self.db.close_position(symbol, execution_price, commission=commission)
            del self.positions[symbol]
            position_closed = True
        else:
            position['quantity'] = remaining_quantity
            position['current_price'] = execution_price
            self.db.update_position(
                symbol,
                quantity=remaining_quantity,
                current_price=execution_price
            )
        
        return {
            'success': True,
            'quantity': quantity,
            'net_proceeds': net_proceeds,
            'commission': commission,
            'position_closed': position_closed
        }
    
    def _simulate_slippage(self, price: float, order_type: str, is_sell: bool = False) -> float:
        """
        Simulate slippage based on order type.
        
        Args:
            price: Original price
            order_type: Order type ('market', 'limit', etc.)
            is_sell: Whether this is a sell order
            
        Returns:
            Adjusted price with slippage
        """
        if not self.slippage_enabled:
            return price
        
        # Market orders get higher slippage
        if order_type == 'market':
            slippage_pct = self.market_order_slippage
        else:
            # Random slippage within range
            slippage_pct = random.uniform(self.slippage_min, self.slippage_max)
        
        # For sells, slippage reduces price (negative)
        if is_sell:
            return price * (1 - slippage_pct)
        else:
            return price * (1 + slippage_pct)
    
    def _calculate_commission(self, price: float, quantity: float) -> float:
        """
        Calculate trading commission.
        
        Args:
            price: Price per unit
            quantity: Number of units
            
        Returns:
            Commission amount
        """
        trade_value = price * quantity
        return trade_value * self.commission_rate
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all open positions."""
        return self.positions.copy()
    
    def update_position_prices(self, market_data: Dict[str, float]):
        """
        Update position prices with current market data.
        
        Args:
            market_data: Dictionary of {symbol: current_price}
        """
        for symbol, current_price in market_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                entry_price = position['entry_price']
                quantity = position['quantity']
                side = position['side']
                
                # Calculate unrealized PnL
                multiplier = 1 if side == 'BUY' else -1
                unrealized_pnl = (current_price - entry_price) * quantity * multiplier
                
                # Update position
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                
                # Update in database
                self.db.update_position_price(symbol, current_price)
        
        # Update portfolio value
        self.get_portfolio_value()
    
    def check_stop_loss_take_profit(self):
        """Check and execute stop-loss/take-profit orders."""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            current_price = position.get('current_price', position['entry_price'])
            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')
            
            trigger_reason = None
            
            if stop_loss and current_price <= stop_loss:
                trigger_reason = 'stop_loss'
                positions_to_close.append((symbol, current_price, trigger_reason))
            
            elif take_profit and current_price >= take_profit:
                trigger_reason = 'take_profit'
                positions_to_close.append((symbol, current_price, trigger_reason))
        
        # Execute closes
        for symbol, price, reason in positions_to_close:
            self.logger.info(f"Auto-closing {symbol} due to {reason} @ ${price:.2f}")
            self.close_position(symbol, price, reason=reason)
    
    def close_position(self, symbol: str, current_price: float, reason: str = 'manual'):
        """
        Close a position manually.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            reason: Reason for closure
        """
        if symbol not in self.positions:
            self.logger.warning(f"Cannot close position: {symbol} not found")
            return
        
        position = self.positions[symbol]
        quantity = position['quantity']
        
        # Execute sell order
        result = self.execute_sell_order(symbol, quantity, current_price)
        
        if result['success']:
            self.logger.info(f"Position closed: {symbol} (reason: {reason})")
        else:
            self.logger.error(f"Failed to close position {symbol}: {result.get('error')}")
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            trades = self.db.get_all_trades()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'total_return': 0.0,
                    'total_pnl': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': None,
                    'sortino_ratio': None
                }
            
            # Calculate basic stats
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
            win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
            
            wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
            losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
            
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            
            total_wins = sum(wins)
            total_losses = abs(sum(losses))  # Use absolute value for profit factor calculation
            profit_factor = total_wins / total_losses if total_losses > 0 else (total_wins if total_wins > 0 else 0.0)
            
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            total_return = ((self.total_equity - self.initial_balance) / self.initial_balance * 100.0) if self.initial_balance > 0 else 0.0
            
            # Calculate max drawdown from equity curve
            snapshots = self.db.get_snapshots()
            max_drawdown = 0.0
            if snapshots:
                equity_values = [s['total_equity'] for s in snapshots]
                peak = equity_values[0]
                for equity in equity_values:
                    if equity > peak:
                        peak = equity
                    drawdown = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            # Calculate Sharpe ratio (simplified - would need returns series)
            sharpe_ratio = None
            sortino_ratio = None
            
            if len(snapshots) >= 30:  # Need sufficient data points
                # Simple Sharpe calculation using daily returns
                returns = []
                prev_equity = self.initial_balance
                for snap in sorted(snapshots, key=lambda x: x['timestamp']):
                    equity = snap['total_equity']
                    if prev_equity > 0:
                        daily_return = (equity - prev_equity) / prev_equity
                        returns.append(daily_return)
                    prev_equity = equity
                
                if returns and len(returns) > 1:
                    import statistics
                    mean_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0
                    
                    # Annualized Sharpe (assuming 252 trading days)
                    if std_return > 0:
                        sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)
                        
                        # Sortino ratio (only downside deviation)
                        downside_returns = [r for r in returns if r < 0]
                        if downside_returns and len(downside_returns) > 1:
                            downside_std = statistics.stdev(downside_returns)
                            if downside_std > 0:
                                sortino_ratio = (mean_return / downside_std) * (252 ** 0.5)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }
        
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {str(e)}")
            return {}
    
    def get_daily_pnl(self) -> float:
        """Calculate PnL for current day."""
        try:
            # Get today's midnight timestamp
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_str = today.isoformat()
            
            # Get all snapshots before today
            snapshots = self.db.get_snapshots(end_date=today_str, limit=1)
            
            if snapshots:
                # Get the most recent snapshot before today
                previous_snapshot = snapshots[0]
                previous_equity = previous_snapshot.get('total_equity', self.initial_balance)
                return self.total_equity - previous_equity
            
            # If no previous snapshot, return 0 (or compare to initial balance)
            return self.total_equity - self.initial_balance
        except Exception as e:
            self.logger.error(f"Failed to calculate daily PnL: {str(e)}")
            return 0.0
    
    def get_total_pnl(self) -> float:
        """Get total PnL."""
        return self.total_equity - self.initial_balance
    
    def get_pnl_percentage(self) -> float:
        """Get PnL as percentage."""
        if self.initial_balance > 0:
            return ((self.total_equity - self.initial_balance) / self.initial_balance) * 100.0
        return 0.0
    
    def execute_signal(self, signal: Dict, current_price: float, 
                      account_balance: Optional[float] = None) -> Dict:
        """
        Execute a trading signal.
        
        Args:
            signal: Signal dictionary from SignalGenerator
            current_price: Current market price
            account_balance: Optional account balance (uses current if not provided)
            
        Returns:
            Execution result dictionary
        """
        signal_type = signal.get('type')
        symbol = signal.get('symbol')
        
        if not symbol:
            return {
                'success': False,
                'error': 'Signal missing symbol'
            }
        
        # Determine order side
        if signal_type in ['BUY', 'STRONG_BUY']:
            # Calculate position size
            if account_balance is None:
                account_balance = self.cash_balance
            
            # Use position size from signal or calculate
            position_size = signal.get('position_size')
            if position_size is None:
                # Default to small position
                position_size = account_balance * 0.1  # 10% of balance
            
            quantity = position_size / current_price
            
            return self.execute_buy_order(symbol, quantity, current_price, signal=signal)
        
        elif signal_type in ['SELL', 'STRONG_SELL']:
            # Sell entire position if exists
            if symbol in self.positions:
                quantity = self.positions[symbol]['quantity']
                return self.execute_sell_order(symbol, quantity, current_price, signal=signal)
            else:
                return {
                    'success': False,
                    'error': f'No position to sell for {symbol}'
                }
        
        else:  # HOLD
            return {
                'success': False,
                'error': 'HOLD signal - no action taken'
            }
    
    def can_execute_order(self, symbol: str, quantity: float, price: float, side: str) -> Tuple[bool, str]:
        """
        Check if an order can be executed.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            side: Order side ('buy' or 'sell')
            
        Returns:
            Tuple of (can_execute: bool, reason: str)
        """
        try:
            self._validate_order_params(symbol, quantity, price)
            
            if side.lower() == 'buy':
                total_cost = price * quantity
                commission = self._calculate_commission(price, quantity)
                total_required = total_cost + commission
                
                if total_required > self.cash_balance:
                    return False, f'Insufficient funds: need ${total_required:.2f}, have ${self.cash_balance:.2f}'
                
                return True, 'OK'
            
            elif side.lower() == 'sell':
                if symbol not in self.positions:
                    return False, f'No position found for {symbol}'
                
                position = self.positions[symbol]
                if quantity > position['quantity']:
                    return False, f'Insufficient quantity: have {position["quantity"]}, need {quantity}'
                
                return True, 'OK'
            
            else:
                return False, f'Invalid side: {side}'
        
        except Exception as e:
            return False, str(e)
    
    def get_account_summary(self) -> Dict:
        """Get account summary."""
        positions_value = sum(
            pos['quantity'] * pos.get('current_price', pos['entry_price'])
            for pos in self.positions.values()
        )
        total_equity = self.cash_balance + positions_value
        
        return {
            'cash_balance': self.cash_balance,
            'total_equity': total_equity,
            'positions_value': positions_value,
            'num_positions': len(self.positions),
            'total_pnl': total_equity - self.initial_balance,
            'pnl_percentage': ((total_equity - self.initial_balance) / self.initial_balance * 100.0) if self.initial_balance > 0 else 0.0,
            'available_buying_power': self.cash_balance,
            'initial_balance': self.initial_balance
        }
    
    def get_trade_history(self, limit: Optional[int] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Get trade history with optional filters."""
        if symbol:
            return self.db.get_trades_by_symbol(symbol, limit=limit)
        else:
            return self.db.get_all_trades(limit=limit)
    
    def _validate_order_params(self, symbol: str, quantity: float, price: float):
        """Validate order parameters."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive: {quantity}")
        
        if price <= 0:
            raise ValueError(f"Price must be positive: {price}")

