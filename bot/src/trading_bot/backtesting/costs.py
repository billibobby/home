"""
Transaction Cost Model

Implements realistic transaction cost modeling for backtesting.
"""

import numpy as np
from typing import Dict


class TransactionCostModel:
    """
    Models transaction costs including commission, slippage, spread, and market impact.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the transaction cost model.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Load configuration
        cost_config = config.get('backtesting.transaction_costs', {})
        self.commission_pct = cost_config.get('commission_pct', 0.001)
        self.slippage_bps = cost_config.get('slippage_bps', 5)
        self.spread_bps = cost_config.get('spread_bps', 2)
        self.market_impact_enabled = cost_config.get('market_impact_enabled', True)
        self.market_impact_coeff = cost_config.get('market_impact_coeff', 0.01)
        
        # Validate parameters
        if self.commission_pct < 0:
            raise ValueError("commission_pct must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be non-negative")
        
        self.logger.info(
            f"TransactionCostModel initialized: commission={self.commission_pct:.4f}, "
            f"slippage={self.slippage_bps}bps, spread={self.spread_bps}bps"
        )
    
    def calculate_cost(self, price: float, quantity: float, avg_volume: float,
                      side: str, order_type: str = 'market', volatility: float = 0.02) -> Dict:
        """
        Calculate transaction cost for an order.
        
        Returns per-share adjustments and separate commission as trade-level fee.
        
        Args:
            price: Stock price
            quantity: Number of shares
            avg_volume: Average daily volume (for market impact calculation)
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            volatility: Price volatility (for slippage calculation)
            
        Returns:
            Dictionary with cost breakdown:
            - commission: Trade-level fee (not per-share)
            - slippage_per_share: Per-share slippage adjustment
            - spread_half: Half-spread per share (paid on each side)
            - impact_per_share: Per-share market impact
            - effective_price: Price after per-share adjustments (excludes commission)
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        trade_value = price * quantity
        
        # Calculate commission (trade-level, not per-share)
        commission = self.calculate_commission(trade_value)
        
        # Calculate per-share adjustments
        slippage_per_share = self.calculate_slippage_per_share(price, volatility, order_type)
        spread_half = self.calculate_spread_per_share(price, avg_volume) / 2.0  # Half-spread each side
        impact_per_share = 0.0
        
        if self.market_impact_enabled and avg_volume > 0:
            impact_per_share = self.calculate_market_impact_per_share(quantity, avg_volume, price, volatility)
            # Warn if market impact is too high (unrealistic large order)
            if impact_per_share / price > 0.01:  # > 1% impact per share
                self.logger.warning(
                    f"High market impact detected: {impact_per_share/price*100:.2f}% per share "
                    f"for {quantity} shares (avg volume: {avg_volume:.0f})"
                )
        
        # Effective price applies only per-share adjustments (excludes commission)
        effective_price = self.get_effective_price(price, slippage_per_share, spread_half, impact_per_share, side)
        
        # Total cost for reporting (per-share costs * quantity + commission)
        total_cost = (slippage_per_share + spread_half + impact_per_share) * quantity + commission
        
        return {
            'total_cost': total_cost,
            'commission': commission,
            'slippage_per_share': slippage_per_share,
            'spread_half': spread_half,
            'impact_per_share': impact_per_share,
            'effective_price': effective_price,
            'cost_pct': (total_cost / trade_value * 100) if trade_value > 0 else 0.0
        }
    
    def calculate_commission(self, trade_value: float) -> float:
        """
        Calculate commission cost.
        
        Args:
            trade_value: Total trade value
            
        Returns:
            Commission amount
        """
        return trade_value * self.commission_pct
    
    def calculate_slippage_per_share(self, price: float, volatility: float,
                                     order_type: str) -> float:
        """
        Calculate slippage cost per share.
        
        Args:
            price: Stock price
            volatility: Price volatility
            order_type: 'market' or 'limit'
            
        Returns:
            Slippage per share
        """
        # Base slippage per share
        base_slippage_per_share = price * (self.slippage_bps / 10000)
        
        # Market orders have higher slippage
        if order_type == 'market':
            slippage_multiplier = 1.5
        else:
            slippage_multiplier = 1.0
        
        # Adjust for volatility (higher volatility = higher slippage)
        volatility_factor = 1.0 + (volatility * 10)  # Scale volatility
        
        slippage_per_share = base_slippage_per_share * slippage_multiplier * volatility_factor
        
        return slippage_per_share
    
    def calculate_spread_per_share(self, price: float, avg_volume: float = None) -> float:
        """
        Calculate bid-ask spread cost per share (full spread).
        
        Spread is responsive to liquidity - lower volume stocks have wider spreads.
        
        Args:
            price: Stock price
            avg_volume: Average daily volume (optional, for liquidity adjustment)
            
        Returns:
            Full spread per share (half-spread is paid on each side)
        """
        base_spread = price * (self.spread_bps / 10000)
        
        # Adjust spread based on liquidity if volume provided
        if avg_volume is not None and avg_volume > 0:
            # Normalize volume (typical high-volume stock: 10M+ shares/day)
            # Low volume stocks (< 1M) get wider spreads
            volume_threshold = 1000000  # 1M shares
            
            if avg_volume < volume_threshold:
                # Increase spread for low liquidity (inverse relationship)
                liquidity_factor = 1.0 + (volume_threshold - avg_volume) / volume_threshold
                # Cap at 3x base spread for very low liquidity
                liquidity_factor = min(liquidity_factor, 3.0)
                base_spread *= liquidity_factor
        
        return base_spread
    
    def calculate_market_impact_per_share(self, quantity: float, avg_volume: float,
                                         price: float, volatility: float) -> float:
        """
        Calculate market impact per share using square root model.
        
        Formula: price * market_impact_coeff * sqrt(quantity / avg_volume) * volatility
        
        Args:
            quantity: Number of shares
            avg_volume: Average daily volume
            price: Stock price
            volatility: Price volatility
            
        Returns:
            Market impact per share
        """
        if avg_volume <= 0:
            return 0.0
        
        # Square root model: impact scales with sqrt(order_size / avg_volume)
        participation_rate = quantity / avg_volume
        impact_per_share = price * self.market_impact_coeff * np.sqrt(participation_rate) * volatility
        
        return impact_per_share
    
    def get_effective_price(self, price: float, slippage_per_share: float,
                           spread_half: float, impact_per_share: float, side: str) -> float:
        """
        Calculate effective execution price after per-share adjustments.
        
        Commission is excluded from effective price and should be subtracted from cash separately.
        
        Args:
            price: Original price
            slippage_per_share: Per-share slippage adjustment
            spread_half: Half-spread per share
            impact_per_share: Per-share market impact
            side: 'buy' or 'sell'
            
        Returns:
            Effective price (after per-share adjustments, excluding commission)
        """
        # Aggregate per-share adjustments
        per_share_adjustment = slippage_per_share + spread_half + impact_per_share
        
        if side == 'buy':
            # Buy orders pay more (price + per-share adjustments)
            return price + per_share_adjustment
        else:
            # Sell orders receive less (price - per-share adjustments)
            return price - per_share_adjustment

