"""
Unit Tests for Transaction Cost Model
"""

import unittest
from unittest.mock import Mock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.backtesting.costs import TransactionCostModel


class TestTransactionCostModel(unittest.TestCase):
    """Test cases for TransactionCostModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'backtesting.transaction_costs.commission_pct': 0.001,
            'backtesting.transaction_costs.slippage_bps': 5,
            'backtesting.transaction_costs.spread_bps': 2,
            'backtesting.transaction_costs.market_impact_enabled': True,
            'backtesting.transaction_costs.market_impact_coeff': 0.01,
        }.get(key, default))
        
        self.mock_logger = Mock()
        self.cost_model = TransactionCostModel(self.mock_config, self.mock_logger)
    
    def test_commission_calculation_basic(self):
        """Test commission calculation."""
        trade_value = 1000.0
        commission = self.cost_model.calculate_commission(trade_value)
        self.assertAlmostEqual(commission, 1.0, places=2)
    
    def test_slippage_market_order(self):
        """Test slippage for market order."""
        price = 100.0
        quantity = 10.0
        volatility = 0.02
        
        slippage = self.cost_model.calculate_slippage(price, quantity, volatility, 'market')
        self.assertGreater(slippage, 0)
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        price = 100.0
        spread = self.cost_model.calculate_spread(price)
        self.assertAlmostEqual(spread, 0.02, places=2)
    
    def test_total_cost_buy_order(self):
        """Test total cost for buy order."""
        price = 100.0
        quantity = 10.0
        avg_volume = 1000000.0
        
        cost_info = self.cost_model.calculate_cost(
            price, quantity, avg_volume, 'buy', 'market', volatility=0.02
        )
        
        self.assertGreater(cost_info['total_cost'], 0)
        self.assertGreater(cost_info['effective_price'], price)
        self.assertIn('commission', cost_info)
        self.assertIn('slippage', cost_info)
        self.assertIn('spread', cost_info)
    
    def test_total_cost_sell_order(self):
        """Test total cost for sell order."""
        price = 100.0
        quantity = 10.0
        avg_volume = 1000000.0
        
        cost_info = self.cost_model.calculate_cost(
            price, quantity, avg_volume, 'sell', 'market', volatility=0.02
        )
        
        self.assertGreater(cost_info['total_cost'], 0)
        self.assertLess(cost_info['effective_price'], price)


if __name__ == '__main__':
    unittest.main()



