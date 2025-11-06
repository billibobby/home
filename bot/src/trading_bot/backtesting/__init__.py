"""
Backtesting Module

Walk-forward backtesting framework for validating trading strategies.
"""

from trading_bot.backtesting.walk_forward import WalkForwardBacktest
from trading_bot.backtesting.costs import TransactionCostModel
from trading_bot.backtesting.metrics import PerformanceMetrics
from trading_bot.backtesting.results import BacktestResults
from trading_bot.backtesting.robustness import RobustnessValidator
from trading_bot.backtesting.visualizations import BacktestVisualizer

__all__ = [
    'WalkForwardBacktest',
    'TransactionCostModel',
    'PerformanceMetrics',
    'BacktestResults',
    'RobustnessValidator',
    'BacktestVisualizer'
]



