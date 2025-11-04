"""
Trading Execution and Strategy Modules

This module provides functionality for:
- Generating trading signals from predictions
- Implementing trading strategies
- Managing positions and risk
"""

from trading_bot.trading.signal_generator import SignalGenerator, SignalType
from trading_bot.trading.strategy import XGBoostStrategy

__all__ = [
    'SignalGenerator',
    'SignalType',
    'XGBoostStrategy',
]

