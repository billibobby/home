"""
Trading Execution and Strategy Modules

This module provides functionality for:
- Generating trading signals from predictions
- Implementing trading strategies
- Managing positions and risk
- Paper trading simulation
"""

from trading_bot.trading.signal_generator import SignalGenerator, SignalType
from trading_bot.trading.strategy import XGBoostStrategy
from trading_bot.trading.paper_trading import PaperTradingEngine

__all__ = [
    'SignalGenerator',
    'SignalType',
    'XGBoostStrategy',
    'PaperTradingEngine',
]

