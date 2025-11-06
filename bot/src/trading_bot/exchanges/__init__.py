"""
Exchange Interface Module

Provides a unified interface for different trading exchanges (paper trading, Alpaca, etc.)
through an adapter pattern. This allows strategies to work with any exchange implementation.
"""

from trading_bot.exchanges.exchange_interface import ExchangeInterface
from trading_bot.exchanges.paper_exchange import PaperTradingExchange
from trading_bot.exchanges.alpaca_exchange import AlpacaExchange
from trading_bot.exchanges.alpaca_client import AlpacaClient
from trading_bot.exchanges.alpaca_stream import AlpacaStream
from trading_bot.exchanges.factory import create_exchange

__all__ = [
    'ExchangeInterface',
    'PaperTradingExchange',
    'AlpacaExchange',
    'AlpacaClient',
    'AlpacaStream',
    'create_exchange',
]




