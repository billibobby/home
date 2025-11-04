"""
AI Trading Bot - Main Package
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Expose key modules for easier imports
from trading_bot.config_loader import Config
from trading_bot.logger import setup_logger, get_logger

__all__ = ['Config', 'setup_logger', 'get_logger', '__version__']

