"""
Data Collection and Processing Modules

This module provides functionality for:
- Fetching stock market data (historical and real-time)
- Engineering features from raw OHLCV data
- Preprocessing data for ML models
"""

from trading_bot.data.stock_fetcher import StockDataFetcher
from trading_bot.data.feature_engineer import FeatureEngineer
from trading_bot.data.preprocessor import DataPreprocessor

__all__ = [
    'StockDataFetcher',
    'FeatureEngineer',
    'DataPreprocessor',
]

