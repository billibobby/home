"""
Data Collection and Processing Modules

This module provides functionality for:
- Fetching stock market data (historical and real-time)
- Engineering features from raw OHLCV data
- Preprocessing data for ML models
- Database persistence for trades, positions, and performance metrics
- Fetching sentiment data from news APIs
"""

from trading_bot.data.stock_fetcher import StockDataFetcher
from trading_bot.data.feature_engineer import FeatureEngineer
from trading_bot.data.preprocessor import DataPreprocessor
from trading_bot.data.database_manager import DatabaseManager
from trading_bot.data.sentiment_fetcher import SentimentDataFetcher

__all__ = [
    'StockDataFetcher',
    'FeatureEngineer',
    'DataPreprocessor',
    'DatabaseManager',
    'SentimentDataFetcher',
]

