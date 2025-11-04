"""
Machine Learning Models for Trading Predictions

This module provides functionality for:
- Training XGBoost models with GPU support
- Loading models and generating predictions
- Managing model lifecycle and versioning
"""

from trading_bot.models.xgboost_trainer import XGBoostTrainer
from trading_bot.models.xgboost_predictor import XGBoostPredictor
from trading_bot.models.model_manager import ModelManager

__all__ = [
    'XGBoostTrainer',
    'XGBoostPredictor',
    'ModelManager',
]

