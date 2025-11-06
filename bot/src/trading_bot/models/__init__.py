"""
Machine Learning Models for Trading Predictions

This module provides functionality for:
- Training XGBoost models with GPU support
- Loading models and generating predictions
- Managing model lifecycle and versioning
- Hyperparameter optimization with Optuna
- Ensemble optimization
"""

from trading_bot.models.xgboost_trainer import XGBoostTrainer
from trading_bot.models.xgboost_predictor import XGBoostPredictor
from trading_bot.models.model_manager import ModelManager
from trading_bot.models.optimizer import XGBoostOptimizer
from trading_bot.models.ensemble_optimizer import EnsembleOptimizer
from trading_bot.models.optimization_monitor import OptimizationMonitor
from trading_bot.models.param_analyzer import ParameterAnalyzer

__all__ = [
    'XGBoostTrainer',
    'XGBoostPredictor',
    'ModelManager',
    'XGBoostOptimizer',
    'EnsembleOptimizer',
    'OptimizationMonitor',
    'ParameterAnalyzer',
]

