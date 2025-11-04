"""
Tests for configuration environment variable overrides.
"""

import os
import pytest
from trading_bot.config_loader import Config


def test_get_with_env_override():
    """Test that environment variables override YAML config."""
    # Set environment variable
    os.environ['LOGGING_LEVEL'] = 'DEBUG'
    
    try:
        # Reload config to pick up env var
        config = Config()
        config.reload()
        
        # Should use env var, not YAML
        level = config.get('logging.level')
        assert level == 'DEBUG' or level == 'DEBUG'  # Env vars are strings
        
    finally:
        # Cleanup
        if 'LOGGING_LEVEL' in os.environ:
            del os.environ['LOGGING_LEVEL']


def test_env_override_naming_convention():
    """Test dot-to-underscore conversion for env vars."""
    os.environ['TRADING_MAX_POSITIONS'] = '10'
    os.environ['MODELS_PREDICTION_BUY_THRESHOLD'] = '0.7'
    
    try:
        config = Config()
        config.reload()
        
        max_pos = config.get('trading.max_positions')
        buy_threshold = config.get('models.prediction.buy_threshold')
        
        assert max_pos == 10 or max_pos == '10'
        assert buy_threshold == 0.7 or buy_threshold == '0.7'
        
    finally:
        if 'TRADING_MAX_POSITIONS' in os.environ:
            del os.environ['TRADING_MAX_POSITIONS']
        if 'MODELS_PREDICTION_BUY_THRESHOLD' in os.environ:
            del os.environ['MODELS_PREDICTION_BUY_THRESHOLD']

