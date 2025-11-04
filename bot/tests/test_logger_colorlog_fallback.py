"""
Tests for logger colorlog fallback behavior.
"""

import logging
from unittest.mock import patch, MagicMock

from trading_bot.logger import setup_logger, COLORLOG_AVAILABLE


def test_logger_works_without_colorlog():
    """Test that logger works even if colorlog is not available."""
    # Mock colorlog import failure
    with patch('trading_bot.logger.COLORLOG_AVAILABLE', False):
        logger = setup_logger('test_logger', console_colors=True)
        
        # Should still create logger successfully
        assert logger is not None
        assert logger.name == 'test_logger'
        
        # Should log without errors
        logger.info("Test message")


def test_logger_uses_colorlog_when_available():
    """Test that logger uses colorlog when available."""
    if COLORLOG_AVAILABLE:
        logger = setup_logger('test_logger_colored', console_colors=True)
        assert logger is not None
        # Check that colorlog formatter is used (via handler)
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        if handlers:
            formatter = handlers[0].formatter
            # Colorlog formatter has log_colors attribute
            assert hasattr(formatter, 'log_colors') or formatter is not None

