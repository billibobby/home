"""
Unit Tests for Logging Setup

Tests logger initialization, log levels, handlers, and formatting.
"""

import unittest
import os
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.logger import setup_logger, get_logger, set_log_level, shutdown_logging


class TestLogger(unittest.TestCase):
    """Test cases for logging setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary log directory
        self.temp_log_dir = tempfile.mkdtemp()
        
        # Clear any existing loggers
        shutdown_logging()
    
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown logging and remove handlers
        shutdown_logging()
        
        # Clean up temporary directory
        if os.path.exists(self.temp_log_dir):
            shutil.rmtree(self.temp_log_dir)
    
    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a logger instance."""
        logger = setup_logger('test_logger', log_dir=self.temp_log_dir)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'test_logger')
    
    def test_setup_logger_with_log_level(self):
        """Test logger initialization with specific log level."""
        logger = setup_logger('test_logger', log_level='DEBUG', log_dir=self.temp_log_dir)
        
        # Logger should be set to DEBUG level
        self.assertEqual(logger.level, logging.DEBUG)
    
    def test_logger_has_handlers(self):
        """Test that logger has both console and file handlers."""
        logger = setup_logger('test_logger', log_dir=self.temp_log_dir)
        
        # Should have 2 handlers: console and file
        self.assertEqual(len(logger.handlers), 2)
        
        # Check handler types
        handler_types = [type(h).__name__ for h in logger.handlers]
        self.assertIn('StreamHandler', handler_types)
        self.assertIn('RotatingFileHandler', handler_types)
    
    def test_log_file_created(self):
        """Test that log file is created in the correct directory."""
        logger = setup_logger('test_logger', log_dir=self.temp_log_dir)
        logger.info("Test log message")
        
        # Check that log file exists
        log_files = list(Path(self.temp_log_dir).glob('*.log'))
        self.assertGreater(len(log_files), 0, "Log file should be created")
    
    def test_log_file_rotation_settings(self):
        """Test that file handler has correct rotation settings."""
        logger = setup_logger('test_logger', log_dir=self.temp_log_dir)
        
        # Find the RotatingFileHandler
        file_handler = None
        for handler in logger.handlers:
            if handler.__class__.__name__ == 'RotatingFileHandler':
                file_handler = handler
                break
        
        self.assertIsNotNone(file_handler)
        self.assertEqual(file_handler.maxBytes, 10 * 1024 * 1024)  # 10 MB
        self.assertEqual(file_handler.backupCount, 5)
    
    def test_custom_rotation_settings(self):
        """Test that custom rotation settings from config are applied."""
        custom_max_bytes = 5 * 1024 * 1024  # 5 MB
        custom_backup_count = 3
        
        logger = setup_logger('test_logger', 
                            log_dir=self.temp_log_dir,
                            max_bytes=custom_max_bytes,
                            backup_count=custom_backup_count)
        
        # Find the RotatingFileHandler
        file_handler = None
        for handler in logger.handlers:
            if handler.__class__.__name__ == 'RotatingFileHandler':
                file_handler = handler
                break
        
        self.assertIsNotNone(file_handler)
        self.assertEqual(file_handler.maxBytes, custom_max_bytes)
        self.assertEqual(file_handler.backupCount, custom_backup_count)
    
    def test_get_logger(self):
        """Test get_logger function returns logger instance."""
        # Setup main logger first
        setup_logger('trading_bot', log_dir=self.temp_log_dir)
        
        # Get child logger
        logger = get_logger('test_module')
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertIn('trading_bot', logger.name)
    
    def test_logger_writes_messages(self):
        """Test that logger actually writes messages to file."""
        logger = setup_logger('test_logger', log_dir=self.temp_log_dir)
        
        test_message = "Test log message for verification"
        logger.info(test_message)
        
        # Read log file and check for message
        log_files = list(Path(self.temp_log_dir).glob('*.log'))
        self.assertGreater(len(log_files), 0)
        
        with open(log_files[0], 'r') as f:
            log_content = f.read()
        
        self.assertIn(test_message, log_content)
    
    def test_different_log_levels(self):
        """Test that different log levels are handled correctly."""
        logger = setup_logger('test_logger', log_level='DEBUG', log_dir=self.temp_log_dir)
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Read log file
        log_files = list(Path(self.temp_log_dir).glob('*.log'))
        with open(log_files[0], 'r') as f:
            log_content = f.read()
        
        # All messages should be in the file (DEBUG and above)
        self.assertIn("Debug message", log_content)
        self.assertIn("Info message", log_content)
        self.assertIn("Warning message", log_content)
        self.assertIn("Error message", log_content)
        self.assertIn("Critical message", log_content)
    
    def test_set_log_level(self):
        """Test changing log level dynamically."""
        logger = setup_logger('test_logger', log_level='INFO', log_dir=self.temp_log_dir)
        
        # Change log level
        set_log_level('ERROR')
        
        # Console handler should now be at ERROR level
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not hasattr(handler, 'maxBytes'):
                # This is the console handler, not the file handler
                self.assertEqual(handler.level, logging.ERROR)
    
    def test_logger_singleton_behavior(self):
        """Test that requesting the same logger returns cached instance."""
        logger1 = setup_logger('test_logger', log_dir=self.temp_log_dir)
        logger2 = setup_logger('test_logger', log_dir=self.temp_log_dir)
        
        self.assertIs(logger1, logger2)
    
    def test_log_directory_creation(self):
        """Test that log directory is created if it doesn't exist."""
        new_log_dir = os.path.join(self.temp_log_dir, 'nested', 'logs')
        
        logger = setup_logger('test_logger', log_dir=new_log_dir)
        
        self.assertTrue(os.path.exists(new_log_dir))
    
    def test_shutdown_logging(self):
        """Test that shutdown_logging closes all handlers."""
        logger = setup_logger('test_logger', log_dir=self.temp_log_dir)
        
        initial_handler_count = len(logger.handlers)
        self.assertGreater(initial_handler_count, 0)
        
        shutdown_logging()
        
        # After shutdown, handlers should be removed
        self.assertEqual(len(logger.handlers), 0)
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'development'}, clear=True)
    def test_environment_based_log_level_development(self):
        """Test that log level defaults to DEBUG in development environment."""
        logger = setup_logger('test_logger', log_level=None, log_dir=self.temp_log_dir)
        
        # In development, default should be DEBUG
        # Check that console handler is set appropriately
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not hasattr(handler, 'maxBytes'):
                # Console handler should be at DEBUG level in development
                self.assertEqual(handler.level, logging.DEBUG)
                break
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'}, clear=True)
    def test_environment_based_log_level_production(self):
        """Test that log level defaults to INFO in production environment."""
        logger = setup_logger('test_logger', log_level=None, log_dir=self.temp_log_dir)
        
        # In production, default should be INFO
        # Check that console handler is set appropriately
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not hasattr(handler, 'maxBytes'):
                # Console handler should be at INFO level in production
                self.assertEqual(handler.level, logging.INFO)
                break
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'WARNING'}, clear=True)
    def test_explicit_log_level_overrides_environment(self):
        """Test that explicit LOG_LEVEL environment variable overrides environment-based default."""
        logger = setup_logger('test_logger', log_level=None, log_dir=self.temp_log_dir)
        
        # Explicit LOG_LEVEL should override environment-based default
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not hasattr(handler, 'maxBytes'):
                self.assertEqual(handler.level, logging.WARNING)
                break
    
    def test_console_colors_disabled(self):
        """Test that console colors can be disabled."""
        logger = setup_logger('test_logger', 
                            log_dir=self.temp_log_dir, 
                            console_colors=False)
        
        # Check that console handler uses standard formatter, not ColoredFormatter
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not hasattr(handler, 'maxBytes'):
                formatter_class = type(handler.formatter).__name__
                # Should be standard Formatter, not ColoredFormatter
                self.assertEqual(formatter_class, 'Formatter')
                break


if __name__ == '__main__':
    unittest.main()
