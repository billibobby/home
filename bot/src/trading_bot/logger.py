"""
Logging Setup Module

Configures Python's logging framework with dual output:
1. Console handler with colored output (INFO and above)
2. Rotating file handler (DEBUG and above)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import colorlog, fallback to standard logging if unavailable
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

from trading_bot.utils.paths import get_writable_app_dir


# Global logger cache
_loggers = {}


class UILogHandler(logging.Handler):
    """
    Thread-safe log handler that emits log records to a GUI via a queue.
    
    GUI applications should create a UILogHandler, pass it a queue.Queue,
    and add it to the logger. The GUI can then poll the queue from its
    event loop to display log messages in the UI.
    
    Example:
        import queue
        log_queue = queue.Queue()
        ui_handler = UILogHandler(log_queue)
        logger.addHandler(ui_handler)
        
        # In GUI event loop:
        while not log_queue.empty():
            record = log_queue.get()
            display_log_in_ui(record.getMessage())
    """
    
    def __init__(self, log_queue):
        """
        Initialize UI log handler.
        
        Args:
            log_queue: A queue.Queue instance for thread-safe log passing
        """
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        """
        Emit a log record by putting it in the queue.
        
        Args:
            record: LogRecord instance
        """
        try:
            self.log_queue.put_nowait(record)
        except Exception:
            # Silently drop if queue is full to avoid blocking
            self.handleError(record)


def setup_logger(name: str = 'trading_bot', 
                 log_level: str = None,
                 log_dir: str = None,
                 max_bytes: int = None,
                 backup_count: int = None,
                 console_colors: bool = None,
                 log_format: str = None,
                 enable_console: bool = True) -> logging.Logger:
    """
    Set up and configure a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        console_colors: Whether to use colored console output
        log_format: Custom log format string
        enable_console: Whether to enable console logging (default True, set False for GUI)
        
    Returns:
        Configured logger instance
    """
    # Return cached logger if it exists
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Import Config here to avoid circular imports
    from trading_bot.config_loader import Config
    config = Config()
    
    # Determine log level with environment-based auto-selection
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL')
        if log_level is None:
            # Auto-select based on ENVIRONMENT variable
            # First try get_env with uppercase, then fallback to get() with dot notation
            environment = config.get_env('ENVIRONMENT')
            if environment is None:
                environment = config.get('environment', 'development')
            if environment.lower() == 'development':
                log_level = 'DEBUG'
            elif environment.lower() == 'production':
                log_level = 'INFO'
            else:
                log_level = 'INFO'
    
    log_level = log_level.upper()
    level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    
    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Read configuration from Config with fallbacks
    if log_dir is None:
        # Try config first, then fall back to user-writable directory
        log_dir = config.get('logging.dir')
        if log_dir is None:
            log_dir = get_writable_app_dir('logs')
    
    if max_bytes is None:
        max_bytes = config.get('logging.file_rotation.max_bytes', 10 * 1024 * 1024)  # 10 MB default
    
    if backup_count is None:
        backup_count = config.get('logging.file_rotation.backup_count', 5)
    
    if console_colors is None:
        console_colors = config.get('logging.console_colors', True)
    
    if log_format is None:
        log_format = config.get('logging.format', 
                                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s')
    
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create file formatter
    file_formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console formatter (with or without colors)
    if console_colors and COLORLOG_AVAILABLE:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        if console_colors and not COLORLOG_AVAILABLE:
            logger.warning("colorlog not available, falling back to standard formatter")
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler (INFO and above) - only if enabled
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation (DEBUG and above)
    log_filename = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y-%m-%d")}.log')
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Cache the logger
    _loggers[name] = logger
    
    logger.info(f"Logger '{name}' initialized with level {log_level}")
    logger.debug(f"Log file: {log_filename}")
    logger.debug(f"File rotation: max_bytes={max_bytes}, backup_count={backup_count}")
    logger.debug(f"Console colors: {console_colors}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for a specific module.
    
    Args:
        name: Module name for the logger
        
    Returns:
        Logger instance
    """
    # Create a child logger if parent exists
    if 'trading_bot' in _loggers:
        return logging.getLogger(f'trading_bot.{name}')
    else:
        # Setup main logger if not exists
        setup_logger()
        return logging.getLogger(f'trading_bot.{name}')


def set_log_level(level: str):
    """
    Change log level for all loggers.
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    for logger in _loggers.values():
        logger.setLevel(log_level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setLevel(log_level)


def shutdown_logging():
    """Shutdown all loggers and close handlers."""
    for logger in _loggers.values():
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    
    _loggers.clear()
    logging.shutdown()

