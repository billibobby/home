"""
Trading Bot Application Service Layer

Provides a unified service interface for bot initialization and control.
Can be used by both CLI and GUI entry points.
"""

import sys
import platform
import threading
from pathlib import Path
from typing import Optional, Callable
from enum import Enum

from trading_bot.config_loader import Config
from trading_bot.logger import setup_logger, get_logger, shutdown_logging
from trading_bot.utils.helpers import ensure_dir, validate_api_keys
from trading_bot.utils.exceptions import ConfigurationError, TradingBotError


class BotStatus(Enum):
    """Bot status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class BotApp:
    """
    Main trading bot application service.
    
    Encapsulates configuration, logging, and bot lifecycle management.
    Can be controlled by CLI or GUI interfaces.
    
    Example (CLI):
        app = BotApp()
        app.initialize()
        app.start()
        # ... do work ...
        app.stop()
    
    Example (GUI):
        app = BotApp()
        app.initialize(enable_console=False, status_callback=update_splash)
        # Start in background thread
        worker = threading.Thread(target=app.start)
        worker.start()
        # ... GUI event loop ...
        app.stop()
    """
    
    def __init__(self):
        """Initialize bot application."""
        self._status = BotStatus.UNINITIALIZED
        self._config: Optional[Config] = None
        self._logger = None
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
    
    def initialize(self, 
                   enable_console: bool = True,
                   status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Initialize the bot application.
        
        Args:
            enable_console: Whether to enable console logging (False for GUI)
            status_callback: Optional callback for initialization status messages
            
        Returns:
            True if initialization successful, False otherwise
        """
        self._status = BotStatus.INITIALIZING
        
        try:
            # Notify status
            self._notify(status_callback, "Initializing AI Trading Bot...")
            
            # Initialize configuration system
            self._config = Config()
            
            # Set up logging system
            log_level = self._config.get('logging.level')
            self._logger = setup_logger('trading_bot', 
                                       log_level=log_level,
                                       enable_console=enable_console)
            
            # Log startup information
            self._logger.info("=" * 70)
            self._logger.info("AI TRADING BOT - STARTING UP")
            self._logger.info("=" * 70)
            self._logger.info(f"Python Version: {platform.python_version()}")
            self._logger.info(f"Platform: {platform.system()} {platform.release()}")
            self._logger.info(f"Environment: {self._config.get_env('ENVIRONMENT', 'development')}")
            self._logger.info(f"Log Level: {log_level if log_level else 'AUTO'}")
            
            # Validate and ensure required directories exist
            self._notify(status_callback, "Validating directory structure...")
            self._logger.info("Validating directory structure...")
            required_dirs = ['data', 'models', 'logs', 'config']
            for directory in required_dirs:
                ensure_dir(directory)
                self._logger.debug(f"Directory validated: {directory}/")
            
            # Validate API keys
            self._notify(status_callback, "Checking API key configuration...")
            self._logger.info("Checking API key configuration...")
            api_status = validate_api_keys()
            for exchange, is_valid in api_status.items():
                status = "[OK] Configured" if is_valid else "[--] Not configured"
                self._logger.info(f"  {exchange.capitalize()}: {status}")
            
            # Load trading configuration
            self._notify(status_callback, "Loading trading parameters...")
            self._logger.info("Loading trading parameters...")
            trading_config = {
                'position_size': self._config.get('trading.default_position_size', 100),
                'risk_percentage': self._config.get('trading.risk_percentage', 2),
                'max_positions': self._config.get('trading.max_positions', 5),
            }
            for key, value in trading_config.items():
                self._logger.debug(f"  {key}: {value}")
            
            # Log model configuration
            self._notify(status_callback, "Loading model configuration...")
            self._logger.info("Loading model configuration...")
            model_config = {
                'default_model': self._config.get('models.default_model', 'random_forest'),
                'buy_threshold': self._config.get('models.prediction.buy_threshold', 0.6),
                'sell_threshold': self._config.get('models.prediction.sell_threshold', 0.6),
            }
            for key, value in model_config.items():
                self._logger.debug(f"  {key}: {value}")
            
            self._logger.info("=" * 70)
            self._logger.info("INITIALIZATION COMPLETE")
            self._logger.info("=" * 70)
            
            self._notify(status_callback, "Initialization complete!")
            self._status = BotStatus.READY
            return True
            
        except ConfigurationError as e:
            error_msg = f"Configuration error: {e}"
            if self._logger:
                self._logger.error(error_msg)
            self._notify(status_callback, error_msg)
            self._status = BotStatus.ERROR
            return False
            
        except Exception as e:
            error_msg = f"Initialization error: {e}"
            if self._logger:
                self._logger.exception(error_msg)
            self._notify(status_callback, error_msg)
            self._status = BotStatus.ERROR
            return False
    
    def start(self, blocking: bool = True) -> bool:
        """
        Start the trading bot.
        
        Args:
            blocking: If True, blocks until stop() is called.
                     If False, returns immediately (for background workers)
            
        Returns:
            True if started successfully, False otherwise
        """
        if self._status != BotStatus.READY:
            if self._logger:
                self._logger.error(f"Cannot start bot in {self._status.value} state")
            return False
        
        try:
            self._status = BotStatus.RUNNING
            self._stop_event.clear()
            
            if self._logger:
                self._logger.info("Trading bot started!")
                self._logger.info("Waiting for implementation of trading modules...")
                self._logger.warning("This is the foundation phase. Trading functionality will be")
                self._logger.warning("implemented in future phases.")
            
            if blocking:
                # Block until stop is called
                self._stop_event.wait()
            
            return True
            
        except Exception as e:
            if self._logger:
                self._logger.exception(f"Error starting bot: {e}")
            self._status = BotStatus.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Stop the trading bot gracefully.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self._status != BotStatus.RUNNING:
            if self._logger:
                self._logger.warning(f"Bot is not running (status: {self._status.value})")
            return False
        
        try:
            if self._logger:
                self._logger.info("Stopping trading bot...")
            
            # Signal stop event
            self._stop_event.set()
            
            # Wait for worker thread if it exists
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
            
            self._status = BotStatus.STOPPED
            
            if self._logger:
                self._logger.info("Trading bot stopped successfully")
            
            return True
            
        except Exception as e:
            if self._logger:
                self._logger.exception(f"Error stopping bot: {e}")
            self._status = BotStatus.ERROR
            return False
    
    def status(self) -> BotStatus:
        """
        Get current bot status.
        
        Returns:
            Current BotStatus
        """
        return self._status
    
    def get_config(self) -> Optional[Config]:
        """
        Get the configuration object.
        
        Returns:
            Config instance or None if not initialized
        """
        return self._config
    
    def get_logger(self):
        """
        Get the logger instance.
        
        Returns:
            Logger instance or None if not initialized
        """
        return self._logger
    
    def shutdown(self):
        """
        Shutdown the bot application and cleanup resources.
        """
        if self._status == BotStatus.RUNNING:
            self.stop()
        
        if self._logger:
            self._logger.info("Shutting down bot application...")
        
        shutdown_logging()
        self._status = BotStatus.UNINITIALIZED
    
    def _notify(self, callback: Optional[Callable[[str], None]], message: str):
        """
        Send notification via callback if provided.
        
        Args:
            callback: Optional status callback function
            message: Status message to send
        """
        if callback:
            try:
                callback(message)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Status callback failed: {e}")

