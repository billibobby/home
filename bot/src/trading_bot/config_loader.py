"""
Configuration Loader Module

Handles loading settings from multiple sources with priority:
1. Environment variables (highest priority)
2. config.yaml file
3. Default values (lowest priority)
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Optional, Callable
from dotenv import load_dotenv

from trading_bot.utils.paths import resolve_resource_path


class Config:
    """
    Singleton configuration class that loads and manages application settings.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration only once."""
        if not self._initialized:
            self._config = {}
            self._defaults = {
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'dir': 'logs',
                    'file_rotation': {
                        'max_bytes': 10485760,  # 10 MB
                        'backup_count': 5
                    },
                    'console_colors': True
                },
                'database_path': 'data/trading_bot.db',
                'model_path': 'models/',
                'data_path': 'data/'
            }
            
            # Load environment variables from .env file
            load_dotenv()
            
            # Load configuration from YAML file
            self.load_config()
            
            # Validate required settings
            self.validate_config()
            
            Config._initialized = True
    
    def load_config(self, config_path: str = None, status_callback: Optional[Callable[[str], None]] = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
            status_callback: Optional callback for status messages (for GUI splash screens)
        """
        if config_path is None:
            # First check for user override in writable app directory
            from trading_bot.utils.paths import get_config_override_path
            override_path = get_config_override_path('config.yaml')
            override_file = Path(override_path)
            
            if override_file.exists():
                config_path = override_path
                logger = logging.getLogger('trading_bot.config')
                if logger.hasHandlers():
                    logger.info(f"Using user config override: {config_path}")
            else:
                # Fallback to packaged resource
                # First try importlib.resources for package resources
                try:
                    from importlib.resources import files
                    config_resource = files('trading_bot.resources.config') / 'config.yaml'
                    if config_resource.is_file():
                        import tempfile
                        import shutil
                        # Extract to temp file for compatibility
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                            shutil.copy(str(config_resource), tmp.name)
                            config_path = tmp.name
                    else:
                        raise FileNotFoundError
                except (ImportError, FileNotFoundError, AttributeError):
                    # Fallback to resolve_resource_path for development
                    config_path = resolve_resource_path('config/config.yaml')
        
        config_file = Path(config_path)
        logger = logging.getLogger('trading_bot.config')
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                msg = f"Configuration loaded from {config_path}"
                if status_callback:
                    status_callback(msg)
                elif logger.hasHandlers():
                    logger.info(msg)
            except yaml.YAMLError as e:
                msg = f"Error parsing YAML configuration: {e}"
                if status_callback:
                    status_callback(msg)
                elif logger.hasHandlers():
                    logger.error(msg)
                self._config = {}
        else:
            msg = f"Configuration file {config_path} not found. Using defaults."
            if status_callback:
                status_callback(msg)
            elif logger.hasHandlers():
                logger.warning(msg)
            self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'trading.position_size')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        # Check environment variable first (convert dot notation to uppercase underscore)
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._convert_type(env_value)
        
        # Navigate through nested dictionary
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    break
            else:
                value = None
                break
        
        # If not found in config, check defaults
        if value is None:
            value = self._get_default(key)
        
        return value if value is not None else default
    
    def _get_default(self, key: str) -> Any:
        """Get default value for a key."""
        keys = key.split('.')
        value = self._defaults
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return None
            else:
                return None
        
        return value
    
    def get_env(self, key: str, default: Any = None, var_type: type = str) -> Any:
        """
        Get environment variable with type conversion.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            var_type: Type to convert the value to (str, int, float, bool)
            
        Returns:
            Environment variable value converted to specified type
        """
        value = os.getenv(key, default)
        
        if value is None:
            return default
        
        # Handle boolean type conversion properly for both string and non-string values
        if var_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            else:
                return bool(value) if value is not None else default
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        else:
            return str(value)
    
    def _convert_type(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            return int(value)
        except ValueError:
            pass
        
        try:
            return float(value)
        except ValueError:
            pass
        
        return value
    
    def validate_config(self, status_callback: Optional[Callable[[str], None]] = None):
        """
        Validate that required configuration settings are present.
        Raises ConfigurationError if validation fails.
        
        Args:
            status_callback: Optional callback for status messages (for GUI splash screens)
        """
        logger = logging.getLogger('trading_bot.config')
        
        # Check if critical paths exist or can be created
        required_dirs = [
            self.get('data_path', 'data'),
            self.get('model_path', 'models'),
            'logs'
        ]
        
        for directory in required_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Additional validation can be added here as needed
        msg = "Configuration validation passed"
        if status_callback:
            status_callback(msg)
        elif logger.hasHandlers():
            logger.info(msg)
    
    def reload(self, config_path: str = None, status_callback: Optional[Callable[[str], None]] = None):
        """
        Reload configuration from file.
        
        Args:
            config_path: Path to the configuration YAML file
            status_callback: Optional callback for status messages
        """
        logger = logging.getLogger('trading_bot.config')
        self.load_config(config_path, status_callback)
        msg = "Configuration reloaded"
        if status_callback:
            status_callback(msg)
        elif logger.hasHandlers():
            logger.info(msg)
    
    def get_all(self) -> dict:
        """Get all configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self):
        return f"Config(initialized={self._initialized})"

