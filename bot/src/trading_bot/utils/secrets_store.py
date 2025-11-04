"""
Secure Secrets Storage

Uses OS keyring for storing sensitive API credentials instead of plaintext .env files.
Falls back to environment variables for headless/CI environments.
"""

import os
import logging
from typing import Optional, Dict

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False


# Service name for keyring storage
SERVICE_NAME = "AITradingBot"


def is_keyring_available() -> bool:
    """
    Check if keyring is available on this system.
    
    Returns:
        True if keyring is available, False otherwise
    """
    return KEYRING_AVAILABLE


def store_api_key(exchange: str, key_name: str, key_value: str) -> bool:
    """
    Store an API key securely in the OS keyring.
    
    Args:
        exchange: Exchange name (e.g., 'binance', 'coinbase')
        key_name: Key identifier (e.g., 'api_key', 'api_secret')
        key_value: The actual API key value
        
    Returns:
        True if stored successfully, False otherwise
        
    Example:
        >>> store_api_key('binance', 'api_key', 'your_key_here')
        True
    """
    if not KEYRING_AVAILABLE:
        logger = logging.getLogger('trading_bot.secrets')
        logger.warning("Keyring not available. Cannot store API key securely.")
        return False
    
    try:
        username = f"{exchange}_{key_name}"
        keyring.set_password(SERVICE_NAME, username, key_value)
        return True
    except Exception as e:
        logger = logging.getLogger('trading_bot.secrets')
        logger.error(f"Failed to store API key in keyring: {e}")
        return False


def get_api_key(exchange: str, key_name: str) -> Optional[str]:
    """
    Retrieve an API key from keyring, falling back to environment variables.
    
    Priority:
    1. OS keyring (most secure)
    2. Environment variables (for headless/CI)
    
    Args:
        exchange: Exchange name (e.g., 'binance', 'coinbase')
        key_name: Key identifier (e.g., 'api_key', 'api_secret')
        
    Returns:
        API key value or None if not found
        
    Example:
        >>> get_api_key('binance', 'api_key')
        'your_key_from_keyring'
    """
    logger = logging.getLogger('trading_bot.secrets')
    
    # Try keyring first
    if KEYRING_AVAILABLE:
        try:
            username = f"{exchange}_{key_name}"
            key_value = keyring.get_password(SERVICE_NAME, username)
            if key_value:
                logger.debug(f"Retrieved {exchange} {key_name} from keyring")
                return key_value
        except Exception as e:
            logger.warning(f"Failed to retrieve from keyring: {e}")
    
    # Fall back to environment variables
    env_var_name = f"{exchange.upper()}_{key_name.upper()}"
    key_value = os.getenv(env_var_name)
    if key_value:
        logger.debug(f"Retrieved {exchange} {key_name} from environment variable")
        return key_value
    
    logger.debug(f"No {exchange} {key_name} found in keyring or environment")
    return None


def delete_api_key(exchange: str, key_name: str) -> bool:
    """
    Delete an API key from the OS keyring.
    
    Args:
        exchange: Exchange name
        key_name: Key identifier
        
    Returns:
        True if deleted successfully, False otherwise
    """
    if not KEYRING_AVAILABLE:
        return False
    
    try:
        username = f"{exchange}_{key_name}"
        keyring.delete_password(SERVICE_NAME, username)
        return True
    except Exception as e:
        logger = logging.getLogger('trading_bot.secrets')
        logger.error(f"Failed to delete API key from keyring: {e}")
        return False


def validate_api_keys() -> Dict[str, bool]:
    """
    Check if required API keys are present in keyring or environment.
    
    Returns:
        Dictionary with validation results for each exchange
        
    Example:
        >>> validate_api_keys()
        {'binance': True, 'coinbase': False, 'alpaca': True, 'alpha_vantage': True}
    """
    required_keys = {
        'binance': ['api_key', 'api_secret'],
        'coinbase': ['api_key', 'api_secret'],
        'alpaca': ['api_key', 'api_secret'],
        'alpha_vantage': ['api_key']
    }
    
    results = {}
    
    for exchange, keys in required_keys.items():
        results[exchange] = all(get_api_key(exchange, key) for key in keys)
    
    return results


def migrate_from_env_to_keyring(exchange: str, key_mapping: Dict[str, str]) -> bool:
    """
    Migrate API keys from environment variables to keyring.
    
    Useful for one-time migration from .env to secure storage.
    
    Args:
        exchange: Exchange name
        key_mapping: Dict mapping key names to env var names
                    e.g., {'api_key': 'BINANCE_API_KEY', 'api_secret': 'BINANCE_API_SECRET'}
        
    Returns:
        True if migration successful, False otherwise
        
    Example:
        >>> migrate_from_env_to_keyring('binance', {
        ...     'api_key': 'BINANCE_API_KEY',
        ...     'api_secret': 'BINANCE_API_SECRET'
        ... })
        True
    """
    if not KEYRING_AVAILABLE:
        return False
    
    logger = logging.getLogger('trading_bot.secrets')
    success = True
    
    for key_name, env_var in key_mapping.items():
        value = os.getenv(env_var)
        if value:
            if store_api_key(exchange, key_name, value):
                logger.info(f"Migrated {exchange} {key_name} to keyring")
            else:
                logger.error(f"Failed to migrate {exchange} {key_name}")
                success = False
        else:
            logger.warning(f"No value found for {env_var} in environment")
    
    return success


def list_stored_keys() -> Dict[str, list]:
    """
    List all exchanges and keys currently stored in the keyring.
    
    Returns:
        Dictionary mapping exchange names to lists of stored keys
        
    Note:
        This is a best-effort function and may not work on all keyring backends.
    """
    if not KEYRING_AVAILABLE:
        return {}
    
    # This is platform-dependent and may not work on all systems
    # For now, we'll return the expected structure
    return {
        'info': 'Use validate_api_keys() to check which exchanges are configured'
    }

