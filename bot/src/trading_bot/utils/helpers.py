"""
Utility Helper Functions

Common utility functions used across the application.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Optional, Callable, Any


def ensure_dir(path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_to_datetime(timestamp: int, unit: str = 'ms') -> datetime:
    """
    Convert Unix timestamp to datetime object.
    
    Args:
        timestamp: Unix timestamp
        unit: Time unit ('s' for seconds, 'ms' for milliseconds)
        
    Returns:
        datetime object
    """
    if unit == 'ms':
        timestamp = timestamp / 1000.0
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime, unit: str = 'ms') -> int:
    """
    Convert datetime object to Unix timestamp.
    
    Args:
        dt: datetime object
        unit: Time unit ('s' for seconds, 'ms' for milliseconds)
        
    Returns:
        Unix timestamp as integer
    """
    timestamp = int(dt.timestamp())
    if unit == 'ms':
        timestamp *= 1000
    return timestamp


def format_currency(amount: float, currency: str = 'USD', decimals: int = 2) -> str:
    """
    Format monetary values for display.
    
    Args:
        amount: Monetary amount
        currency: Currency code
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'BTC': '₿',
        'ETH': 'Ξ'
    }
    
    symbol = symbols.get(currency, currency + ' ')
    return f"{symbol}{amount:,.{decimals}f}"


def validate_api_keys() -> dict:
    """
    Check if required API keys are present in keyring or environment.
    
    Uses the secrets_store module which checks keyring first,
    then falls back to environment variables.
    
    Returns:
        Dictionary with validation results for each exchange
    """
    from trading_bot.utils.secrets_store import validate_api_keys as secrets_validate
    return secrets_validate()


def retry_on_failure(max_attempts: int = 3, 
                    delay: float = 1.0, 
                    backoff: float = 2.0,
                    exceptions: tuple = (Exception,),
                    logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay (exponential backoff)
        exceptions: Tuple of exceptions to catch and retry
        logger: Optional logger instance (if None, uses default retry logger)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            current_delay = delay
            
            # Get logger (use provided or default)
            if logger is None:
                retry_logger = logging.getLogger('trading_bot.retry')
            else:
                retry_logger = logger
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        retry_logger.error(
                            f"Max retries ({max_attempts}) reached for {func.__name__}: {str(e)}"
                        )
                        raise
                    
                    retry_logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay:.1f} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        
        return wrapper
    return decorator


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def truncate_string(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

