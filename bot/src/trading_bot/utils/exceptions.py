"""
Custom Exception Classes

Custom exceptions for better error handling and debugging across the application.
"""


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize exception with message and optional details.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(TradingBotError):
    """
    Raised when there are configuration loading or validation issues.
    
    Examples:
        - Missing required configuration file
        - Invalid YAML syntax
        - Missing required environment variables
        - Invalid configuration values
    """
    pass


class APIError(TradingBotError):
    """
    Raised when exchange or broker API calls fail.
    
    Examples:
        - Connection timeout
        - Authentication failure
        - Rate limit exceeded
        - Invalid API response
        - Exchange maintenance
    """
    
    def __init__(self, message: str, status_code: int = None, exchange: str = None, details: dict = None):
        """
        Initialize API error with additional context.
        
        Args:
            message: Error message
            status_code: HTTP status code
            exchange: Exchange name where error occurred
            details: Additional error details
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.exchange = exchange
    
    def __str__(self):
        parts = [self.message]
        if self.exchange:
            parts.append(f"Exchange: {self.exchange}")
        if self.status_code:
            parts.append(f"Status Code: {self.status_code}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class DataError(TradingBotError):
    """
    Raised when data fetching or processing fails.
    
    Examples:
        - Missing historical data
        - Data validation failure
        - Corrupt data files
        - Database connection issues
        - Data format mismatch
    """
    
    def __init__(self, message: str, symbol: str = None, timeframe: str = None, details: dict = None):
        """
        Initialize data error with symbol and timeframe context.
        
        Args:
            message: Error message
            symbol: Trading symbol
            timeframe: Data timeframe
            details: Additional error details
        """
        super().__init__(message, details)
        self.symbol = symbol
        self.timeframe = timeframe
    
    def __str__(self):
        parts = [self.message]
        if self.symbol:
            parts.append(f"Symbol: {self.symbol}")
        if self.timeframe:
            parts.append(f"Timeframe: {self.timeframe}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class ModelError(TradingBotError):
    """
    Raised when ML model operations fail.
    
    Examples:
        - Model loading failure
        - Model training error
        - Invalid model parameters
        - Prediction generation failure
        - Model file not found
    """
    
    def __init__(self, message: str, model_name: str = None, model_version: str = None, details: dict = None):
        """
        Initialize model error with model context.
        
        Args:
            message: Error message
            model_name: Name of the model
            model_version: Version of the model
            details: Additional error details
        """
        super().__init__(message, details)
        self.model_name = model_name
        self.model_version = model_version
    
    def __str__(self):
        parts = [self.message]
        if self.model_name:
            parts.append(f"Model: {self.model_name}")
        if self.model_version:
            parts.append(f"Version: {self.model_version}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class TradingError(TradingBotError):
    """
    Raised when order execution or trading operations fail.
    
    Examples:
        - Order placement failure
        - Insufficient balance
        - Invalid order parameters
        - Position sizing error
        - Risk limit exceeded
    """
    
    def __init__(self, message: str, symbol: str = None, order_type: str = None, 
                 amount: float = None, details: dict = None):
        """
        Initialize trading error with order context.
        
        Args:
            message: Error message
            symbol: Trading symbol
            order_type: Type of order (market, limit, etc.)
            amount: Order amount
            details: Additional error details
        """
        super().__init__(message, details)
        self.symbol = symbol
        self.order_type = order_type
        self.amount = amount
    
    def __str__(self):
        parts = [self.message]
        if self.symbol:
            parts.append(f"Symbol: {self.symbol}")
        if self.order_type:
            parts.append(f"Order Type: {self.order_type}")
        if self.amount:
            parts.append(f"Amount: {self.amount}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class ValidationError(TradingBotError):
    """
    Raised when input validation fails.
    
    Examples:
        - Invalid symbol format
        - Invalid timeframe
        - Invalid parameter values
        - Missing required fields
    """
    pass


class DatabaseError(TradingBotError):
    """
    Raised when database operations fail.
    
    Examples:
        - Connection failure
        - Query execution error
        - Data integrity violation
        - Transaction rollback
    """
    pass

