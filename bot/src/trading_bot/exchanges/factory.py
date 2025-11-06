"""
Exchange Factory

Provides a single entry point for creating exchange instances, abstracting the selection logic.
"""

from typing import Optional

from trading_bot.exchanges.exchange_interface import ExchangeInterface
from trading_bot.exchanges.paper_exchange import PaperTradingExchange
from trading_bot.exchanges.alpaca_exchange import AlpacaExchange
from trading_bot.utils.exceptions import DatabaseError


def create_exchange(config, logger, db_manager=None, exchange_type: Optional[str] = None) -> ExchangeInterface:
    """
    Create an exchange instance based on configuration.
    
    Args:
        config: Configuration object
        logger: Logger instance
        db_manager: DatabaseManager instance (required for paper trading)
        exchange_type: Optional override for exchange type ('paper', 'alpaca')
        
    Returns:
        ExchangeInterface instance
        
    Raises:
        ValueError: If exchange_type is invalid
        DatabaseError: If db_manager is required but not provided
    """
    # Determine exchange type
    if exchange_type:
        selected_type = exchange_type.lower()
    else:
        # Check config for Alpaca enabled
        alpaca_enabled = config.get('exchanges.alpaca.enabled', False)
        if alpaca_enabled is None:
            # Fallback to deprecated config location
            alpaca_enabled = config.get('api.exchanges.alpaca.enabled', False)
        
        if alpaca_enabled:
            selected_type = 'alpaca'
        else:
            selected_type = 'paper'
    
    # Create appropriate exchange
    if selected_type == 'paper':
        if db_manager is None:
            raise DatabaseError("db_manager is required for paper trading exchange")
        
        exchange = PaperTradingExchange(config, logger, db_manager)
        logger.info("Created PaperTradingExchange")
        
    elif selected_type == 'alpaca':
        try:
            exchange = AlpacaExchange(config, logger)
            logger.info(f"Created AlpacaExchange ({exchange.get_exchange_name()})")
        except ValueError as e:
            logger.error(f"Failed to create AlpacaExchange: {str(e)}")
            logger.warning("Falling back to paper trading")
            if db_manager is None:
                raise DatabaseError("db_manager is required for paper trading fallback")
            exchange = PaperTradingExchange(config, logger, db_manager)
            logger.info("Created PaperTradingExchange as fallback")
        
    else:
        raise ValueError(f"Unknown exchange type: {exchange_type}")
    
    return exchange




