"""
AI Trading Bot - CLI Entry Point

This is the CLI entry point for the trading bot application.
Uses the BotApp service layer for initialization and control.
"""

import sys
from trading_bot.app import BotApp
from trading_bot.utils.exceptions import ConfigurationError, TradingBotError


def main():
    """
    Main function to initialize and run the trading bot via CLI.
    """
    app = BotApp()
    
    try:
        # Initialize with console logging enabled (CLI mode)
        print("Initializing AI Trading Bot...")
        if not app.initialize(enable_console=True):
            print("Failed to initialize bot")
            sys.exit(1)
        
        logger = app.get_logger()
        
        # Show next steps
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Implement data collection modules (src/data/)")
        logger.info("  2. Develop ML models (src/models/)")
        logger.info("  3. Build trading logic (src/trading/)")
        logger.info("  4. Add backtesting framework")
        logger.info("  5. Implement live trading mode")
        logger.info("")
        logger.info("Application initialized successfully. Press Ctrl+C to exit.")
        
        # Start the bot (blocking)
        app.start(blocking=True)
        
    except ConfigurationError as e:
        logger = app.get_logger()
        if logger:
            logger.error(f"Configuration error: {e}")
        else:
            print(f"Configuration error: {e}")
        sys.exit(1)
        
    except TradingBotError as e:
        logger = app.get_logger()
        if logger:
            logger.error(f"Trading bot error: {e}")
        else:
            print(f"Trading bot error: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger = app.get_logger()
        if logger:
            logger.info("")
            logger.info("Shutdown signal received...")
            logger.info("Closing trading bot gracefully...")
        else:
            print("\nShutdown signal received.")
        
        app.stop()
        app.shutdown()
        
        if logger:
            logger.info("Goodbye!")
        else:
            print("Goodbye!")
        sys.exit(0)
        
    except Exception as e:
        logger = app.get_logger()
        if logger:
            logger.exception(f"Unexpected error occurred: {e}")
        else:
            print(f"Unexpected error: {e}")
        
        app.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()

