"""
GUI Entry Point for Trading Bot

Bootstraps the graphical user interface for the trading bot.
"""

import sys


def main():
    """
    Main entry point for the GUI application.
    """
    try:
        from trading_bot.gui.main_window import create_application
    except ImportError as e:
        print(f"Error: Failed to import GUI components: {e}")
        print("\nThe GUI requires PySide6. Install it with:")
        print("  pip install PySide6")
        print("\nOr install the bot with GUI extras:")
        print("  pip install -e .[gui]")
        sys.exit(1)
    
    try:
        app, window = create_application()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

