#!/usr/bin/env python3
"""
AI Trading Bot GUI Launcher

Simple launcher script for the trading bot GUI.
Double-click this file to launch the GUI.

Note: For development, it's recommended to use:
  pip install -e .[gui]
  trading-bot-gui
"""

import sys

try:
    print("=" * 60)
    print("AI Trading Bot - XGBoost ML Trading System")
    print("=" * 60)
    print("\nLaunching GUI...")
    print("If the window doesn't appear, check for errors below.\n")
    
    # Import from installed package (no path manipulation needed)
    from trading_bot.gui_main import main
    main()
    
except ImportError as e:
    print("\n" + "=" * 60)
    print("ERROR: Missing dependencies")
    print("=" * 60)
    print(f"\nError: {e}")
    print("\nPlease install the package:")
    print("  pip install -e .[gui]")
    print("\nThen use the console script:")
    print("  trading-bot-gui")
    print("\nOr run directly:")
    print("  python -m trading_bot.gui_main")
    print("\n" + "=" * 60)
    input("\nPress Enter to exit...")
    sys.exit(1)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR: Failed to launch GUI")
    print("=" * 60)
    print(f"\nError: {e}")
    print("\nCheck the error message above for details.")
    print("\n" + "=" * 60)
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)

