"""
Trading Bot GUI Components

PySide6-based graphical user interface for the trading bot.
"""

try:
    from trading_bot.gui.main_window import MainWindow, create_application
    from trading_bot.gui.model_tab import ModelManagementTab
    from trading_bot.gui.predictions_tab import PredictionsDashboard
    from trading_bot.gui.strategy_tab import StrategyMonitor
    
    __all__ = [
        'MainWindow',
        'create_application',
        'ModelManagementTab',
        'PredictionsDashboard',
        'StrategyMonitor',
    ]
except ImportError:
    # PySide6 not available
    __all__ = []
