"""
Main Window for Trading Bot GUI

Provides a graphical interface for controlling and monitoring the trading bot.
Uses PySide6 for cross-platform GUI support.
"""

import sys
import queue
import threading
import logging
from typing import Optional

from trading_bot.app import BotApp, BotStatus
from trading_bot.logger import UILogHandler

# Check for PySide6 availability
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QTextEdit, QLabel, QStatusBar, QMenuBar, QMenu,
        QGroupBox, QGridLayout, QMessageBox, QDialog, QDialogButtonBox,
        QLineEdit, QFormLayout, QTabWidget, QSplashScreen
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QObject
    from PySide6.QtGui import QAction, QIcon, QPixmap, QFont
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    # Create dummy base class for when PySide6 is not available
    class QMainWindow:
        pass
    class QObject:
        pass
    class QApplication:
        pass
    Signal = None
    Qt = None


class LogSignals(QObject if PYSIDE6_AVAILABLE else object):
    """Signals for thread-safe log message passing."""
    if PYSIDE6_AVAILABLE:
        log_message = Signal(str, str)  # (level, message)
    else:
        # No-op when PySide6 not available
        def __init__(self):
            pass
        log_message = None


class MainWindow(QMainWindow):
    """
    Main window for the Trading Bot GUI.
    
    Features:
    - Start/Stop bot control
    - Real-time log display
    - Configuration management
    - API key management (using keyring)
    - Status monitoring
    """
    
    def __init__(self):
        """Initialize the main window."""
        if not PYSIDE6_AVAILABLE:
            raise ImportError("PySide6 is required for GUI. Install with: pip install PySide6")
        
        super().__init__()
        
        self.bot_app: Optional[BotApp] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.log_queue = queue.Queue(maxsize=1000)
        self.log_signals = LogSignals()
        self.logger = logging.getLogger(__name__)
        
        # Initialize tabs as None first
        self.model_tab = None
        self.predictions_tab = None
        self.strategy_tab = None
        
        self._init_ui()
        self._setup_log_handler()
        self._setup_timers()
        
        # Initialize bot in background
        self._initialize_bot()
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AI Trading Bot")
        self.setMinimumSize(1000, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create status display
        status_group = QGroupBox("Bot Status")
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Create control buttons
        control_group = QGroupBox("Bot Control")
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Bot")
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Bot")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # XGBoost Model Management tab
        try:
            from trading_bot.gui.model_tab import ModelManagementTab
            self.model_tab = ModelManagementTab(None)  # Will be set after bot init
            self.model_tab.model_loaded.connect(self._on_model_loaded)
            self.tab_widget.addTab(self.model_tab, "🤖 Models")
        except ImportError as e:
            self.model_tab = None
            self.logger.warning(f"Could not load model tab: {e}")
        
        # Predictions Dashboard tab
        try:
            from trading_bot.gui.predictions_tab import PredictionsDashboard
            self.predictions_tab = PredictionsDashboard(None)  # Will be set after bot init
            self.tab_widget.addTab(self.predictions_tab, "🔮 Predictions")
        except ImportError as e:
            self.predictions_tab = None
            self.logger.warning(f"Could not load predictions tab: {e}")
        
        # Strategy Monitor tab
        try:
            from trading_bot.gui.strategy_tab import StrategyMonitor
            self.strategy_tab = StrategyMonitor(None)  # Will be set after bot init
            self.tab_widget.addTab(self.strategy_tab, "📊 Strategy")
        except ImportError as e:
            self.strategy_tab = None
            self.logger.warning(f"Could not load strategy tab: {e}")
        
        # Log display tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_display)
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_display.clear)
        log_layout.addWidget(clear_log_btn)
        
        self.tab_widget.addTab(log_widget, "📋 Logs")
        
        # Configuration tab
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.addWidget(QLabel("Configuration management coming soon..."))
        self.tab_widget.addTab(config_widget, "⚙️ Configuration")
        
        # API Keys tab
        api_widget = QWidget()
        api_layout = QVBoxLayout(api_widget)
        api_layout.addWidget(QLabel("API key management via OS keyring coming soon..."))
        api_layout.addWidget(QLabel("Use the secrets_store module programmatically for now."))
        self.tab_widget.addTab(api_widget, "🔑 API Keys")
        
        main_layout.addWidget(self.tab_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_log_handler(self):
        """Set up log handler for GUI display."""
        # Connect log signal to slot
        self.log_signals.log_message.connect(self._append_log_message)
    
    def _setup_timers(self):
        """Set up timers for periodic updates."""
        # Timer for processing log queue
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self._process_log_queue)
        self.log_timer.start(100)  # Process every 100ms
        
        # Timer for status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)  # Update every 500ms
    
    def _initialize_bot(self):
        """Initialize the bot application in the background."""
        def init_worker():
            self.bot_app = BotApp()
            success = self.bot_app.initialize(
                enable_console=False,
                status_callback=self._status_callback
            )
            
            if success:
                # Add UI log handler
                logger = self.bot_app.get_logger()
                if logger:
                    ui_handler = UILogHandler(self.log_queue)
                    ui_handler.setLevel(logging.INFO)
                    logger.addHandler(ui_handler)
                
                # Set bot_app reference in tabs
                if self.model_tab:
                    self.model_tab.bot_app = self.bot_app
                    self.model_tab._initialize_model_manager()
                
                if self.predictions_tab:
                    self.predictions_tab.bot_app = self.bot_app
                
                if self.strategy_tab:
                    self.strategy_tab.bot_app = self.bot_app
        
        init_thread = threading.Thread(target=init_worker, daemon=True)
        init_thread.start()
    
    def _status_callback(self, message: str):
        """Callback for initialization status messages."""
        # Add to log queue
        self.log_queue.put(logging.LogRecord(
            name="init",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        ))
    
    def _process_log_queue(self):
        """Process log messages from the queue."""
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get_nowait()
                level_name = record.levelname
                message = record.getMessage()
                self.log_signals.log_message.emit(level_name, message)
            except queue.Empty:
                break
    
    def _append_log_message(self, level: str, message: str):
        """Append a log message to the display (runs in main thread)."""
        # Color-code by level
        colors = {
            'DEBUG': '#808080',
            'INFO': '#000000',
            'WARNING': '#FFA500',
            'ERROR': '#FF0000',
            'CRITICAL': '#8B0000'
        }
        color = colors.get(level, '#000000')
        
        self.log_display.append(
            f'<span style="color: {color};">[{level}] {message}</span>'
        )
    
    def _update_status(self):
        """Update status display."""
        if not self.bot_app:
            return
        
        status = self.bot_app.status()
        
        # Update status label
        status_text = f"Status: {status.value.capitalize()}"
        self.status_label.setText(status_text)
        
        # Update button states
        if status == BotStatus.READY:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        elif status == BotStatus.RUNNING:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
    
    def _on_start_clicked(self):
        """Handle start button click."""
        if not self.bot_app:
            return
        
        def start_worker():
            self.bot_app.start(blocking=True)
        
        self.worker_thread = threading.Thread(target=start_worker, daemon=True)
        self.worker_thread.start()
        
        self.status_bar.showMessage("Bot started")
    
    def _on_stop_clicked(self):
        """Handle stop button click."""
        if not self.bot_app:
            return
        
        self.bot_app.stop()
        self.status_bar.showMessage("Bot stopped")
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AI Trading Bot",
            "<h2>AI Trading Bot</h2>"
            "<p>Version 0.1.0</p>"
            "<p>An AI-powered trading bot for cryptocurrency and stock markets.</p>"
            "<p>Built with Python and PySide6.</p>"
        )
    
    def _on_model_loaded(self, model_name):
        """Handle model loaded event."""
        if self.model_tab and self.predictions_tab:
            # Get the loaded predictor and pass it to predictions tab
            predictor = self.model_tab.get_loaded_predictor()
            if predictor:
                self.predictions_tab.set_predictor(predictor)
                
                # Initialize strategy
                if self.strategy_tab:
                    try:
                        from trading_bot.trading.strategy import XGBoostStrategy
                        from trading_bot.trading.signal_generator import SignalGenerator
                        
                        config = self.bot_app.get_config()
                        logger = self.bot_app.get_logger()
                        
                        signal_gen = SignalGenerator(config, logger)
                        strategy = XGBoostStrategy(config, logger, predictor, signal_gen)
                        
                        self.strategy_tab.set_strategy(strategy)
                    except Exception as e:
                        logger.error(f"Failed to initialize strategy: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.bot_app and self.bot_app.status() == BotStatus.RUNNING:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Bot is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.bot_app.stop()
                self.bot_app.shutdown()
                event.accept()
            else:
                event.ignore()
        else:
            if self.bot_app:
                self.bot_app.shutdown()
            event.accept()


def create_application():
    """
    Create and return the Qt application and main window.
    
    Returns:
        Tuple of (QApplication, MainWindow)
    """
    app = QApplication(sys.argv)
    app.setApplicationName("AI Trading Bot")
    app.setOrganizationName("AITradingBot")
    
    window = MainWindow()
    window.show()
    
    return app, window

