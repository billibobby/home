"""
Strategy Monitor Tab for Trading Bot GUI

Monitors trading strategy execution and performance.
"""

from typing import Optional
from datetime import datetime

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QGroupBox, QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox
    )
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class StrategyMonitor(QWidget):
    """
    Monitor for trading strategy execution.
    
    Features:
    - Active positions display
    - Signal history
    - Performance metrics
    - Strategy status
    """
    
    def __init__(self, bot_app):
        """
        Initialize the strategy monitor.
        
        Args:
            bot_app: BotApp instance
        """
        super().__init__()
        self.bot_app = bot_app
        self.strategy = None
        
        self._init_ui()
        self._setup_timers()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Strategy Monitor")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QHBoxLayout()
        
        # Total signals
        signals_box = QGroupBox("Total Signals")
        signals_layout = QVBoxLayout()
        self.total_signals_label = QLabel("0")
        self.total_signals_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.total_signals_label.setAlignment(Qt.AlignCenter)
        signals_layout.addWidget(self.total_signals_label)
        signals_box.setLayout(signals_layout)
        metrics_layout.addWidget(signals_box)
        
        # Buy signals
        buy_box = QGroupBox("Buy Signals")
        buy_layout = QVBoxLayout()
        self.buy_signals_label = QLabel("0")
        self.buy_signals_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.buy_signals_label.setAlignment(Qt.AlignCenter)
        self.buy_signals_label.setStyleSheet("color: green;")
        buy_layout.addWidget(self.buy_signals_label)
        buy_box.setLayout(buy_layout)
        metrics_layout.addWidget(buy_box)
        
        # Sell signals
        sell_box = QGroupBox("Sell Signals")
        sell_layout = QVBoxLayout()
        self.sell_signals_label = QLabel("0")
        self.sell_signals_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.sell_signals_label.setAlignment(Qt.AlignCenter)
        self.sell_signals_label.setStyleSheet("color: red;")
        sell_layout.addWidget(self.sell_signals_label)
        sell_box.setLayout(sell_layout)
        metrics_layout.addWidget(sell_box)
        
        # Active positions
        positions_box = QGroupBox("Active Positions")
        positions_layout = QVBoxLayout()
        self.active_positions_label = QLabel("0")
        self.active_positions_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.active_positions_label.setAlignment(Qt.AlignCenter)
        positions_layout.addWidget(self.active_positions_label)
        positions_box.setLayout(positions_layout)
        metrics_layout.addWidget(positions_box)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Active positions table
        positions_group = QGroupBox("Active Positions")
        positions_table_layout = QVBoxLayout()
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels([
            "Symbol", "Entry Price", "Position Size", "Stop Loss", "Take Profit"
        ])
        positions_table_layout.addWidget(self.positions_table)
        
        positions_group.setLayout(positions_table_layout)
        layout.addWidget(positions_group)
        
        # Signal history table
        signals_group = QGroupBox("Signal History")
        signals_layout = QVBoxLayout()
        
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(6)
        self.signals_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Signal", "Confidence", "Strength", "Reasoning"
        ])
        signals_layout.addWidget(self.signals_table)
        
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_signal_history)
        signals_layout.addWidget(clear_btn)
        
        signals_group.setLayout(signals_layout)
        layout.addWidget(signals_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("📊 Analyze Symbol")
        self.analyze_button.clicked.connect(self._analyze_symbol)
        self.analyze_button.setEnabled(False)
        controls_layout.addWidget(self.analyze_button)
        
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.clicked.connect(self._refresh_display)
        controls_layout.addWidget(self.refresh_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Status
        self.status_label = QLabel("Ready - Initialize strategy to begin")
        layout.addWidget(self.status_label)
    
    def _setup_timers(self):
        """Set up periodic refresh."""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_display)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def set_strategy(self, strategy):
        """
        Set the strategy to monitor.
        
        Args:
            strategy: XGBoostStrategy instance
        """
        self.strategy = strategy
        self.analyze_button.setEnabled(True)
        self.status_label.setText("✅ Strategy initialized")
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh all displays."""
        if not self.strategy:
            return
        
        try:
            # Update positions
            positions = self.strategy.get_active_positions()
            self.active_positions_label.setText(str(len(positions)))
            
            # Update positions table
            self.positions_table.setRowCount(len(positions))
            for i, (symbol, pos) in enumerate(positions.items()):
                self.positions_table.setItem(i, 0, QTableWidgetItem(symbol))
                self.positions_table.setItem(i, 1, QTableWidgetItem(f"${pos['entry_price']:.2f}"))
                self.positions_table.setItem(i, 2, QTableWidgetItem(f"${pos['size']:.2f}"))
                self.positions_table.setItem(i, 3, QTableWidgetItem(f"${pos.get('stop_loss', 0):.2f}"))
                self.positions_table.setItem(i, 4, QTableWidgetItem(f"${pos.get('take_profit', 0):.2f}"))
            
            self.positions_table.resizeColumnsToContents()
            
            # Update signal history
            signals = self.strategy.get_signal_history(limit=50)
            
            # Count signal types
            total = len(signals)
            buy_count = sum(1 for s in signals if 'BUY' in s.get('type', ''))
            sell_count = sum(1 for s in signals if 'SELL' in s.get('type', ''))
            
            self.total_signals_label.setText(str(total))
            self.buy_signals_label.setText(str(buy_count))
            self.sell_signals_label.setText(str(sell_count))
            
            # Update signals table
            self.signals_table.setRowCount(len(signals))
            for i, signal in enumerate(reversed(signals)):
                self.signals_table.setItem(i, 0, QTableWidgetItem(signal.get('timestamp', '')))
                self.signals_table.setItem(i, 1, QTableWidgetItem(signal.get('symbol', 'N/A')))
                self.signals_table.setItem(i, 2, QTableWidgetItem(signal.get('type', 'N/A')))
                self.signals_table.setItem(i, 3, QTableWidgetItem(f"{signal.get('confidence', 0):.2%}"))
                self.signals_table.setItem(i, 4, QTableWidgetItem(f"{signal.get('strength', 0):.2%}"))
                self.signals_table.setItem(i, 5, QTableWidgetItem(signal.get('reasoning', '')))
            
            self.signals_table.resizeColumnsToContents()
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
    
    def _analyze_symbol(self):
        """Analyze a symbol using the strategy."""
        from PySide6.QtWidgets import QInputDialog
        
        symbol, ok = QInputDialog.getText(
            self,
            "Analyze Symbol",
            "Enter stock symbol:"
        )
        
        if ok and symbol:
            try:
                self.status_label.setText(f"Analyzing {symbol}...")
                
                # Run analysis
                decision = self.strategy.analyze(symbol.strip().upper())
                
                if decision:
                    self._show_decision(decision)
                    self._refresh_display()
                    self.status_label.setText(f"✅ Analysis complete: {decision['signal']['type']}")
                else:
                    QMessageBox.information(
                        self,
                        "No Action",
                        f"No actionable signal generated for {symbol}"
                    )
                    self.status_label.setText(f"No action for {symbol}")
                    
            except Exception as e:
                self.status_label.setText(f"❌ Error: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Analysis Error",
                    f"Failed to analyze {symbol}:\n{str(e)}"
                )
    
    def _show_decision(self, decision):
        """Show trading decision dialog."""
        signal = decision['signal']
        
        message = f"Symbol: {decision['symbol']}\n"
        message += f"Current Price: ${decision['current_price']:.2f}\n"
        message += f"\n"
        message += f"Signal: {signal['type']}\n"
        message += f"Confidence: {signal['confidence']:.2%}\n"
        message += f"Strength: {signal.get('strength', 0):.2%}\n"
        message += f"\n"
        
        if decision.get('position_size'):
            message += f"Suggested Position Size: ${decision['position_size']:.2f}\n"
            message += f"Stop Loss: ${decision.get('stop_loss', 0):.2f}\n"
            message += f"Take Profit: ${decision.get('take_profit', 0):.2f}\n"
            message += f"\n"
        
        message += f"Reasoning:\n{signal.get('reasoning', 'N/A')}"
        
        QMessageBox.information(
            self,
            "Trading Decision",
            message
        )
    
    def _clear_signal_history(self):
        """Clear signal history."""
        if self.strategy:
            self.strategy.signal_history.clear()
            self._refresh_display()

