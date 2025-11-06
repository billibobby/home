"""
Backtesting Tab for Trading Bot GUI

Provides interface for running walk-forward backtests.
"""

from typing import Optional
from datetime import datetime, timedelta

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
        QTableWidgetItem, QLabel, QGroupBox, QLineEdit, QDateEdit,
        QComboBox, QMessageBox, QProgressBar, QTabWidget, QSpinBox,
        QFileDialog
    )
    from PySide6.QtCore import Qt, QThread, Signal, QDate
    from PySide6.QtGui import QFont
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class BacktestWorker(QThread):
    """Worker thread for running backtests."""
    
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(object)  # BacktestResults
    error = Signal(str)
    
    def __init__(self, backtest_engine, initial_capital):
        super().__init__()
        self.backtest_engine = backtest_engine
        self.initial_capital = initial_capital
        self.stop_requested = False
    
    def request_stop(self):
        """Request the worker to stop cooperatively."""
        self.stop_requested = True
    
    def run(self):
        """Run backtest in thread."""
        try:
            def progress_callback(current, total, message):
                # Check if stop requested and break out if so
                if self.stop_requested:
                    return
                self.progress.emit(current, total, message)
            
            # Set stop_requested flag on engine if it supports it
            if hasattr(self.backtest_engine, 'stop_requested'):
                self.backtest_engine.stop_requested = False
                
                # Create a closure to check stop_requested
                original_callback = None
                if hasattr(self, '_original_callback'):
                    original_callback = self._original_callback
                
                def wrapped_progress_callback(current, total, message):
                    if self.stop_requested:
                        self.backtest_engine.stop_requested = True
                        return
                    if original_callback:
                        original_callback(current, total, message)
                    else:
                        progress_callback(current, total, message)
                
                # Use wrapped callback
                def progress_callback_wrapper(current, total, message):
                    if self.stop_requested:
                        self.backtest_engine.stop_requested = True
                        return
                    progress_callback(current, total, message)
                
                progress_callback = progress_callback_wrapper
            
            results = self.backtest_engine.run(
                initial_capital=self.initial_capital,
                progress_callback=progress_callback
            )
            
            if not self.stop_requested:
                self.finished.emit(results)
        except Exception as e:
            if not self.stop_requested:
                self.error.emit(str(e))


class BacktestingTab(QWidget):
    """
    Tab for backtesting interface.
    
    Features:
    - Run walk-forward backtests
    - View results and metrics
    - Generate visualizations
    - Export results
    """
    
    def __init__(self, bot_app):
        """
        Initialize the backtesting tab.
        
        Args:
            bot_app: BotApp instance
        """
        super().__init__()
        self.bot_app = bot_app
        self.backtest_worker = None
        self.current_results = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Walk-Forward Backtesting")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        # Input section
        input_group = QGroupBox("Backtest Configuration")
        input_layout = QVBoxLayout()
        
        # First row: Symbol and dates
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit("AAPL")
        self.symbol_input.setMaximumWidth(100)
        row1.addWidget(self.symbol_input)
        
        row1.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addYears(-2))
        self.start_date.setCalendarPopup(True)
        row1.addWidget(self.start_date)
        
        row1.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        row1.addWidget(self.end_date)
        
        row1.addStretch()
        input_layout.addLayout(row1)
        
        # Second row: Model selector
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        row2.addWidget(self.model_combo)
        
        row2.addStretch()
        input_layout.addLayout(row2)
        
        # Third row: Walk-forward parameters
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Train Period (days):"))
        self.train_period = QSpinBox()
        self.train_period.setMinimum(60)
        self.train_period.setMaximum(1000)
        self.train_period.setValue(252)
        row3.addWidget(self.train_period)
        
        row3.addWidget(QLabel("Test Period (days):"))
        self.test_period = QSpinBox()
        self.test_period.setMinimum(5)
        self.test_period.setMaximum(100)
        self.test_period.setValue(21)
        row3.addWidget(self.test_period)
        
        row3.addWidget(QLabel("Step Size (days):"))
        self.step_size = QSpinBox()
        self.step_size.setMinimum(5)
        self.step_size.setMaximum(100)
        self.step_size.setValue(21)
        row3.addWidget(self.step_size)
        
        row3.addStretch()
        input_layout.addLayout(row3)
        
        # Fourth row: Buttons
        row4 = QHBoxLayout()
        self.run_button = QPushButton("▶️ Run Backtest")
        self.run_button.clicked.connect(self._run_backtest)
        row4.addWidget(self.run_button)
        
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.clicked.connect(self._stop_backtest)
        self.stop_button.setEnabled(False)
        row4.addWidget(self.stop_button)
        
        row4.addStretch()
        input_layout.addLayout(row4)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Results section (tabbed)
        results_tabs = QTabWidget()
        
        # Summary tab
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        summary_layout.addWidget(self.metrics_table)
        results_tabs.addTab(summary_widget, "Summary")
        
        # Equity curve tab
        equity_widget = QWidget()
        equity_layout = QVBoxLayout(equity_widget)
        self.equity_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        equity_layout.addWidget(self.equity_canvas)
        results_tabs.addTab(equity_widget, "Equity Curve")
        
        # Drawdown tab
        drawdown_widget = QWidget()
        drawdown_layout = QVBoxLayout(drawdown_widget)
        self.drawdown_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        drawdown_layout.addWidget(self.drawdown_canvas)
        results_tabs.addTab(drawdown_widget, "Drawdown")
        
        # Trades tab
        trades_widget = QWidget()
        trades_layout = QVBoxLayout(trades_widget)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            "Date", "Symbol", "Side", "Entry Price", "Exit Price",
            "Quantity", "PnL", "PnL%"
        ])
        trades_layout.addWidget(self.trades_table)
        results_tabs.addTab(trades_widget, "Trades")
        
        # Periods tab
        periods_widget = QWidget()
        periods_layout = QVBoxLayout(periods_widget)
        self.periods_table = QTableWidget()
        self.periods_table.setColumnCount(5)
        self.periods_table.setHorizontalHeaderLabels([
            "Period", "Start Date", "End Date", "Return %", "Trades"
        ])
        periods_layout.addWidget(self.periods_table)
        results_tabs.addTab(periods_widget, "Periods")
        
        layout.addWidget(results_tabs)
        
        # Actions section
        actions_layout = QHBoxLayout()
        
        self.export_button = QPushButton("💾 Export Results")
        self.export_button.clicked.connect(self._export_results)
        self.export_button.setEnabled(False)
        actions_layout.addWidget(self.export_button)
        
        self.report_button = QPushButton("📊 Generate Report")
        self.report_button.clicked.connect(self._generate_html_report)
        self.report_button.setEnabled(False)
        actions_layout.addWidget(self.report_button)
        
        self.save_plots_button = QPushButton("🖼️ Save Plots")
        self.save_plots_button.clicked.connect(self._save_plots)
        self.save_plots_button.setEnabled(False)
        actions_layout.addWidget(self.save_plots_button)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
    
    def _run_backtest(self):
        """Run backtest."""
        if not self.bot_app:
            QMessageBox.warning(self, "Error", "Bot application not initialized")
            return
        
        # Validate inputs
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Error", "Please enter a symbol")
            return
        
        start_date = self.start_date.date().toString(Qt.ISODate)
        end_date = self.end_date.date().toString(Qt.ISODate)
        
        if start_date >= end_date:
            QMessageBox.warning(self, "Error", "Start date must be before end date")
            return
        
        try:
            from trading_bot.backtesting import WalkForwardBacktest
            from trading_bot.data import FeatureEngineer, StockDataFetcher
            from trading_bot.trading import SignalGenerator
            from trading_bot.models import XGBoostTrainer
            
            config = self.bot_app.get_config()
            logger = self.bot_app.get_logger()
            
            # Fetch data
            self.status_label.setText("Fetching historical data...")
            fetcher = StockDataFetcher(config, logger)
            data = fetcher.fetch_historical_data(symbol, start_date, end_date)
            
            if len(data) < 252:
                QMessageBox.warning(self, "Error", "Insufficient data for backtesting")
                return
            
            # Initialize components
            feature_engineer = FeatureEngineer(config, logger)
            signal_generator = SignalGenerator(config, logger)
            
            # Create backtest engine
            backtest_engine = WalkForwardBacktest(
                data=data,
                config=config,
                logger=logger,
                feature_engineer=feature_engineer,
                signal_generator=signal_generator,
                model_class=XGBoostTrainer
            )
            
            # Set symbol on backtest engine
            backtest_engine.symbol = symbol
            
            # Override window parameters
            backtest_engine.train_period_days = self.train_period.value()
            backtest_engine.test_period_days = self.test_period.value()
            backtest_engine.step_size_days = self.step_size.value()
            
            # Create worker thread
            self.backtest_worker = BacktestWorker(backtest_engine, initial_capital=10000.0)
            self.backtest_worker.backtest_engine = backtest_engine  # Store reference for stop
            self.backtest_worker.progress.connect(self._update_progress)
            self.backtest_worker.finished.connect(self._backtest_finished)
            self.backtest_worker.error.connect(self._backtest_error)
            
            # Update UI
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Running backtest...")
            
            # Start backtest
            self.backtest_worker.start()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start backtest: {str(e)}")
            self.status_label.setText("Error")
    
    def _stop_backtest(self):
        """Stop running backtest cooperatively."""
        if self.backtest_worker and self.backtest_worker.isRunning():
            # Request cooperative stop
            self.backtest_worker.request_stop()
            if hasattr(self.backtest_worker, 'backtest_engine'):
                if hasattr(self.backtest_worker.backtest_engine, 'stop_requested'):
                    self.backtest_worker.backtest_engine.stop_requested = True
            
            # Wait for thread to finish (with timeout)
            if not self.backtest_worker.wait(5000):  # 5 second timeout
                # If still running after timeout, force terminate as last resort
                import logging
                logging.warning("Backtest worker did not stop cooperatively, forcing termination")
                self.backtest_worker.terminate()
                self.backtest_worker.wait()
            
            self.status_label.setText("Backtest stopped")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
    
    def _backtest_finished(self, results):
        """Handle backtest completion."""
        self.current_results = results
        self._display_results(results)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)
        self.report_button.setEnabled(True)
        self.save_plots_button.setEnabled(True)
        self.status_label.setText("Backtest completed")
        self.progress_bar.setValue(self.progress_bar.maximum())
    
    def _backtest_error(self, error_msg: str):
        """Handle backtest error."""
        QMessageBox.critical(self, "Backtest Error", f"Backtest failed: {error_msg}")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Error occurred")
    
    def _display_results(self, results):
        """Display backtest results."""
        # Populate metrics table
        self._populate_metrics_table(results)
        
        # Plot equity curve
        self._plot_equity_curve(results)
        
        # Plot drawdown
        self._plot_drawdown(results)
        
        # Populate trades table
        self._populate_trades_table(results)
        
        # Populate periods table
        self._populate_periods_table(results)
    
    def _populate_metrics_table(self, results):
        """Populate metrics table."""
        metrics = [
            ("Total Return (%)", f"{results.total_return:.2f}"),
            ("Annualized Return (%)", f"{results.annualized_return:.2f}"),
            ("Sharpe Ratio", f"{results.sharpe_ratio:.2f}"),
            ("Sortino Ratio", f"{results.sortino_ratio:.2f}"),
            ("Max Drawdown (%)", f"{results.max_drawdown:.2f}"),
            ("Win Rate (%)", f"{results.win_rate:.2f}"),
            ("Profit Factor", f"{results.profit_factor:.2f}"),
            ("Total Trades", f"{results.total_trades}"),
            ("Winning Trades", f"{results.winning_trades}"),
            ("Losing Trades", f"{results.losing_trades}"),
        ]
        
        self.metrics_table.setRowCount(len(metrics))
        for i, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
    
    def _plot_equity_curve(self, results):
        """Plot equity curve."""
        if len(results.equity_curve) == 0:
            return
        
        ax = self.equity_canvas.figure.subplots()
        ax.clear()
        ax.plot(results.equity_curve.index, results.equity_curve.values, linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Equity Curve')
        ax.grid(True, alpha=0.3)
        self.equity_canvas.draw()
    
    def _plot_drawdown(self, results):
        """Plot drawdown."""
        if len(results.equity_curve) == 0:
            return
        
        running_max = results.equity_curve.expanding().max()
        drawdown = (results.equity_curve - running_max) / running_max * 100
        
        ax = self.drawdown_canvas.figure.subplots()
        ax.clear()
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Chart')
        ax.grid(True, alpha=0.3)
        self.drawdown_canvas.draw()
    
    def _populate_trades_table(self, results):
        """Populate trades table."""
        self.trades_table.setRowCount(len(results.trades))
        
        for i, trade in enumerate(results.trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get('entry_time', ''))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(str(trade.get('symbol', ''))))
            self.trades_table.setItem(i, 2, QTableWidgetItem(str(trade.get('side', ''))))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade.get('entry_price', 0):.2f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('exit_price', 0):.2f}" if trade.get('exit_price') else ''))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.get('quantity', 0):.2f}"))
            self.trades_table.setItem(i, 6, QTableWidgetItem(f"{trade.get('pnl', 0):.2f}" if trade.get('pnl') is not None else ''))
            self.trades_table.setItem(i, 7, QTableWidgetItem(f"{trade.get('pnl_pct', 0):.2f}" if trade.get('pnl_pct') is not None else ''))
    
    def _populate_periods_table(self, results):
        """Populate periods table."""
        self.periods_table.setRowCount(len(results.period_results))
        
        for i, period in enumerate(results.period_results):
            self.periods_table.setItem(i, 0, QTableWidgetItem(str(period.get('window', ''))))
            self.periods_table.setItem(i, 1, QTableWidgetItem(str(period.get('start_date', ''))))
            self.periods_table.setItem(i, 2, QTableWidgetItem(str(period.get('end_date', ''))))
            self.periods_table.setItem(i, 3, QTableWidgetItem(f"{period.get('total_return_pct', 0):.2f}"))
            self.periods_table.setItem(i, 4, QTableWidgetItem(str(period.get('num_trades', 0))))
    
    def _export_results(self):
        """Export results to file."""
        if not self.current_results:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON (*.json);;CSV (*.csv);;Pickle (*.pkl)"
        )
        
        if file_path:
            try:
                format = 'json' if file_path.endswith('.json') else ('csv' if file_path.endswith('.csv') else 'pickle')
                self.current_results.save_to_file(file_path, format=format)
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def _generate_html_report(self):
        """Generate HTML report."""
        if not self.current_results:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML Report", "", "HTML (*.html)"
        )
        
        if file_path:
            try:
                from trading_bot.backtesting import BacktestVisualizer
                config = self.bot_app.get_config()
                logger = self.bot_app.get_logger()
                visualizer = BacktestVisualizer(config, logger)
                visualizer.generate_html_report(self.current_results, file_path)
                QMessageBox.information(self, "Success", f"Report generated: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")
    
    def _save_plots(self):
        """Save all plots to directory."""
        if not self.current_results:
            return
        
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Plots")
        
        if dir_path:
            try:
                from trading_bot.backtesting import BacktestVisualizer
                config = self.bot_app.get_config()
                logger = self.bot_app.get_logger()
                visualizer = BacktestVisualizer(config, logger)
                saved_files = visualizer.plot_all(self.current_results, dir_path)
                QMessageBox.information(self, "Success", f"Plots saved to {dir_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plots: {str(e)}")

