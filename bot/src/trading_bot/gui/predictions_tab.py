"""
Predictions Dashboard Tab for Trading Bot GUI

Displays real-time predictions and signals.
"""

from typing import Optional
from datetime import datetime
from collections import deque

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QGroupBox, QTableWidget, QTableWidgetItem, QLineEdit,
        QComboBox, QMessageBox, QTextEdit
    )
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QFont
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class PredictionsDashboard(QWidget):
    """
    Dashboard for viewing predictions and signals.
    
    Features:
    - Real-time price and prediction display
    - Signal generation (BUY/SELL/HOLD)
    - Confidence scores
    - Prediction history
    """
    
    prediction_made = Signal(dict)  # Emits prediction data
    
    def __init__(self, bot_app):
        """
        Initialize the predictions dashboard.
        
        Args:
            bot_app: BotApp instance
        """
        super().__init__()
        self.bot_app = bot_app
        self.predictor = None
        self.signal_generator = None
        self.data_fetcher = None
        self.feature_engineer = None
        
        self.prediction_history = deque(maxlen=100)
        self.auto_predict_enabled = False
        
        self._init_ui()
        self._setup_timers()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Predictions Dashboard")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        # Input group
        input_group = QGroupBox("Prediction Input")
        input_layout = QHBoxLayout()
        
        input_layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit("AAPL")
        self.symbol_input.setMaximumWidth(100)
        input_layout.addWidget(self.symbol_input)
        
        self.predict_button = QPushButton("🔮 Predict")
        self.predict_button.clicked.connect(self._make_prediction)
        input_layout.addWidget(self.predict_button)
        
        self.auto_predict_button = QPushButton("▶️ Auto Predict")
        self.auto_predict_button.setCheckable(True)
        self.auto_predict_button.toggled.connect(self._toggle_auto_predict)
        input_layout.addWidget(self.auto_predict_button)
        
        input_layout.addStretch()
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Current prediction display
        current_group = QGroupBox("Current Prediction")
        current_layout = QVBoxLayout()
        
        # Metrics display
        metrics_layout = QHBoxLayout()
        
        # Current price
        price_box = QGroupBox("Current Price")
        price_box_layout = QVBoxLayout()
        self.current_price_label = QLabel("--")
        self.current_price_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.current_price_label.setAlignment(Qt.AlignCenter)
        price_box_layout.addWidget(self.current_price_label)
        price_box.setLayout(price_box_layout)
        metrics_layout.addWidget(price_box)
        
        # Predicted price
        pred_box = QGroupBox("Predicted Price")
        pred_box_layout = QVBoxLayout()
        self.predicted_price_label = QLabel("--")
        self.predicted_price_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.predicted_price_label.setAlignment(Qt.AlignCenter)
        pred_box_layout.addWidget(self.predicted_price_label)
        pred_box.setLayout(pred_box_layout)
        metrics_layout.addWidget(pred_box)
        
        # Expected return
        return_box = QGroupBox("Expected Return")
        return_box_layout = QVBoxLayout()
        self.expected_return_label = QLabel("--")
        self.expected_return_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.expected_return_label.setAlignment(Qt.AlignCenter)
        return_box_layout.addWidget(self.expected_return_label)
        return_box.setLayout(return_box_layout)
        metrics_layout.addWidget(return_box)
        
        current_layout.addLayout(metrics_layout)
        
        # Signal display
        signal_layout = QHBoxLayout()
        
        signal_layout.addWidget(QLabel("Signal:"))
        self.signal_label = QLabel("--")
        self.signal_label.setFont(QFont("Arial", 18, QFont.Bold))
        signal_layout.addWidget(self.signal_label)
        
        signal_layout.addWidget(QLabel("Confidence:"))
        self.confidence_label = QLabel("--")
        self.confidence_label.setFont(QFont("Arial", 16))
        signal_layout.addWidget(self.confidence_label)
        
        signal_layout.addWidget(QLabel("Strength:"))
        self.strength_label = QLabel("--")
        self.strength_label.setFont(QFont("Arial", 16))
        signal_layout.addWidget(self.strength_label)
        
        signal_layout.addStretch()
        current_layout.addLayout(signal_layout)
        
        # Reasoning
        self.reasoning_display = QTextEdit()
        self.reasoning_display.setReadOnly(True)
        self.reasoning_display.setMaximumHeight(80)
        current_layout.addWidget(QLabel("Reasoning:"))
        current_layout.addWidget(self.reasoning_display)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # Prediction history table
        history_group = QGroupBox("Prediction History")
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Current $", "Predicted $", "Return %", "Signal", "Confidence"
        ])
        history_layout.addWidget(self.history_table)
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self._clear_history)
        history_layout.addWidget(clear_history_btn)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        # Status
        self.status_label = QLabel("Ready - Load a model to begin")
        layout.addWidget(self.status_label)
    
    def _setup_timers(self):
        """Set up periodic prediction timer."""
        self.auto_predict_timer = QTimer()
        self.auto_predict_timer.timeout.connect(self._auto_predict)
        # Will start when auto-predict is enabled
    
    def set_predictor(self, predictor):
        """
        Set the predictor to use.
        
        Args:
            predictor: XGBoostPredictor instance
        """
        self.predictor = predictor
        
        # Initialize other components
        if self.bot_app:
            try:
                from trading_bot.data import StockDataFetcher, FeatureEngineer
                from trading_bot.trading import SignalGenerator
                
                config = self.bot_app.config
                logger = self.bot_app.get_logger()
                
                self.data_fetcher = StockDataFetcher(config, logger)
                self.feature_engineer = FeatureEngineer(config, logger)
                self.signal_generator = SignalGenerator(config, logger)
                
                self.status_label.setText("✅ Ready to predict")
                self.predict_button.setEnabled(True)
                self.auto_predict_button.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"❌ Error: {str(e)}")
    
    def _make_prediction(self):
        """Make a prediction for the current symbol."""
        if not self.predictor or not self.predictor.is_model_loaded():
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a model first in the Model Management tab."
            )
            return
        
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Invalid Input", "Please enter a symbol")
            return
        
        try:
            self.status_label.setText(f"Fetching data for {symbol}...")
            
            # Fetch latest data
            lookback = self.bot_app.config.get('models.xgboost.lookback_days', 60)
            data = self.data_fetcher.fetch_latest_data(symbol, period=f"{lookback + 30}d")
            
            self.status_label.setText(f"Engineering features...")
            
            # Engineer features
            features = self.feature_engineer.create_features(data)
            features = features.dropna().tail(1)
            
            if len(features) == 0:
                raise Exception("No valid features after preprocessing")
            
            self.status_label.setText(f"Generating prediction...")
            
            # Make prediction
            prediction = self.predictor.predict(features)[0]
            confidence = self.predictor.get_confidence(prediction, features)
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                prediction,
                confidence,
                current_price,
                symbol
            )
            
            # Update display
            self._update_display(symbol, current_price, prediction, confidence, signal)
            
            # Add to history
            self._add_to_history(symbol, current_price, prediction, confidence, signal)
            
            self.status_label.setText(f"✅ Prediction complete for {symbol}")
            
            # Emit signal
            self.prediction_made.emit({
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': prediction,
                'confidence': confidence,
                'signal': signal
            })
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            QMessageBox.critical(
                self,
                "Prediction Error",
                f"Failed to make prediction:\n{str(e)}"
            )
    
    def _update_display(self, symbol, current_price, predicted_price, confidence, signal):
        """Update the current prediction display."""
        # Update prices
        self.current_price_label.setText(f"${current_price:.2f}")
        self.predicted_price_label.setText(f"${predicted_price:.2f}")
        
        # Calculate and display expected return
        expected_return = ((predicted_price - current_price) / current_price) * 100
        self.expected_return_label.setText(f"{expected_return:+.2f}%")
        
        # Color code expected return
        if expected_return > 0:
            self.expected_return_label.setStyleSheet("color: green;")
        elif expected_return < 0:
            self.expected_return_label.setStyleSheet("color: red;")
        else:
            self.expected_return_label.setStyleSheet("color: black;")
        
        # Update signal
        signal_text = signal['type']
        self.signal_label.setText(signal_text)
        
        # Color code signal
        signal_colors = {
            'STRONG_BUY': 'darkgreen',
            'BUY': 'green',
            'HOLD': 'gray',
            'SELL': 'orange',
            'STRONG_SELL': 'red'
        }
        color = signal_colors.get(signal_text, 'black')
        self.signal_label.setStyleSheet(f"color: {color};")
        
        # Update confidence and strength
        self.confidence_label.setText(f"{confidence:.2%}")
        self.strength_label.setText(f"{signal.get('strength', 0):.2%}")
        
        # Update reasoning
        self.reasoning_display.setText(signal.get('reasoning', 'No reasoning available'))
    
    def _add_to_history(self, symbol, current_price, predicted_price, confidence, signal):
        """Add prediction to history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        expected_return = ((predicted_price - current_price) / current_price) * 100
        
        # Add to deque
        self.prediction_history.append({
            'time': timestamp,
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'return': expected_return,
            'signal': signal['type'],
            'confidence': confidence
        })
        
        # Update table
        self.history_table.setRowCount(len(self.prediction_history))
        for i, pred in enumerate(reversed(list(self.prediction_history))):
            row = i
            self.history_table.setItem(row, 0, QTableWidgetItem(pred['time']))
            self.history_table.setItem(row, 1, QTableWidgetItem(pred['symbol']))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"${pred['current_price']:.2f}"))
            self.history_table.setItem(row, 3, QTableWidgetItem(f"${pred['predicted_price']:.2f}"))
            self.history_table.setItem(row, 4, QTableWidgetItem(f"{pred['return']:+.2f}%"))
            self.history_table.setItem(row, 5, QTableWidgetItem(pred['signal']))
            self.history_table.setItem(row, 6, QTableWidgetItem(f"{pred['confidence']:.2%}"))
        
        self.history_table.resizeColumnsToContents()
    
    def _clear_history(self):
        """Clear prediction history."""
        self.prediction_history.clear()
        self.history_table.setRowCount(0)
    
    def _toggle_auto_predict(self, checked):
        """Toggle auto prediction."""
        self.auto_predict_enabled = checked
        
        if checked:
            self.auto_predict_button.setText("⏸️ Pause")
            self.auto_predict_timer.start(60000)  # Predict every minute
            self.status_label.setText("🔄 Auto-predict enabled (every 60s)")
        else:
            self.auto_predict_button.setText("▶️ Auto Predict")
            self.auto_predict_timer.stop()
            self.status_label.setText("Auto-predict disabled")
    
    def _auto_predict(self):
        """Automatically make prediction."""
        if self.auto_predict_enabled:
            self._make_prediction()

