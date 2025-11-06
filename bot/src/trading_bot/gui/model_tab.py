"""
Model Management Tab for Trading Bot GUI

Provides interface for managing XGBoost models.
"""

from typing import Optional
import webbrowser
import sys

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
        QTableWidgetItem, QLabel, QGroupBox, QTextEdit, QFileDialog,
        QMessageBox, QComboBox, QProgressBar
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QThread
    from PySide6.QtGui import QFont
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class ModelManagementTab(QWidget):
    """
    Tab for managing ML models.
    
    Features:
    - List available models
    - View model metadata
    - Load/unload models
    - Upload new models
    - Delete old models
    """
    
    model_loaded = Signal(str)  # Emits model name when loaded
    
    def __init__(self, bot_app):
        """
        Initialize the model management tab.
        
        Args:
            bot_app: BotApp instance
        """
        super().__init__()
        self.bot_app = bot_app
        self.model_manager = None
        self.predictor = None
        self.current_model = None
        
        self._init_ui()
        self._setup_timers()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("XGBoost Model Management")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        # Model selector group
        selector_group = QGroupBox("Model Selection")
        selector_layout = QHBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_selected)
        selector_layout.addWidget(QLabel("Select Model:"))
        selector_layout.addWidget(self.model_combo, 1)
        
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.clicked.connect(self._refresh_models)
        selector_layout.addWidget(self.refresh_button)
        
        self.load_button = QPushButton("📥 Load Model")
        self.load_button.clicked.connect(self._load_model)
        self.load_button.setEnabled(False)
        selector_layout.addWidget(self.load_button)
        
        self.unload_button = QPushButton("❌ Unload")
        self.unload_button.clicked.connect(self._unload_model)
        self.unload_button.setEnabled(False)
        selector_layout.addWidget(self.unload_button)
        
        selector_group.setLayout(selector_layout)
        layout.addWidget(selector_group)
        
        # Model info display
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(200)
        self.info_display.setFont(QFont("Courier", 9))
        info_layout.addWidget(self.info_display)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Available models table
        table_group = QGroupBox("Available Models")
        table_layout = QVBoxLayout()
        
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(6)
        self.models_table.setHorizontalHeaderLabels([
            "Filename", "Symbol", "Version", "Date", "Type", "Status"
        ])
        self.models_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.models_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table_layout.addWidget(self.models_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        self.upload_button = QPushButton("📤 Upload Model Files")
        self.upload_button.clicked.connect(self._upload_model)
        actions_layout.addWidget(self.upload_button)
        
        self.train_button = QPushButton("🚀 Open Training Notebook")
        self.train_button.clicked.connect(self._open_training_notebook)
        actions_layout.addWidget(self.train_button)
        
        self.analysis_button = QPushButton("📊 Open Feature Analysis Notebook")
        self.analysis_button.clicked.connect(self._open_feature_analysis_notebook)
        actions_layout.addWidget(self.analysis_button)
        
        self.optimize_button = QPushButton("🔧 Optimize Hyperparameters")
        self.optimize_button.clicked.connect(self._optimize_hyperparameters)
        actions_layout.addWidget(self.optimize_button)
        
        self.delete_button = QPushButton("🗑️ Delete Old Models")
        self.delete_button.clicked.connect(self._delete_old_models)
        actions_layout.addWidget(self.delete_button)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Initialize model manager
        self._initialize_model_manager()
    
    def _setup_timers(self):
        """Set up periodic refresh."""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
    
    def _initialize_model_manager(self):
        """Initialize the model manager."""
        try:
            from trading_bot.models import ModelManager, XGBoostPredictor
            
            if self.bot_app:
                config = self.bot_app.config
                logger = self.bot_app.get_logger()
                
                self.model_manager = ModelManager(config, logger)
                self.predictor = XGBoostPredictor(config, logger)
                
                self._refresh_models()
                self.status_label.setText("✅ Model manager initialized")
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
    
    def _refresh_models(self):
        """Refresh the list of available models."""
        if not self.model_manager:
            return
        
        try:
            models = self.model_manager.list_available_models('xgboost')
            
            # Update combo box
            self.model_combo.clear()
            self.model_combo.addItem("-- Select a model --")
            for model in models:
                self.model_combo.addItem(model['filename'])
            
            # Update table
            self.models_table.setRowCount(len(models))
            for i, model in enumerate(models):
                self.models_table.setItem(i, 0, QTableWidgetItem(model.get('filename', 'N/A')))
                self.models_table.setItem(i, 1, QTableWidgetItem(model.get('symbol', 'N/A')))
                self.models_table.setItem(i, 2, QTableWidgetItem(model.get('version', 'N/A')))
                self.models_table.setItem(i, 3, QTableWidgetItem(model.get('date', 'N/A')))
                
                # Get type from metadata if available
                model_type = 'N/A'
                if 'metadata' in model:
                    model_type = model['metadata'].get('target_type', 'N/A')
                self.models_table.setItem(i, 4, QTableWidgetItem(model_type))
                
                # Check if this is the loaded model
                status = '🟢 Loaded' if model['filename'] == self.current_model else ''
                self.models_table.setItem(i, 5, QTableWidgetItem(status))
            
            self.models_table.resizeColumnsToContents()
            self.status_label.setText(f"✅ Found {len(models)} models")
            
        except Exception as e:
            self.status_label.setText(f"❌ Error refreshing: {str(e)}")
    
    def _auto_refresh(self):
        """Auto-refresh models list."""
        if self.model_manager:
            self._refresh_models()
    
    def _on_model_selected(self, model_name):
        """Handle model selection from combo box."""
        if model_name and model_name != "-- Select a model --":
            self.load_button.setEnabled(True)
            self._display_model_info(model_name)
        else:
            self.load_button.setEnabled(False)
            self.info_display.clear()
    
    def _on_table_selection_changed(self):
        """Handle table selection change."""
        selected = self.models_table.selectedItems()
        if selected:
            row = selected[0].row()
            model_name = self.models_table.item(row, 0).text()
            self.model_combo.setCurrentText(model_name)
    
    def _display_model_info(self, model_name):
        """Display information about a model."""
        if not self.model_manager:
            return
        
        try:
            models = self.model_manager.list_available_models('xgboost')
            model_info = next((m for m in models if m['filename'] == model_name), None)
            
            if not model_info:
                return
            
            info_text = f"📋 Model: {model_name}\n"
            info_text += f"📍 Path: {model_info.get('path', 'N/A')}\n"
            info_text += f"🏷️ Symbol: {model_info.get('symbol', 'N/A')}\n"
            info_text += f"📌 Version: {model_info.get('version', 'N/A')}\n"
            info_text += f"📅 Date: {model_info.get('date', 'N/A')}\n"
            
            if 'metadata' in model_info:
                metadata = model_info['metadata']
                info_text += f"\n📊 Metadata:\n"
                info_text += f"  Type: {metadata.get('target_type', 'N/A')}\n"
                info_text += f"  Features: {metadata.get('n_features', 'N/A')}\n"
                info_text += f"  Training Date: {metadata.get('training_date', 'N/A')}\n"
                
                if 'metrics' in metadata:
                    info_text += f"\n📈 Performance:\n"
                    for metric, value in metadata['metrics'].items():
                        info_text += f"  {metric}: {value:.4f}\n"
            
            self.info_display.setText(info_text)
            
        except Exception as e:
            self.info_display.setText(f"Error loading model info: {str(e)}")
    
    def _load_model(self):
        """Load the selected model."""
        model_name = self.model_combo.currentText()
        if not model_name or model_name == "-- Select a model --":
            return
        
        try:
            self.status_label.setText(f"Loading {model_name}...")
            
            # Load model files
            model_info = self.model_manager.load_model(model_name)
            self.predictor.load_model(
                model_info['model_path'],
                model_info['metadata_path'],
                model_info['scaler_path']
            )
            
            self.current_model = model_name
            self.unload_button.setEnabled(True)
            
            self._refresh_models()  # Update status in table
            self.status_label.setText(f"✅ Loaded: {model_name}")
            
            # Emit signal
            self.model_loaded.emit(model_name)
            
            QMessageBox.information(
                self,
                "Success",
                f"Model '{model_name}' loaded successfully!"
            )
            
        except Exception as e:
            self.status_label.setText(f"❌ Error loading model")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load model:\n{str(e)}"
            )
    
    def _unload_model(self):
        """Unload the current model."""
        if self.current_model:
            self.predictor = None
            self.current_model = None
            self.unload_button.setEnabled(False)
            self._refresh_models()
            self.status_label.setText("Model unloaded")
            
            # Reinitialize predictor
            from trading_bot.models import XGBoostPredictor
            config = self.bot_app.config
            logger = self.bot_app.get_logger()
            self.predictor = XGBoostPredictor(config, logger)
    
    def _upload_model(self):
        """Upload new model files."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Model files (*.json *.pkl)")
        
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            # TODO: Implement file copying to models directory
            QMessageBox.information(
                self,
                "Upload",
                f"Selected {len(files)} files. Copy them manually to the models/ directory for now."
            )
    
    def _open_training_notebook(self):
        """Open the training notebook in browser (for Colab)."""
        try:
            import os
            notebook_path = os.path.abspath("notebooks/xgboost_training_colab.ipynb")
            
            reply = QMessageBox.question(
                self,
                "Open Training Notebook",
                "This will open the training notebook file.\n\n"
                "To train with GPU:\n"
                "1. Upload this file to Google Colab\n"
                "2. Enable T4 GPU runtime\n"
                "3. Run all cells\n"
                "4. Download the generated model files\n\n"
                "Open notebook file location?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if os.path.exists(notebook_path):
                    folder = os.path.dirname(notebook_path)
                    webbrowser.open(f'file:///{folder}')
                else:
                    QMessageBox.warning(
                        self,
                        "Not Found",
                        f"Notebook not found at: {notebook_path}"
                    )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open notebook: {str(e)}")
    
    def _open_feature_analysis_notebook(self):
        """Open the feature analysis notebook in Jupyter."""
        try:
            import os
            import subprocess
            from pathlib import Path
            
            # Find notebook path
            project_root = Path(__file__).parent.parent.parent.parent
            notebook_path = project_root / "notebooks" / "feature_analysis.ipynb"
            
            if not notebook_path.exists():
                # Try alternative path
                notebook_path = Path("bot/notebooks/feature_analysis.ipynb")
                if not notebook_path.exists():
                    notebook_path = Path("notebooks/feature_analysis.ipynb")
            
            if notebook_path.exists():
                # Try to open with Jupyter
                try:
                    # Check if Jupyter is available
                    result = subprocess.run(
                        ['jupyter', '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        # Launch Jupyter notebook
                        subprocess.Popen(
                            ['jupyter', 'notebook', str(notebook_path)],
                            cwd=str(notebook_path.parent),
                            shell=False
                        )
                        QMessageBox.information(
                            self,
                            "Notebook Opening",
                            f"Opening feature analysis notebook in Jupyter...\n\n"
                            f"Notebook: {notebook_path.name}\n"
                            f"Location: {notebook_path.parent}"
                        )
                    else:
                        raise FileNotFoundError("Jupyter not found")
                        
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    # Fallback: open file location
                    reply = QMessageBox.question(
                        self,
                        "Jupyter Not Found",
                        "Jupyter Notebook is not installed or not in PATH.\n\n"
                        "Would you like to:\n"
                        "1. Open the notebook file location\n"
                        "2. Install Jupyter (pip install jupyter)\n\n"
                        "Open notebook file location?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        folder = str(notebook_path.parent)
                        if os.name == 'nt':  # Windows
                            os.startfile(folder)
                        elif os.name == 'posix':  # macOS/Linux
                            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', folder])
                        else:
                            webbrowser.open(f'file:///{folder}')
            else:
                QMessageBox.warning(
                    self,
                    "Not Found",
                    f"Feature analysis notebook not found.\n\n"
                    f"Expected at: {notebook_path}\n\n"
                    f"Please ensure the notebook exists in the notebooks/ directory."
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open notebook: {str(e)}"
            )
    
    def _delete_old_models(self):
        """Delete old model versions."""
        reply = QMessageBox.question(
            self,
            "Delete Old Models",
            "This will delete old model versions, keeping only the 3 most recent.\n\n"
            "Are you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                deleted = self.model_manager.delete_old_models(keep_latest_n=3)
                self._refresh_models()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Deleted {deleted} old model(s)"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete models: {str(e)}"
                )
    
    def _optimize_hyperparameters(self):
        """Show optimization configuration dialog."""
        self._show_optimization_dialog()
    
    def _show_optimization_dialog(self):
        """Create and display optimization configuration dialog."""
        if not PYSIDE6_AVAILABLE:
            QMessageBox.warning(
                self,
                "Not Available",
                "Optimization requires PySide6 GUI support"
            )
            return
        
        from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDateEdit, QSpinBox, QComboBox, QDialogButtonBox
        from PySide6.QtCore import QDate
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Optimize Hyperparameters")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout(dialog)
        
        # Symbol
        symbol_input = QLineEdit()
        symbol_input.setPlaceholderText("e.g., AAPL")
        layout.addRow("Symbol:", symbol_input)
        
        # Start Date
        start_date = QDateEdit()
        start_date.setDate(QDate.currentDate().addYears(-2))
        start_date.setCalendarPopup(True)
        layout.addRow("Start Date:", start_date)
        
        # End Date
        end_date = QDateEdit()
        end_date.setDate(QDate.currentDate())
        end_date.setCalendarPopup(True)
        layout.addRow("End Date:", end_date)
        
        # Number of Trials
        n_trials = QSpinBox()
        n_trials.setMinimum(10)
        n_trials.setMaximum(1000)
        n_trials.setValue(100)
        layout.addRow("Number of Trials:", n_trials)
        
        # Objective
        objective = QComboBox()
        objective.addItems(['sharpe', 'sortino', 'returns', 'calmar'])
        objective.setCurrentText('sharpe')
        layout.addRow("Objective:", objective)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            params = {
                'symbol': symbol_input.text(),
                'start_date': start_date.date().toString('yyyy-MM-dd'),
                'end_date': end_date.date().toString('yyyy-MM-dd'),
                'n_trials': n_trials.value(),
                'objective': objective.currentText()
            }
            self._run_optimization(params)
    
    def _run_optimization(self, params):
        """Execute optimization in background thread."""
        if not PYSIDE6_AVAILABLE:
            return
        
        from PySide6.QtCore import QThread
        
        class OptimizationThread(QThread):
            finished = Signal(dict)
            progress = Signal(int, int, str)
            
            def __init__(self, params, bot_app):
                super().__init__()
                self.params = params
                self.bot_app = bot_app
                self.optimizer_ref = None
            
            def run(self):
                try:
                    from trading_bot.data.stock_fetcher import StockDataFetcher
                    from trading_bot.data.feature_engineer import FeatureEngineer
                    from trading_bot.models.optimizer import XGBoostOptimizer
                    from trading_bot.models.optimization_monitor import OptimizationMonitor
                    from trading_bot.models.param_analyzer import ParameterAnalyzer
                    from trading_bot.data.preprocessor import DataPreprocessor
                    from sklearn.model_selection import train_test_split
                    import pandas as pd
                    
                    config = self.bot_app.config
                    logger = self.bot_app.get_logger()
                    
                    # Fetch data
                    fetcher = StockDataFetcher(config, logger)
                    data = fetcher.fetch_historical_data(
                        symbol=self.params['symbol'],
                        start_date=pd.to_datetime(self.params['start_date']),
                        end_date=pd.to_datetime(self.params['end_date']),
                        interval='1d'
                    )
                    
                    if data is None or len(data) == 0:
                        self.finished.emit({'error': 'No data fetched'})
                        return
                    
                    # Create features
                    feature_engineer = FeatureEngineer(config, logger)
                    features_df = feature_engineer.create_features(data)
                    
                    # Prepare data
                    preprocessor = DataPreprocessor(config, logger)
                    X, y = preprocessor.prepare_training_data(features_df, target_column='Close')
                    
                    valid_mask = ~(X.isna().any(axis=1) | y.isna())
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Create optimizer
                    optimizer = XGBoostOptimizer(
                        X_train, y_train, X_val, y_val,
                        config, logger, objective=self.params['objective']
                    )
                    
                    # Create monitor
                    monitor = OptimizationMonitor(config, logger)
                    monitor.start_time = datetime.now()
                    
                    # Store optimizer reference for GUI updates
                    self.optimizer_ref = optimizer
                    optimizer.monitor = monitor
                    
                    # Progress callback
                    def progress_callback(study, trial):
                        monitor._on_trial_complete(study, trial)
                        best_str = f"{study.best_value:.4f}" if study.best_value is not None else "N/A"
                        self.progress.emit(
                            monitor.completed_trials,
                            self.params['n_trials'],
                            f"Trial {trial.number}: Best = {best_str}"
                        )
                    
                    # Run optimization
                    best_params = optimizer.optimize(
                        n_trials=self.params['n_trials'],
                        callbacks=[progress_callback]
                    )
                    
                    # Analyze
                    analyzer = ParameterAnalyzer(optimizer.study, config, logger)
                    insights = analyzer.generate_insights()
                    
                    # Store optimizer reference for GUI
                    self.optimizer_ref = optimizer
                    
                    self.finished.emit({
                        'success': True,
                        'best_params': best_params,
                        'best_value': optimizer.best_value,
                        'insights': insights,
                        'optimizer': optimizer
                    })
                    
                except Exception as e:
                    self.finished.emit({'error': str(e)})
        
        # Create progress dialog with real-time plots
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtCore import QTimer
        
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Optimizing...")
        progress_dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(progress_dialog)
        
        # Status label
        status_label = QLabel("Optimization in progress. Please wait...")
        layout.addWidget(status_label)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(params['n_trials'])
        progress_bar.setValue(0)
        layout.addWidget(progress_bar)
        
        # Best value label
        best_label = QLabel("Best Value: N/A")
        layout.addWidget(best_label)
        
        # ETA label
        eta_label = QLabel("ETA: Calculating...")
        layout.addWidget(eta_label)
        
        # Web view for optimization history plot
        web_view = QWebEngineView()
        layout.addWidget(web_view)
        
        # Store references for updates
        progress_dialog.status_label = status_label
        progress_dialog.progress_bar = progress_bar
        progress_dialog.best_label = best_label
        progress_dialog.eta_label = eta_label
        progress_dialog.web_view = web_view
        progress_dialog.optimizer_ref = None
        
        progress_dialog.show()
        
        # Create and start thread
        self.optimization_thread = OptimizationThread(params, self.bot_app)
        self.optimization_thread.progress.connect(
            lambda current, total, message: self._update_optimization_progress(
                current, total, message, progress_dialog
            )
        )
        self.optimization_thread.finished.connect(
            lambda result: self._on_optimization_complete(result, progress_dialog)
        )
        
        # Store optimizer reference when available
        def store_optimizer_ref(result):
            if 'optimizer' in result and result['optimizer']:
                progress_dialog.optimizer_ref = result['optimizer']
        
        self.optimization_thread.finished.connect(store_optimizer_ref)
        self.optimization_thread.start()
        
        # Set up timer to update plots periodically
        update_timer = QTimer()
        update_timer.timeout.connect(lambda: self._update_optimization_plot(progress_dialog))
        update_timer.start(2000)  # Update every 2 seconds
        progress_dialog.update_timer = update_timer
    
    def _update_optimization_progress(self, current, total, message, progress_dialog=None):
        """Update progress bar during optimization."""
        if progress_dialog:
            progress_dialog.progress_bar.setValue(current)
            progress_dialog.progress_bar.setMaximum(total)
            progress_dialog.status_label.setText(f"Trial {current}/{total}: {message}")
            
            # Extract best value from message if available
            if "Best = " in message:
                try:
                    best_str = message.split("Best = ")[1].split()[0]
                    progress_dialog.best_label.setText(f"Best Value: {best_str}")
                except:
                    pass
        else:
            # Fallback to status label
            self.status_label.setText(f"Optimizing: {current}/{total} - {message}")
    
    def _update_optimization_plot(self, progress_dialog):
        """Update optimization history plot in real-time."""
        if not hasattr(progress_dialog, 'optimizer_ref') or progress_dialog.optimizer_ref is None:
            return
        
        try:
            optimizer = progress_dialog.optimizer_ref
            if optimizer.study and len(optimizer.study.trials) > 0:
                # Generate plot
                fig = optimizer.plot_optimization_history()
                html_content = fig.to_html()
                
                # Update web view
                progress_dialog.web_view.setHtml(html_content)
                
                # Update ETA
                from trading_bot.models.optimization_monitor import OptimizationMonitor
                if hasattr(optimizer, 'monitor') and optimizer.monitor:
                    monitor = optimizer.monitor
                    eta = monitor.estimate_time_remaining(
                        optimizer.study,
                        monitor.start_time,
                        progress_dialog.progress_bar.maximum()
                    )
                    progress_dialog.eta_label.setText(f"ETA: {eta}")
        except Exception as e:
            # Silently fail - plot updates are best-effort
            pass
    
    def _on_optimization_complete(self, results, progress_dialog):
        """Handle optimization completion."""
        # Stop update timer
        if hasattr(progress_dialog, 'update_timer'):
            progress_dialog.update_timer.stop()
        
        # Final plot update
        if 'optimizer' in results and results['optimizer']:
            try:
                fig = results['optimizer'].plot_optimization_history()
                html_content = fig.to_html()
                progress_dialog.web_view.setHtml(html_content)
            except:
                pass
        
        progress_dialog.close()
        
        if 'error' in results:
            QMessageBox.critical(
                self,
                "Optimization Error",
                f"Optimization failed:\n{results['error']}"
            )
            return
        
        if results.get('success'):
            # Show results dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Optimization Complete")
            msg.setText(
                f"Optimization completed!\n\n"
                f"Best Value: {results['best_value']:.4f}\n\n"
                f"Best Parameters:\n" +
                "\n".join([f"  {k}: {v}" for k, v in results['best_params'].items()])
            )
            
            msg.setDetailedText(results.get('insights', ''))
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Save)
            
            reply = msg.exec()
            
            if reply == QMessageBox.Save:
                # Offer to train with best params
                reply2 = QMessageBox.question(
                    self,
                    "Train with Best Parameters?",
                    "Would you like to train a model with these optimized parameters?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply2 == QMessageBox.Yes:
                    # Update config and trigger training
                    # This would require additional implementation
                    QMessageBox.information(
                        self,
                        "Info",
                        "Training with optimized parameters will be implemented in the training notebook."
                    )
    
    def get_loaded_predictor(self):
        """Get the currently loaded predictor."""
        if self.predictor and self.predictor.is_model_loaded():
            return self.predictor
        return None

