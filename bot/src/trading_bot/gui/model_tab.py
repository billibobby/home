"""
Model Management Tab for Trading Bot GUI

Provides interface for managing XGBoost models.
"""

from typing import Optional
import webbrowser

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
        QTableWidgetItem, QLabel, QGroupBox, QTextEdit, QFileDialog,
        QMessageBox, QComboBox, QProgressBar
    )
    from PySide6.QtCore import Qt, QTimer, Signal
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
    
    def get_loaded_predictor(self):
        """Get the currently loaded predictor."""
        if self.predictor and self.predictor.is_model_loaded():
            return self.predictor
        return None

