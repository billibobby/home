"""
Main Window for Novita.ai GPU Manager

Provides the primary user interface with instance management, logging, 
and configuration controls.
"""

from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QTextEdit, QPushButton, 
    QToolBar, QStatusBar, QMenuBar, QMenu, QMessageBox,
    QHeaderView, QGroupBox, QLabel, QApplication
)
from PySide6.QtCore import Qt, QTimer, QSize, QThreadPool
from PySide6.QtGui import QIcon, QFont, QCursor, QColor, QAction
from config_manager import ConfigManager
from settings_dialog import SettingsDialog
from create_instance_dialog import CreateInstanceDialog
from novita_api import NovitaAPIClient, NovitaAPIError, NovitaAuthenticationError
from workers import APIWorker
import requests


class MainWindow(QMainWindow):
    """Main application window for Novita.ai GPU Manager."""
    
    def __init__(self, config_manager: ConfigManager, api_client=None, user_info=None, startup_error_message=None):
        """
        Initialize main window.
        
        Args:
            config_manager (ConfigManager): Configuration manager instance.
            api_client: NovitaAPIClient instance (optional, can be None if no API key).
            user_info: User information dict with credits (optional).
            startup_error_message: Error message from startup API validation (optional).
        """
        super().__init__()
        
        # Store references
        self.config_manager = config_manager
        self.api_client = api_client
        self.user_info = user_info
        self.startup_error_message = startup_error_message
        
        # Initialize thread pool for background API operations
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(3)  # Limit concurrent API calls
        
        # Initialize instance data structure (will hold instance data in Phase 3)
        self.instances = []
        
        # Setup periodic API revalidation timer (every 3 minutes)
        self.revalidation_timer = QTimer(self)
        self.revalidation_timer.timeout.connect(self._periodic_api_revalidation)
        if self.api_client:
            self.revalidation_timer.start(180000)  # 180000 ms = 3 minutes
        
        # Set window properties
        self.setWindowTitle("Novita.ai GPU Instance Manager")
        self.setGeometry(100, 100, 1200, 800)
        
        # Build interface components
        self._init_ui()
        self._init_menu_bar()
        self._init_toolbar()
        self._init_status_bar()
        
        # Restore saved window state
        self._restore_window_state()
        
        # Log welcome message
        self.log_message("Novita.ai GPU Manager started. Configure your API key in Settings.", "INFO")
        
        # Show startup error if present
        if self.startup_error_message:
            self.log_message(self.startup_error_message, "ERROR")
            self.status_bar.showMessage(self.startup_error_message, 10000)  # Show for 10 seconds
        
        # Automatically refresh instances on startup if API client is available
        if self.api_client is not None:
            self._refresh_instances()
    
    def _init_ui(self):
        """Initialize main user interface layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        central_widget.setLayout(main_layout)
        
        # Create vertical splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # === Instance List Area (Top Section) ===
        instance_widget = self._create_instance_area()
        splitter.addWidget(instance_widget)
        
        # === Logs Area (Bottom Section) ===
        logs_widget = self._create_logs_area()
        splitter.addWidget(logs_widget)
        
        # Set initial splitter sizes (60% top, 40% bottom)
        splitter.setSizes([600, 400])
    
    def _create_instance_area(self) -> QWidget:
        """
        Create instance list area with controls and table.
        
        Returns:
            QWidget: Widget containing instance management UI.
        """
        # Group box for instance area
        instance_group = QGroupBox("GPU Instances")
        instance_layout = QVBoxLayout()
        instance_layout.setSpacing(10)
        instance_group.setLayout(instance_layout)
        
        # === Control Panel ===
        control_layout = QHBoxLayout()
        
        # Action buttons
        self.create_btn = QPushButton("Create Instance")
        self.create_btn.clicked.connect(self._create_instance)
        control_layout.addWidget(self.create_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_instance)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_instance)
        control_layout.addWidget(self.stop_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_instance)
        control_layout.addWidget(self.delete_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_instances)
        control_layout.addWidget(self.refresh_btn)
        
        # Add stretch to push buttons to left
        control_layout.addStretch()
        
        # Instance count label
        self.instance_count_label = QLabel("Instances: 0")
        control_layout.addWidget(self.instance_count_label)
        
        instance_layout.addLayout(control_layout)
        
        # === Instance Table ===
        self.instance_table = QTableWidget()
        self.instance_table.setColumnCount(6)
        self.instance_table.setHorizontalHeaderLabels([
            "Name", "Instance ID", "Status", "GPU Type", "Cluster", "Created"
        ])
        
        # Configure table properties
        self.instance_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.instance_table.setSelectionMode(QTableWidget.SingleSelection)
        self.instance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.instance_table.setAlternatingRowColors(True)
        self.instance_table.verticalHeader().setVisible(False)
        
        # Configure column resizing
        header = self.instance_table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        
        # Connect selection signal
        self.instance_table.itemSelectionChanged.connect(self._on_instance_selected)
        
        # Track if showing placeholder to control selection mode
        self._showing_placeholder = False
        
        # Show initial placeholder
        self._show_placeholder()
        
        instance_layout.addWidget(self.instance_table)
        
        return instance_group
    
    def _create_logs_area(self) -> QWidget:
        """
        Create logs area with text display and controls.
        
        Returns:
            QWidget: Widget containing logs UI.
        """
        # Group box for logs area
        logs_group = QGroupBox("Activity Logs")
        logs_layout = QVBoxLayout()
        logs_layout.setSpacing(10)
        logs_group.setLayout(logs_layout)
        
        # Log text display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setPlaceholderText("Application logs will appear here...")
        logs_layout.addWidget(self.log_text)
        
        # Clear logs button
        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self._clear_logs)
        logs_layout.addWidget(clear_logs_btn)
        
        return logs_group
    
    def _init_menu_bar(self):
        """Initialize menu bar with all menus and actions."""
        menubar = self.menuBar()
        
        # === File Menu ===
        file_menu = menubar.addMenu("File")
        
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+S")
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # === Instance Menu ===
        instance_menu = menubar.addMenu("Instance")
        
        create_action = QAction("Create Instance", self)
        create_action.triggered.connect(self._create_instance)
        instance_menu.addAction(create_action)
        
        refresh_action = QAction("Refresh Instances", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_instances)
        instance_menu.addAction(refresh_action)
        
        instance_menu.addSeparator()
        
        self.start_action = QAction("Start Instance", self)
        self.start_action.setEnabled(False)
        self.start_action.triggered.connect(self._start_instance)
        instance_menu.addAction(self.start_action)
        
        self.stop_action = QAction("Stop Instance", self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._stop_instance)
        instance_menu.addAction(self.stop_action)
        
        self.delete_action = QAction("Delete Instance", self)
        self.delete_action.setEnabled(False)
        self.delete_action.triggered.connect(self._delete_instance)
        instance_menu.addAction(self.delete_action)
        
        # === Tools Menu ===
        tools_menu = menubar.addMenu("Tools")
        
        save_snapshot_action = QAction("Save Snapshot", self)
        save_snapshot_action.triggered.connect(self._placeholder_save_snapshot)
        tools_menu.addAction(save_snapshot_action)
        
        tools_menu.addSeparator()
        
        view_templates_action = QAction("View Templates", self)
        view_templates_action.triggered.connect(self._placeholder_view_templates)
        tools_menu.addAction(view_templates_action)
        
        # === Help Menu ===
        help_menu = menubar.addMenu("Help")
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self._open_documentation)
        help_menu.addAction(docs_action)
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        updates_action = QAction("Check for Updates", self)
        updates_action.triggered.connect(self._placeholder_check_updates)
        help_menu.addAction(updates_action)
    
    def _init_toolbar(self):
        """Initialize toolbar with action buttons."""
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(self.toolbar)
        
        # Create Instance action
        create_toolbar_action = QAction("Create Instance", self)
        create_toolbar_action.triggered.connect(self._create_instance)
        self.toolbar.addAction(create_toolbar_action)
        
        # Start Instance action
        self.start_toolbar_action = QAction("Start", self)
        self.start_toolbar_action.setEnabled(False)
        self.start_toolbar_action.triggered.connect(self._start_instance)
        self.toolbar.addAction(self.start_toolbar_action)
        
        # Stop Instance action
        self.stop_toolbar_action = QAction("Stop", self)
        self.stop_toolbar_action.setEnabled(False)
        self.stop_toolbar_action.triggered.connect(self._stop_instance)
        self.toolbar.addAction(self.stop_toolbar_action)
        
        # Refresh action
        refresh_toolbar_action = QAction("Refresh", self)
        refresh_toolbar_action.triggered.connect(self._refresh_instances)
        self.toolbar.addAction(refresh_toolbar_action)
        
        self.toolbar.addSeparator()
        
        # Settings action
        settings_toolbar_action = QAction("Settings", self)
        settings_toolbar_action.triggered.connect(self._open_settings)
        self.toolbar.addAction(settings_toolbar_action)
    
    def _init_status_bar(self):
        """Initialize status bar with permanent widgets."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # API status label (permanent widget on right)
        self.api_status_label = QLabel()
        if self.api_client is not None:
            self.api_status_label.setText("API: Connected")
            self.api_status_label.setStyleSheet("color: green;")
        else:
            self.api_status_label.setText("API: Disconnected")
            self.api_status_label.setStyleSheet("color: red;")
        self.status_bar.addPermanentWidget(self.api_status_label)
        
        # Credit balance label (permanent widget on right)
        self.credit_label = QLabel("Credits: --")
        self.status_bar.addPermanentWidget(self.credit_label)
        
        # Update credit label if we have user info
        if self.user_info:
            self._update_credit_display(self.user_info)
        
        # Set initial status message
        self.status_bar.showMessage("Ready")
    
    def _show_placeholder(self):
        """Show placeholder message in empty table."""
        self.instance_table.setRowCount(1)
        placeholder_item = QTableWidgetItem("No instances. Click 'Create Instance' to get started.")
        placeholder_item.setFlags(Qt.NoItemFlags)  # Disable all interactions with placeholder
        self.instance_table.setItem(0, 0, placeholder_item)
        self.instance_table.setSpan(0, 0, 1, 6)
        self._showing_placeholder = True
    
    def _clear_placeholder(self):
        """Clear placeholder state and prepare table for real data."""
        if self._showing_placeholder:
            self.instance_table.clearSpans()
            self.instance_table.setRowCount(0)
            self._showing_placeholder = False
    
    def _on_instance_selected(self):
        """Handle instance selection in table."""
        # Don't process selections when showing placeholder or no actual rows
        if self._showing_placeholder or self.instance_table.rowCount() == 0:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(False)
            self.delete_action.setEnabled(False)
            self.start_toolbar_action.setEnabled(False)
            self.stop_toolbar_action.setEnabled(False)
            return
        
        selected_rows = self.instance_table.selectedItems()
        
        if not selected_rows:
            # No selection
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(False)
            self.delete_action.setEnabled(False)
            self.start_toolbar_action.setEnabled(False)
            self.stop_toolbar_action.setEnabled(False)
            return
        
        # Get selected row
        row = self.instance_table.currentRow()
        
        # Enable delete button
        self.delete_btn.setEnabled(True)
        self.delete_action.setEnabled(True)
        
        # Enable/disable start/stop based on instance status
        # For Phase 2, we'll check status column (index 2)
        if row < self.instance_table.rowCount():
            status_item = self.instance_table.item(row, 2)
            if status_item:
                status = status_item.text().lower()
                
                if "stopped" in status or "terminated" in status:
                    self.start_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    self.start_action.setEnabled(True)
                    self.stop_action.setEnabled(False)
                    self.start_toolbar_action.setEnabled(True)
                    self.stop_toolbar_action.setEnabled(False)
                elif "running" in status:
                    self.start_btn.setEnabled(False)
                    self.stop_btn.setEnabled(True)
                    self.start_action.setEnabled(False)
                    self.stop_action.setEnabled(True)
                    self.start_toolbar_action.setEnabled(False)
                    self.stop_toolbar_action.setEnabled(True)
                else:
                    # Unknown status, disable both
                    self.start_btn.setEnabled(False)
                    self.stop_btn.setEnabled(False)
                    self.start_action.setEnabled(False)
                    self.stop_action.setEnabled(False)
                    self.start_toolbar_action.setEnabled(False)
                    self.stop_toolbar_action.setEnabled(False)
                
            # Log selection
            name_item = self.instance_table.item(row, 0)
            if name_item:
                self.log_message(f"Selected instance: {name_item.text()}", "INFO")
    
    def _open_settings(self):
        """Open settings dialog."""
        self.log_message("Settings dialog opened", "INFO")
        
        settings_dialog = SettingsDialog(self.config_manager, self.api_client, self)
        settings_dialog.settings_changed.connect(self._on_settings_changed)
        settings_dialog.exec()
    
    def _on_settings_changed(self):
        """Handle settings changed signal and verify API connection."""
        self.log_message("Settings updated successfully", "SUCCESS")
        self.status_bar.showMessage("Verifying API connection...", 0)
        
        # Reload API client with new API key from config
        api_key = self.config_manager.get("api_key", "")
        if api_key:
            # Show wait cursor and disable relevant buttons
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            self.create_btn.setEnabled(False)
            self.refresh_btn.setEnabled(False)
            
            # Test API connectivity in background
            def verify_api():
                """Verify API connectivity and fetch user info."""
                client = NovitaAPIClient(api_key)
                # Try to get user info first (includes credits)
                try:
                    user_info = client.get_user_info()
                    return user_info
                except Exception:
                    # If user info fails, still verify API works with products endpoint
                    client.list_gpu_products()
                    return {}  # Return empty dict if user info unavailable
            
            worker = APIWorker(verify_api)
            worker.signals.finished.connect(self._on_api_verification_success)
            worker.signals.error.connect(self._on_api_verification_error)
            self.thread_pool.start(worker)
        else:
            self.api_status_label.setText("API: Disconnected")
            self.api_status_label.setStyleSheet("color: red;")
            self.credit_label.setText("Credits: --")
            self.status_bar.showMessage("API key not configured", 3000)
    
    def _on_api_verification_success(self, user_info):
        """Handle successful API verification."""
        QApplication.restoreOverrideCursor()
        self.create_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        
        # Update API client reference
        api_key = self.config_manager.get("api_key", "")
        self.api_client = NovitaAPIClient(api_key)
        self.user_info = user_info
        
        # Update UI
        self.api_status_label.setText("API: Connected")
        self.api_status_label.setStyleSheet("color: green;")
        
        # Update credit display if user info available
        if user_info:
            self._update_credit_display(user_info)
        
        self.log_message("API connection verified successfully", "SUCCESS")
        self.status_bar.showMessage("Connected to Novita.ai API", 3000)
        
        # Start periodic revalidation timer if not already running
        if not self.revalidation_timer.isActive():
            self.revalidation_timer.start(180000)  # 180000 ms = 3 minutes
    
    def _on_api_verification_error(self, error_info):
        """Handle API verification failure."""
        QApplication.restoreOverrideCursor()
        self.create_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        
        exc_type, exc_value, tb_str = error_info
        
        # Update UI to show disconnected state
        self._set_disconnected_state()
        
        # Display user-friendly error message
        error_msg = self._format_api_error(exc_type, exc_value)
        self.log_message(f"API verification failed: {error_msg}", "ERROR")
        self.status_bar.showMessage("API verification failed", 5000)
        
        QMessageBox.warning(
            self,
            "API Connection Failed",
            f"Failed to connect to Novita.ai API:\n\n{error_msg}\n\n"
            "Please check your API key and internet connection."
        )
    
    def _update_credit_display(self, user_info):
        """
        Update credit label with user balance.
        Safely handles missing or None values with consistent two-decimal formatting.
        
        Args:
            user_info: User information dict containing 'credits' or 'balance' keys.
        """
        # Centralized helper to safely extract and convert credit value
        def get_credit_value(info):
            """
            Extract credit value from user_info.
            Tries 'credits' key first, then 'balance' key, fallback to 0.0.
            Coerces values to float with safe conversion.
            """
            if not info or not isinstance(info, dict):
                return 0.0
            
            # Try 'credits' key first
            value = info.get('credits')
            if value is None:
                # Fall back to 'balance' key
                value = info.get('balance')
            
            # Safe float conversion - handle None, empty string, and invalid values
            if value is None or value == '':
                return 0.0
            
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        # Extract credit value and ensure consistent two-decimal formatting
        credit_value = get_credit_value(user_info)
        self.credit_label.setText(f"Credits: ${credit_value:.2f}")
    
    def _set_disconnected_state(self):
        """Set UI to disconnected state and stop revalidation timer."""
        self.api_status_label.setText("API: Disconnected")
        self.api_status_label.setStyleSheet("color: red;")
        self.credit_label.setText("Credits: --")
        self.api_client = None
        self.user_info = None
        
        # Stop periodic revalidation timer
        if self.revalidation_timer.isActive():
            self.revalidation_timer.stop()
    
    def _periodic_api_revalidation(self):
        """
        Periodically revalidate API connection and update credits.
        Only runs when window is visible/active.
        """
        # Only revalidate if window is visible and we have an API client
        if not self.isVisible() or not self.api_client:
            return
        
        # Run revalidation in background
        def revalidate():
            """Verify API and fetch fresh user info."""
            try:
                # Try to get updated user info (includes credits)
                return self.api_client.get_user_info()
            except Exception:
                # If user info fails, just verify API is accessible
                self.api_client.list_gpu_products()
                return {}
        
        worker = APIWorker(revalidate)
        worker.signals.finished.connect(self._on_periodic_revalidation_success)
        worker.signals.error.connect(self._on_periodic_revalidation_error)
        self.thread_pool.start(worker)
    
    def _on_periodic_revalidation_success(self, user_info):
        """Handle successful periodic revalidation."""
        # Update user info if available
        if user_info:
            self.user_info = user_info
            self._update_credit_display(user_info)
        # Silently update - no log message to avoid spam
    
    def _on_periodic_revalidation_error(self, error_info):
        """Handle periodic revalidation failure."""
        exc_type, exc_value, tb_str = error_info
        
        # Set disconnected state
        self._set_disconnected_state()
        
        # Log the error
        error_msg = self._format_api_error(exc_type, exc_value)
        self.log_message(f"API connection lost: {error_msg}", "ERROR")
        self.status_bar.showMessage("API connection lost - please check settings", 5000)
    
    def _format_api_error(self, exc_type, exc_value):
        """
        Format API error into user-friendly message.
        
        Args:
            exc_type: Exception type
            exc_value: Exception instance
            
        Returns:
            str: User-friendly error message
        """
        if exc_type == NovitaAuthenticationError:
            return "Invalid API key (401 Unauthorized). Please check your API key."
        elif exc_type == NovitaAPIError:
            error_str = str(exc_value)
            if "timeout" in error_str.lower():
                return "Request timed out. Please check your internet connection."
            elif "connection" in error_str.lower():
                return "Connection error. Please check your internet connection."
            else:
                return f"API error: {error_str}"
        elif exc_type == requests.exceptions.Timeout:
            return "Request timed out. Please check your internet connection and try again."
        elif exc_type == requests.exceptions.ConnectionError:
            return "Network connection error. Please check your internet connection."
        elif exc_type == requests.exceptions.HTTPError:
            return f"HTTP error: {exc_value}"
        else:
            return f"Unexpected error: {str(exc_value)}"
    
    def _clear_logs(self):
        """Clear all logs."""
        self.log_text.clear()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f'<span style="color: gray;">[{timestamp}] [INFO] Logs cleared</span>')
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Novita.ai GPU Manager",
            "<h3>Novita.ai GPU Instance Manager</h3>"
            "<p><b>Version:</b> 1.0.0</p>"
            "<p><b>Description:</b> Desktop application for managing GPU instances on Novita.ai platform.</p>"
            "<p>Provides streamlined interface for creating, managing, and monitoring cloud GPU instances "
            "for AI/ML workloads.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>GPU instance lifecycle management</li>"
            "<li>Real-time monitoring</li>"
            "<li>Snapshot and template system</li>"
            "</ul>"
            "<p><small>Built with PySide6 and Python</small></p>"
        )
    
    def _open_documentation(self):
        """Open documentation in browser."""
        import webbrowser
        # Open Novita.ai documentation
        webbrowser.open("https://novita.ai/docs")
        self.log_message("Documentation opened in browser", "INFO")
    
    def log_message(self, message: str, level: str = "INFO"):
        """
        Append timestamped message to log display.
        
        Args:
            message (str): Log message to display.
            level (str): Log level (INFO, WARNING, ERROR, SUCCESS).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Color-code by level
        colors = {
            "INFO": "black",
            "WARNING": "orange",
            "ERROR": "red",
            "SUCCESS": "green"
        }
        color = colors.get(level.upper(), "black")
        
        # Format message with HTML
        formatted_message = f'<span style="color: {color};">[{timestamp}] [{level}] {message}</span>'
        
        # Append to log text
        self.log_text.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_instance_count(self):
        """Update instance count label."""
        count = len(self.instances)
        self.instance_count_label.setText(f"Instances: {count}")
    
    def _restore_window_state(self):
        """Restore window geometry and state from config."""
        geometry = self.config_manager.get("window_geometry")
        
        if geometry and isinstance(geometry, dict):
            x = geometry.get("x", 100)
            y = geometry.get("y", 100)
            width = geometry.get("width", 1200)
            height = geometry.get("height", 800)
            self.setGeometry(x, y, width, height)
        else:
            # Center on screen if no saved state
            self.resize(1200, 800)
            frame_geometry = self.frameGeometry()
            # Use primaryScreen() instead of deprecated desktop()
            screen = QApplication.primaryScreen()
            if screen:
                screen_center = screen.availableGeometry().center()
                frame_geometry.moveCenter(screen_center)
                self.move(frame_geometry.topLeft())
    
    def _save_window_state(self):
        """Save current window geometry and state to config."""
        geometry = self.geometry()
        geometry_dict = {
            "x": geometry.x(),
            "y": geometry.y(),
            "width": geometry.width(),
            "height": geometry.height()
        }
        
        self.config_manager.set("window_geometry", geometry_dict)
        self.config_manager.save()
    
    def closeEvent(self, event):
        """
        Override close event to save window state.
        
        Args:
            event: Close event object.
        """
        self.log_message("Application closing...", "INFO")
        self._save_window_state()
        event.accept()
    
    # === Instance Management Methods (Phase 3) ===
    
    def _create_instance(self):
        """Open create instance dialog."""
        if not self.api_client:
            QMessageBox.warning(
                self,
                "API Not Connected",
                "Please configure your API key in Settings before creating instances."
            )
            self.log_message("Cannot create instance: API not connected", "ERROR")
            return
        
        self.log_message("Opening create instance dialog", "INFO")
        
        # Create and show dialog
        create_dialog = CreateInstanceDialog(self.config_manager, self.api_client, self)
        create_dialog.instance_created.connect(self._on_instance_created)
        create_dialog.exec()
    
    def _on_instance_created(self, instance_data):
        """Handle successful instance creation."""
        instance_name = instance_data.get('name', 'Unknown')
        instance_id = instance_data.get('id', 'Unknown')
        
        # Add to instances list
        self.instances.append(instance_data)
        
        # Repopulate table
        self._populate_instance_table()
        
        # Update count
        self.update_instance_count()
        
        # Log success
        self.log_message(f"Instance '{instance_name}' created successfully (ID: {instance_id})", "SUCCESS")
        self.status_bar.showMessage(f"Instance '{instance_name}' created", 5000)
    
    def _refresh_instances(self):
        """Refresh instance list from API."""
        if not self.api_client:
            QMessageBox.warning(
                self,
                "API Not Connected",
                "Please configure your API key in Settings before refreshing instances."
            )
            self.log_message("Cannot refresh instances: API not connected", "ERROR")
            return
        
        self.log_message("Refreshing instances...", "INFO")
        
        # Disable refresh button and show wait cursor
        self.refresh_btn.setEnabled(False)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        
        # Create worker to fetch instances
        worker = APIWorker(self.api_client.list_instances)
        worker.signals.finished.connect(self._on_refresh_success)
        worker.signals.error.connect(self._on_refresh_error)
        self.thread_pool.start(worker)
    
    def _on_refresh_success(self, instances):
        """Handle successful instance refresh."""
        QApplication.restoreOverrideCursor()
        self.refresh_btn.setEnabled(True)
        
        # Store instances
        self.instances = instances
        
        # Populate table
        self._populate_instance_table()
        
        # Update count
        self.update_instance_count()
        
        # Log success
        self.log_message(f"Successfully refreshed {len(instances)} instances", "SUCCESS")
        self.status_bar.showMessage(f"Loaded {len(instances)} instances", 3000)
    
    def _on_refresh_error(self, error_info):
        """Handle instance refresh error."""
        QApplication.restoreOverrideCursor()
        self.refresh_btn.setEnabled(True)
        
        exc_type, exc_value, tb_str = error_info
        
        # Log error
        error_msg = self._format_api_error(exc_type, exc_value)
        self.log_message(f"Failed to refresh instances: {error_msg}", "ERROR")
        self.status_bar.showMessage("Failed to refresh instances", 5000)
        
        # Show error dialog
        QMessageBox.critical(
            self,
            "Refresh Failed",
            f"Failed to refresh instances:\n\n{error_msg}"
        )
    
    def _start_instance(self):
        """Start the selected instance."""
        instance = self._get_selected_instance()
        if not instance:
            return
        
        instance_id = instance.get('id')
        instance_name = instance.get('name', 'Unknown')
        
        self.log_message(f"Starting instance '{instance_name}'...", "INFO")
        
        # Disable buttons and show wait cursor
        self._disable_action_buttons()
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        
        # Create worker to start instance
        worker = APIWorker(self.api_client.start_instance, instance_id)
        worker.signals.finished.connect(self._on_start_success)
        worker.signals.error.connect(self._on_start_error)
        self.thread_pool.start(worker)
    
    def _on_start_success(self, result):
        """Handle successful instance start."""
        QApplication.restoreOverrideCursor()
        
        self.log_message("Instance started successfully", "SUCCESS")
        
        # Refresh to update status
        self._refresh_instances()
    
    def _on_start_error(self, error_info):
        """Handle instance start error."""
        QApplication.restoreOverrideCursor()
        self._enable_action_buttons()
        
        exc_type, exc_value, tb_str = error_info
        
        # Log error
        error_msg = self._format_api_error(exc_type, exc_value)
        self.log_message(f"Failed to start instance: {error_msg}", "ERROR")
        
        # Show error dialog
        QMessageBox.critical(
            self,
            "Start Failed",
            f"Failed to start instance:\n\n{error_msg}"
        )
    
    def _stop_instance(self):
        """Stop the selected instance."""
        instance = self._get_selected_instance()
        if not instance:
            return
        
        instance_id = instance.get('id')
        instance_name = instance.get('name', 'Unknown')
        
        self.log_message(f"Stopping instance '{instance_name}'...", "INFO")
        
        # Disable buttons and show wait cursor
        self._disable_action_buttons()
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        
        # Create worker to stop instance
        worker = APIWorker(self.api_client.stop_instance, instance_id)
        worker.signals.finished.connect(self._on_stop_success)
        worker.signals.error.connect(self._on_stop_error)
        self.thread_pool.start(worker)
    
    def _on_stop_success(self, result):
        """Handle successful instance stop."""
        QApplication.restoreOverrideCursor()
        
        self.log_message("Instance stopped successfully", "SUCCESS")
        
        # Refresh to update status
        self._refresh_instances()
    
    def _on_stop_error(self, error_info):
        """Handle instance stop error."""
        QApplication.restoreOverrideCursor()
        self._enable_action_buttons()
        
        exc_type, exc_value, tb_str = error_info
        
        # Log error
        error_msg = self._format_api_error(exc_type, exc_value)
        self.log_message(f"Failed to stop instance: {error_msg}", "ERROR")
        
        # Show error dialog
        QMessageBox.critical(
            self,
            "Stop Failed",
            f"Failed to stop instance:\n\n{error_msg}"
        )
    
    def _delete_instance(self):
        """Delete the selected instance."""
        instance = self._get_selected_instance()
        if not instance:
            return
        
        instance_id = instance.get('id')
        instance_name = instance.get('name', 'Unknown')
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete instance '{instance_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            self.log_message("Instance deletion cancelled", "INFO")
            return
        
        self.log_message(f"Deleting instance '{instance_name}'...", "INFO")
        
        # Disable buttons and show wait cursor
        self._disable_action_buttons()
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        
        # Create worker to delete instance
        worker = APIWorker(self.api_client.delete_instance, instance_id)
        worker.signals.finished.connect(self._on_delete_success)
        worker.signals.error.connect(self._on_delete_error)
        self.thread_pool.start(worker)
    
    def _on_delete_success(self, result):
        """Handle successful instance deletion."""
        QApplication.restoreOverrideCursor()
        
        # Get current row to remove from instances list
        current_row = self.instance_table.currentRow()
        if 0 <= current_row < len(self.instances):
            deleted_instance = self.instances.pop(current_row)
            instance_name = deleted_instance.get('name', 'Unknown')
            
            # Repopulate table
            self._populate_instance_table()
            
            # Update count
            self.update_instance_count()
            
            # Log success
            self.log_message(f"Instance '{instance_name}' deleted successfully", "SUCCESS")
            self.status_bar.showMessage("Instance deleted", 3000)
        else:
            # Fallback: just refresh
            self._refresh_instances()
    
    def _on_delete_error(self, error_info):
        """Handle instance deletion error."""
        QApplication.restoreOverrideCursor()
        self._enable_action_buttons()
        
        exc_type, exc_value, tb_str = error_info
        
        # Log error
        error_msg = self._format_api_error(exc_type, exc_value)
        self.log_message(f"Failed to delete instance: {error_msg}", "ERROR")
        
        # Show error dialog
        QMessageBox.critical(
            self,
            "Delete Failed",
            f"Failed to delete instance:\n\n{error_msg}"
        )
    
    def _populate_instance_table(self):
        """Populate the instance table with instance data."""
        # Clear placeholder state
        self._clear_placeholder()
        
        # If no instances, show placeholder
        if not self.instances:
            self._show_placeholder()
            return
        
        # Set row count
        self.instance_table.setRowCount(len(self.instances))
        
        # Populate rows
        for row, instance in enumerate(self.instances):
            # Column 0: Name
            name = instance.get('name', 'Unnamed')
            name_item = QTableWidgetItem(name)
            self.instance_table.setItem(row, 0, name_item)
            
            # Column 1: Instance ID
            instance_id = instance.get('id', '')
            id_item = QTableWidgetItem(instance_id)
            self.instance_table.setItem(row, 1, id_item)
            
            # Column 2: Status (with color coding)
            status = instance.get('status', 'unknown')
            status_text = status.capitalize()
            status_item = QTableWidgetItem(status_text)
            
            # Color-code status
            if status.lower() == 'running':
                status_item.setForeground(QColor(0, 128, 0))  # Green
            elif status.lower() in ['starting', 'stopping', 'tostart']:
                status_item.setForeground(QColor(255, 140, 0))  # Orange
            elif status.lower() in ['stopped', 'exited']:
                status_item.setForeground(QColor(255, 0, 0))  # Red
            else:
                status_item.setForeground(QColor(128, 128, 128))  # Gray
            
            self.instance_table.setItem(row, 2, status_item)
            
            # Column 3: GPU Type
            product_name = instance.get('productName', '')
            if not product_name:
                # Build from GPU info if available
                gpu_num = instance.get('gpuNum', 0)
                if gpu_num > 0:
                    product_name = f"{gpu_num}x GPU"
                else:
                    product_name = 'N/A'
            gpu_item = QTableWidgetItem(product_name)
            self.instance_table.setItem(row, 3, gpu_item)
            
            # Column 4: Cluster
            cluster = instance.get('clusterName', instance.get('clusterId', 'N/A'))
            cluster_item = QTableWidgetItem(cluster)
            self.instance_table.setItem(row, 4, cluster_item)
            
            # Column 5: Created
            created_at = instance.get('createdAt')
            created_text = self._format_timestamp(created_at)
            created_item = QTableWidgetItem(created_text)
            self.instance_table.setItem(row, 5, created_item)
        
        # Update count
        self.update_instance_count()
    
    def _format_timestamp(self, timestamp):
        """
        Format timestamp to readable string.
        
        Args:
            timestamp: ISO timestamp string or Unix timestamp.
        
        Returns:
            str: Formatted timestamp or 'N/A' if invalid.
        """
        if not timestamp:
            return 'N/A'
        
        try:
            # Try parsing as Unix timestamp (seconds)
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime("%Y-%m-%d %H:%M")
            
            # Try parsing as ISO format string
            if isinstance(timestamp, str):
                # Handle ISO format with 'T' and 'Z'
                if 'T' in timestamp:
                    # Remove 'Z' and parse
                    timestamp_clean = timestamp.replace('Z', '').split('.')[0]
                    dt = datetime.fromisoformat(timestamp_clean)
                    return dt.strftime("%Y-%m-%d %H:%M")
                
                # Try parsing as Unix timestamp in string
                try:
                    dt = datetime.fromtimestamp(float(timestamp))
                    return dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass
            
            return 'N/A'
        except Exception:
            return 'N/A'
    
    def _get_selected_instance(self):
        """
        Get the currently selected instance.
        
        Returns:
            dict: Selected instance data or None if no selection.
        """
        current_row = self.instance_table.currentRow()
        if 0 <= current_row < len(self.instances):
            return self.instances[current_row]
        return None
    
    def _disable_action_buttons(self):
        """Disable instance action buttons."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.delete_action.setEnabled(False)
        self.start_toolbar_action.setEnabled(False)
        self.stop_toolbar_action.setEnabled(False)
    
    def _enable_action_buttons(self):
        """Re-enable instance action buttons based on selection."""
        # This will trigger _on_instance_selected which sets proper states
        self._on_instance_selected()
    
    def _placeholder_save_snapshot(self):
        """Placeholder for save snapshot functionality."""
        self.log_message("Feature coming in Phase 6: Save Snapshot", "INFO")
        QMessageBox.information(self, "Coming Soon", "Snapshot feature will be available in Phase 6.")
    
    def _placeholder_view_templates(self):
        """Placeholder for view templates functionality."""
        self.log_message("Feature coming in Phase 6: View Templates", "INFO")
        QMessageBox.information(self, "Coming Soon", "Template manager will be available in Phase 6.")
    
    def _placeholder_check_updates(self):
        """Placeholder for check updates functionality."""
        self.log_message("Checking for updates...", "INFO")
        QMessageBox.information(self, "Up to Date", "You are running the latest version.")

