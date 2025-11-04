"""
Settings Dialog for Novita.ai GPU Manager

Provides user interface for configuring API credentials, GPU preferences, 
and Docker settings with persistent storage.
"""

import re
import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, 
    QComboBox, QPushButton, QLabel, QDialogButtonBox, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThreadPool
from config_manager import ConfigManager
from workers import APIWorker


class SettingsDialog(QDialog):
    """Settings dialog for user configuration."""
    
    # Custom signal emitted when settings are saved successfully
    settings_changed = Signal()
    
    def __init__(self, config_manager: ConfigManager, api_client=None, parent=None):
        """
        Initialize settings dialog.
        
        Args:
            config_manager (ConfigManager): Configuration manager instance.
            api_client: NovitaAPIClient instance (optional, can be None if no API key).
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        
        # Store config manager reference
        self.config_manager = config_manager
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize thread pool for background operations
        self.thread_pool = QThreadPool()
        
        # Configure dialog
        self.setWindowTitle("Settings - Novita.ai GPU Manager")
        self.setModal(True)
        self.resize(500, 400)
        
        # Build interface and load settings
        self._init_ui()
        self._load_settings()
    
    def _init_ui(self):
        """Initialize user interface layout and widgets."""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # === Authentication Section ===
        auth_group = QGroupBox("Authentication")
        auth_layout = QFormLayout()
        auth_layout.setSpacing(10)
        
        # API Key field with show/hide toggle
        api_key_container = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter your Novita.ai API key")
        api_key_container.addWidget(self.api_key_input)
        
        # Show/Hide toggle button
        self.api_key_toggle_btn = QPushButton("Show")
        self.api_key_toggle_btn.setFixedWidth(60)
        self.api_key_toggle_btn.clicked.connect(self._toggle_api_key_visibility)
        api_key_container.addWidget(self.api_key_toggle_btn)
        
        auth_layout.addRow("API Key:", api_key_container)
        
        # Help label with link
        help_label = QLabel(
            '<small><a href="https://novita.ai/dashboard">Get your API key from novita.ai dashboard</a></small>'
        )
        help_label.setOpenExternalLinks(True)
        help_label.setStyleSheet("color: gray;")
        auth_layout.addRow("", help_label)
        
        auth_group.setLayout(auth_layout)
        main_layout.addWidget(auth_group)
        
        # === GPU Preferences Section ===
        gpu_group = QGroupBox("GPU Preferences")
        gpu_layout = QFormLayout()
        gpu_layout.setSpacing(10)
        
        # Default GPU Product combo box
        self.gpu_product_combo = QComboBox()
        self.gpu_product_combo.addItem("Select default GPU (optional)")
        gpu_layout.addRow("Default GPU Product:", self.gpu_product_combo)
        
        # Default Cluster combo box
        self.cluster_combo = QComboBox()
        self.cluster_combo.addItem("Select default cluster (optional)")
        gpu_layout.addRow("Default Cluster:", self.cluster_combo)
        
        # Load GPU products and clusters if API client is available
        if self.api_client:
            self._load_gpu_products()
            self._load_clusters()
        
        # Info label
        info_label = QLabel(
            "<small>These are optional defaults for quick instance creation</small>"
        )
        info_label.setStyleSheet("color: gray;")
        info_label.setWordWrap(True)
        gpu_layout.addRow("", info_label)
        
        gpu_group.setLayout(gpu_layout)
        main_layout.addWidget(gpu_group)
        
        # === Docker Configuration Section ===
        docker_group = QGroupBox("Docker Configuration")
        docker_layout = QFormLayout()
        docker_layout.setSpacing(10)
        
        # Docker Image URL
        self.docker_image_input = QLineEdit()
        self.docker_image_input.setPlaceholderText("Docker image URL for ComfyUI")
        docker_layout.addRow("Docker Image URL:", self.docker_image_input)
        
        # Help label
        docker_help_label = QLabel(
            "<small>Default ComfyUI image with CUDA support.<br>"
            "Can be changed per-instance during creation.</small>"
        )
        docker_help_label.setStyleSheet("color: gray;")
        docker_help_label.setWordWrap(True)
        docker_layout.addRow("", docker_help_label)
        
        docker_group.setLayout(docker_layout)
        main_layout.addWidget(docker_group)
        
        # Add spacing before buttons
        main_layout.addSpacing(10)
        
        # === Button Box ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._save_settings)
        button_box.rejected.connect(self.reject)
        
        # Make Save button default (highlighted)
        save_button = button_box.button(QDialogButtonBox.Save)
        save_button.setDefault(True)
        
        main_layout.addWidget(button_box)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def _load_settings(self):
        """Load current settings from config manager and populate fields."""
        config = self.config_manager.get_all()
        
        # Load API key
        api_key = config.get("api_key", "")
        self.api_key_input.setText(api_key)
        
        # Load Docker image URL
        docker_image = config.get("docker_image_url", "ghcr.io/ai-dock/comfyui:latest-cuda")
        self.docker_image_input.setText(docker_image)
        
        # Load GPU preferences (will be restored after API data loads)
        self.default_gpu_product_id = config.get("default_gpu_product_id", "")
        self.default_cluster_id = config.get("default_cluster_id", "")
    
    def _save_settings(self):
        """Validate and save settings to config manager."""
        try:
            # Get values from inputs
            api_key = self.api_key_input.text().strip()
            docker_image = self.docker_image_input.text().strip()
            
            # Validate API key (relaxed - only check non-empty)
            if not api_key:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "API key cannot be empty. Please enter a valid API key."
                )
                self.api_key_input.setFocus()
                return
            
            # Validate Docker image URL - basic validation
            if not docker_image:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Docker image URL cannot be empty."
                )
                self.docker_image_input.setFocus()
                return
            
            # Length check
            if len(docker_image) > 255:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Docker image URL is too long (max 255 characters)."
                )
                self.docker_image_input.setFocus()
                return
            
            # Best-effort Docker image reference validation using regex
            # This is a simplified validation that catches common errors but may not catch all invalid formats.
            # The backend/Docker daemon will perform final validation.
            # Matches: [registry/][namespace/]repository[:tag|@digest]
            # Examples: ubuntu, ubuntu:latest, myregistry.com/myimage:v1.0, ghcr.io/user/repo:tag
            docker_image_pattern = re.compile(
                r'^'
                r'(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*(?::[0-9]+)?/))?'  # Optional registry
                r'(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*'  # Optional namespace(s)
                r'[a-z0-9]+(?:[._-][a-z0-9]+)*'  # Repository name
                r'(?::[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127})?'  # Optional :tag
                r'(?:@[a-zA-Z0-9:]+)?'  # Optional @digest
                r'$',
                re.IGNORECASE
            )
            
            # Show warning but allow save if format looks suspicious
            if not docker_image_pattern.match(docker_image):
                reply = QMessageBox.question(
                    self,
                    "Docker Image Format Warning",
                    "Docker image URL format appears unusual:\n\n"
                    f"{docker_image}\n\n"
                    "Expected formats:\n"
                    "  • repository\n"
                    "  • repository:tag\n"
                    "  • registry/repository:tag\n"
                    "  • registry.com:port/namespace/repository:tag\n\n"
                    "Do you want to save it anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    self.docker_image_input.setFocus()
                    return
            
            # Create config dictionary with updated values
            config_dict = {
                "api_key": api_key,
                "docker_image_url": docker_image
            }
            
            # Add GPU preferences if not using placeholder values
            gpu_product_idx = self.gpu_product_combo.currentIndex()
            if gpu_product_idx > 0:  # Skip placeholder at index 0
                product_id = self.gpu_product_combo.itemData(gpu_product_idx, Qt.UserRole)
                if product_id:
                    config_dict["default_gpu_product_id"] = product_id
            
            cluster_idx = self.cluster_combo.currentIndex()
            if cluster_idx > 0:  # Skip placeholder at index 0
                cluster_id = self.cluster_combo.itemData(cluster_idx, Qt.UserRole)
                if cluster_id:
                    config_dict["default_cluster_id"] = cluster_id
            
            # Update and save configuration
            self.config_manager.update(config_dict)
            save_success = self.config_manager.save()
            
            if not save_success:
                raise Exception("Failed to save configuration file")
            
            # Emit signal to notify main window
            self.settings_changed.emit()
            
            # Show success message
            QMessageBox.information(
                self,
                "Success",
                "Settings saved successfully!"
            )
            
            # Close dialog
            self.accept()
            
        except Exception as e:
            # Handle save errors
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save settings: {str(e)}"
            )
    
    def _toggle_api_key_visibility(self):
        """Toggle API key field visibility between password and normal mode."""
        if self.api_key_input.echoMode() == QLineEdit.Password:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.api_key_toggle_btn.setText("Hide")
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.api_key_toggle_btn.setText("Show")
    
    def _load_gpu_products(self):
        """Load GPU products from API in background."""
        if not self.api_client:
            return
        
        # Show loading state
        self.gpu_product_combo.clear()
        self.gpu_product_combo.addItem("Loading GPU products...")
        self.gpu_product_combo.setEnabled(False)
        
        # Create worker to fetch products
        worker = APIWorker(self.api_client.list_gpu_products)
        worker.signals.finished.connect(self._on_gpu_products_loaded)
        worker.signals.error.connect(self._on_gpu_products_error)
        self.thread_pool.start(worker)
    
    def _on_gpu_products_loaded(self, products):
        """Handle successful GPU products loading."""
        self.gpu_product_combo.clear()
        self.gpu_product_combo.addItem("Select default GPU (optional)", None)
        
        for product in products:
            # Format display text
            name = product.get('name', 'Unknown')
            price = product.get('price', 0)
            display_text = f"{name} - ${price}/hr"
            
            # Add item with product ID stored
            product_id = product.get('id')
            self.gpu_product_combo.addItem(display_text, product_id)
        
        self.gpu_product_combo.setEnabled(True)
        self.logger.info(f"Loaded {len(products)} GPU products")
        
        # Restore previously selected product
        self._restore_gpu_selection()
    
    def _on_gpu_products_error(self, error_info):
        """Handle GPU products loading error."""
        exc_type, exc_value, tb_str = error_info
        self.gpu_product_combo.clear()
        self.gpu_product_combo.addItem("Failed to load GPU products")
        self.gpu_product_combo.setEnabled(True)
        
        self.logger.warning(f"Failed to load GPU products: {exc_value}")
    
    def _load_clusters(self):
        """Load clusters from API in background."""
        if not self.api_client:
            return
        
        # Show loading state (keep placeholder item)
        while self.cluster_combo.count() > 1:
            self.cluster_combo.removeItem(1)
        self.cluster_combo.addItem("Loading clusters...")
        self.cluster_combo.setEnabled(False)
        
        # Create worker to fetch clusters
        worker = APIWorker(self.api_client.list_clusters)
        worker.signals.finished.connect(self._on_clusters_loaded)
        worker.signals.error.connect(self._on_clusters_error)
        self.thread_pool.start(worker)
    
    def _on_clusters_loaded(self, clusters):
        """Handle successful clusters loading."""
        # Clear all except the first "Select default cluster (optional)" item
        while self.cluster_combo.count() > 1:
            self.cluster_combo.removeItem(1)
        
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            cluster_id = cluster.get('id', '')
            
            # Add item with cluster ID stored
            self.cluster_combo.addItem(cluster_name, cluster_id)
        
        self.cluster_combo.setEnabled(True)
        self.logger.info(f"Loaded {len(clusters)} clusters")
        
        # Restore previously selected cluster
        self._restore_cluster_selection()
    
    def _on_clusters_error(self, error_info):
        """Handle clusters loading error."""
        exc_type, exc_value, tb_str = error_info
        # Keep optional item, add error message
        while self.cluster_combo.count() > 1:
            self.cluster_combo.removeItem(1)
        self.cluster_combo.addItem("Failed to load clusters")
        self.cluster_combo.setEnabled(True)
        
        self.logger.warning(f"Failed to load clusters: {exc_value}")
    
    def _restore_gpu_selection(self):
        """Restore previously selected GPU product."""
        if not hasattr(self, 'default_gpu_product_id') or not self.default_gpu_product_id:
            return
        
        # Find item with matching product ID
        for i in range(self.gpu_product_combo.count()):
            item_data = self.gpu_product_combo.itemData(i, Qt.UserRole)
            if item_data == self.default_gpu_product_id:
                self.gpu_product_combo.setCurrentIndex(i)
                break
    
    def _restore_cluster_selection(self):
        """Restore previously selected cluster."""
        if not hasattr(self, 'default_cluster_id') or not self.default_cluster_id:
            return
        
        # Find item with matching cluster ID
        for i in range(self.cluster_combo.count()):
            item_data = self.cluster_combo.itemData(i, Qt.UserRole)
            if item_data == self.default_cluster_id:
                self.cluster_combo.setCurrentIndex(i)
                break

