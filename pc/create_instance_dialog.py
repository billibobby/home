"""
Create Instance Dialog for Novita.ai GPU Manager

Provides user interface for creating new GPU instances with form validation
and API-populated dropdowns for GPU products and clusters.
"""

import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, 
    QComboBox, QPushButton, QLabel, QDialogButtonBox, QGroupBox, 
    QMessageBox, QListWidget, QInputDialog, QApplication
)
from PySide6.QtCore import Qt, Signal, QThreadPool
from PySide6.QtGui import QCursor
from config_manager import ConfigManager
from novita_api import NovitaAPIClient
from workers import APIWorker


class CreateInstanceDialog(QDialog):
    """Dialog for creating new GPU instances."""
    
    # Custom signal emitted when instance is created successfully
    instance_created = Signal(dict)
    
    def __init__(self, config_manager: ConfigManager, api_client: NovitaAPIClient, parent=None):
        """
        Initialize create instance dialog.
        
        Args:
            config_manager (ConfigManager): Configuration manager instance.
            api_client (NovitaAPIClient): API client for Novita.ai.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        
        # Store references
        self.config_manager = config_manager
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize thread pool for background operations
        self.thread_pool = QThreadPool()
        
        # Store environment variables
        self.env_vars = []
        
        # Configure dialog
        self.setWindowTitle("Create GPU Instance - Novita.ai")
        self.setModal(True)
        self.resize(600, 500)
        
        # Build interface
        self._init_ui()
        
        # Load GPU products and clusters in background
        self._load_gpu_products()
        self._load_clusters()
    
    def _init_ui(self):
        """Initialize user interface layout and widgets."""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # === Instance Details Section ===
        details_group = QGroupBox("Instance Details")
        details_layout = QFormLayout()
        details_layout.setSpacing(10)
        
        # Instance Name field
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("my-gpu-instance")
        details_layout.addRow("Instance Name:", self.name_input)
        
        name_help = QLabel("<small>Optional - will auto-generate if left empty</small>")
        name_help.setStyleSheet("color: gray;")
        details_layout.addRow("", name_help)
        
        # Docker Image URL field
        self.image_input = QLineEdit()
        default_image = self.config_manager.get('docker_image_url', 'ghcr.io/ai-dock/comfyui:latest-cuda')
        self.image_input.setText(default_image)
        details_layout.addRow("Docker Image URL:", self.image_input)
        
        image_help = QLabel("<small>Docker image to run on the instance</small>")
        image_help.setStyleSheet("color: gray;")
        details_layout.addRow("", image_help)
        
        details_group.setLayout(details_layout)
        main_layout.addWidget(details_group)
        
        # === GPU Configuration Section ===
        gpu_group = QGroupBox("GPU Configuration")
        gpu_layout = QFormLayout()
        gpu_layout.setSpacing(10)
        
        # GPU Product dropdown
        self.gpu_product_combo = QComboBox()
        self.gpu_product_combo.addItem("Loading GPU products...")
        self.gpu_product_combo.setEnabled(False)
        self.gpu_product_combo.currentIndexChanged.connect(self._on_gpu_product_changed)
        gpu_layout.addRow("GPU Product:", self.gpu_product_combo)
        
        # Cluster dropdown
        self.cluster_combo = QComboBox()
        self.cluster_combo.addItem("Auto-select", None)  # First option for auto-select
        self.cluster_combo.addItem("Loading clusters...")
        self.cluster_combo.setEnabled(False)
        gpu_layout.addRow("Cluster:", self.cluster_combo)
        
        gpu_group.setLayout(gpu_layout)
        main_layout.addWidget(gpu_group)
        
        # === Advanced Options Section ===
        advanced_group = QGroupBox("Advanced Options (Optional)")
        advanced_layout = QFormLayout()
        advanced_layout.setSpacing(10)
        
        # Ports field
        self.ports_input = QLineEdit()
        self.ports_input.setPlaceholderText("8080/http, 6006/tcp")
        advanced_layout.addRow("Ports:", self.ports_input)
        
        ports_help = QLabel("<small>Port mappings (e.g., 8080/http, 6006/tcp)</small>")
        ports_help.setStyleSheet("color: gray;")
        advanced_layout.addRow("", ports_help)
        
        # Command field
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("python app.py")
        advanced_layout.addRow("Command:", self.command_input)
        
        command_help = QLabel("<small>Override container startup command</small>")
        command_help.setStyleSheet("color: gray;")
        advanced_layout.addRow("", command_help)
        
        # Environment Variables section
        env_label = QLabel("Environment Variables:")
        advanced_layout.addRow(env_label)
        
        self.env_list = QListWidget()
        self.env_list.setMaximumHeight(100)
        advanced_layout.addRow(self.env_list)
        
        env_buttons_layout = QHBoxLayout()
        add_env_btn = QPushButton("Add Variable")
        add_env_btn.clicked.connect(self._add_environment_variable)
        env_buttons_layout.addWidget(add_env_btn)
        
        remove_env_btn = QPushButton("Remove Selected")
        remove_env_btn.clicked.connect(self._remove_environment_variable)
        env_buttons_layout.addWidget(remove_env_btn)
        env_buttons_layout.addStretch()
        
        advanced_layout.addRow(env_buttons_layout)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
        
        # === Button Box ===
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        
        # Rename OK button to "Create"
        create_button = self.button_box.button(QDialogButtonBox.Ok)
        create_button.setText("Create")
        create_button.setDefault(True)
        create_button.setEnabled(False)  # Disabled until GPU product is selected
        self.create_button = create_button
        
        self.button_box.accepted.connect(self._create_instance)
        self.button_box.rejected.connect(self.reject)
        
        main_layout.addWidget(self.button_box)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def _load_gpu_products(self):
        """Load GPU products from API in background."""
        if not self.api_client:
            self.gpu_product_combo.clear()
            self.gpu_product_combo.addItem("API client not available")
            return
        
        # Create worker to fetch products
        worker = APIWorker(self.api_client.list_gpu_products)
        worker.signals.finished.connect(self._on_gpu_products_loaded)
        worker.signals.error.connect(self._on_gpu_products_error)
        self.thread_pool.start(worker)
    
    def _on_gpu_products_loaded(self, products):
        """Handle successful GPU products loading."""
        self.gpu_product_combo.clear()
        self.gpu_product_combo.addItem("Select GPU product", None)
        
        for product in products:
            # Format display text
            name = product.get('name', 'Unknown')
            gpu_num = product.get('gpuNum', 0)
            cpu_per_gpu = product.get('cpuPerGpu', 0)
            memory_per_gpu = product.get('memoryPerGpu', 0)
            price = product.get('price', 0)
            
            display_text = f"{name} - {gpu_num}x GPU, {cpu_per_gpu} CPU, {memory_per_gpu}GB RAM - ${price}/hr"
            
            # Add item with product data stored
            self.gpu_product_combo.addItem(display_text, product)
        
        self.gpu_product_combo.setEnabled(True)
        self.logger.info(f"Loaded {len(products)} GPU products")
    
    def _on_gpu_products_error(self, error_info):
        """Handle GPU products loading error."""
        exc_type, exc_value, tb_str = error_info
        self.gpu_product_combo.clear()
        self.gpu_product_combo.addItem("Failed to load GPU products")
        self.gpu_product_combo.setEnabled(False)
        self.create_button.setEnabled(False)
        
        self.logger.warning(f"Failed to load GPU products: {exc_value}")
        QMessageBox.warning(
            self,
            "Error Loading Products",
            f"Failed to load GPU products:\n\n{exc_value}\n\n"
            "Please check your API connection and try again."
        )
    
    def _load_clusters(self):
        """Load clusters from API in background."""
        if not self.api_client:
            self.cluster_combo.clear()
            self.cluster_combo.addItem("Auto-select", None)
            self.cluster_combo.addItem("API client not available")
            return
        
        # Create worker to fetch clusters
        worker = APIWorker(self.api_client.list_clusters)
        worker.signals.finished.connect(self._on_clusters_loaded)
        worker.signals.error.connect(self._on_clusters_error)
        self.thread_pool.start(worker)
    
    def _on_clusters_loaded(self, clusters):
        """Handle successful clusters loading."""
        # Clear all except the first "Auto-select" item
        while self.cluster_combo.count() > 1:
            self.cluster_combo.removeItem(1)
        
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            cluster_id = cluster.get('id', '')
            
            display_text = f"{cluster_name} ({cluster_id})"
            
            # Add item with cluster data stored
            self.cluster_combo.addItem(display_text, cluster)
        
        self.cluster_combo.setEnabled(True)
        self.logger.info(f"Loaded {len(clusters)} clusters")
    
    def _on_clusters_error(self, error_info):
        """Handle clusters loading error."""
        exc_type, exc_value, tb_str = error_info
        # Keep auto-select option, add error message
        while self.cluster_combo.count() > 1:
            self.cluster_combo.removeItem(1)
        self.cluster_combo.addItem("Failed to load clusters")
        self.cluster_combo.setEnabled(True)  # Still allow auto-select
        
        self.logger.warning(f"Failed to load clusters: {exc_value}")
    
    def _on_gpu_product_changed(self, index):
        """Handle GPU product selection change."""
        # Enable Create button if a valid product is selected (index > 0)
        product_data = self.gpu_product_combo.itemData(index)
        self.create_button.setEnabled(product_data is not None)
    
    def _add_environment_variable(self):
        """Add an environment variable."""
        # Ask for key
        key, ok = QInputDialog.getText(
            self,
            "Add Environment Variable",
            "Variable Name:"
        )
        
        if not ok or not key.strip():
            return
        
        key = key.strip()
        
        # Ask for value
        value, ok = QInputDialog.getText(
            self,
            "Add Environment Variable",
            f"Value for '{key}':"
        )
        
        if not ok:
            return
        
        # Add to list
        self.env_vars.append({"key": key, "value": value})
        self.env_list.addItem(f"{key}={value}")
        
        self.logger.info(f"Added environment variable: {key}")
    
    def _remove_environment_variable(self):
        """Remove selected environment variable."""
        current_row = self.env_list.currentRow()
        if current_row >= 0:
            self.env_list.takeItem(current_row)
            del self.env_vars[current_row]
            self.logger.info("Removed environment variable")
    
    def _validate_ports_format(self, ports_str: str) -> str:
        """
        Validate ports format string.
        
        Args:
            ports_str: Ports string to validate (e.g., "8080/http, 6006/tcp").
        
        Returns:
            str: Error message if invalid, empty string if valid.
        """
        if not ports_str or not ports_str.strip():
            return ""
        
        # Split by comma and validate each port spec
        for port_spec in ports_str.split(','):
            port_spec = port_spec.strip()
            if not port_spec:
                continue
            
            # Expected format: "port/protocol" or just "port"
            if '/' in port_spec:
                parts = port_spec.split('/', 1)
                if len(parts) != 2:
                    return f"Invalid format: '{port_spec}'"
                
                port_str, protocol = parts
                port_str = port_str.strip()
                protocol = protocol.strip().lower()
                
                # Validate port number
                try:
                    port = int(port_str)
                    if port < 1 or port > 65535:
                        return f"Port number {port} out of range (1-65535)"
                except ValueError:
                    return f"Invalid port number: '{port_str}'"
                
                # Validate protocol
                if protocol not in ['tcp', 'http', 'https', 'udp']:
                    return f"Invalid protocol: '{protocol}' (use tcp, http, https, or udp)"
            else:
                # Just port number
                try:
                    port = int(port_spec)
                    if port < 1 or port > 65535:
                        return f"Port number {port} out of range (1-65535)"
                except ValueError:
                    return f"Invalid port number: '{port_spec}'"
        
        return ""  # Valid
    
    def _create_instance(self):
        """Create instance with validation."""
        # Validate inputs
        image_url = self.image_input.text().strip()
        if not image_url:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Docker image URL cannot be empty."
            )
            self.image_input.setFocus()
            return
        
        # Get selected GPU product
        gpu_index = self.gpu_product_combo.currentIndex()
        product_data = self.gpu_product_combo.itemData(gpu_index)
        
        if not product_data:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please select a GPU product."
            )
            self.gpu_product_combo.setFocus()
            return
        
        product_id = product_data.get('id')
        if not product_id:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Invalid GPU product selected."
            )
            return
        
        # Build parameters
        name = self.name_input.text().strip() or None
        
        # Get cluster ID (None if auto-select is chosen)
        cluster_index = self.cluster_combo.currentIndex()
        cluster_data = self.cluster_combo.itemData(cluster_index)
        cluster_id = cluster_data.get('id') if cluster_data else None
        
        # Validate and normalize ports input
        ports_input = self.ports_input.text().strip()
        ports = None
        if ports_input:
            # Validate port format before sending to API
            validation_error = self._validate_ports_format(ports_input)
            if validation_error:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    f"Invalid ports format:\n\n{validation_error}\n\n"
                    "Expected format: port/protocol, port/protocol\n"
                    "Example: 8080/http, 6006/tcp"
                )
                self.ports_input.setFocus()
                return
            ports = ports_input  # Pass as string; API client will parse it
        
        command = self.command_input.text().strip() or None
        envs = self.env_vars if self.env_vars else None
        
        # Show wait cursor and disable buttons
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.button_box.setEnabled(False)
        
        # Create instance in background
        def create():
            return self.api_client.create_instance(
                product_id=product_id,
                image_url=image_url,
                name=name,
                cluster_id=cluster_id,
                ports=ports,
                envs=envs,
                command=command
            )
        
        worker = APIWorker(create)
        worker.signals.finished.connect(self._on_create_success)
        worker.signals.error.connect(self._on_create_error)
        self.thread_pool.start(worker)
    
    def _on_create_success(self, instance_data):
        """Handle successful instance creation."""
        QApplication.restoreOverrideCursor()
        self.button_box.setEnabled(True)
        
        instance_name = instance_data.get('name', 'Unknown')
        instance_id = instance_data.get('id', 'Unknown')
        
        self.logger.info(f"Successfully created instance: {instance_name} ({instance_id})")
        
        # Emit signal with instance data
        self.instance_created.emit(instance_data)
        
        # Show success message
        QMessageBox.information(
            self,
            "Success",
            f"Instance '{instance_name}' created successfully!\n\n"
            f"Instance ID: {instance_id}"
        )
        
        # Close dialog
        self.accept()
    
    def _on_create_error(self, error_info):
        """Handle instance creation error."""
        QApplication.restoreOverrideCursor()
        self.button_box.setEnabled(True)
        
        exc_type, exc_value, tb_str = error_info
        
        self.logger.error(f"Failed to create instance: {exc_value}")
        
        # Show error message
        QMessageBox.critical(
            self,
            "Creation Failed",
            f"Failed to create instance:\n\n{exc_value}\n\n"
            "Please check your configuration and try again."
        )

