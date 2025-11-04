"""
Configuration Manager for Novita.ai GPU Manager

Handles persistent storage of application settings in JSON format.
Stores configuration in user's home directory: ~/.novita_gpu_manager/config.json

Security Note: API keys are stored in plain text. This is acceptable for a local 
desktop application, but consider encryption for sensitive deployments.
"""

import os
import json
import logging
from pathlib import Path


class ConfigManager:
    """Manages application configuration with persistent JSON storage."""
    
    def __init__(self):
        """
        Initialize configuration manager.
        
        Creates config directory if it doesn't exist and loads existing configuration.
        Falls back to defaults if config file doesn't exist or is invalid.
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Define config file path in user's home directory
        home_dir = Path.home()
        self.config_dir = home_dir / ".novita_gpu_manager"
        self.config_file = self.config_dir / "config.json"
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Set file permissions to user-only (Unix-like systems only)
        # Skip on Windows to avoid noisy warnings
        if os.name != 'nt':
            try:
                if not self.config_file.exists():
                    self.config_file.touch(mode=0o600)
            except Exception as e:
                self.logger.warning(f"Could not set file permissions: {e}")
        
        # Track if we've already warned about permissions on this session
        self._permission_warning_logged = False
        
        # Track if we've logged Windows permission info
        self._windows_permission_info_logged = False
        
        # Default configuration
        self.default_config = {
            "config_version": 1,  # Schema version for future migrations
            "api_key": "",
            "default_gpu_product_id": "",
            "default_cluster_id": "",
            "docker_image_url": "ghcr.io/ai-dock/comfyui:latest-cuda",
            "window_geometry": None,
            "window_state": None
        }
        
        # Initialize config with defaults
        self.config = self.default_config.copy()
        
        # Load existing configuration
        self.load()
    
    def load(self) -> dict:
        """
        Load configuration from JSON file.
        
        Merges loaded config with defaults to ensure all keys exist.
        Handles file not found and JSON decode errors gracefully.
        
        Returns:
            dict: Loaded configuration dictionary with defaults for missing keys.
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    
                # Validate that loaded config is a dictionary
                if not isinstance(loaded_config, dict):
                    self.logger.warning("Invalid config file format. Using defaults.")
                    self.config = self.default_config.copy()
                    return self.config
                
                # Merge with defaults (defaults for missing keys)
                self.config = self.default_config.copy()
                self.config.update(loaded_config)
                
                # Perform schema migration if needed
                self._migrate_config()
                
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.logger.info("No config file found. Using defaults.")
                self.config = self.default_config.copy()
                
        except FileNotFoundError:
            self.logger.info("Config file not found. Using defaults.")
            self.config = self.default_config.copy()
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse config file: {e}. Using defaults.")
            self.config = self.default_config.copy()
            
        except Exception as e:
            self.logger.error(f"Unexpected error loading config: {e}. Using defaults.")
            self.config = self.default_config.copy()
        
        return self.config
    
    def save(self, config: dict = None) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config (dict, optional): Configuration dictionary to save. 
                                    If None, uses current self.config.
        
        Returns:
            bool: True if save successful, False otherwise.
        """
        try:
            # Use provided config or current config
            if config is not None:
                # Merge provided config with existing config
                self.config.update(config)
            
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Write JSON to file with indentation for readability
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            
            # Set file permissions to user-only (Unix-like systems only)
            # Skip on Windows to avoid noisy warnings
            if os.name != 'nt':
                try:
                    os.chmod(self.config_file, 0o600)
                except Exception as e:
                    # Only log warning once per session to avoid spam
                    if not self._permission_warning_logged:
                        self.logger.warning(f"Could not set file permissions: {e}")
                        self._permission_warning_logged = True
            else:
                # On Windows, log info about permission handling once per session
                if not self._windows_permission_info_logged:
                    self.logger.info(
                        "File permission hardening is skipped on Windows. "
                        "Please ensure your Windows user account is properly secured. "
                        "Consider using Windows file encryption (EFS) or BitLocker for additional protection."
                    )
                    self._windows_permission_info_logged = True
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except IOError as e:
            self.logger.error(f"Failed to save config file: {e}")
            return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error saving config: {e}")
            return False
    
    def get(self, key: str, default=None):
        """
        Retrieve specific configuration value by key.
        
        Args:
            key (str): Configuration key to retrieve.
            default: Default value to return if key doesn't exist.
        
        Returns:
            any: Configuration value or default if key not found.
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value) -> None:
        """
        Update specific configuration value by key.
        
        Note: Does NOT auto-save. Call save() explicitly to persist changes.
        
        Args:
            key (str): Configuration key to update.
            value: New value for the key.
        """
        self.config[key] = value
    
    def get_all(self) -> dict:
        """
        Get complete configuration dictionary.
        
        Returns:
            dict: Complete configuration dictionary.
        """
        return self.config.copy()
    
    def update(self, config_dict: dict) -> None:
        """
        Batch update multiple configuration values.
        
        Note: Does NOT auto-save. Call save() explicitly to persist changes.
        
        Args:
            config_dict (dict): Dictionary of key-value pairs to update.
        """
        self.config.update(config_dict)
    
    def _migrate_config(self) -> None:
        """
        Migrate configuration to current schema version.
        
        Handles backward compatibility by applying incremental migrations.
        """
        current_version = self.config.get("config_version", 0)
        target_version = self.default_config["config_version"]
        
        if current_version < target_version:
            self.logger.info(f"Migrating config from version {current_version} to {target_version}")
            
            # Apply migrations incrementally
            # Migration from version 0 to 1: Add config_version field
            if current_version < 1:
                self.config["config_version"] = 1
                self.logger.info("Applied migration: Added config_version field")
            
            # Future migrations go here:
            # if current_version < 2:
            #     # Migration logic for v1 -> v2
            #     self.config["new_field"] = "default_value"
            #     self.config["config_version"] = 2
            
            # Save migrated config
            if current_version != target_version:
                self.save()
                self.logger.info(f"Config migration complete: v{current_version} -> v{target_version}")

