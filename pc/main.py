"""
Novita.ai GPU Instance Manager - Application Entry Point

Desktop application for managing GPU instances on Novita.ai platform.
Launches PySide6 GUI with configuration management and API integration.
"""

import os
import sys
import logging
import re
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from config_manager import ConfigManager
from main_window import MainWindow
from novita_api import NovitaAPIClient, NovitaAPIError, NovitaAuthenticationError


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from log messages."""
    
    # Patterns that might indicate API keys or secrets
    SENSITIVE_PATTERNS = [
        (re.compile(r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{20,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
        (re.compile(r'(bearer\s+)([a-zA-Z0-9_\-\.]{20,})', re.IGNORECASE), r'\1***REDACTED***'),
        (re.compile(r'(authorization["\']?\s*[:=]\s*["\']?)([^"\'>\s]{20,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
        (re.compile(r'\bsk_[a-zA-Z0-9_\-]{30,}\b'), r'sk_***REDACTED***'),  # Common API key prefix
        (re.compile(r'(password["\']?\s*[:=]\s*["\']?)([^"\'>\s]+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
        (re.compile(r'(secret["\']?\s*[:=]\s*["\']?)([^"\'>\s]+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    ]
    
    def filter(self, record):
        """Redact sensitive data from log record."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            msg = record.msg
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


# Configure logging with sensitive data filter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add sensitive data filter to root logger
for handler in logging.root.handlers:
    handler.addFilter(SensitiveDataFilter())


def main():
    """Main application entry point."""
    try:
        # === Phase 1: Configuration Loading ===
        logger.info("Starting Novita.ai GPU Manager...")
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Try to get API key from config first, then environment
        api_key = config_manager.get('api_key', '').strip()
        if not api_key:
            api_key = os.getenv('NOVITA_API_KEY', '').strip()
        
        # Log API key status (without logging the actual key)
        if api_key:
            logger.info(f"API key found (length: {len(api_key)} characters)")
        else:
            logger.warning("No API key configured. Please set API key in Settings.")
        
        # === Phase 2: API Client Initialization ===
        api_client = None
        user_info = None
        startup_error_message = None  # Track any startup errors to show in UI
        
        if api_key:
            try:
                # Create API client and test connectivity
                api_client = NovitaAPIClient(api_key)
                
                # Try to get user info (includes credits)
                try:
                    user_info = api_client.get_user_info()
                    # Log success with credit balance
                    credits = user_info.get('credits', user_info.get('balance', 0))
                    logger.info(f"API connectivity test successful. Credits: ${credits:.2f}")
                except Exception as user_info_error:
                    # If user info fails, still verify API works with products endpoint
                    logger.warning(f"Could not fetch user info: {user_info_error}")
                    products = api_client.list_gpu_products()
                    logger.info(f"API connectivity test successful. Found {len(products)} GPU products available.")
                    user_info = None  # Set to None if we couldn't get it
                
            except NovitaAuthenticationError as e:
                error_detail = str(e)
                logger.error(f"API authentication failed: {error_detail}")
                logger.warning("Invalid API key. Please update in Settings.")
                startup_error_message = "Invalid API key - Authentication failed (401 Unauthorized)"
                api_client = None
                user_info = None
                
            except NovitaAPIError as e:
                error_detail = str(e)
                logger.error(f"API connectivity test failed: {error_detail}")
                logger.warning("API client initialization failed. You can still use the UI.")
                startup_error_message = f"API connection failed: {error_detail}"
                api_client = None
                user_info = None
                
            except Exception as e:
                error_detail = str(e)
                logger.error(f"Unexpected error during API initialization: {error_detail}")
                startup_error_message = f"Unexpected error: {error_detail}"
                api_client = None
                user_info = None
        else:
            logger.info("API client not initialized. Configure API key in Settings.")
        
        # === Phase 3: PySide6 Application Launch ===
        
        # Enable high DPI scaling (with compatibility checks for Qt6)
        if hasattr(Qt, "ApplicationAttribute") and hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "ApplicationAttribute") and hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("Novita.ai GPU Manager")
        app.setApplicationVersion("1.0.0")
        
        # Create and show main window, passing user_info and any startup errors
        window = MainWindow(config_manager, api_client, user_info, startup_error_message)
        window.show()
        
        logger.info("Application window displayed")
        
        # Start event loop (PySide6 uses exec() without underscore)
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        
        # Try to show error dialog if Qt is initialized
        try:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle("Fatal Error")
            error_box.setText("An unexpected error occurred:")
            error_box.setDetailedText(str(e))
            error_box.exec()
        except:
            # If Qt isn't initialized, just print to console
            print(f"FATAL ERROR: {e}", file=sys.stderr)
        
        sys.exit(1)


if __name__ == "__main__":
    main()

