"""
Path Resolution Utilities

Provides packaging-aware path resolution for frozen executables and development.
Handles resource paths for config files and writable app directories for logs/data.
"""

import sys
import os
from pathlib import Path
from typing import Union


def resolve_resource_path(relative_path: str) -> str:
    """
    Resolve path to a resource file that's bundled with the application.
    
    Works both in development (CWD-relative) and when frozen as an executable
    (PyInstaller sets sys._MEIPASS to the temporary extraction directory).
    
    Args:
        relative_path: Relative path to the resource (e.g., 'config/config.yaml')
        
    Returns:
        Absolute path to the resource file
        
    Examples:
        >>> resolve_resource_path('config/config.yaml')
        '/path/to/extracted/config/config.yaml'  # when frozen
        '/path/to/project/config/config.yaml'    # when in development
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as a frozen executable (PyInstaller)
        # sys._MEIPASS is the temporary directory where bundled files are extracted
        base_path = Path(sys._MEIPASS)
    else:
        # Running in development mode
        # Find project root by ascending from current file until we find a marker
        current_path = Path(__file__).resolve()
        base_path = None
        
        # Markers that indicate project root
        markers = ['config', 'src', 'setup.py', 'README.md']
        
        for parent in [current_path] + list(current_path.parents):
            # Check if this directory contains any of the markers
            if any((parent / marker).exists() for marker in markers):
                # Additional check: if 'config' marker, ensure it's a directory
                if (parent / 'config').exists() and (parent / 'config').is_dir():
                    base_path = parent
                    break
                # If 'src' marker, go up one level to project root
                elif (parent / 'src').exists() and (parent / 'src').is_dir():
                    base_path = parent
                    break
                # If setup.py or README.md, this is the root
                elif (parent / 'setup.py').exists() or (parent / 'README.md').exists():
                    base_path = parent
                    break
        
        # Fallback: use parent.parent.parent.parent if markers not found
        if base_path is None:
            base_path = current_path.parent.parent.parent.parent
    
    return str(base_path / relative_path)


def get_writable_app_dir(subdir: str = '') -> str:
    """
    Get a user-writable directory for the application.
    
    Returns platform-specific writable locations:
    - Windows: %APPDATA%/AITradingBot/<subdir>
    - macOS: ~/Library/Application Support/AITradingBot/<subdir>
    - Linux: ~/.local/share/ai-trading-bot/<subdir>
    
    Creates the directory if it doesn't exist.
    
    Args:
        subdir: Optional subdirectory name (e.g., 'logs', 'data', 'models')
        
    Returns:
        Absolute path to the writable directory
        
    Examples:
        >>> get_writable_app_dir('logs')
        'C:/Users/username/AppData/Roaming/AITradingBot/logs'  # Windows
        >>> get_writable_app_dir('logs')
        '/home/username/.local/share/ai-trading-bot/logs'       # Linux
        >>> get_writable_app_dir('logs')
        '/Users/username/Library/Application Support/AITradingBot/logs'  # macOS
    """
    system = sys.platform
    
    if system == 'win32':
        # Windows: Use APPDATA
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        app_dir = base / 'AITradingBot'
    elif system == 'darwin':
        # macOS: Use ~/Library/Application Support
        app_dir = Path.home() / 'Library' / 'Application Support' / 'AITradingBot'
    else:
        # Linux and other Unix-like systems: Use XDG_DATA_HOME or ~/.local/share
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
        app_dir = base / 'ai-trading-bot'
    
    # Add subdirectory if specified
    if subdir:
        app_dir = app_dir / subdir
    
    # Create directory if it doesn't exist
    app_dir.mkdir(parents=True, exist_ok=True)
    
    return str(app_dir)


def get_config_override_path(filename: str = 'config.yaml') -> str:
    """
    Get path for user-specific config overrides (writable location).
    
    Useful for allowing users to override bundled configs without modifying
    the installation directory.
    
    Args:
        filename: Config filename
        
    Returns:
        Path to user-specific config file
    """
    config_dir = get_writable_app_dir('config')
    return str(Path(config_dir) / filename)


def is_frozen() -> bool:
    """
    Check if the application is running as a frozen executable.
    
    Returns:
        True if frozen, False if running in development
    """
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

