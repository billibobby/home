"""
Tests for path resolution utilities.
"""

import sys
from pathlib import Path
from unittest.mock import patch

from trading_bot.utils.paths import resolve_resource_path, get_writable_app_dir


def test_resolve_resource_path_development():
    """Test resource path resolution in development mode."""
    # Mock non-frozen state
    with patch('sys.frozen', False), patch.dict(sys.modules, {'_MEIPASS': None}):
        path = resolve_resource_path('config/config.yaml')
        
        # Should return a valid path
        assert isinstance(path, str)
        assert 'config' in path
        assert 'config.yaml' in path


def test_resolve_resource_path_frozen():
    """Test resource path resolution when frozen."""
    # Mock frozen executable
    with patch('sys.frozen', True):
        with patch('sys._MEIPASS', '/tmp/frozen_app'):
            path = resolve_resource_path('config/config.yaml')
            
            # Should use _MEIPASS path
            assert '/tmp/frozen_app' in path
            assert 'config.yaml' in path


def test_get_writable_app_dir():
    """Test writable app directory creation."""
    app_dir = get_writable_app_dir('test')
    
    # Should return a valid path
    assert isinstance(app_dir, str)
    assert Path(app_dir).exists()
    
    # Should be writable
    assert Path(app_dir).is_dir()

