"""
Tests for ModelManager scaler path resolution.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from trading_bot.models.model_manager import ModelManager


def test_model_manager_prefers_metadata_scaler_path():
    """Test that ModelManager prefers scaler path from metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        
        # Create model file
        model_file = model_dir / 'test_model.json'
        model_file.write_text('{"test": "model"}')
        
        # Create metadata with scaler path
        metadata = {
            'scaler_file': str(model_dir / 'custom_scaler.pkl'),
            'model_file': str(model_file)
        }
        metadata_file = model_dir / 'test_model_metadata.json'
        metadata_file.write_text(json.dumps(metadata))
        
        # Create the scaler file
        scaler_file = model_dir / 'custom_scaler.pkl'
        scaler_file.write_text('scaler data')
        
        # Create ModelManager
        config = Mock()
        logger = Mock()
        manager = ModelManager(config, logger)
        manager.model_dirs = [model_dir]
        
        # Load model
        result = manager.load_model('test_model')
        
        # Should use scaler from metadata
        assert result['scaler_path'] == str(scaler_file)


def test_model_manager_fallback_to_glob():
    """Test that ModelManager falls back to glob if metadata missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        
        # Create model file
        model_file = model_dir / 'test_model.json'
        model_file.write_text('{"test": "model"}')
        
        # Create scaler file (no metadata)
        scaler_file = model_dir / 'test_model_scaler.pkl'
        scaler_file.write_text('scaler data')
        
        # Create ModelManager
        config = Mock()
        logger = Mock()
        manager = ModelManager(config, logger)
        manager.model_dirs = [model_dir]
        
        # Load model
        result = manager.load_model('test_model')
        
        # Should find scaler via glob
        assert result['scaler_path'] == str(scaler_file)

