"""
Unit Tests for Configuration Loader

Tests configuration loading from YAML files, environment variables,
defaults, and singleton pattern enforcement.
"""

import unittest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.config_loader import Config
from trading_bot.utils.exceptions import ConfigurationError


class TestConfigLoader(unittest.TestCase):
    """Test cases for Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton instance before each test
        Config._instance = None
        Config._initialized = False
        
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        self.test_config = {
            'logging': {
                'level': 'DEBUG',
                'format': 'test_format'
            },
            'trading': {
                'position_size': 500,
                'risk_percentage': 3
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        
        # Clean up any directories created during tests
        for dir_name in ['data', 'models', 'logs']:
            test_dir = os.path.join(self.temp_dir, dir_name)
            if os.path.exists(test_dir):
                os.rmdir(test_dir)
        
        os.rmdir(self.temp_dir)
        
        # Reset singleton
        Config._instance = None
        Config._initialized = False
    
    def test_singleton_pattern(self):
        """Test that Config implements singleton pattern."""
        config1 = Config()
        config2 = Config()
        
        self.assertIs(config1, config2, "Config should be a singleton")
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        config = Config()
        config.load_config(self.config_path)
        
        self.assertEqual(config.get('logging.level'), 'DEBUG')
        self.assertEqual(config.get('trading.position_size'), 500)
    
    def test_get_nested_config(self):
        """Test retrieving nested configuration values with dot notation."""
        config = Config()
        config.load_config(self.config_path)
        
        self.assertEqual(config.get('trading.risk_percentage'), 3)
        self.assertEqual(config.get('logging.format'), 'test_format')
    
    def test_get_with_default(self):
        """Test retrieving non-existent config with default value."""
        config = Config()
        config.load_config(self.config_path)
        
        result = config.get('nonexistent.key', default='default_value')
        self.assertEqual(result, 'default_value')
    
    @patch.dict(os.environ, {'TRADING_POSITION_SIZE': '1000'})
    def test_environment_variable_override(self):
        """Test that environment variables override config file values."""
        config = Config()
        config.load_config(self.config_path)
        
        # Environment variable should override config file
        result = config.get('trading.position_size')
        self.assertEqual(result, 1000)
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'ERROR'})
    def test_get_env(self):
        """Test retrieving environment variables with type conversion."""
        config = Config()
        
        result = config.get_env('LOG_LEVEL')
        self.assertEqual(result, 'ERROR')
    
    @patch.dict(os.environ, {'MAX_POSITIONS': '10'})
    def test_get_env_with_type_conversion(self):
        """Test environment variable type conversion."""
        config = Config()
        
        result = config.get_env('MAX_POSITIONS', var_type=int)
        self.assertEqual(result, 10)
        self.assertIsInstance(result, int)
    
    @patch.dict(os.environ, {'ENABLE_FEATURE': 'true'})
    def test_get_env_boolean(self):
        """Test boolean environment variable conversion."""
        config = Config()
        
        result = config.get_env('ENABLE_FEATURE', var_type=bool)
        self.assertTrue(result)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_boolean_with_default(self):
        """Test boolean environment variable with boolean default when env var is missing."""
        config = Config()
        
        # Test with True default
        result = config.get_env('MISSING_BOOL_VAR', default=True, var_type=bool)
        self.assertTrue(result)
        self.assertIsInstance(result, bool)
        
        # Test with False default
        result = config.get_env('MISSING_BOOL_VAR', default=False, var_type=bool)
        self.assertFalse(result)
        self.assertIsInstance(result, bool)
    
    @patch.dict(os.environ, {'BOOL_STRING_TRUE': 'true', 'BOOL_STRING_FALSE': 'false'})
    def test_get_env_boolean_string_parsing(self):
        """Test boolean string parsing for environment variables."""
        config = Config()
        
        # Test string 'true'
        result = config.get_env('BOOL_STRING_TRUE', var_type=bool)
        self.assertTrue(result)
        
        # Test string 'false'
        result = config.get_env('BOOL_STRING_FALSE', var_type=bool)
        self.assertFalse(result)
    
    def test_missing_config_file(self):
        """Test behavior when config file is missing."""
        config = Config()
        config.load_config('nonexistent_config.yaml')
        
        # Should not raise exception, just use defaults
        result = config.get('logging.level', 'INFO')
        self.assertEqual(result, 'INFO')
    
    def test_validate_config_creates_directories(self):
        """Test that validate_config creates required directories in temp location."""
        # Save the current directory
        original_dir = os.getcwd()
        
        try:
            # Change to temp directory
            os.chdir(self.temp_dir)
            
            config = Config()
            config.validate_config()
            
            # Check that required directories exist in temp directory
            self.assertTrue(Path(self.temp_dir, 'data').exists())
            self.assertTrue(Path(self.temp_dir, 'models').exists())
            self.assertTrue(Path(self.temp_dir, 'logs').exists())
        finally:
            # Always restore original directory
            os.chdir(original_dir)
    
    def test_reload_config(self):
        """Test reloading configuration."""
        config = Config()
        config.load_config(self.config_path)
        
        initial_value = config.get('trading.position_size')
        
        # Modify config file
        self.test_config['trading']['position_size'] = 2000
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Reload and check new value
        config.reload(self.config_path)
        new_value = config.get('trading.position_size')
        
        self.assertNotEqual(initial_value, new_value)
        self.assertEqual(new_value, 2000)
    
    def test_get_all(self):
        """Test retrieving all configuration as dictionary."""
        config = Config()
        config.load_config(self.config_path)
        
        all_config = config.get_all()
        
        self.assertIsInstance(all_config, dict)
        self.assertIn('logging', all_config)
        self.assertIn('trading', all_config)
    
    def test_type_conversion(self):
        """Test automatic type conversion for string values."""
        config = Config()
        
        # Test boolean conversion
        self.assertTrue(config._convert_type('true'))
        self.assertFalse(config._convert_type('false'))
        
        # Test integer conversion
        self.assertEqual(config._convert_type('42'), 42)
        
        # Test float conversion
        self.assertEqual(config._convert_type('3.14'), 3.14)
        
        # Test string remains string
        self.assertEqual(config._convert_type('hello'), 'hello')


if __name__ == '__main__':
    unittest.main()
