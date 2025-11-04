"""
Unit Tests for Utility Functions

Tests helper functions including directory creation, timestamp conversions,
currency formatting, API key validation, and retry decorator.
"""

import unittest
import os
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.utils.helpers import (
    ensure_dir,
    timestamp_to_datetime,
    datetime_to_timestamp,
    format_currency,
    validate_api_keys,
    retry_on_failure,
    calculate_percentage_change,
    truncate_string,
    safe_divide
)


class TestUtilityHelpers(unittest.TestCase):
    """Test cases for utility helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_ensure_dir_creates_directory(self):
        """Test that ensure_dir creates a directory."""
        test_dir = os.path.join(self.temp_dir, 'test_directory')
        
        result = ensure_dir(test_dir)
        
        self.assertTrue(os.path.exists(test_dir))
        self.assertIsInstance(result, Path)
    
    def test_ensure_dir_nested_directories(self):
        """Test creating nested directories."""
        nested_dir = os.path.join(self.temp_dir, 'level1', 'level2', 'level3')
        
        result = ensure_dir(nested_dir)
        
        self.assertTrue(os.path.exists(nested_dir))
    
    def test_ensure_dir_existing_directory(self):
        """Test that ensure_dir handles existing directories."""
        test_dir = os.path.join(self.temp_dir, 'existing')
        os.makedirs(test_dir)
        
        # Should not raise an exception
        result = ensure_dir(test_dir)
        
        self.assertTrue(os.path.exists(test_dir))
    
    def test_timestamp_to_datetime_milliseconds(self):
        """Test converting millisecond timestamp to datetime."""
        timestamp_ms = 1609459200000  # 2021-01-01 00:00:00 UTC (may vary by timezone)
        
        result = timestamp_to_datetime(timestamp_ms, unit='ms')
        
        self.assertIsInstance(result, datetime)
        # Year could be 2020 or 2021 depending on timezone
        self.assertIn(result.year, [2020, 2021])
        # Month and day should be around Jan 1 (could be Dec 31 in some timezones)
        self.assertIn(result.month, [12, 1])
    
    def test_timestamp_to_datetime_seconds(self):
        """Test converting second timestamp to datetime."""
        timestamp_s = 1609459200  # 2021-01-01 00:00:00 UTC (may vary by timezone)
        
        result = timestamp_to_datetime(timestamp_s, unit='s')
        
        self.assertIsInstance(result, datetime)
        # Year could be 2020 or 2021 depending on timezone
        self.assertIn(result.year, [2020, 2021])
    
    def test_datetime_to_timestamp_milliseconds(self):
        """Test converting datetime to millisecond timestamp."""
        dt = datetime(2021, 1, 1, 0, 0, 0)
        
        result = datetime_to_timestamp(dt, unit='ms')
        
        self.assertIsInstance(result, int)
        self.assertGreater(result, 1000000000000)  # Should be in milliseconds
    
    def test_datetime_to_timestamp_seconds(self):
        """Test converting datetime to second timestamp."""
        dt = datetime(2021, 1, 1, 0, 0, 0)
        
        result = datetime_to_timestamp(dt, unit='s')
        
        self.assertIsInstance(result, int)
        self.assertLess(result, 2000000000)  # Should be in seconds
    
    def test_format_currency_usd(self):
        """Test formatting USD currency."""
        result = format_currency(1234.56, 'USD')
        
        self.assertEqual(result, '$1,234.56')
    
    def test_format_currency_btc(self):
        """Test formatting BTC currency."""
        result = format_currency(0.12345678, 'BTC', decimals=8)
        
        self.assertIn('0.12345678', result)
    
    def test_format_currency_large_amount(self):
        """Test formatting large amounts with thousands separator."""
        result = format_currency(1000000.99, 'USD')
        
        self.assertEqual(result, '$1,000,000.99')
    
    @patch.dict(os.environ, {
        'BINANCE_API_KEY': 'test_key',
        'BINANCE_API_SECRET': 'test_secret'
    })
    def test_validate_api_keys_binance_valid(self):
        """Test API key validation when Binance keys are present."""
        result = validate_api_keys()
        
        self.assertTrue(result['binance'])
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_api_keys_none_configured(self):
        """Test API key validation when no keys are configured."""
        result = validate_api_keys()
        
        self.assertFalse(result['binance'])
        self.assertFalse(result['coinbase'])
        self.assertFalse(result['alpaca'])
    
    def test_retry_on_failure_success(self):
        """Test retry decorator with successful function."""
        call_count = [0]
        
        @retry_on_failure(max_attempts=3, delay=0.1)
        def successful_function():
            call_count[0] += 1
            return "success"
        
        result = successful_function()
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 1)
    
    def test_retry_on_failure_eventual_success(self):
        """Test retry decorator with function that succeeds after failures."""
        call_count = [0]
        
        @retry_on_failure(max_attempts=3, delay=0.1)
        def eventually_successful():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_successful()
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)
    
    def test_retry_on_failure_max_attempts(self):
        """Test retry decorator raises exception after max attempts."""
        call_count = [0]
        
        @retry_on_failure(max_attempts=3, delay=0.1)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")
        
        with self.assertRaises(ValueError):
            always_fails()
        
        self.assertEqual(call_count[0], 3)
    
    def test_calculate_percentage_change_positive(self):
        """Test calculating positive percentage change."""
        result = calculate_percentage_change(100, 150)
        
        self.assertEqual(result, 50.0)
    
    def test_calculate_percentage_change_negative(self):
        """Test calculating negative percentage change."""
        result = calculate_percentage_change(100, 80)
        
        self.assertEqual(result, -20.0)
    
    def test_calculate_percentage_change_zero_old_value(self):
        """Test percentage change with zero old value."""
        result = calculate_percentage_change(0, 100)
        
        self.assertEqual(result, 0.0)
    
    def test_truncate_string_short(self):
        """Test truncating string that's shorter than max length."""
        text = "Short text"
        result = truncate_string(text, max_length=50)
        
        self.assertEqual(result, text)
    
    def test_truncate_string_long(self):
        """Test truncating long string."""
        text = "This is a very long string that needs to be truncated"
        result = truncate_string(text, max_length=20)
        
        self.assertEqual(len(result), 20)
        self.assertTrue(result.endswith('...'))
    
    def test_safe_divide_normal(self):
        """Test safe division with normal values."""
        result = safe_divide(10, 2)
        
        self.assertEqual(result, 5.0)
    
    def test_safe_divide_by_zero(self):
        """Test safe division by zero returns default."""
        result = safe_divide(10, 0, default=0.0)
        
        self.assertEqual(result, 0.0)
    
    def test_safe_divide_custom_default(self):
        """Test safe division with custom default value."""
        result = safe_divide(10, 0, default=-1.0)
        
        self.assertEqual(result, -1.0)


if __name__ == '__main__':
    unittest.main()

