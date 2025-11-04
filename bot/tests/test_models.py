"""
Unit Tests for Model Training, Prediction, and Management

Tests XGBoost trainer, predictor, and model manager.
"""

import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.models import XGBoostPredictor, ModelManager
from trading_bot.utils.exceptions import ModelError, ValidationError


class TestXGBoostPredictor(unittest.TestCase):
    """Test cases for XGBoostPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_logger = Mock()
        
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model files
        self.model_path = Path(self.temp_dir) / 'test_model.json'
        self.metadata_path = Path(self.temp_dir) / 'test_model_metadata.json'
        self.scaler_path = Path(self.temp_dir) / 'test_scaler.pkl'
        
        # Create mock metadata
        self.metadata = {
            'target_type': 'regression',
            'training_date': '2024-10-30T00:00:00',
            'feature_names': ['feature1', 'feature2', 'feature3'],
            'metrics': {'rmse': 2.5, 'r2': 0.85}
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test XGBoostPredictor initialization."""
        predictor = XGBoostPredictor(self.mock_config, self.mock_logger)
        
        self.assertIsNotNone(predictor)
        self.assertIsNone(predictor.model)
        self.assertIsNone(predictor.scaler)
        self.assertFalse(predictor.is_model_loaded())
    
    @patch('trading_bot.models.xgboost_predictor.xgb')
    @patch('trading_bot.models.xgboost_predictor.joblib')
    def test_load_model_success(self, mock_joblib, mock_xgb):
        """Test successful model loading."""
        # Create mock model and scaler
        mock_model = Mock()
        mock_scaler = Mock()
        
        mock_xgb.XGBRegressor.return_value = mock_model
        mock_joblib.load.return_value = mock_scaler
        
        # Create dummy files
        self.model_path.touch()
        self.scaler_path.touch()
        
        predictor = XGBoostPredictor(self.mock_config, self.mock_logger)
        predictor.load_model(
            str(self.model_path),
            str(self.metadata_path),
            str(self.scaler_path)
        )
        
        self.assertTrue(predictor.is_model_loaded())
        self.assertIsNotNone(predictor.metadata)
        self.assertEqual(predictor.feature_names, ['feature1', 'feature2', 'feature3'])
    
    def test_load_model_file_not_found(self):
        """Test model loading with missing file."""
        predictor = XGBoostPredictor(self.mock_config, self.mock_logger)
        
        with self.assertRaises(ModelError):
            predictor.load_model(
                'nonexistent_model.json',
                str(self.metadata_path),
                str(self.scaler_path)
            )
    
    @patch('trading_bot.models.xgboost_predictor.xgb')
    @patch('trading_bot.models.xgboost_predictor.joblib')
    def test_predict_success(self, mock_joblib, mock_xgb):
        """Test successful prediction."""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict.return_value = np.array([105.5])
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[0.5, 0.3, 0.7]])
        
        mock_xgb.XGBRegressor.return_value = mock_model
        mock_joblib.load.return_value = mock_scaler
        
        # Create files
        self.model_path.touch()
        self.scaler_path.touch()
        
        predictor = XGBoostPredictor(self.mock_config, self.mock_logger)
        predictor.load_model(
            str(self.model_path),
            str(self.metadata_path),
            str(self.scaler_path)
        )
        
        # Test prediction
        test_features = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0]
        })
        
        predictions = predictor.predict(test_features)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 1)
    
    @patch('trading_bot.models.xgboost_predictor.xgb')
    @patch('trading_bot.models.xgboost_predictor.joblib')
    def test_validate_features_success(self, mock_joblib, mock_xgb):
        """Test feature validation with correct features."""
        mock_model = Mock()
        mock_scaler = Mock()
        mock_xgb.XGBRegressor.return_value = mock_model
        mock_joblib.load.return_value = mock_scaler
        
        self.model_path.touch()
        self.scaler_path.touch()
        
        predictor = XGBoostPredictor(self.mock_config, self.mock_logger)
        predictor.load_model(
            str(self.model_path),
            str(self.metadata_path),
            str(self.scaler_path)
        )
        
        valid_features = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0]
        })
        
        # Should not raise exception
        predictor.validate_features(valid_features)
    
    @patch('trading_bot.models.xgboost_predictor.xgb')
    @patch('trading_bot.models.xgboost_predictor.joblib')
    def test_validate_features_missing(self, mock_joblib, mock_xgb):
        """Test feature validation with missing features."""
        mock_model = Mock()
        mock_scaler = Mock()
        mock_xgb.XGBRegressor.return_value = mock_model
        mock_joblib.load.return_value = mock_scaler
        
        self.model_path.touch()
        self.scaler_path.touch()
        
        predictor = XGBoostPredictor(self.mock_config, self.mock_logger)
        predictor.load_model(
            str(self.model_path),
            str(self.metadata_path),
            str(self.scaler_path)
        )
        
        invalid_features = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0]
            # Missing feature3
        })
        
        with self.assertRaises(ValidationError):
            predictor.validate_features(invalid_features)
    
    def test_predict_without_loading(self):
        """Test prediction without loading model first."""
        predictor = XGBoostPredictor(self.mock_config, self.mock_logger)
        
        test_features = pd.DataFrame({'feature1': [1.0]})
        
        with self.assertRaises(ModelError):
            predictor.predict(test_features)


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(return_value='xgboost_stock_v1.json')
        self.mock_logger = Mock()
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('trading_bot.models.model_manager.get_writable_app_dir')
    @patch('trading_bot.models.model_manager.resolve_resource_path')
    def test_initialization(self, mock_resolve, mock_get_dir):
        """Test ModelManager initialization."""
        mock_get_dir.return_value = self.temp_dir
        mock_resolve.return_value = self.temp_dir
        
        manager = ModelManager(self.mock_config, self.mock_logger)
        
        self.assertIsNotNone(manager)
        self.assertGreater(len(manager.model_dirs), 0)
    
    @patch('trading_bot.models.model_manager.get_writable_app_dir')
    @patch('trading_bot.models.model_manager.resolve_resource_path')
    def test_list_available_models(self, mock_resolve, mock_get_dir):
        """Test listing available models."""
        mock_get_dir.return_value = self.temp_dir
        mock_resolve.return_value = self.temp_dir
        
        # Create test model files
        model_dir = Path(self.temp_dir)
        model_file = model_dir / 'xgboost_AAPL_v1_20241030.json'
        model_file.touch()
        
        metadata_file = model_dir / 'xgboost_AAPL_v1_20241030_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({'symbol': 'AAPL', 'version': 'v1'}, f)
        
        manager = ModelManager(self.mock_config, self.mock_logger)
        models = manager.list_available_models('xgboost')
        
        self.assertGreater(len(models), 0)
        self.assertEqual(models[0]['symbol'], 'AAPL')
    
    @patch('trading_bot.models.model_manager.get_writable_app_dir')
    @patch('trading_bot.models.model_manager.resolve_resource_path')
    def test_get_latest_model(self, mock_resolve, mock_get_dir):
        """Test getting latest model."""
        mock_get_dir.return_value = self.temp_dir
        mock_resolve.return_value = self.temp_dir
        
        # Create multiple model files
        model_dir = Path(self.temp_dir)
        
        for date in ['20241001', '20241015', '20241030']:
            model_file = model_dir / f'xgboost_AAPL_v1_{date}.json'
            model_file.touch()
            
            metadata_file = model_dir / f'xgboost_AAPL_v1_{date}_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump({'symbol': 'AAPL', 'version': 'v1', 'date': date}, f)
        
        manager = ModelManager(self.mock_config, self.mock_logger)
        latest = manager.get_latest_model(symbol='AAPL')
        
        self.assertIsNotNone(latest)
        self.assertEqual(latest['date'], '20241030')
    
    @patch('trading_bot.models.model_manager.get_writable_app_dir')
    @patch('trading_bot.models.model_manager.resolve_resource_path')
    def test_validate_model(self, mock_resolve, mock_get_dir):
        """Test model validation."""
        mock_get_dir.return_value = self.temp_dir
        mock_resolve.return_value = self.temp_dir
        
        # Create valid model file
        model_file = Path(self.temp_dir) / 'test_model.json'
        with open(model_file, 'w') as f:
            json.dump({'test': 'data'}, f)
        
        manager = ModelManager(self.mock_config, self.mock_logger)
        
        # Valid file
        self.assertTrue(manager.validate_model(str(model_file)))
        
        # Non-existent file
        self.assertFalse(manager.validate_model('nonexistent.json'))
    
    @patch('trading_bot.models.model_manager.get_writable_app_dir')
    @patch('trading_bot.models.model_manager.resolve_resource_path')
    def test_get_model_info(self, mock_resolve, mock_get_dir):
        """Test reading model metadata."""
        mock_get_dir.return_value = self.temp_dir
        mock_resolve.return_value = self.temp_dir
        
        model_dir = Path(self.temp_dir)
        model_file = model_dir / 'test_model.json'
        model_file.touch()
        
        metadata_file = model_dir / 'test_model_metadata.json'
        test_metadata = {
            'symbol': 'AAPL',
            'version': 'v1',
            'training_date': '2024-10-30'
        }
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f)
        
        manager = ModelManager(self.mock_config, self.mock_logger)
        info = manager.get_model_info(str(model_file))
        
        self.assertIsNotNone(info)
        self.assertEqual(info['symbol'], 'AAPL')
        self.assertEqual(info['version'], 'v1')


if __name__ == '__main__':
    unittest.main()

