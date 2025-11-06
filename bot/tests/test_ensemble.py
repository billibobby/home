"""
Unit Tests for Ensemble Optimization

Tests EnsembleOptimizer class.
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import optuna

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.models.ensemble_optimizer import EnsembleOptimizer
from trading_bot.utils.exceptions import ModelError


class TestEnsembleOptimizer(unittest.TestCase):
    """Test cases for EnsembleOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'models.optimization.ensemble.n_models': 5,
            'models.optimization.ensemble.diversity_weight': 0.1
        }.get(key, default))
        
        self.mock_logger = Mock()
        
        # Create a mock study with trials
        self.study = optuna.create_study(direction='maximize')
        
        # Add some trials
        for i in range(10):
            trial = self.study.ask()
            trial.suggest_int('max_depth', 3, 10)
            trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
            value = np.random.rand() * 2.0
            self.study.tell(trial, value)
        
        # Create synthetic data
        n_samples = 50
        n_features = 10
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = pd.Series(np.random.randn(n_samples))
        self.X_val = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_val = pd.Series(np.random.randn(20))
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Verify ensemble optimizer initializes with study."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger, n_models=3)
        
        self.assertIsNotNone(ensemble)
        self.assertEqual(ensemble.n_models, 3)
    
    def test_get_top_trials(self):
        """Test extraction of top N trials."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger, n_models=3)
        
        top_trials = ensemble._get_top_trials(3)
        
        self.assertLessEqual(len(top_trials), 3)
        self.assertGreater(len(top_trials), 0)
        
        # Verify trials are sorted (best first)
        if len(top_trials) > 1:
            for i in range(len(top_trials) - 1):
                if top_trials[i].value is not None and top_trials[i+1].value is not None:
                    self.assertGreaterEqual(top_trials[i].value, top_trials[i+1].value)
    
    @patch('trading_bot.models.ensemble_optimizer.XGBoostTrainer')
    def test_train_models_from_trials(self, mock_trainer_class):
        """Test model training from trial parameters."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger, n_models=3)
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        top_trials = ensemble._get_top_trials(3)
        
        models = ensemble._train_models_from_trials(top_trials, self.X_train, self.y_train)
        
        self.assertGreater(len(models), 0)
        self.assertLessEqual(len(models), 3)
    
    def test_diversity_calculation(self):
        """Test diversity score calculation between models."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger)
        
        # Create mock models with different predictions
        model1 = Mock()
        model1.model = Mock()
        model1.model.predict = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        
        model2 = Mock()
        model2.model = Mock()
        model2.model.predict = Mock(return_value=np.array([1.1, 2.1, 3.1]))  # High correlation
        
        diversity = ensemble._calculate_diversity([model1, model2], self.X_val)
        
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
        # High correlation should result in low diversity
        self.assertLess(diversity, 0.5)
    
    def test_diversity_calculation_different(self):
        """Test diversity with very different predictions."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger)
        
        # Create mock models with very different predictions
        model1 = Mock()
        model1.model = Mock()
        model1.model.predict = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        
        model2 = Mock()
        model2.model = Mock()
        model2.model.predict = Mock(return_value=np.array([10.0, 20.0, 30.0]))  # Different scale
        
        diversity = ensemble._calculate_diversity([model1, model2], self.X_val)
        
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    @patch('trading_bot.models.ensemble_optimizer.XGBoostTrainer')
    def test_weight_optimization(self, mock_trainer_class):
        """Test ensemble weight optimization."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger, n_models=2)
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.model = Mock()
        mock_trainer.model.predict = Mock(return_value=np.random.randn(20))
        mock_trainer_class.return_value = mock_trainer
        
        top_trials = ensemble._get_top_trials(2)
        models = ensemble._train_models_from_trials(top_trials, self.X_train, self.y_train)
        
        if len(models) > 0:
            weights = ensemble._optimize_weights(models, self.X_val, self.y_val)
            
            self.assertIsNotNone(weights)
            self.assertAlmostEqual(weights.sum(), 1.0, places=5)
            self.assertTrue(all(w >= 0 for w in weights))
    
    @patch('trading_bot.models.ensemble_optimizer.XGBoostTrainer')
    def test_predict_ensemble(self, mock_trainer_class):
        """Test ensemble prediction with weighted averaging."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger, n_models=2)
        
        # Mock models
        mock_model1 = Mock()
        mock_model1.model = Mock()
        mock_model1.model.predict = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        
        mock_model2 = Mock()
        mock_model2.model = Mock()
        mock_model2.model.predict = Mock(return_value=np.array([2.0, 3.0, 4.0]))
        
        ensemble.models = [mock_model1, mock_model2]
        ensemble.weights = np.array([0.6, 0.4])
        
        predictions = ensemble.predict_ensemble(self.X_val)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.X_val))
        # Weighted average: 0.6 * [1,2,3] + 0.4 * [2,3,4] = [1.4, 2.4, 3.4]
        expected_first = 0.6 * 1.0 + 0.4 * 2.0
        self.assertAlmostEqual(predictions[0], expected_first, places=5)
    
    def test_predict_ensemble_not_trained(self):
        """Test predict_ensemble raises error if not trained."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger)
        
        with self.assertRaises(ModelError):
            ensemble.predict_ensemble(self.X_val)
    
    def test_get_ensemble_info(self):
        """Test ensemble info retrieval."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger)
        
        info = ensemble.get_ensemble_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('n_models', info)
        self.assertIn('diversity_score', info)
        self.assertIn('weights', info)
        self.assertIn('model_params', info)
    
    @patch('trading_bot.models.ensemble_optimizer.XGBoostTrainer')
    def test_save_ensemble(self, mock_trainer_class):
        """Test ensemble persistence - saving ensemble."""
        ensemble = EnsembleOptimizer(self.study, self.mock_config, self.mock_logger, n_models=2)
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.model = Mock()
        mock_trainer.train = Mock()
        mock_trainer.save_model = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Train models
        top_trials = ensemble._get_top_trials(2)
        models = ensemble._train_models_from_trials(top_trials, self.X_train, self.y_train)
        
        if len(models) > 0:
            ensemble.models = models
            ensemble.weights = np.array([0.5, 0.5])
            ensemble.diversity_score = 0.5
            
            # Save ensemble
            save_path = str(Path(self.temp_dir) / "test_ensemble")
            ensemble.save_ensemble(save_path)
            
            # Verify files were created
            import os
            self.assertTrue(os.path.exists(save_path))
            # Check for weights file
            weights_file = Path(save_path) / "ensemble_weights.pkl"
            self.assertTrue(weights_file.exists())


if __name__ == '__main__':
    unittest.main()

