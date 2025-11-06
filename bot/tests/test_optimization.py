"""
Unit Tests for Hyperparameter Optimization

Tests XGBoostOptimizer, OptimizationMonitor, and ParameterAnalyzer.
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_bot.models.optimizer import XGBoostOptimizer
from trading_bot.models.optimization_monitor import OptimizationMonitor
from trading_bot.models.param_analyzer import ParameterAnalyzer
from trading_bot.utils.exceptions import ModelError


class TestXGBoostOptimizer(unittest.TestCase):
    """Test cases for XGBoostOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'models.optimization.parameter_ranges': {
                'max_depth': [3, 10],
                'learning_rate': [0.001, 0.3],
                'n_estimators': [50, 500],
                'subsample': [0.6, 1.0],
                'colsample_bytree': [0.6, 1.0],
                'min_child_weight': [1, 10],
                'gamma': [0, 5.0],
                'reg_alpha': [0, 10.0],
                'reg_lambda': [0, 10.0]
            },
            'backtesting.walk_forward.train_period_days': 252,
            'backtesting.walk_forward.test_period_days': 21,
            'backtesting.walk_forward.step_size_days': 21,
            'models.optimization.pruning.enabled': True,
            'models.optimization.pruning.warmup_steps': 10,
            'models.optimization.pruning.n_startup_trials': 5,
            'models.optimization.monitoring.save_plots': False
        }.get(key, default))
        
        self.mock_logger = Mock()
        
        # Create synthetic data
        n_samples = 100
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
        """Verify optimizer initializes correctly with data and config."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger, objective='sharpe'
        )
        
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.objective, 'sharpe')
    
    @patch('trading_bot.models.optimizer.XGBoostTrainer')
    def test_suggest_params(self, mock_trainer_class):
        """Test parameter suggestion within defined ranges."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        
        # Create a mock trial
        mock_trial = Mock()
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: (low + high) // 2)
        mock_trial.suggest_float = Mock(side_effect=lambda name, low, high, log=False: (low + high) / 2)
        
        params = optimizer._suggest_params(mock_trial)
        
        self.assertIn('max_depth', params)
        self.assertIn('learning_rate', params)
        self.assertIn('n_estimators', params)
        self.assertGreaterEqual(params['max_depth'], 3)
        self.assertLessEqual(params['max_depth'], 10)
    
    @patch('trading_bot.models.optimizer.XGBoostTrainer')
    def test_objective_function(self, mock_trainer_class):
        """Test objective function returns valid metric value."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.target_type = 'regression'
        mock_trainer.train = Mock()
        mock_trainer.evaluate = Mock(return_value={'r2': 0.85, 'rmse': 1.5})
        mock_trainer_class.return_value = mock_trainer
        
        # Create mock trial
        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: (low + high) // 2)
        mock_trial.suggest_float = Mock(side_effect=lambda name, low, high, log=False: (low + high) / 2)
        mock_trial.report = Mock()
        mock_trial.should_prune = Mock(return_value=False)
        
        value = optimizer.objective_function(mock_trial)
        
        self.assertIsInstance(value, (int, float))
    
    @patch('trading_bot.models.optimizer.XGBoostTrainer')
    def test_optimize_completes(self, mock_trainer_class):
        """Test optimization runs to completion."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.target_type = 'regression'
        mock_trainer.train = Mock()
        mock_trainer.evaluate = Mock(return_value={'r2': 0.85})
        mock_trainer_class.return_value = mock_trainer
        
        # Run optimization with minimal trials
        best_params = optimizer.optimize(n_trials=3)
        
        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
    
    def test_get_best_params(self):
        """Verify best params extraction."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        
        # Set best params manually
        optimizer.best_params = {'max_depth': 6, 'learning_rate': 0.1}
        
        best_params = optimizer.get_best_params()
        
        self.assertEqual(best_params, {'max_depth': 6, 'learning_rate': 0.1})
    
    def test_get_best_params_not_optimized(self):
        """Test get_best_params raises error if not optimized."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        
        with self.assertRaises(ModelError):
            optimizer.get_best_params()
    
    def test_multiple_objectives(self):
        """Test different objective functions."""
        objectives = ['sharpe', 'sortino', 'returns', 'calmar']
        
        for objective in objectives:
            optimizer = XGBoostOptimizer(
                self.X_train, self.y_train, self.X_val, self.y_val,
                self.mock_config, self.mock_logger, objective=objective
            )
            self.assertEqual(optimizer.objective, objective)
    
    def test_save_load_study(self):
        """Test saving and loading study persistence."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        
        # Create a mock study
        import optuna
        optimizer.study = optuna.create_study(direction='maximize')
        
        # Add a trial
        trial = optimizer.study.ask()
        trial.suggest_int('max_depth', 3, 10)
        optimizer.study.tell(trial, 1.5)
        
        # Save study
        study_path = str(Path(self.temp_dir) / "test_study.pkl")
        optimizer.save_study(study_path)
        
        # Create new optimizer and load study
        optimizer2 = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        optimizer2.load_study(study_path)
        
        # Verify study was loaded
        self.assertIsNotNone(optimizer2.study)
        self.assertEqual(len(optimizer2.study.trials), 1)
        self.assertEqual(optimizer2.best_value, 1.5)
        self.assertIn('max_depth', optimizer2.best_params)
    
    @patch('trading_bot.models.optimizer.XGBoostTrainer')
    def test_pruning_logic(self, mock_trainer_class):
        """Test that pruning logic works correctly."""
        optimizer = XGBoostOptimizer(
            self.X_train, self.y_train, self.X_val, self.y_val,
            self.mock_config, self.mock_logger
        )
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.target_type = 'regression'
        mock_trainer.train = Mock()
        mock_trainer.model = Mock()
        mock_trainer.model.predict = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        mock_trainer_class.return_value = mock_trainer
        
        # Create mock trial that should be pruned
        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: (low + high) // 2)
        mock_trial.suggest_float = Mock(side_effect=lambda name, low, high, log=False: (low + high) / 2)
        mock_trial.report = Mock()
        mock_trial.should_prune = Mock(return_value=True)  # Simulate pruning
        
        # Objective function should raise TrialPruned
        with self.assertRaises(optuna.TrialPruned):
            optimizer.objective_function(mock_trial)


class TestOptimizationMonitor(unittest.TestCase):
    """Test cases for OptimizationMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get = Mock(side_effect=lambda key, default=None: {
            'models.optimization.monitoring.checkpoint_interval': 10,
            'models.optimization.notifications.enabled': False
        }.get(key, default))
        
        self.mock_logger = Mock()
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_callback_creation(self):
        """Test callback function creation."""
        monitor = OptimizationMonitor(self.mock_config, self.mock_logger, self.temp_dir)
        
        callback = monitor.create_callback()
        
        self.assertIsNotNone(callback)
        self.assertTrue(callable(callback))
    
    def test_progress_tracking(self):
        """Test progress info updates."""
        monitor = OptimizationMonitor(self.mock_config, self.mock_logger, self.temp_dir)
        monitor.completed_trials = 5
        monitor.best_value = 1.5
        
        progress_info = monitor.get_progress_info()
        
        self.assertEqual(progress_info['completed_trials'], 5)
        self.assertEqual(progress_info['best_value'], 1.5)
    
    def test_eta_calculation(self):
        """Test time remaining estimation."""
        monitor = OptimizationMonitor(self.mock_config, self.mock_logger, self.temp_dir)
        monitor.trial_durations = [10.0, 12.0, 8.0]
        monitor.completed_trials = 3
        
        # Create mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(3)]
        
        eta = monitor.estimate_time_remaining(mock_study, None, 10)
        
        self.assertIsInstance(eta, str)


class TestParameterAnalyzer(unittest.TestCase):
    """Test cases for ParameterAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        import optuna
        
        self.mock_config = Mock()
        self.mock_logger = Mock()
        
        # Create a mock study with some trials
        self.study = optuna.create_study(direction='maximize')
        
        # Add some trials
        for i in range(5):
            trial = self.study.ask()
            trial.suggest_int('max_depth', 3, 10)
            trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
            value = np.random.rand() * 2.0
            self.study.tell(trial, value)
    
    def test_importance_calculation(self):
        """Test parameter importance ranking."""
        analyzer = ParameterAnalyzer(self.study, self.mock_config, self.mock_logger)
        
        importances = analyzer.analyze_importance()
        
        self.assertIsInstance(importances, dict)
    
    def test_optimal_regions(self):
        """Test optimal region identification."""
        analyzer = ParameterAnalyzer(self.study, self.mock_config, self.mock_logger)
        
        optimal_ranges = analyzer.find_optimal_regions()
        
        self.assertIsInstance(optimal_ranges, dict)
    
    def test_insights_generation(self):
        """Test insight text generation."""
        analyzer = ParameterAnalyzer(self.study, self.mock_config, self.mock_logger)
        
        insights = analyzer.generate_insights()
        
        self.assertIsInstance(insights, str)
        self.assertIn('PARAMETER IMPORTANCE', insights.upper())


if __name__ == '__main__':
    unittest.main()

