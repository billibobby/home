"""
XGBoost Model Trainer Module

Trains XGBoost models with GPU support (designed for Google Colab).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path

from trading_bot.utils.exceptions import ModelError


class XGBoostTrainer:
    """
    Trains XGBoost models for stock price prediction.
    
    Supports both GPU (Colab) and CPU (local) training.
    Handles regression and classification tasks.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the XGBoost trainer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Load configuration
        self.target_type = config.get('models.xgboost.target_type', 'regression')
        self.device = config.get('models.xgboost.device', 'cpu')
        
        # Hyperparameters
        self.max_depth = config.get('models.xgboost.max_depth', 6)
        self.learning_rate = config.get('models.xgboost.learning_rate', 0.1)
        self.n_estimators = config.get('models.xgboost.n_estimators', 100)
        self.subsample = config.get('models.xgboost.subsample', 0.8)
        self.colsample_bytree = config.get('models.xgboost.colsample_bytree', 0.8)
        self.min_child_weight = config.get('models.xgboost.min_child_weight', 1)
        self.gamma = config.get('models.xgboost.gamma', 0)
        self.reg_alpha = config.get('models.xgboost.reg_alpha', 0)
        self.reg_lambda = config.get('models.xgboost.reg_lambda', 1)
        
        # Training parameters
        self.cv_folds = config.get('models.training.cross_validation_folds', 5)
        self.random_state = config.get('models.training.random_state', 42)
        
        self.model = None
        self.feature_names = None
        self.training_history = {}
        
        self.logger.info(f"XGBoost trainer initialized (target: {self.target_type}, device: {self.device})")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train XGBoost model with validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary with training metrics
            
        Raises:
            ModelError: If training fails
        """
        try:
            import xgboost as xgb
            
            self.logger.info(f"Starting XGBoost training with {len(X_train)} samples")
            
            # Get model parameters
            params = self._get_model_params()
            
            # Store feature names
            self.feature_names = list(X_train.columns)
            
            # Create model
            if self.target_type == 'regression':
                self.model = xgb.XGBRegressor(**params)
            else:
                self.model = xgb.XGBClassifier(**params)
            
            # Setup validation
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.logger.info(f"Using validation set with {len(X_val)} samples")
            
            # Train model
            if eval_set:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=True
                )
            else:
                self.model.fit(X_train, y_train, verbose=True)
            
            # Store training history
            if hasattr(self.model, 'evals_result'):
                self.training_history = self.model.evals_result()
            
            self.logger.info("XGBoost training completed successfully")
            
            # Return training summary
            return {
                'n_samples': len(X_train),
                'n_features': len(self.feature_names),
                'target_type': self.target_type,
                'device': self.device
            }
            
        except ImportError:
            raise ModelError("xgboost library not installed. Install with: pip install xgboost")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}")
    
    def _get_model_params(self) -> Dict:
        """
        Build XGBoost parameters from configuration.
        
        Returns:
            Dictionary of model parameters
        """
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': getattr(self, 'min_child_weight', 1),
            'gamma': getattr(self, 'gamma', 0),
            'reg_alpha': getattr(self, 'reg_alpha', 0),
            'reg_lambda': getattr(self, 'reg_lambda', 1),
            'random_state': self.random_state,
        }
        
        # Add device-specific parameters
        # Accept 'gpu', 'cuda', or 'gpu_hist' for GPU training
        if self.device in ('gpu', 'cuda', 'gpu_hist'):
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'
            self.logger.info("Configured for GPU training (tree_method='gpu_hist', device='cuda')")
        else:
            params['tree_method'] = 'hist'
            params['device'] = 'cpu'
            self.logger.info("Configured for CPU training")
        
        return params
    
    def _setup_gpu_training(self) -> bool:
        """
        Configure GPU parameters if available.
        
        Returns:
            True if GPU is available, False otherwise
        """
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            
            if gpu_available:
                self.logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                return True
            else:
                self.logger.warning("GPU not available, falling back to CPU")
                return False
                
        except ImportError:
            self.logger.warning("PyTorch not installed, cannot check GPU availability")
            return False
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            ModelError: If evaluation fails
        """
        if self.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error, r2_score,
                accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            )
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            metrics = {}
            
            if self.target_type == 'regression':
                # Regression metrics
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                metrics['r2'] = r2_score(y_test, y_pred)
                
                # MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                metrics['mape'] = mape
                
                self.logger.info(f"Evaluation - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
                
            else:
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # ROC-AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    except:
                        pass
                
                self.logger.info(f"Evaluation - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise ModelError(f"Evaluation failed: {str(e)}")
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Export model and metadata.
        
        Args:
            filepath: Path to save model (JSON format)
            metadata: Additional metadata to save
            
        Raises:
            ModelError: If saving fails
        """
        if self.model is None:
            raise ModelError("No model to save")
        
        try:
            # Save model as JSON
            self.model.save_model(filepath)
            self.logger.info(f"Model saved to {filepath}")
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata_dict = {
                'model_file': filepath,
                'target_type': self.target_type,
                'device': self.device,
                'training_date': datetime.now().isoformat(),
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names) if self.feature_names else 0,
                'hyperparameters': {
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'n_estimators': self.n_estimators,
                    'subsample': self.subsample,
                    'colsample_bytree': self.colsample_bytree,
                },
                **metadata
            }
            
            # Save scaler path if provided in metadata, otherwise infer from model path
            if 'scaler_file' not in metadata_dict:
                # Infer scaler path: replace .json with _scaler.pkl or use scaler.pkl in same dir
                model_dir = Path(filepath).parent
                model_stem = Path(filepath).stem
                scaler_path = model_dir / f"{model_stem}_scaler.pkl"
                metadata_dict['scaler_file'] = str(scaler_path)
            
            # Save metadata JSON
            metadata_file = filepath.replace('.json', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            self.logger.info(f"Metadata saved to {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise ModelError(f"Failed to save model: {str(e)}")
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Visualize feature importance.
        
        Args:
            top_n: Number of top features to display
            
        Raises:
            ModelError: If plotting fails
        """
        if self.model is None:
            raise ModelError("Model not trained")
        
        try:
            import matplotlib.pyplot as plt
            
            # Get feature importance
            importance = self.model.feature_importances_
            
            # Create DataFrame for sorting
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Feature importance plot displayed")
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
        except Exception as e:
            self.logger.warning(f"Failed to plot feature importance: {str(e)}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Returns:
            DataFrame with features and their importance scores
            
        Raises:
            ModelError: If model not trained
        """
        if self.model is None:
            raise ModelError("Model not trained")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                                 n_trials: int = 100, objective: str = 'sharpe',
                                 timeout: Optional[int] = None) -> Dict:
        """
        Main optimization entry point that creates XGBoostOptimizer and runs optimization.
        
        Args:
            X: Features
            y: Target
            n_trials: Number of optimization trials
            objective: Optimization objective ('sharpe', 'sortino', 'returns', 'calmar')
            timeout: Maximum time in seconds (None for no limit)
            
        Returns:
            Dictionary of best hyperparameters
            
        Raises:
            ModelError: If optimization fails
        """
        try:
            from trading_bot.models.optimizer import XGBoostOptimizer
            from trading_bot.models.optimization_monitor import OptimizationMonitor
            
            self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
            
            # Split data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            # Create optimizer
            optimizer = XGBoostOptimizer(
                X_train, y_train, X_val, y_val,
                self.config, self.logger, objective=objective
            )
            
            # Create monitor
            monitor = OptimizationMonitor(self.config, self.logger)
            monitor.start_time = datetime.now()
            callback = monitor.create_callback()
            
            # Run optimization
            best_params = optimizer.optimize(n_trials=n_trials, timeout=timeout, callbacks=[callback])
            
            # Optionally save optimization history
            opt_config = self.config.get('models.optimization.monitoring', {})
            if opt_config.get('save_plots', False):
                plot_dir = Path(opt_config.get('plot_dir', 'optimization_plots'))
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                # Save optimization history plot
                fig = optimizer.plot_optimization_history()
                fig.write_html(str(plot_dir / 'optimization_history.html'))
                
                # Save parameter importance plot
                fig = optimizer.plot_param_importances()
                fig.write_html(str(plot_dir / 'param_importance.html'))
            
            self.logger.info("Hyperparameter optimization completed")
            
            return best_params
            
        except ImportError as e:
            raise ModelError(f"Optimization dependencies not available: {str(e)}")
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise ModelError(f"Hyperparameter optimization failed: {str(e)}")
    
    def train_with_optimization(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: Optional[pd.DataFrame] = None,
                                y_val: Optional[pd.Series] = None,
                                optimize: bool = True, n_trials: int = 100) -> Dict:
        """
        Convenience method that optionally runs optimization before training.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            optimize: Whether to run optimization before training
            n_trials: Number of optimization trials (if optimize=True)
            
        Returns:
            Dictionary with training metrics
        """
        if optimize:
            self.logger.info("Running hyperparameter optimization before training")
            # Combine train and val for optimization
            if X_val is not None and y_val is not None:
                X_combined = pd.concat([X_train, X_val], ignore_index=True)
                y_combined = pd.concat([y_train, y_val], ignore_index=True)
            else:
                X_combined = X_train
                y_combined = y_train
            
            # Get optimization objective from config
            opt_config = self.config.get('models.optimization', {})
            objective = opt_config.get('objective', 'sharpe')
            
            # Run optimization
            best_params = self.optimize_hyperparameters(
                X_combined, y_combined, n_trials=n_trials, objective=objective
            )
            
            # Update trainer with optimized params
            self.set_hyperparameters(best_params)
        
        # Train with (possibly optimized) parameters
        return self.train(X_train, y_train, X_val, y_val)
    
    def set_hyperparameters(self, params: Dict) -> None:
        """
        Update trainer's hyperparameters from dictionary.
        
        Args:
            params: Dictionary of hyperparameters to set
            
        Raises:
            ModelError: If validation fails
        """
        # Validate parameter ranges
        valid_params = {
            'max_depth': (1, 20),
            'learning_rate': (0.001, 1.0),
            'n_estimators': (1, 10000),
            'subsample': (0.1, 1.0),
            'colsample_bytree': (0.1, 1.0),
            'min_child_weight': (0, 100),
            'gamma': (0, 100),
            'reg_alpha': (0, 100),
            'reg_lambda': (0, 100)
        }
        
        for param_name, value in params.items():
            if param_name in valid_params:
                min_val, max_val = valid_params[param_name]
                if not (min_val <= value <= max_val):
                    raise ModelError(
                        f"Parameter {param_name}={value} out of range [{min_val}, {max_val}]"
                    )
                setattr(self, param_name, value)
            else:
                self.logger.warning(f"Unknown parameter: {param_name}, skipping")
        
        self.logger.info(f"Hyperparameters updated: {params}")
    
    def get_hyperparameters(self) -> Dict:
        """
        Return current hyperparameters as dictionary.
        
        Returns:
            Dictionary of current hyperparameters
        """
        return {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': getattr(self, 'min_child_weight', None),
            'gamma': getattr(self, 'gamma', None),
            'reg_alpha': getattr(self, 'reg_alpha', None),
            'reg_lambda': getattr(self, 'reg_lambda', None)
        }
    
    def backtest(self, data: pd.DataFrame, symbol: str = 'UNKNOWN',
                 walk_forward_config: Dict = None) -> Optional[object]:
        """
        Run walk-forward backtest on historical data.
        
        Args:
            data: Historical OHLCV DataFrame with Date column
            symbol: Trading symbol
            walk_forward_config: Optional walk-forward configuration overrides
            
        Returns:
            BacktestResults object or None if backtest fails
            
        Raises:
            ModelError: If backtest fails
        """
        try:
            from trading_bot.backtesting import WalkForwardBacktest, BacktestResults
            from trading_bot.data import FeatureEngineer
            from trading_bot.trading import SignalGenerator
            
            self.logger.info(f"Starting backtest for {symbol}")
            
            # Initialize dependencies
            feature_engineer = FeatureEngineer(self.config, self.logger)
            signal_generator = SignalGenerator(self.config, self.logger)
            
            # Create backtest engine
            backtest_engine = WalkForwardBacktest(
                data=data,
                config=self.config,
                logger=self.logger,
                feature_engineer=feature_engineer,
                signal_generator=signal_generator,
                model_class=self.__class__
            )
            
            # Set symbol on backtest engine
            backtest_engine.symbol = symbol
            
            # Override config if provided
            if walk_forward_config:
                for key, value in walk_forward_config.items():
                    if hasattr(backtest_engine, key):
                        setattr(backtest_engine, key, value)
            
            # Run backtest
            results = backtest_engine.run()
            
            # Validate results
            validation_config = self.config.get('backtesting.validation', {})
            min_sharpe = validation_config.get('min_sharpe_ratio', 1.5)
            max_dd = validation_config.get('max_drawdown_pct', 15)
            min_win_rate = validation_config.get('min_win_rate', 0.50)
            min_trades = validation_config.get('min_trades', 100)
            
            # Check validation criteria
            validation_passed = True
            warnings = []
            
            if results.sharpe_ratio < min_sharpe:
                warnings.append(f"Sharpe ratio {results.sharpe_ratio:.2f} below minimum {min_sharpe}")
                validation_passed = False
            
            if abs(results.max_drawdown) > max_dd:
                warnings.append(f"Max drawdown {abs(results.max_drawdown):.2f}% exceeds maximum {max_dd}%")
                validation_passed = False
            
            if results.win_rate < min_win_rate * 100:
                warnings.append(f"Win rate {results.win_rate:.2f}% below minimum {min_win_rate * 100:.2f}%")
                validation_passed = False
            
            if results.total_trades < min_trades:
                warnings.append(f"Total trades {results.total_trades} below minimum {min_trades}")
                validation_passed = False
            
            # Log warnings
            for warning in warnings:
                self.logger.warning(f"Backtest validation: {warning}")
            
            # Log summary
            self.logger.info(
                f"Backtest completed: Sharpe={results.sharpe_ratio:.2f}, "
                f"Return={results.total_return:.2f}%, Drawdown={results.max_drawdown:.2f}%, "
                f"Win Rate={results.win_rate:.2f}%, Trades={results.total_trades}"
            )
            
            # Optionally raise exception if validation fails
            if not validation_passed and validation_config.get('raise_on_failure', False):
                raise ModelError(f"Backtest validation failed: {', '.join(warnings)}")
            
            return results
            
        except ImportError as e:
            self.logger.error(f"Backtesting module not available: {str(e)}")
            raise ModelError(f"Backtesting module not available: {str(e)}")
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise ModelError(f"Backtest failed: {str(e)}")

