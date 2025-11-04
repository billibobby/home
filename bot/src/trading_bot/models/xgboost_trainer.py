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

