"""
XGBoost Model Predictor Module

Loads trained XGBoost models and generates predictions (CPU inference).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import json
from pathlib import Path
from datetime import datetime

from trading_bot.utils.exceptions import ModelError, ValidationError
from trading_bot.utils.paths import resolve_resource_path, get_writable_app_dir


class XGBoostPredictor:
    """
    Loads and uses trained XGBoost models for predictions.
    
    Handles model loading, feature validation, and prediction generation.
    Always uses CPU for inference (fast enough for real-time).
    """
    
    def __init__(self, config, logger):
        """
        Initialize the XGBoost predictor.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
        self.logger.info("XGBoost predictor initialized")
    
    def load_model(self, model_path: str, metadata_path: str, 
                   scaler_path: str) -> None:
        """
        Load trained model, metadata, and scaler.
        
        Args:
            model_path: Path to model JSON file
            metadata_path: Path to metadata JSON file
            scaler_path: Path to scaler pickle file
            
        Raises:
            ModelError: If loading fails
        """
        try:
            import xgboost as xgb
            import joblib
            
            # Load model
            if not Path(model_path).exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            # Determine model type from metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            target_type = self.metadata.get('target_type', 'regression')
            
            # Create appropriate model instance
            if target_type == 'regression':
                self.model = xgb.XGBRegressor()
            else:
                self.model = xgb.XGBClassifier()
            
            # Load model weights
            self.model.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            
            # Load scaler
            if not Path(scaler_path).exists():
                raise ModelError(f"Scaler file not found: {scaler_path}")
            
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Scaler loaded from {scaler_path}")
            
            # Extract feature names
            self.feature_names = self.metadata.get('feature_names', [])
            
            if not self.feature_names:
                self.logger.warning("No feature names in metadata")
            
            self.logger.info(f"Model ready for predictions (features: {len(self.feature_names)})")
            
        except ImportError as e:
            raise ModelError(f"Required library not installed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions on new data.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Array of predictions
            
        Raises:
            ModelError: If prediction fails
            ValidationError: If feature validation fails
        """
        if not self.is_model_loaded():
            raise ModelError("Model not loaded. Call load_model() first.")
        
        try:
            # Validate features
            self.validate_features(features_df)
            
            # Select and order features with reindexing to ensure correct order and handle missing columns
            X = features_df.reindex(columns=self.feature_names, fill_value=0)
            
            # Ensure numeric dtypes
            X = X.astype(float)
            
            # Apply scaler
            X_scaled = self.scaler.transform(X)
            
            # Generate predictions
            predictions = self.model.predict(X_scaled)
            
            self.logger.debug(f"Generated {len(predictions)} predictions")
            
            return predictions
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Array of prediction probabilities
            
        Raises:
            ModelError: If prediction fails or model is regression
        """
        if not self.is_model_loaded():
            raise ModelError("Model not loaded")
        
        target_type = self.metadata.get('target_type', 'regression')
        if target_type != 'classification':
            raise ModelError("predict_proba only available for classification models")
        
        try:
            # Validate features
            self.validate_features(features_df)
            
            # Select and order features with reindexing to ensure correct order and handle missing columns
            X = features_df.reindex(columns=self.feature_names, fill_value=0)
            
            # Ensure numeric dtypes
            X = X.astype(float)
            
            # Apply scaler
            X_scaled = self.scaler.transform(X)
            
            # Generate probability predictions
            probabilities = self.model.predict_proba(X_scaled)
            
            self.logger.debug(f"Generated probabilities for {len(probabilities)} samples")
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {str(e)}")
            raise ModelError(f"Probability prediction failed: {str(e)}")
    
    def get_confidence(self, prediction: float, features_df: pd.DataFrame = None) -> float:
        """
        Calculate prediction confidence.
        
        For classification: uses probability from predict_proba
        For regression: uses a heuristic based on feature variance
        
        Args:
            prediction: The prediction value
            features_df: Optional features for confidence calculation (must be single-row DataFrame)
            
        Returns:
            Confidence score (0-1) for single-row input, or array for multi-row
        """
        target_type = self.metadata.get('target_type', 'regression')
        
        if target_type == 'classification' and features_df is not None:
            try:
                # Validate single row
                if len(features_df) != 1:
                    raise ValidationError(
                        f"get_confidence expects exactly 1 row, got {len(features_df)}",
                        details={'rows': len(features_df)}
                    )
                
                probabilities = self.predict_proba(features_df)
                # Confidence is the max probability for the single row
                confidence = np.max(probabilities, axis=1)[0]
                return float(confidence)
            except ValidationError:
                raise
            except:
                return 0.5
        else:
            # For regression, return a default confidence
            # In practice, you might want to use model uncertainty quantification
            return 0.7
    
    def validate_features(self, features_df: pd.DataFrame) -> None:
        """
        Ensure features match training features.
        
        Args:
            features_df: DataFrame with features
            
        Raises:
            ValidationError: If feature validation fails
        """
        if not self.feature_names:
            self.logger.warning("No feature names to validate against")
            return
        
        # Check if all required features are present
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        
        if missing_features:
            raise ValidationError(
                f"Missing features: {missing_features}",
                details={
                    'expected': self.feature_names,
                    'received': list(features_df.columns),
                    'missing': missing_features
                }
            )
        
        # Check for extra features (just warn)
        extra_features = [f for f in features_df.columns if f not in self.feature_names]
        if extra_features:
            self.logger.debug(f"Extra features (will be ignored): {extra_features}")
        
        self.logger.debug("Feature validation passed")
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is ready for predictions.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.scaler is not None
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata and information.
        
        Returns:
            Dictionary with model information
        """
        if self.metadata is None:
            return {}
        
        info = {
            'target_type': self.metadata.get('target_type', 'unknown'),
            'training_date': self.metadata.get('training_date', 'unknown'),
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'loaded': self.is_model_loaded()
        }
        
        # Add training metrics if available
        if 'metrics' in self.metadata:
            info['metrics'] = self.metadata['metrics']
        
        return info
    
    def check_model_age(self, retrain_interval_days: int) -> bool:
        """
        Check if model needs retraining based on age.
        
        Args:
            retrain_interval_days: Maximum age in days
            
        Returns:
            True if model is outdated, False otherwise
        """
        if self.metadata is None:
            return True
        
        training_date_str = self.metadata.get('training_date')
        if not training_date_str:
            return True
        
        try:
            training_date = datetime.fromisoformat(training_date_str)
            age_days = (datetime.now() - training_date).days
            
            if age_days > retrain_interval_days:
                self.logger.warning(
                    f"Model is {age_days} days old (threshold: {retrain_interval_days})"
                )
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not parse training date: {str(e)}")
            return True

