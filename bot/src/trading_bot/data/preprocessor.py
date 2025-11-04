"""
Data Preprocessing Module

Handles data preprocessing including scaling, train/test splitting, and target creation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import joblib
from pathlib import Path

from trading_bot.utils.exceptions import DataError, ValidationError


class DataPreprocessor:
    """
    Prepares data for ML model training and inference.
    
    Handles scaling, splitting, and target variable creation.
    Critical: Use same scaler for training and inference.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Load configuration
        self.test_size = config.get('models.training.test_size', 0.2)
        self.validation_size = config.get('models.training.validation_size', 0.1)
        self.random_state = config.get('models.training.random_state', 42)
        self.scaling_method = config.get('models.features.scaling_method', 'standard')
        self.target_type = config.get('models.xgboost.target_type', 'regression')
        
        self.scaler = None
        
        self.logger.info(f"Data preprocessor initialized with {self.scaling_method} scaling")
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                             target_column: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y).
        
        Args:
            features_df: DataFrame with all features
            target_column: Column name for target variable
            
        Returns:
            Tuple of (X, y)
        """
        # Create target based on target_type
        if self.target_type == 'regression':
            y = self.create_regression_target(features_df, target_column)
        else:
            y = self.create_classification_target(features_df, target_column)
        
        # Remove target and date columns from features
        exclude_cols = ['Date', target_column]
        if 'target' in features_df.columns:
            exclude_cols.append('target')
        
        X = features_df.drop(columns=[col for col in exclude_cols if col in features_df.columns])
        
        # Align X and y (remove rows where target is NaN)
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = None, 
                        shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                       pd.Series, pd.Series]:
        """
        Split data into training and test sets (time-series aware).
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Fraction of data for testing (default from config)
            shuffle: Whether to shuffle (should be False for time-series)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.test_size
        
        if shuffle:
            self.logger.warning("Shuffling disabled for time-series data")
            shuffle = False
        
        # Calculate split index
        split_idx = int(len(X) * (1 - test_size))
        
        # Split data (most recent data as test set)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        self.logger.info(f"Train/test split: train={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """
        Fit scaler on training data.
        
        Args:
            X_train: Training features
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValidationError(f"Unknown scaling method: {self.scaling_method}")
        
        self.scaler.fit(X_train)
        self.logger.info(f"Fitted {self.scaling_method} scaler on training data")
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted scaler to features.
        
        Args:
            X: Features to scale
            
        Returns:
            Scaled features as numpy array
            
        Raises:
            ValidationError: If scaler not fitted
        """
        if self.scaler is None:
            raise ValidationError("Scaler not fitted. Call fit_scaler() first.")
        
        X_scaled = self.scaler.transform(X)
        self.logger.debug(f"Transformed {len(X)} samples")
        
        return X_scaled
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse scaling transformation.
        
        Args:
            X_scaled: Scaled features
            
        Returns:
            Original scale features
            
        Raises:
            ValidationError: If scaler not fitted
        """
        if self.scaler is None:
            raise ValidationError("Scaler not fitted")
        
        X_original = self.scaler.inverse_transform(X_scaled)
        return X_original
    
    def save_scaler(self, filepath: str) -> None:
        """
        Serialize scaler to file.
        
        Args:
            filepath: Path to save scaler
            
        Raises:
            ValidationError: If scaler not fitted
        """
        if self.scaler is None:
            raise ValidationError("No scaler to save")
        
        joblib.dump(self.scaler, filepath)
        self.logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """
        Load pre-fitted scaler from file.
        
        Args:
            filepath: Path to scaler file
            
        Raises:
            DataError: If loading fails
        """
        try:
            self.scaler = joblib.load(filepath)
            self.logger.info(f"Scaler loaded from {filepath}")
        except Exception as e:
            raise DataError(f"Failed to load scaler: {str(e)}")
    
    def create_regression_target(self, df: pd.DataFrame, 
                                target_col: str = 'Close', 
                                shift: int = -1) -> pd.Series:
        """
        Create regression target (next day's price).
        
        Args:
            df: DataFrame with price data
            target_col: Column to predict
            shift: Number of periods ahead (-1 = next day)
            
        Returns:
            Target Series
        """
        # Shift prices to get next day's price
        target = df[target_col].shift(shift)
        
        self.logger.debug(f"Created regression target with shift={shift}")
        return target
    
    def create_classification_target(self, df: pd.DataFrame, 
                                    target_col: str = 'Close',
                                    threshold: float = 0.0) -> pd.Series:
        """
        Create binary classification target (up=1, down=0).
        
        Args:
            df: DataFrame with price data
            target_col: Column to use for direction
            threshold: Threshold for up/down classification
            
        Returns:
            Target Series with binary values
        """
        # Calculate next day return
        returns = df[target_col].pct_change().shift(-1)
        
        # Classify as up (1) or down (0)
        target = (returns > threshold).astype(int)
        
        self.logger.debug(f"Created classification target (threshold={threshold})")
        return target
    
    def create_multiclass_target(self, df: pd.DataFrame,
                                 target_col: str = 'Close',
                                 thresholds: Tuple[float, float] = (-0.02, 0.02)) -> pd.Series:
        """
        Create multi-class target (strong_down, down, neutral, up, strong_up).
        
        Args:
            df: DataFrame with price data
            target_col: Column to use
            thresholds: Tuple of (lower, upper) thresholds
            
        Returns:
            Target Series with class labels
        """
        # Calculate next day return
        returns = df[target_col].pct_change().shift(-1)
        
        lower_thresh, upper_thresh = thresholds
        
        # Classify into 5 categories
        target = pd.Series(index=df.index, dtype=int)
        target[returns < lower_thresh * 2] = 0  # strong_down
        target[(returns >= lower_thresh * 2) & (returns < lower_thresh)] = 1  # down
        target[(returns >= lower_thresh) & (returns <= upper_thresh)] = 2  # neutral
        target[(returns > upper_thresh) & (returns <= upper_thresh * 2)] = 3  # up
        target[returns > upper_thresh * 2] = 4  # strong_up
        
        self.logger.debug(f"Created multi-class target with {target.nunique()} classes")
        return target
    
    def validate_features_target(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate feature-target alignment and data quality.
        
        Args:
            X: Features
            y: Target
            
        Raises:
            ValidationError: If validation fails
        """
        # Check length alignment
        if len(X) != len(y):
            raise ValidationError(f"Feature-target length mismatch: X={len(X)}, y={len(y)}")
        
        # Check for NaN values
        X_nan_count = X.isnull().sum().sum()
        y_nan_count = y.isnull().sum()
        
        if X_nan_count > 0:
            self.logger.warning(f"Found {X_nan_count} NaN values in features")
        
        if y_nan_count > 0:
            self.logger.warning(f"Found {y_nan_count} NaN values in target")
        
        # Check for infinite values
        X_inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if X_inf_count > 0:
            self.logger.warning(f"Found {X_inf_count} infinite values in features")
        
        # Log data statistics
        self.logger.info(f"Data validation: X shape={X.shape}, y shape={y.shape}")
        
        if self.target_type == 'classification':
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")

