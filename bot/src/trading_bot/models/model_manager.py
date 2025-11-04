"""
Model Manager Module

Manages model lifecycle, versioning, and selection.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from trading_bot.utils.exceptions import ModelError
from trading_bot.utils.paths import resolve_resource_path, get_writable_app_dir


class ModelManager:
    """
    Manages ML model files, versions, and metadata.
    
    Handles model discovery, selection, validation, and cleanup.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Setup model directories
        self.model_dirs = []
        
        # Check writable user directory
        try:
            user_model_dir = get_writable_app_dir('models')
            self.model_dirs.append(Path(user_model_dir))
            self.logger.info(f"User model directory: {user_model_dir}")
        except Exception as e:
            self.logger.warning(f"Could not access user model directory: {str(e)}")
        
        # Check bundled models directory
        try:
            bundled_model_dir = Path(resolve_resource_path('models'))
            if bundled_model_dir.exists():
                self.model_dirs.append(bundled_model_dir)
                self.logger.info(f"Bundled model directory: {bundled_model_dir}")
        except Exception as e:
            self.logger.warning(f"Could not access bundled model directory: {str(e)}")
        
        self.logger.info(f"Model manager initialized with {len(self.model_dirs)} directories")
    
    def list_available_models(self, model_type: str = 'xgboost') -> List[Dict]:
        """
        List all models in model directories.
        
        Args:
            model_type: Type of model to list (e.g., 'xgboost')
            
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        for model_dir in self.model_dirs:
            if not model_dir.exists():
                continue
            
            # Find model files
            pattern = f"{model_type}_*.json"
            for model_file in model_dir.glob(pattern):
                try:
                    # Parse model filename
                    model_info = self._parse_model_filename(model_file.name)
                    
                    if model_info:
                        model_info['path'] = str(model_file)
                        model_info['directory'] = str(model_dir)
                        
                        # Try to load metadata
                        metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                model_info['metadata'] = metadata
                        
                        models.append(model_info)
                        
                except Exception as e:
                    self.logger.warning(f"Could not process {model_file}: {str(e)}")
        
        # Sort by date (newest first)
        models.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        self.logger.info(f"Found {len(models)} {model_type} models")
        
        return models
    
    def get_latest_model(self, symbol: str = None, 
                        model_type: str = 'xgboost') -> Optional[Dict]:
        """
        Find most recent model for a symbol.
        
        Args:
            symbol: Stock symbol (optional, if None returns latest of any symbol)
            model_type: Type of model
            
        Returns:
            Dictionary with model information, or None if not found
        """
        models = self.list_available_models(model_type)
        
        if not models:
            self.logger.warning(f"No {model_type} models found")
            return None
        
        # Filter by symbol if specified
        if symbol:
            models = [m for m in models if m.get('symbol') == symbol]
            
            if not models:
                self.logger.warning(f"No models found for symbol {symbol}")
                return None
        
        # Return the first (most recent) model
        latest = models[0]
        self.logger.info(f"Latest model: {latest.get('filename', 'unknown')}")
        
        return latest
    
    def load_model(self, model_name: str) -> Dict:
        """
        Load specific model by name.
        
        Args:
            model_name: Model filename (with or without extension)
            
        Returns:
            Dictionary with model paths
            
        Raises:
            ModelError: If model not found
        """
        # Add .json extension if not present
        if not model_name.endswith('.json'):
            model_name += '.json'
        
        # Search in all model directories
        for model_dir in self.model_dirs:
            model_path = model_dir / model_name
            
            if model_path.exists():
                # Find associated files
                base_name = model_path.stem
                metadata_path = model_dir / f"{base_name}_metadata.json"
                
                # Load metadata first to get scaler path if available
                metadata = None
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        self.logger.warning(f"Could not load metadata: {str(e)}")
                else:
                    self.logger.warning(f"Metadata file not found for {model_name}")
                
                # Find scaler file - prefer metadata path, then fallback to glob
                scaler_path = None
                if metadata and 'scaler_file' in metadata:
                    scaler_path = Path(metadata['scaler_file'])
                    if not scaler_path.is_absolute():
                        # Make relative to metadata file location
                        scaler_path = model_dir / scaler_path
                    if not scaler_path.exists():
                        self.logger.warning(f"Scaler file from metadata not found: {scaler_path}")
                        scaler_path = None
                
                # Fallback to glob search if metadata path not available
                if scaler_path is None or not scaler_path.exists():
                    for scaler_file in model_dir.glob(f"*scaler*.pkl"):
                        if base_name in scaler_file.name or scaler_file.stem.startswith('scaler'):
                            scaler_path = scaler_file
                            break
                
                if not scaler_path or not scaler_path.exists():
                    raise ModelError(f"Scaler file not found for {model_name}")
                
                return {
                    'model_path': str(model_path),
                    'metadata_path': str(metadata_path) if metadata_path.exists() else None,
                    'scaler_path': str(scaler_path)
                }
        
        raise ModelError(f"Model not found: {model_name}")
    
    def validate_model(self, model_path: str) -> bool:
        """
        Check model file integrity.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            model_file = Path(model_path)
            
            # Check file exists
            if not model_file.exists():
                self.logger.error(f"Model file does not exist: {model_path}")
                return False
            
            # Check file size (should be > 0)
            if model_file.stat().st_size == 0:
                self.logger.error(f"Model file is empty: {model_path}")
                return False
            
            # Try to load as JSON (basic format check)
            try:
                with open(model_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                # XGBoost JSON format might not be standard JSON
                pass
            
            self.logger.debug(f"Model validation passed: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def get_model_info(self, model_path: str) -> Optional[Dict]:
        """
        Read metadata without loading full model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with model information, or None if not found
        """
        try:
            model_file = Path(model_path)
            metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
            
            if not metadata_file.exists():
                self.logger.warning(f"Metadata file not found: {metadata_file}")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add file info
            metadata['file_size'] = model_file.stat().st_size
            metadata['modified_date'] = datetime.fromtimestamp(
                model_file.stat().st_mtime
            ).isoformat()
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Could not read model info: {str(e)}")
            return None
    
    def delete_old_models(self, keep_latest_n: int = 3, 
                         model_type: str = 'xgboost') -> int:
        """
        Cleanup old model versions.
        
        Args:
            keep_latest_n: Number of recent models to keep
            model_type: Type of model
            
        Returns:
            Number of models deleted
        """
        models = self.list_available_models(model_type)
        
        if len(models) <= keep_latest_n:
            self.logger.info(f"Only {len(models)} models found, no cleanup needed")
            return 0
        
        # Delete models beyond keep_latest_n
        models_to_delete = models[keep_latest_n:]
        deleted_count = 0
        
        for model in models_to_delete:
            try:
                model_path = Path(model['path'])
                
                # Delete model file
                if model_path.exists():
                    model_path.unlink()
                    deleted_count += 1
                
                # Delete metadata file
                metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Delete scaler file
                base_name = model_path.stem
                for scaler_file in model_path.parent.glob(f"*{base_name}*scaler*.pkl"):
                    scaler_file.unlink()
                
                self.logger.info(f"Deleted old model: {model_path.name}")
                
            except Exception as e:
                self.logger.warning(f"Could not delete model: {str(e)}")
        
        self.logger.info(f"Deleted {deleted_count} old models")
        return deleted_count
    
    def _parse_model_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse model filename to extract information.
        
        Expected format: xgboost_{symbol}_{version}_{date}.json
        
        Args:
            filename: Model filename
            
        Returns:
            Dictionary with parsed information
        """
        # Remove extension
        name = filename.replace('.json', '')
        
        # Try to parse standard format
        # Pattern: model_type_symbol_version_date
        parts = name.split('_')
        
        if len(parts) < 2:
            return None
        
        info = {
            'filename': filename,
            'model_type': parts[0]
        }
        
        # Try to extract symbol, version, date
        if len(parts) >= 2:
            info['symbol'] = parts[1]
        
        if len(parts) >= 3:
            info['version'] = parts[2]
        
        if len(parts) >= 4:
            info['date'] = parts[3]
        
        return info
    
    def get_default_model(self) -> Optional[str]:
        """
        Get default model from configuration.
        
        Returns:
            Model filename, or None if not specified
        """
        model_file = self.config.get('models.xgboost.model_file')
        
        if model_file:
            self.logger.info(f"Default model from config: {model_file}")
            return model_file
        
        return None

