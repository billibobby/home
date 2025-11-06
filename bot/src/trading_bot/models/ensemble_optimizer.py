"""
Ensemble Optimizer Module

Optimizes ensemble of top-performing models with diversity weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.optimize import minimize
import optuna

from trading_bot.models.xgboost_trainer import XGBoostTrainer
from trading_bot.backtesting.metrics import PerformanceMetrics
from trading_bot.utils.exceptions import ModelError


class EnsembleOptimizer:
    """
    Optimizes ensemble of top-N models with diversity weighting.
    
    Trains multiple models from top trials and finds optimal weighted combination.
    """
    
    def __init__(self, study: optuna.Study, config, logger, n_models: int = 5):
        """
        Initialize ensemble optimizer.
        
        Args:
            study: Completed Optuna study
            config: Configuration object
            logger: Logger instance
            n_models: Number of top models to include in ensemble
        """
        self.study = study
        self.config = config
        self.logger = logger
        self.n_models = n_models
        
        # Storage
        self.models = []
        self.weights = None
        self.diversity_score = None
        
        # Get ensemble config
        ensemble_config = config.get('models.optimization.ensemble', {})
        self.diversity_weight = ensemble_config.get('diversity_weight', 0.1)
        
        self.logger.info(f"EnsembleOptimizer initialized with n_models={n_models}")
    
    def _get_top_trials(self, n: int) -> List[optuna.Trial]:
        """
        Extract top N trials from study.
        
        Args:
            n: Number of top trials to extract
            
        Returns:
            List of top trial objects
        """
        # Get completed trials sorted by value
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) == 0:
            raise ModelError("No completed trials in study")
        
        # Sort by value (best first)
        if self.study.direction == 'maximize':
            sorted_trials = sorted(completed_trials, key=lambda t: t.value if t.value is not None else -1e10, reverse=True)
        else:
            sorted_trials = sorted(completed_trials, key=lambda t: t.value if t.value is not None else 1e10)
        
        return sorted_trials[:n]
    
    def _train_models_from_trials(self, trials: List[optuna.Trial],
                                   X_train: pd.DataFrame, y_train: pd.Series) -> List[XGBoostTrainer]:
        """
        Train models with parameters from trials.
        
        Args:
            trials: List of trial objects
            X_train: Training features
            y_train: Training target
            
        Returns:
            List of trained XGBoostTrainer instances
        """
        models = []
        
        for i, trial in enumerate(trials):
            self.logger.info(f"Training model {i+1}/{len(trials)} from trial {trial.number}")
            
            try:
                trainer = XGBoostTrainer(self.config, self.logger)
                trainer.set_hyperparameters(trial.params)
                trainer.train(X_train, y_train)
                models.append(trainer)
            except Exception as e:
                self.logger.warning(f"Failed to train model from trial {trial.number}: {str(e)}")
                continue
        
        if len(models) == 0:
            raise ModelError("Failed to train any models from trials")
        
        return models
    
    def _calculate_diversity(self, models: List[XGBoostTrainer], X_val: pd.DataFrame) -> float:
        """
        Calculate diversity score between models using prediction correlation.
        
        Args:
            models: List of trained models
            X_val: Validation features
            
        Returns:
            Diversity score (0 = identical, 1 = completely different)
        """
        if len(models) < 2:
            return 0.0
        
        # Get predictions from all models
        predictions = []
        for model in models:
            try:
                pred = model.model.predict(X_val)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Failed to get predictions from model: {str(e)}")
                continue
        
        if len(predictions) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if len(correlations) == 0:
            return 0.0
        
        # Diversity = 1 - mean correlation
        mean_correlation = np.mean(correlations)
        diversity = 1 - abs(mean_correlation)
        
        return max(0.0, min(1.0, diversity))  # Clamp between 0 and 1
    
    def _optimize_weights(self, models: List[XGBoostTrainer], X_val: pd.DataFrame,
                         y_val: pd.Series) -> np.ndarray:
        """
        Find optimal weighted combination using scipy.optimize.
        
        Args:
            models: List of trained models
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Optimal weights array
        """
        # Get predictions from all models
        model_predictions = []
        for model in models:
            try:
                pred = model.model.predict(X_val)
                model_predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Failed to get predictions: {str(e)}")
                continue
        
        if len(model_predictions) == 0:
            raise ModelError("No valid predictions from models")
        
        model_predictions = np.array(model_predictions)  # Shape: (n_models, n_samples)
        
        # Calculate diversity
        diversity = self._calculate_diversity(models, X_val)
        
        # Create signal generator for converting predictions to signals
        from trading_bot.trading.signal_generator import SignalGenerator
        signal_generator = SignalGenerator(self.config, self.logger)
        
        # Create performance metrics calculator
        from trading_bot.backtesting.metrics import PerformanceMetrics
        metrics_calculator = PerformanceMetrics(self.config, self.logger)
        
        # Objective function: minimize negative Sharpe ratio (with diversity bonus)
        def objective(weights):
            # Normalize weights
            weights = weights / (weights.sum() + 1e-10)
            
            # Weighted prediction
            weighted_pred = np.dot(weights, model_predictions)
            
            # Convert predictions to signals and simulate trades
            initial_capital = 10000.0
            cash = initial_capital
            positions = {}
            trades = []
            equity_curve = [initial_capital]
            
            # Simulate trades using weighted predictions
            for i in range(len(X_val)):
                prediction = weighted_pred[i]
                current_price = y_val.iloc[i] if i < len(y_val) else prediction
                
                # Calculate confidence (simplified: use prediction variance or fixed value)
                confidence = 0.7  # Default confidence
                
                # Generate signal
                signal = signal_generator.generate_signal(
                    prediction, confidence, current_price, symbol='ENSEMBLE'
                )
                
                # Check if signal is actionable
                if not signal_generator.should_execute_signal(signal):
                    equity_curve.append(cash + sum(pos.get('quantity', 0) * current_price for pos in positions.values()))
                    continue
                
                # Simulate trade execution
                position_size_pct = self.config.get('trading.position_size_percentage', 10) / 100
                position_value = cash * position_size_pct
                
                if signal['type'] in ['BUY', 'STRONG_BUY'] and cash > position_value:
                    quantity = position_value / current_price
                    # Simple commission model
                    commission = position_value * 0.001  # 0.1% commission
                    cash -= position_value + commission
                    positions[i] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': i
                    }
                elif signal['type'] in ['SELL', 'STRONG_SELL'] and len(positions) > 0:
                    # Close all positions
                    for pos_idx, position in list(positions.items()):
                        quantity = position['quantity']
                        entry_price = position['entry_price']
                        commission = current_price * quantity * 0.001
                        pnl = (current_price - entry_price) * quantity - commission
                        cash += current_price * quantity - commission
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': i,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'quantity': quantity,
                            'pnl': pnl
                        })
                        del positions[pos_idx]
                
                # Update equity curve
                equity_curve.append(cash + sum(pos.get('quantity', 0) * current_price for pos in positions.values()))
            
            # Close remaining positions
            if len(positions) > 0 and len(X_val) > 0:
                final_price = y_val.iloc[-1] if len(y_val) > 0 else current_price
                for pos_idx, position in list(positions.items()):
                    quantity = position['quantity']
                    entry_price = position['entry_price']
                    commission = final_price * quantity * 0.001
                    pnl = (final_price - entry_price) * quantity - commission
                    cash += final_price * quantity - commission
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': len(X_val) - 1,
                        'entry_price': entry_price,
                        'exit_price': final_price,
                        'quantity': quantity,
                        'pnl': pnl
                    })
                    del positions[pos_idx]
            
            # Calculate Sharpe ratio from equity curve
            if len(equity_curve) > 1:
                equity_series = pd.Series(equity_curve)
                daily_returns = equity_series.pct_change().dropna()
                if len(daily_returns) > 0:
                    sharpe = metrics_calculator.calculate_sharpe_ratio(daily_returns)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
            
            # Negative Sharpe (we minimize, so negative) + diversity bonus
            score = -sharpe - self.diversity_weight * diversity
            
            return score
        
        # Constraints: weights sum to 1, all positive
        n_models = len(models)
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
            
            if result.success:
                weights = result.x
                weights = weights / (weights.sum() + 1e-10)  # Normalize
                return weights
            else:
                self.logger.warning("Optimization did not converge, using equal weights")
                return np.ones(n_models) / n_models
        except Exception as e:
            self.logger.warning(f"Weight optimization failed: {str(e)}, using equal weights")
            return np.ones(n_models) / n_models
    
    def optimize_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with ensemble info
        """
        self.logger.info(f"Optimizing ensemble with top {self.n_models} models")
        
        # Get top trials
        top_trials = self._get_top_trials(self.n_models)
        self.logger.info(f"Selected {len(top_trials)} top trials")
        
        # Train models
        self.models = self._train_models_from_trials(top_trials, X_train, y_train)
        self.logger.info(f"Trained {len(self.models)} models")
        
        # Calculate diversity
        self.diversity_score = self._calculate_diversity(self.models, X_val)
        self.logger.info(f"Diversity score: {self.diversity_score:.4f}")
        
        # Optimize weights
        self.weights = self._optimize_weights(self.models, X_val, y_val)
        self.logger.info(f"Optimized weights: {self.weights}")
        
        return self.get_ensemble_info()
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make weighted predictions using ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Weighted predictions array
        """
        if self.models is None or len(self.models) == 0:
            raise ModelError("Ensemble not trained. Call optimize_ensemble() first.")
        
        if self.weights is None:
            raise ModelError("Weights not optimized. Call optimize_ensemble() first.")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.model.predict(X)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Failed to get prediction from model: {str(e)}")
                continue
        
        if len(predictions) == 0:
            raise ModelError("No valid predictions from ensemble models")
        
        # Weighted average
        predictions_array = np.array(predictions)  # Shape: (n_models, n_samples)
        weighted_pred = np.dot(self.weights[:len(predictions_array)], predictions_array)
        
        return weighted_pred
    
    def get_ensemble_info(self) -> Dict:
        """
        Return dictionary with model params, weights, and diversity score.
        
        Returns:
            Dictionary with ensemble information
        """
        if self.models is None or len(self.models) == 0:
            return {
                'n_models': 0,
                'diversity_score': 0.0,
                'weights': [],
                'model_params': []
            }
        
        # Get model parameters from trials
        top_trials = self._get_top_trials(self.n_models)
        
        model_params = []
        for i, trial in enumerate(top_trials[:len(self.models)]):
            model_params.append({
                'trial_number': trial.number,
                'trial_value': trial.value,
                'params': trial.params.copy()
            })
        
        return {
            'n_models': len(self.models),
            'diversity_score': self.diversity_score if self.diversity_score is not None else 0.0,
            'weights': self.weights.tolist() if self.weights is not None else [],
            'model_params': model_params
        }
    
    def save_ensemble(self, path: str) -> None:
        """
        Save ensemble models and weights.
        
        Args:
            path: Directory path to save ensemble
        """
        if self.models is None or len(self.models) == 0:
            raise ModelError("No ensemble to save. Call optimize_ensemble() first.")
        
        import joblib
        from pathlib import Path as PathLib
        
        save_dir = PathLib(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for i, model in enumerate(self.models):
            model_path = save_dir / f"ensemble_model_{i}.json"
            model.save_model(str(model_path))
        
        # Save weights and info
        ensemble_data = {
            'weights': self.weights.tolist() if self.weights is not None else [],
            'diversity_score': self.diversity_score,
            'ensemble_info': self.get_ensemble_info()
        }
        
        weights_path = save_dir / "ensemble_weights.pkl"
        joblib.dump(ensemble_data, weights_path)
        
        self.logger.info(f"Ensemble saved to {path}")

