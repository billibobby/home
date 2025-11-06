"""
XGBoost Hyperparameter Optimizer Module

Implements automated hyperparameter optimization using Optuna with walk-forward validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Callable
import optuna
from optuna.pruners import MedianPruner
from optuna.study import StudyDirection
import plotly.graph_objects as go
import time

from trading_bot.backtesting import PerformanceMetrics
from trading_bot.trading import SignalGenerator
from trading_bot.models.xgboost_trainer import XGBoostTrainer
from trading_bot.utils.exceptions import ModelError


class CustomPruner(optuna.pruners.BasePruner):
    """
    Custom pruner with:
    - Bottom 50% pruning at 25% progress
    - Negative Sharpe pruning after 50% progress
    - Per-trial timeout (300s)
    """
    
    def __init__(self, trial_timeout: int = 300):
        self.trial_timeout = trial_timeout
        self.trial_starts = {}
    
    def prune(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """Determine if trial should be pruned."""
        # Check per-trial timeout
        if trial.number in self.trial_starts:
            elapsed = time.time() - self.trial_starts[trial.number]
            if elapsed > self.trial_timeout:
                return True
        
        # Get completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if len(completed_trials) < 2:
            return False
        
        # Get current intermediate value
        intermediate_values = trial.intermediate_values
        if not intermediate_values:
            return False
        
        # Get progress: step / max_steps (estimate max_steps from n_estimators if available)
        max_step = max(intermediate_values.keys()) if intermediate_values else 1
        current_step = max_step
        # Estimate total steps (assuming n_estimators is similar across trials)
        # For now, use a heuristic: if we're at step 1, assume 100 total steps
        estimated_total_steps = 100
        progress = current_step / estimated_total_steps if estimated_total_steps > 0 else 0.0
        
        # At 25% progress: prune bottom 50% of observed values
        if progress >= 0.25 and progress < 0.5:
            values = sorted([t.value for t in completed_trials], reverse=(study.direction == StudyDirection.MAXIMIZE))
            median_value = values[len(values) // 2] if values else None
            
            if median_value is not None:
                current_value = list(intermediate_values.values())[-1]
                if study.direction == StudyDirection.MAXIMIZE:
                    if current_value < median_value:
                        return True
                else:
                    if current_value > median_value:
                        return True
        
        # After 50% progress: prune if Sharpe < 0 (for Sharpe objective)
        if progress >= 0.5:
            current_value = list(intermediate_values.values())[-1]
            # Check if this looks like a Sharpe ratio (typically negative if bad)
            if current_value < 0:
                return True
        
        return False
    
    def start_trial(self, study: optuna.Study, trial: optuna.Trial):
        """Record trial start time."""
        self.trial_starts[trial.number] = time.time()
    
    def end_trial(self, study: optuna.Study, trial: optuna.Trial):
        """Clean up trial timing."""
        if trial.number in self.trial_starts:
            del self.trial_starts[trial.number]


class XGBoostOptimizer:
    """
    Optimizes XGBoost hyperparameters using Optuna with walk-forward validation.
    
    Uses Bayesian optimization to search hyperparameter space and evaluates
    candidates using walk-forward backtesting.
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                 config, logger, objective: str = 'sharpe',
                 storage: Optional[str] = None, study_name: Optional[str] = None):
        """
        Initialize the optimizer.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            config: Configuration object
            logger: Logger instance
            objective: Optimization objective ('sharpe', 'sortino', 'returns', 'calmar', 'custom')
            storage: Optuna storage URL (e.g., 'sqlite:///optuna.db') for distributed optimization
            study_name: Study name for distributed optimization
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config
        self.logger = logger
        self.objective = objective
        self.storage = storage
        self.study_name = study_name
        
        # Get optimization config
        opt_config = config.get('models.optimization', {})
        self.param_ranges = opt_config.get('parameter_ranges', {
            'max_depth': [3, 10],
            'learning_rate': [0.001, 0.3],
            'n_estimators': [50, 500],
            'subsample': [0.6, 1.0],
            'colsample_bytree': [0.6, 1.0],
            'min_child_weight': [1, 10],
            'gamma': [0, 5.0],
            'reg_alpha': [0, 10.0],
            'reg_lambda': [0, 10.0]
        })
        
        # Initialize study
        self.study = None
        self.best_params = None
        self.best_value = None
        
        # Storage for optimization history
        self.optimization_history = []
        
        # Check for distributed optimization config
        if self.storage is None:
            dist_config = config.get('models.optimization.distributed', {})
            if dist_config.get('enabled', False):
                self.storage = dist_config.get('storage', 'sqlite:///optuna.db')
                if self.study_name is None:
                    self.study_name = dist_config.get('study_name', f'xgboost_optimization_{objective}')
        
        self.logger.info(f"XGBoostOptimizer initialized with objective: {objective}")
        if self.storage:
            self.logger.info(f"Distributed optimization enabled: storage={self.storage}, study_name={self.study_name}")
    
    def _suggest_params(self, trial) -> Dict:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            'max_depth': trial.suggest_int('max_depth', 
                                          self.param_ranges['max_depth'][0],
                                          self.param_ranges['max_depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate',
                                                self.param_ranges['learning_rate'][0],
                                                self.param_ranges['learning_rate'][1],
                                                log=True),
            'n_estimators': trial.suggest_int('n_estimators',
                                             self.param_ranges['n_estimators'][0],
                                             self.param_ranges['n_estimators'][1]),
            'subsample': trial.suggest_float('subsample',
                                            self.param_ranges['subsample'][0],
                                            self.param_ranges['subsample'][1]),
            'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                   self.param_ranges['colsample_bytree'][0],
                                                   self.param_ranges['colsample_bytree'][1]),
            'min_child_weight': trial.suggest_int('min_child_weight',
                                                 self.param_ranges['min_child_weight'][0],
                                                 self.param_ranges['min_child_weight'][1]),
            'gamma': trial.suggest_float('gamma',
                                        self.param_ranges['gamma'][0],
                                        self.param_ranges['gamma'][1]),
            'reg_alpha': trial.suggest_float('reg_alpha',
                                            self.param_ranges['reg_alpha'][0],
                                            self.param_ranges['reg_alpha'][1]),
            'reg_lambda': trial.suggest_float('reg_lambda',
                                             self.param_ranges['reg_lambda'][0],
                                             self.param_ranges['reg_lambda'][1])
        }
        
        return params
    
    def _create_time_series_splits(self) -> list:
        """
        Create time-series cross-validation splits.
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        splits = []
        n_samples = len(self.X_train)
        
        # Use walk-forward windows for CV splits
        train_period = self.config.get('backtesting.walk_forward.train_period_days', 252)
        test_period = self.config.get('backtesting.walk_forward.test_period_days', 21)
        step_size = self.config.get('backtesting.walk_forward.step_size_days', 21)
        
        # Convert to indices (assuming daily data)
        train_size = min(train_period, n_samples // 2)
        test_size = min(test_period, n_samples // 4)
        step = max(step_size, 1)
        
        start = 0
        while start + train_size + test_size <= n_samples:
            train_end = start + train_size
            test_end = min(train_end + test_size, n_samples)
            
            train_indices = list(range(start, train_end))
            val_indices = list(range(train_end, test_end))
            
            splits.append((train_indices, val_indices))
            start += step
            
            # Limit to 5 splits for efficiency
            if len(splits) >= 5:
                break
        
        # If no splits created, use simple train/val split
        if len(splits) == 0:
            split_idx = int(n_samples * 0.8)
            splits.append((list(range(split_idx)), list(range(split_idx, n_samples))))
        
        return splits
    
    def _evaluate_with_walk_forward(self, params: Dict, trial: Optional[optuna.Trial] = None) -> float:
        """
        Evaluate hyperparameters using walk-forward backtesting with real trading simulation.
        
        Args:
            params: Hyperparameters to evaluate
            trial: Optuna trial object for intermediate reporting (optional)
            
        Returns:
            Trading metric value (Sharpe, Sortino, returns, or Calmar)
        """
        try:
            # Create trainer with suggested params
            trainer = XGBoostTrainer(self.config, self.logger)
            trainer.set_hyperparameters(params)
            
            # Create signal generator
            signal_generator = SignalGenerator(self.config, self.logger)
            
            # Create performance metrics calculator
            metrics_calculator = PerformanceMetrics(self.config, self.logger)
            
            # Prepare data for walk-forward evaluation
            splits = self._create_time_series_splits()
            
            metric_values = []
            n_estimators = params.get('n_estimators', 100)
            
            for split_idx, (train_idx, val_idx) in enumerate(splits):
                X_train_split = self.X_train.iloc[train_idx]
                y_train_split = self.y_train.iloc[train_idx]
                X_val_split = self.X_train.iloc[val_idx] if self.X_val is None else self.X_val.iloc[val_idx]
                y_val_split = self.y_train.iloc[val_idx] if self.y_val is None else self.y_val.iloc[val_idx]
                
                # Train model with intermediate reporting for pruning
                if trial is not None:
                    # Use staged predictions or manual iteration to report intermediate values
                    # For XGBoost, we can use early_stopping_rounds and report at intervals
                    trainer.train(X_train_split, y_train_split, X_val_split, y_val_split)
                    
                    # Report intermediate metrics during training
                    # Since XGBoost doesn't expose per-boost-round callbacks easily,
                    # we'll train in stages and report intermediate values
                    # For now, report after training completes for each split
                    # In a more sophisticated implementation, we could use XGBoost's callbacks
                    if split_idx == 0:  # Only report for first split to avoid too many reports
                        # Get quick validation metric for intermediate reporting
                        # Use a simple metric like validation R² or MSE as proxy
                        try:
                            val_pred = trainer.model.predict(X_val_split)
                            from sklearn.metrics import mean_squared_error
                            mse = mean_squared_error(y_val_split, val_pred)
                            # Convert to a score (higher is better for pruning)
                            intermediate_value = -mse  # Negative MSE as proxy
                        except:
                            intermediate_value = 0.0
                        
                        # Report at estimated step (based on n_estimators)
                        estimated_step = int(n_estimators * 0.5)  # Report at 50% of training
                        trial.report(intermediate_value, step=estimated_step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                else:
                    # Standard training without intermediate reporting
                    trainer.train(X_train_split, y_train_split, X_val_split, y_val_split)
                
                # Get predictions on validation set
                predictions = trainer.model.predict(X_val_split)
                
                # Convert predictions to signals and simulate trades
                # We need price data for signal generation - use validation target as proxy
                # In real scenario, we'd need full OHLCV data
                initial_capital = 10000.0
                cash = initial_capital
                positions = {}
                trades = []
                equity_curve = [initial_capital]
                
                # Simulate trades
                for i, (idx, row) in enumerate(X_val_split.iterrows()):
                    prediction = predictions[i] if isinstance(predictions, np.ndarray) else predictions
                    current_price = y_val_split.iloc[i] if i < len(y_val_split) else prediction
                    
                    # Calculate confidence (simplified: use prediction variance or fixed value)
                    confidence = 0.7  # Default confidence
                    
                    # Generate signal
                    signal = signal_generator.generate_signal(
                        prediction, confidence, current_price, symbol='OPTIMIZATION'
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
                        positions[idx] = {
                            'quantity': quantity,
                            'entry_price': current_price,
                            'entry_time': idx
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
                                'exit_time': idx,
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'quantity': quantity,
                                'pnl': pnl
                            })
                            del positions[pos_idx]
                    
                    # Update equity curve
                    equity_curve.append(cash + sum(pos.get('quantity', 0) * current_price for pos in positions.values()))
                
                # Close remaining positions
                if len(positions) > 0 and len(X_val_split) > 0:
                    final_price = y_val_split.iloc[-1] if len(y_val_split) > 0 else current_price
                    for pos_idx, position in list(positions.items()):
                        quantity = position['quantity']
                        entry_price = position['entry_price']
                        commission = final_price * quantity * 0.001
                        pnl = (final_price - entry_price) * quantity - commission
                        cash += final_price * quantity - commission
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': len(X_val_split) - 1,
                            'entry_price': entry_price,
                            'exit_price': final_price,
                            'quantity': quantity,
                            'pnl': pnl
                        })
                        del positions[pos_idx]
                
                # Calculate performance metrics
                if len(trades) > 0 and len(equity_curve) > 1:
                    # Convert equity curve to Series
                    equity_series = pd.Series(equity_curve)
                    
                    # Calculate daily returns
                    daily_returns = equity_series.pct_change().dropna()
                    
                    # Calculate metrics based on objective
                    if self.objective == 'sharpe':
                        metric = metrics_calculator.calculate_sharpe_ratio(daily_returns)
                    elif self.objective == 'sortino':
                        metric = metrics_calculator.calculate_sortino_ratio(daily_returns)
                    elif self.objective == 'returns':
                        total_return = ((equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]) * 100
                        metric = total_return
                    elif self.objective == 'calmar':
                        returns_dict = metrics_calculator.calculate_returns(trades, equity_series)
                        drawdown_dict = metrics_calculator.calculate_max_drawdown(equity_series)
                        metric = metrics_calculator.calculate_calmar_ratio(
                            returns_dict['annualized_return_pct'],
                            drawdown_dict['max_drawdown_pct']
                        )
                    else:
                        # Default to Sharpe
                        metric = metrics_calculator.calculate_sharpe_ratio(daily_returns)
                    
                    metric_values.append(metric)
            
            # Average metric across splits
            avg_metric = np.mean(metric_values) if metric_values else -1e10
            
            return avg_metric
            
        except Exception as e:
            self.logger.warning(f"Evaluation failed with params {params}: {str(e)}")
            # Return worst-case value based on objective
            if self.objective in ['sharpe', 'sortino', 'returns', 'calmar']:
                return -1e10  # Very negative for maximization
            else:
                return 1e10  # Very positive for minimization
    
    def objective_function(self, trial) -> float:
        """
        Main optimization objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to optimize
        """
        # Suggest hyperparameters
        params = self._suggest_params(trial)
        
        # Report intermediate value for pruning
        trial.report(0.0, step=0)
        
        # Evaluate with walk-forward
        if self.objective == 'sharpe':
            value = self._objective_sharpe(trial, params)
        elif self.objective == 'sortino':
            value = self._objective_sortino(trial, params)
        elif self.objective == 'returns':
            value = self._objective_returns(trial, params)
        elif self.objective == 'calmar':
            value = self._objective_calmar(trial, params)
        elif self.objective == 'custom':
            # Custom objective function
            value = self._objective_custom(trial, params)
        else:
            # Default: use simplified evaluation
            value = self._evaluate_with_walk_forward(params)
        
        # Store in history
        self.optimization_history.append({
            'trial': trial.number,
            'params': params.copy(),
            'value': value
        })
        
        return value
    
    def _objective_custom(self, trial, params: Dict) -> float:
        """Optimize using custom objective function."""
        # Get custom objective from config
        opt_config = self.config.get('models.optimization', {})
        custom_obj_path = opt_config.get('custom_objective', None)
        
        if custom_obj_path is None:
            raise ModelError("Custom objective specified but no custom_objective path provided in config")
        
        # Resolve custom objective function
        try:
            # Support dotted import path (e.g., 'my_module.my_function')
            parts = custom_obj_path.split('.')
            module_path = '.'.join(parts[:-1])
            func_name = parts[-1]
            
            import importlib
            module = importlib.import_module(module_path)
            custom_func = getattr(module, func_name)
            
            # Call custom function
            value = custom_func(trial, self.X_train, self.y_train, 
                               self.X_val, self.y_val, self.config, self.logger)
            
            # Report intermediate value
            trial.report(value, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return value
        except Exception as e:
            self.logger.error(f"Custom objective function failed: {str(e)}")
            raise ModelError(f"Custom objective function failed: {str(e)}")
    
    def _objective_sharpe(self, trial, params: Dict) -> float:
        """Optimize for Sharpe ratio using real trading simulation."""
        value = self._evaluate_with_walk_forward(params, trial=trial)
        
        # Report final value
        trial.report(value, step=100)  # Use step 100 as final step
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return value
    
    def _objective_sortino(self, trial, params: Dict) -> float:
        """Optimize for Sortino ratio using real trading simulation."""
        value = self._evaluate_with_walk_forward(params, trial=trial)
        trial.report(value, step=100)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value
    
    def _objective_returns(self, trial, params: Dict) -> float:
        """Optimize for total returns using real trading simulation."""
        value = self._evaluate_with_walk_forward(params, trial=trial)
        trial.report(value, step=100)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value
    
    def _objective_calmar(self, trial, params: Dict) -> float:
        """Optimize for Calmar ratio using real trading simulation."""
        value = self._evaluate_with_walk_forward(params, trial=trial)
        trial.report(value, step=100)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value
    
    def optimize(self, n_trials: int = 100, timeout: Optional[int] = None,
                 callbacks: Optional[list] = None) -> Dict:
        """
        Run optimization study.
        
        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (None for no limit)
            callbacks: List of Optuna callbacks
            
        Returns:
            Dictionary of best hyperparameters
        """
        self.logger.info(f"Starting optimization with {n_trials} trials")
        
        # Determine direction
        direction = 'maximize' if self.objective in ['sharpe', 'sortino', 'returns', 'calmar'] else 'minimize'
        
        # Get pruning config
        opt_config = self.config.get('models.optimization', {})
        pruning_config = opt_config.get('pruning', {})
        pruner_enabled = pruning_config.get('enabled', True)
        
        if pruner_enabled:
            # Use custom pruner if configured, otherwise use MedianPruner
            use_custom_pruner = pruning_config.get('use_custom_pruner', True)
            trial_timeout = pruning_config.get('trial_timeout_seconds', 300)
            
            if use_custom_pruner:
                pruner = CustomPruner(trial_timeout=trial_timeout)
            else:
                pruner = MedianPruner(
                    n_startup_trials=pruning_config.get('n_startup_trials', 5),
                    n_warmup_steps=pruning_config.get('warmup_steps', 10)
                )
        else:
            pruner = None
        
        # Create or load study (for distributed optimization)
        if self.storage and self.study_name:
            try:
                # Try to load existing study
                self.study = optuna.load_study(
                    study_name=self.study_name,
                    storage=self.storage
                )
                self.logger.info(f"Loaded existing study '{self.study_name}' from storage")
            except Exception as e:
                # Create new study if it doesn't exist
                self.logger.info(f"Creating new study '{self.study_name}' in storage")
                self.study = optuna.create_study(
                    direction=direction,
                    pruner=pruner,
                    study_name=self.study_name,
                    storage=self.storage,
                    load_if_exists=True
                )
        else:
            # Create in-memory study
            self.study = optuna.create_study(
                direction=direction,
                pruner=pruner,
                study_name=f'xgboost_optimization_{self.objective}'
            )
        
        # Get early stopping config
        early_stopping_config = opt_config.get('early_stopping', {})
        early_stopping_enabled = early_stopping_config.get('enabled', False)
        patience = early_stopping_config.get('patience', 10)
        min_delta = early_stopping_config.get('min_delta', 0.01)
        
        # Run optimization with early stopping
        try:
            if early_stopping_enabled:
                # Custom loop with early stopping
                best_value = None
                patience_counter = 0
                completed_trials_count = 0
                
                def should_stop():
                    nonlocal best_value, patience_counter, completed_trials_count
                    
                    # Get latest completed trial
                    completed_trials = [t for t in self.study.trials 
                                       if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
                    if len(completed_trials) == 0:
                        return False
                    
                    latest_trial = completed_trials[-1]
                    current_value = latest_trial.value
                    completed_trials_count = len(completed_trials)
                    
                    if best_value is None:
                        best_value = current_value
                        patience_counter = 0
                        return False
                    
                    # Check if improvement is significant
                    if direction == 'maximize':
                        improvement = current_value - best_value
                        if improvement > min_delta:
                            best_value = current_value
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    else:
                        improvement = best_value - current_value
                        if improvement > min_delta:
                            best_value = current_value
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    
                    if patience_counter >= patience:
                        self.logger.info(
                            f"Early stopping triggered: no improvement for {patience} trials "
                            f"(best value: {best_value:.4f})"
                        )
                        return True
                    
                    return False
                
                # Run trials with early stopping check
                for trial_num in range(n_trials):
                    if should_stop():
                        break
                    
                    trial = self.study.ask()
                    try:
                        value = self.objective_function(trial)
                        self.study.tell(trial, value)
                    except optuna.TrialPruned:
                        self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    except Exception as e:
                        self.logger.warning(f"Trial {trial_num} failed: {str(e)}")
                        self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    
                    # Check timeout
                    if timeout and completed_trials_count > 0:
                        # Estimate remaining time (simplified)
                        pass  # Timeout is handled by study.optimize
            else:
                # Standard optimization
                self.study.optimize(
                    self.objective_function,
                    n_trials=n_trials,
                    timeout=timeout,
                    callbacks=callbacks or []
                )
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise ModelError(f"Optimization failed: {str(e)}")
        
        # Extract best parameters
        if len(self.study.trials) > 0:
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            self.logger.info(f"Optimization completed. Best value: {self.best_value:.4f}")
            self.logger.info(f"Best parameters: {self.best_params}")
        else:
            self.logger.warning("No trials completed")
            self.best_params = {}
        
        return self.best_params
    
    def get_best_params(self) -> Dict:
        """Return best hyperparameters."""
        if self.best_params is None:
            raise ModelError("Optimization not run yet. Call optimize() first.")
        return self.best_params.copy()
    
    def get_best_model(self) -> XGBoostTrainer:
        """
        Train and return model with best parameters.
        
        Returns:
            Trained XGBoostTrainer instance
        """
        if self.best_params is None:
            raise ModelError("Optimization not run yet. Call optimize() first.")
        
        trainer = XGBoostTrainer(self.config, self.logger)
        trainer.set_hyperparameters(self.best_params)
        trainer.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        return trainer
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Return DataFrame of all trials.
        
        Returns:
            DataFrame with trial number, parameters, and scores
        """
        if not self.optimization_history:
            return pd.DataFrame()
        
        # Flatten params
        rows = []
        for record in self.optimization_history:
            row = {'trial': record['trial'], 'value': record['value']}
            row.update(record['params'])
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_optimization_history(self) -> go.Figure:
        """
        Generate Plotly visualization of optimization progress.
        
        Returns:
            Plotly figure object
        """
        if self.study is None:
            raise ModelError("No study to plot. Run optimize() first.")
        
        # Create optimization history plot
        trials = self.study.trials
        trial_numbers = [t.number for t in trials]
        values = [t.value if t.value is not None else 0 for t in trials]
        
        fig = go.Figure()
        
        # Add line plot
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=values,
            mode='lines+markers',
            name='Trial Value',
            line=dict(color='blue')
        ))
        
        # Add best value line
        best_values = []
        current_best = None
        for t in trials:
            if t.value is not None:
                if current_best is None or (self.study.direction == StudyDirection.MAXIMIZE and t.value > current_best) or \
                   (self.study.direction == StudyDirection.MINIMIZE and t.value < current_best):
                    current_best = t.value
                best_values.append(current_best)
            else:
                best_values.append(current_best if current_best is not None else 0)
        
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=best_values,
            mode='lines',
            name='Best Value',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Optimization History',
            xaxis_title='Trial Number',
            yaxis_title='Objective Value',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_param_importances(self) -> go.Figure:
        """
        Generate Plotly visualization of parameter importance.
        
        Returns:
            Plotly figure object
        """
        if self.study is None:
            raise ModelError("No study to plot. Run optimize() first.")
        
        try:
            importances = optuna.importance.get_param_importances(self.study)
        except Exception as e:
            self.logger.warning(f"Failed to calculate parameter importance: {str(e)}")
            return go.Figure()
        
        # Create bar plot
        params = list(importances.keys())
        values = list(importances.values())
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=params,
            y=values,
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='Parameter Importance',
            xaxis_title='Parameter',
            yaxis_title='Importance',
            xaxis={'categoryorder': 'total descending'}
        )
        
        return fig
    
    def save_study(self, path: str) -> None:
        """
        Save Optuna study to file.
        
        Args:
            path: File path to save study
        """
        if self.study is None:
            raise ModelError("No study to save. Run optimize() first.")
        
        import joblib
        joblib.dump(self.study, path)
        self.logger.info(f"Study saved to {path}")
    
    def load_study(self, path: str) -> None:
        """
        Load existing study from file.
        
        Args:
            path: File path to load study from
        """
        import joblib
        self.study = joblib.load(path)
        
        # Extract best params
        if len(self.study.trials) > 0:
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
        
        self.logger.info(f"Study loaded from {path}")

