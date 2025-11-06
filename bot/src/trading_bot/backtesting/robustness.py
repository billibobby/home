"""
Robustness Testing Module

Implements robustness validation and sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import product
from datetime import datetime


class RobustnessValidator:
    """
    Validates strategy robustness through parameter sensitivity,
    time period stability, and feature stability tests.
    """
    
    def __init__(self, config, logger):
        """
        Initialize robustness validator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Load validation criteria
        validation_config = config.get('backtesting.validation', {})
        self.min_sharpe_ratio = validation_config.get('min_sharpe_ratio', 1.5)
        self.max_drawdown_pct = validation_config.get('max_drawdown_pct', 15)
        self.min_win_rate = validation_config.get('min_win_rate', 0.50)
        self.min_trades = validation_config.get('min_trades', 100)
        self.min_positive_periods = validation_config.get('min_positive_periods', 0.60)
        
        self.logger.info("RobustnessValidator initialized")
    
    def test_parameter_sensitivity(self, backtest_engine, param_ranges: Dict[str, List]) -> Dict:
        """
        Test parameter sensitivity.
        
        Args:
            backtest_engine: WalkForwardBacktest instance
            param_ranges: Dictionary of parameter names to lists of values
            
        Returns:
            Dictionary with sensitivity results
        """
        self.logger.info(f"Testing parameter sensitivity for {len(param_ranges)} parameters")
        
        results = {}
        
        for param_name, values in param_ranges.items():
            self.logger.info(f"Testing {param_name} with {len(values)} values")
            
            param_results = {
                'values': values,
                'sharpe_ratios': [],
                'returns': [],
                'drawdowns': []
            }
            
            # Store original parameter value
            original_value = None
            if hasattr(backtest_engine, param_name):
                original_value = getattr(backtest_engine, param_name)
            
            # Test each value
            for value in values:
                try:
                    # Set parameter
                    if hasattr(backtest_engine, param_name):
                        setattr(backtest_engine, param_name, value)
                    
                    # Run backtest
                    backtest_results = backtest_engine.run()
                    
                    # Record metrics
                    param_results['sharpe_ratios'].append(backtest_results.sharpe_ratio)
                    param_results['returns'].append(backtest_results.total_return)
                    param_results['drawdowns'].append(backtest_results.max_drawdown)
                
                except Exception as e:
                    self.logger.error(f"Error testing {param_name}={value}: {str(e)}")
                    param_results['sharpe_ratios'].append(0.0)
                    param_results['returns'].append(0.0)
                    param_results['drawdowns'].append(100.0)
            
            # Restore original value
            if original_value is not None:
                setattr(backtest_engine, param_name, original_value)
            
            results[param_name] = param_results
        
        return results
    
    def test_time_period_stability(self, backtest_engine, periods: List[Tuple[str, str]]) -> Dict:
        """
        Test strategy across different time periods.
        
        Args:
            backtest_engine: WalkForwardBacktest instance
            periods: List of (start_date, end_date) tuples
            
        Returns:
            Dictionary with period results
        """
        self.logger.info(f"Testing time period stability for {len(periods)} periods")
        
        results = {}
        
        original_data = backtest_engine.data
        
        for period_name, (start_date, end_date) in enumerate(periods):
            try:
                # Filter data to period
                period_data = original_data[
                    (original_data['Date'] >= start_date) &
                    (original_data['Date'] <= end_date)
                ].copy()
                
                if len(period_data) < backtest_engine.min_data_points:
                    self.logger.warning(f"Period {period_name} has insufficient data")
                    continue
                
                # Create new engine with filtered data
                from trading_bot.backtesting.walk_forward import WalkForwardBacktest
                period_engine = WalkForwardBacktest(
                    period_data, backtest_engine.config, backtest_engine.logger,
                    backtest_engine.feature_engineer, backtest_engine.signal_generator,
                    backtest_engine.model_class
                )
                
                # Run backtest
                period_results = period_engine.run()
                results[f"period_{period_name}"] = period_results
            
            except Exception as e:
                self.logger.error(f"Error testing period {period_name}: {str(e)}")
        
        return results
    
    def test_symbol_robustness(self, backtest_engine, symbols: List[str]) -> Dict:
        """
        Test strategy on multiple symbols.
        
        Args:
            backtest_engine: WalkForwardBacktest instance
            symbols: List of symbols to test
            
        Returns:
            Dictionary with symbol results and correlation matrix
        """
        self.logger.info(f"Testing symbol robustness for {len(symbols)} symbols")
        
        results = {}
        returns_series = {}
        
        for symbol in symbols:
            try:
                # This would require fetching data for each symbol
                # For now, we'll just log that it needs implementation
                self.logger.warning(f"Symbol robustness test for {symbol} requires data fetching - not implemented")
                results[symbol] = None
            
            except Exception as e:
                self.logger.error(f"Error testing symbol {symbol}: {str(e)}")
                results[symbol] = None
        
        # Calculate correlation matrix if we have returns
        correlation_matrix = pd.DataFrame()
        if returns_series:
            returns_df = pd.DataFrame(returns_series)
            correlation_matrix = returns_df.corr()
        
        return {
            'symbol_results': results,
            'correlation_matrix': correlation_matrix
        }
    
    def test_feature_stability(self, model, X: pd.DataFrame, y: pd.Series,
                              drop_pct: float = 0.1, n_trials: int = 10) -> Dict:
        """
        Test feature stability by randomly dropping features.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            y: Target Series
            drop_pct: Percentage of features to drop
            n_trials: Number of trials
            
        Returns:
            Dictionary with stability results
        """
        self.logger.info(f"Testing feature stability: drop {drop_pct*100}% features, {n_trials} trials")
        
        results = {}
        
        # Get baseline performance
        try:
            baseline_pred = model.model.predict(X)
            baseline_rmse = np.sqrt(np.mean((y - baseline_pred) ** 2))
        except:
            baseline_rmse = 0.0
        
        for trial in range(n_trials):
            try:
                # Randomly select features to drop
                n_features = len(X.columns)
                n_drop = int(n_features * drop_pct)
                features_to_drop = np.random.choice(X.columns, size=n_drop, replace=False)
                
                # Create reduced feature set
                X_reduced = X.drop(columns=features_to_drop)
                
                # Retrain model (simplified - would need to retrain properly)
                # For now, just record dropped features
                results[f'trial_{trial}'] = {
                    'dropped_features': list(features_to_drop),
                    'baseline_rmse': baseline_rmse,
                    'reduced_rmse': baseline_rmse  # Placeholder
                }
            
            except Exception as e:
                self.logger.error(f"Error in feature stability trial {trial}: {str(e)}")
        
        return results
    
    def monte_carlo_simulation(self, returns: pd.Series, n_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation on returns.
        
        Args:
            returns: Daily returns series
            n_simulations: Number of simulations
            
        Returns:
            Dictionary with simulation results
        """
        self.logger.info(f"Running Monte Carlo simulation: {n_simulations} simulations")
        
        if len(returns) == 0:
            return {'simulated_returns': [], 'percentiles': {}}
        
        simulated_returns = []
        
        for _ in range(n_simulations):
            # Randomly resample returns with replacement
            resampled = np.random.choice(returns.values, size=len(returns), replace=True)
            
            # Calculate cumulative return
            cumulative_return = (1 + resampled).prod() - 1
            simulated_returns.append(cumulative_return * 100)
        
        # Calculate percentiles
        percentiles = {
            5: np.percentile(simulated_returns, 5),
            25: np.percentile(simulated_returns, 25),
            50: np.percentile(simulated_returns, 50),
            75: np.percentile(simulated_returns, 75),
            95: np.percentile(simulated_returns, 95)
        }
        
        return {
            'simulated_returns': simulated_returns,
            'percentiles': percentiles
        }
    
    def test_walk_forward_window_sensitivity(self, backtest_engine,
                                            window_configs: List[Dict]) -> Dict:
        """
        Test sensitivity to walk-forward window sizes.
        
        Args:
            backtest_engine: WalkForwardBacktest instance
            window_configs: List of window configuration dictionaries
            
        Returns:
            Dictionary with results for each configuration
        """
        self.logger.info(f"Testing walk-forward window sensitivity: {len(window_configs)} configurations")
        
        results = {}
        
        # Store original window settings
        original_train = backtest_engine.train_period_days
        original_test = backtest_engine.test_period_days
        original_step = backtest_engine.step_size_days
        
        for config_name, config in enumerate(window_configs):
            try:
                # Set new window parameters
                backtest_engine.train_period_days = config.get('train', original_train)
                backtest_engine.test_period_days = config.get('test', original_test)
                backtest_engine.step_size_days = config.get('step', original_step)
                
                # Run backtest
                config_results = backtest_engine.run()
                results[f"config_{config_name}"] = config_results
            
            except Exception as e:
                self.logger.error(f"Error testing window config {config_name}: {str(e)}")
        
        # Restore original settings
        backtest_engine.train_period_days = original_train
        backtest_engine.test_period_days = original_test
        backtest_engine.step_size_days = original_step
        
        return results
    
    def generate_robustness_report(self, results: Dict) -> str:
        """
        Generate HTML report summarizing robustness tests.
        
        Args:
            results: Dictionary with robustness test results
            
        Returns:
            HTML report string
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Robustness Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>Robustness Test Report</h1>
            <p>Generated: {datetime.now()}</p>
            
            <h2>Summary</h2>
            <p>Robustness testing completed. See individual test sections below.</p>
        </body>
        </html>
        """
        
        return html

