"""
Walk-Forward Backtesting Engine

Implements walk-forward backtesting with rolling train/test windows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime

from trading_bot.backtesting.results import BacktestResults
from trading_bot.backtesting.costs import TransactionCostModel
from trading_bot.backtesting.metrics import PerformanceMetrics
from trading_bot.data.preprocessor import DataPreprocessor
from trading_bot.utils.exceptions import DataError


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine.
    
    Splits data into rolling train/test windows, trains models, generates predictions,
    simulates trades, and calculates performance metrics.
    """
    
    def __init__(self, data: pd.DataFrame, config, logger,
                 feature_engineer, signal_generator, model_class):
        """
        Initialize walk-forward backtest.
        
        Args:
            data: Historical OHLCV DataFrame with Date column
            config: Configuration object
            logger: Logger instance
            feature_engineer: FeatureEngineer instance
            signal_generator: SignalGenerator instance
            model_class: Model trainer class (e.g., XGBoostTrainer)
        """
        self.data = data.copy()
        self.config = config
        self.logger = logger
        self.feature_engineer = feature_engineer
        self.signal_generator = signal_generator
        self.model_class = model_class
        self.stop_requested = False  # Flag for cooperative cancellation
        self.symbol = getattr(data, 'symbol', None) or getattr(config, 'symbol', None) or 'UNKNOWN'  # Extract symbol if available
        
        # Load walk-forward configuration
        wf_config = config.get('backtesting.walk_forward', {})
        self.train_period_days = wf_config.get('train_period_days', 252)
        self.test_period_days = wf_config.get('test_period_days', 21)
        self.step_size_days = wf_config.get('step_size_days', 21)
        self.retrain_frequency_days = wf_config.get('retrain_frequency_days', 21)
        self.min_data_points = wf_config.get('min_data_points', 252)
        
        # Initialize components
        self.cost_model = TransactionCostModel(config, logger)
        self.metrics_calculator = PerformanceMetrics(config, logger)
        self.preprocessor = DataPreprocessor(config, logger)
        
        # Validate data
        self._validate_data()
        
        # Ensure Date column is datetime
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        self.logger.info(
            f"WalkForwardBacktest initialized: train={self.train_period_days}d, "
            f"test={self.test_period_days}d, step={self.step_size_days}d"
        )
    
    def _validate_data(self):
        """Validate input data has required columns."""
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise DataError(f"Missing required columns: {missing_cols}")
        
        if len(self.data) < self.min_data_points:
            raise DataError(
                f"Insufficient data: need {self.min_data_points} rows, got {len(self.data)}"
            )
    
    def run(self, initial_capital: float = 10000.0,
            progress_callback: Optional[Callable] = None) -> BacktestResults:
        """
        Run walk-forward backtest.
        
        Args:
            initial_capital: Starting capital
            progress_callback: Optional callback for progress updates (current, total, message)
            
        Returns:
            BacktestResults object
        """
        self.logger.info(f"Starting walk-forward backtest with {len(self.data)} data points")
        
        # Create windows
        windows = self._create_windows()
        
        if len(windows) == 0:
            raise DataError("No valid windows created")
        
        self.logger.info(f"Created {len(windows)} walk-forward windows")
        
        # Track state
        all_trades = []
        all_period_results = []
        cash = initial_capital
        positions = {}  # symbol -> {quantity, entry_price, entry_time}
        equity_curve = []
        equity_dates = []
        self.initial_capital = initial_capital  # Store for period metrics
        
        # Track last trained model and predictions for stats
        last_trained_model = None
        all_predictions = []
        all_actuals = []
        
        # Track last equity date to avoid duplicates
        last_equity_date = None
        
        # Process each window
        for i, window in enumerate(windows):
            # Check for stop request
            if self.stop_requested:
                self.logger.info("Stop requested, terminating backtest")
                break
            
            if progress_callback:
                progress_callback(i + 1, len(windows), f"Processing window {i+1}/{len(windows)}")
            
            self.logger.info(f"Processing window {i+1}/{len(windows)}")
            
            try:
                # Get train and test data
                train_data = self.data.iloc[window['train_start_idx']:window['train_end_idx']]
                test_data = self.data.iloc[window['test_start_idx']:window['test_end_idx']]
                
                # Train model (only if retrain needed - will implement retrain frequency later)
                model, scaler = self._train_model(train_data)
                last_trained_model = model  # Track last trained model
                
                # Test model
                predictions_df = self._test_model(model, scaler, test_data)
                
                # Collect predictions and actuals for stats
                if len(predictions_df) > 0:
                    all_predictions.extend(predictions_df['prediction'].values)
                    all_actuals.extend(predictions_df['Close'].values)
                
                # Simulate trades and accumulate daily equity
                window_trades, cash, positions = self._simulate_trades(
                    predictions_df, test_data, cash, positions, window['test_start_idx']
                )
                
                all_trades.extend(window_trades)
                
                # Calculate period metrics
                period_metrics = self._calculate_period_metrics(window_trades, test_data, initial_capital)
                period_metrics['window'] = i + 1
                period_metrics['start_date'] = str(test_data.iloc[0]['Date'])
                period_metrics['end_date'] = str(test_data.iloc[-1]['Date'])
                all_period_results.append(period_metrics)
                
                # Accumulate daily equity across test period
                for idx, row in test_data.iterrows():
                    date = row['Date']
                    daily_close = row['Close']
                    
                    # Skip if we already have equity for this date (window overlap)
                    if last_equity_date is not None and date <= last_equity_date:
                        continue
                    
                    # Calculate mark-to-market equity
                    current_equity = cash + sum(
                        pos['quantity'] * daily_close
                        for pos in positions.values()
                    )
                    
                    equity_curve.append(current_equity)
                    equity_dates.append(date)
                    last_equity_date = date
                
            except Exception as e:
                self.logger.error(f"Error processing window {i+1}: {str(e)}")
                continue
        
        # Create equity curve series
        if len(equity_curve) == 0:
            # Fallback if no equity calculated
            equity_curve = [initial_capital]
            equity_dates = [self.data.iloc[0]['Date']]
        
        equity_series = pd.Series(equity_curve, index=pd.to_datetime(equity_dates))
        
        # Calculate daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        # Aggregate results
        results = self._aggregate_results(
            all_period_results, all_trades, equity_series, daily_returns,
            last_trained_model, all_predictions, all_actuals
        )
        
        self.logger.info("Walk-forward backtest completed")
        
        return results
    
    def _create_windows(self) -> List[Dict]:
        """
        Create walk-forward windows.
        
        Returns:
            List of window dictionaries with indices
        """
        windows = []
        total_rows = len(self.data)
        
        # Calculate indices
        train_start_idx = 0
        
        while train_start_idx + self.train_period_days + self.test_period_days <= total_rows:
            train_end_idx = train_start_idx + self.train_period_days
            test_start_idx = train_end_idx  # Non-overlapping: test starts where train ends
            test_end_idx = min(test_start_idx + self.test_period_days, total_rows)
            
            # Ensure test starts at or after train ends (non-overlapping)
            if test_start_idx >= train_end_idx:
                windows.append({
                    'train_start_idx': train_start_idx,
                    'train_end_idx': train_end_idx,
                    'test_start_idx': test_start_idx,
                    'test_end_idx': test_end_idx
                })
            
            # Roll forward
            train_start_idx += self.step_size_days
            
            # Stop if we can't create another full window
            if train_start_idx + self.train_period_days + self.test_period_days > total_rows:
                break
        
        return windows
    
    def _train_model(self, train_data: pd.DataFrame) -> Tuple[object, object]:
        """
        Train model on training data.
        
        Args:
            train_data: Training data DataFrame
            
        Returns:
            Tuple of (model, scaler)
        """
        # Create features
        features_df = self.feature_engineer.create_features(train_data)
        
        # Prepare training data
        X_train, y_train = self.preprocessor.prepare_training_data(features_df, target_column='Close')
        
        # Handle NaN values
        valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        if len(X_train) == 0:
            raise DataError("No valid training data after feature engineering")
        
        # Fit scaler
        self.preprocessor.fit_scaler(X_train)
        
        # Scale features
        X_train_scaled = self.preprocessor.transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # Create and train model
        model = self.model_class(self.config, self.logger)
        model.train(X_train_scaled_df, y_train)
        
        # Return model and scaler
        return model, self.preprocessor.scaler
    
    def _test_model(self, model, scaler, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions on test data.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            test_data: Test data DataFrame
            
        Returns:
            DataFrame with predictions
        """
        # Create features
        features_df = self.feature_engineer.create_features(test_data)
        
        # Prepare features (remove target columns)
        exclude_cols = ['Date', 'Close']
        X_test = features_df.drop(columns=[col for col in exclude_cols if col in features_df.columns])
        
        # Handle NaN
        X_test = X_test.fillna(method='ffill').fillna(0)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Make predictions
        predictions = model.model.predict(X_test_scaled_df)
        
        # Get prediction confidence (for classification)
        if hasattr(model.model, 'predict_proba'):
            try:
                proba = model.model.predict_proba(X_test_scaled_df)
                confidence = np.max(proba, axis=1)
            except:
                confidence = np.ones(len(predictions)) * 0.5
        else:
            # For regression, use a simple confidence metric based on prediction magnitude
            # This is a simplified approach
            confidence = np.ones(len(predictions)) * 0.7
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': test_data['Date'].values,
            'Close': test_data['Close'].values,
            'prediction': predictions,
            'confidence': confidence
        })
        
        return predictions_df
    
    def _simulate_trades(self, predictions_df: pd.DataFrame, test_data: pd.DataFrame,
                        cash: float, positions: Dict, start_idx: int) -> Tuple[List[Dict], float, Dict]:
        """
        Simulate trades based on predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            test_data: Test data DataFrame
            cash: Current cash balance
            positions: Current positions dict
            start_idx: Starting index in full data
            
        Returns:
            Tuple of (trades_list, updated_cash, updated_positions)
        """
        trades = []
        
        for idx, row in predictions_df.iterrows():
            date = row['Date']
            current_price = row['Close']
            prediction = row['prediction']
            confidence = row['confidence']
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                prediction, confidence, current_price, symbol='UNKNOWN'
            )
            
            # Check if signal is actionable
            if not self.signal_generator.should_execute_signal(signal):
                continue
            
            # Calculate position size (simplified: use fixed percentage of capital)
            position_size_pct = self.config.get('trading.position_size_percentage', 10) / 100
            position_value = cash * position_size_pct
            
            # Get average volume for market impact
            avg_volume = test_data['Volume'].mean()
            
            # Calculate volatility for cost model
            if len(test_data) > 20:
                returns = test_data['Close'].pct_change()
                volatility = returns.std()
            else:
                volatility = 0.02  # Default
            
            # Execute trade based on signal
            if signal['type'] in ['BUY', 'STRONG_BUY']:
                # Calculate quantity
                quantity = position_value / current_price
                
                # Calculate transaction costs
                cost_info = self.cost_model.calculate_cost(
                    current_price, quantity, avg_volume, 'buy', 'market', volatility
                )
                
                effective_price = cost_info['effective_price']
                commission = cost_info['commission']
                total_cost = cost_info['total_cost']
                trade_value = effective_price * quantity + commission  # Include commission
                
                # Check if we have enough cash
                if trade_value <= cash:
                    # Execute buy: subtract effective price * quantity and commission separately
                    cash -= effective_price * quantity
                    cash -= commission  # Subtract commission from cash
                    positions[date] = {
                        'quantity': quantity,
                        'entry_price': effective_price,
                        'entry_time': str(date),
                        'symbol': self.symbol
                    }
                    
                    trades.append({
                        'entry_time': str(date),
                        'exit_time': None,
                        'symbol': self.symbol,
                        'side': 'BUY',
                        'entry_price': effective_price,
                        'exit_price': None,
                        'quantity': quantity,
                        'pnl': None,
                        'pnl_pct': None,
                        'cost': total_cost
                    })
            
            elif signal['type'] in ['SELL', 'STRONG_SELL']:
                # Close existing positions
                for pos_date, position in list(positions.items()):
                    quantity = position['quantity']
                    entry_price = position['entry_price']
                    
                    # Calculate transaction costs
                    cost_info = self.cost_model.calculate_cost(
                        current_price, quantity, avg_volume, 'sell', 'market', volatility
                    )
                    
                    effective_price = cost_info['effective_price']
                    commission = cost_info['commission']
                    total_cost = cost_info['total_cost']
                    
                    # Calculate PnL: effective_price already includes per-share adjustments,
                    # but commission needs to be subtracted separately
                    pnl = (effective_price - entry_price) * quantity - commission
                    pnl_pct = ((effective_price - entry_price) / entry_price) * 100
                    
                    # Execute sell: add effective price * quantity and subtract commission separately
                    cash += effective_price * quantity
                    cash -= commission  # Subtract commission from cash
                    del positions[pos_date]
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': str(date),
                        'symbol': position['symbol'],
                        'side': 'SELL',
                        'entry_price': entry_price,
                        'exit_price': effective_price,
                        'quantity': quantity,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'cost': total_cost
                    })
        
        # Close remaining positions at end of test period
        final_price = test_data.iloc[-1]['Close']
        final_date = test_data.iloc[-1]['Date']
        
        for pos_date, position in list(positions.items()):
            quantity = position['quantity']
            entry_price = position['entry_price']
            
            # Calculate transaction costs
            avg_volume = test_data['Volume'].mean()
            volatility = test_data['Close'].pct_change().std() if len(test_data) > 20 else 0.02
            
            cost_info = self.cost_model.calculate_cost(
                final_price, quantity, avg_volume, 'sell', 'market', volatility
            )
            
            effective_price = cost_info['effective_price']
            commission = cost_info['commission']
            total_cost = cost_info['total_cost']
            
            # Calculate PnL: effective_price already includes per-share adjustments,
            # but commission needs to be subtracted separately
            pnl = (effective_price - entry_price) * quantity - commission
            pnl_pct = ((effective_price - entry_price) / entry_price) * 100
            
            # Execute sell: add effective price * quantity and subtract commission separately
            cash += effective_price * quantity
            cash -= commission  # Subtract commission from cash
            del positions[pos_date]
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': str(final_date),
                'symbol': position['symbol'],
                'side': 'SELL',
                'entry_price': entry_price,
                'exit_price': effective_price,
                'quantity': quantity,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'cost': total_cost
            })
        
        return trades, cash, positions
    
    def _calculate_period_metrics(self, trades: List[Dict], test_data: pd.DataFrame, initial_capital: float) -> Dict:
        """
        Calculate metrics for a single test period.
        
        Args:
            trades: List of trades in this period
            test_data: Test data DataFrame
            initial_capital: Initial capital for return calculation
            
        Returns:
            Dictionary with period metrics
        """
        if len(trades) == 0:
            return {
                'total_return_pct': 0.0,
                'num_trades': 0,
                'win_rate_pct': 0.0,
                'total_pnl': 0.0
            }
        
        # Calculate period return
        total_pnl = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl') is not None)
        
        # Calculate win rate
        completed_trades = [t for t in trades if t.get('pnl') is not None]
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            win_rate_pct = (len(winning_trades) / len(completed_trades)) * 100
        else:
            win_rate_pct = 0.0
        
        # Calculate return percentage using initial_capital
        total_return_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0
        
        return {
            'total_return_pct': total_return_pct,
            'num_trades': len(completed_trades),
            'win_rate_pct': win_rate_pct,
            'total_pnl': total_pnl
        }
    
    def _aggregate_results(self, period_results: List[Dict], all_trades: List[Dict],
                          equity_curve: pd.Series, daily_returns: pd.Series,
                          last_model=None, all_predictions=None, all_actuals=None) -> BacktestResults:
        """
        Aggregate results from all periods.
        
        Args:
            period_results: List of period result dictionaries
            all_trades: List of all trades
            equity_curve: Equity curve series
            daily_returns: Daily returns series
            
        Returns:
            BacktestResults object
        """
        # Calculate overall metrics
        all_metrics = self.metrics_calculator.calculate_all_metrics(
            all_trades, equity_curve, daily_returns
        )
        
        # Extract feature importance if available (from last model)
        feature_importance = {}
        if last_model is not None:
            try:
                if hasattr(last_model, 'get_feature_importance'):
                    importance_df = last_model.get_feature_importance()
                    feature_importance = dict(zip(importance_df['feature'], importance_df['importance']))
                elif hasattr(last_model, 'model') and hasattr(last_model.model, 'feature_importances_'):
                    # XGBoost models have feature_importances_
                    if hasattr(last_model, 'feature_names'):
                        feature_names = last_model.feature_names
                    else:
                        feature_names = [f'feature_{i}' for i in range(len(last_model.model.feature_importances_))]
                    feature_importance = dict(zip(feature_names, last_model.model.feature_importances_))
            except Exception as e:
                self.logger.warning(f"Failed to extract feature importance: {str(e)}")
        
        # Calculate prediction stats
        prediction_rmse = None
        prediction_accuracy = None
        
        if all_predictions is not None and all_actuals is not None and len(all_predictions) > 0:
            try:
                # Determine task type
                target_type = self.config.get('models.xgboost.target_type', 'regression')
                
                if target_type == 'regression':
                    # Calculate RMSE for regression
                    predictions_array = np.array(all_predictions)
                    actuals_array = np.array(all_actuals)
                    # Remove NaN values
                    valid_mask = ~(np.isnan(predictions_array) | np.isnan(actuals_array))
                    if valid_mask.sum() > 0:
                        prediction_rmse = np.sqrt(np.mean((predictions_array[valid_mask] - actuals_array[valid_mask]) ** 2))
                else:
                    # Calculate accuracy for classification
                    predictions_array = np.array(all_predictions)
                    actuals_array = np.array(all_actuals)
                    # Remove NaN values
                    valid_mask = ~(np.isnan(predictions_array) | np.isnan(actuals_array))
                    if valid_mask.sum() > 0:
                        # For classification, predictions are classes, actuals are also classes
                        # Accuracy = number of correct predictions / total predictions
                        correct = (predictions_array[valid_mask] == actuals_array[valid_mask]).sum()
                        prediction_accuracy = correct / valid_mask.sum()
            except Exception as e:
                self.logger.warning(f"Failed to calculate prediction stats: {str(e)}")
        
        # Get start and end dates
        start_date = str(self.data.iloc[0]['Date'])
        end_date = str(self.data.iloc[-1]['Date'])
        
        # Create BacktestResults
        results = BacktestResults(
            total_return=all_metrics.get('total_return_pct', 0.0),
            annualized_return=all_metrics.get('annualized_return_pct', 0.0),
            sharpe_ratio=all_metrics.get('sharpe_ratio', 0.0),
            sortino_ratio=all_metrics.get('sortino_ratio', 0.0),
            calmar_ratio=all_metrics.get('calmar_ratio', 0.0),
            max_drawdown=all_metrics.get('max_drawdown_pct', 0.0),
            max_drawdown_duration=all_metrics.get('max_drawdown_duration_days', 0),
            win_rate=all_metrics.get('win_rate_pct', 0.0),
            profit_factor=all_metrics.get('profit_factor', 0.0),
            total_trades=all_metrics.get('total_trades', 0),
            winning_trades=all_metrics.get('winning_trades', 0),
            losing_trades=all_metrics.get('losing_trades', 0),
            avg_win=all_metrics.get('avg_win', 0.0),
            avg_loss=all_metrics.get('avg_loss', 0.0),
            largest_win=all_metrics.get('largest_win', 0.0),
            largest_loss=all_metrics.get('largest_loss', 0.0),
            avg_holding_period=all_metrics.get('avg_holding_period_days', 0.0),
            expectancy=all_metrics.get('expectancy', 0.0),
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=all_trades,
            period_results=period_results,
            num_periods=len(period_results),
            feature_importance=feature_importance,
            prediction_rmse=prediction_rmse,
            prediction_accuracy=prediction_accuracy,
            backtest_start_date=start_date,
            backtest_end_date=end_date,
            symbol=self.symbol,
            model_type=self.model_class.__name__,
            config_snapshot={}
        )
        
        return results

