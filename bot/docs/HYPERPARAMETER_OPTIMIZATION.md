# Hyperparameter Optimization Guide

## Overview

This guide explains how to use the automated hyperparameter optimization feature for XGBoost models in the trading bot. The optimization uses Optuna, a Bayesian optimization framework, to efficiently search the hyperparameter space and find optimal configurations.

## Quick Start

### Basic Usage

```python
from trading_bot.models import XGBoostTrainer

# Initialize trainer
trainer = XGBoostTrainer(config, logger)

# Optimize hyperparameters
best_params = trainer.optimize_hyperparameters(X, y, n_trials=100, objective='sharpe')

# Train with optimized parameters
trainer.set_hyperparameters(best_params)
trainer.train(X_train, y_train, X_val, y_val)
```

### CLI Command

```bash
python scripts/optimize_model.py \
  --symbol AAPL \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --n-trials 200 \
  --objective sharpe \
  --output best_params.json
```

### GUI Usage

1. Open the trading bot GUI
2. Navigate to the "Model Management" tab
3. Click "🔧 Optimize Hyperparameters"
4. Fill in the optimization parameters:
   - Symbol (e.g., AAPL)
   - Start and end dates
   - Number of trials
   - Objective function
5. Click "OK" to start optimization
6. Review results and optionally train with best parameters

## Optimization Objectives

The optimizer supports multiple objective functions:

- **Sharpe Ratio** (default): Maximizes risk-adjusted returns. Best for balanced risk/return optimization.
- **Sortino Ratio**: Maximizes downside risk-adjusted returns. Focuses on minimizing downside volatility.
- **Total Returns**: Maximizes absolute returns. Ignores risk, focuses purely on profit.
- **Calmar Ratio**: Maximizes return/max_drawdown ratio. Best for minimizing drawdowns.

## Configuration

### Basic Configuration

Edit `config/config.yaml` to customize optimization settings:

```yaml
models:
  optimization:
    enabled: true
    n_trials: 100
    timeout_seconds: 3600
    objective: 'sharpe'
```

### Parameter Search Ranges

Customize the hyperparameter search space:

```yaml
models:
  optimization:
    parameter_ranges:
      max_depth: [3, 10]
      learning_rate: [0.001, 0.3]
      n_estimators: [50, 500]
      subsample: [0.6, 1.0]
      colsample_bytree: [0.6, 1.0]
      min_child_weight: [1, 10]
      gamma: [0, 5.0]
      reg_alpha: [0, 10.0]
      reg_lambda: [0, 10.0]
```

### Early Stopping and Pruning

Configure early stopping and pruning to speed up optimization:

```yaml
models:
  optimization:
    early_stopping:
      enabled: true
      patience: 10  # Stop if no improvement for 10 trials
      min_delta: 0.01
    
    pruning:
      enabled: true
      warmup_steps: 10
      n_startup_trials: 5
```

### Ensemble Optimization

Enable ensemble optimization to combine top-performing models:

```yaml
models:
  optimization:
    ensemble:
      enabled: false
      n_models: 5
      diversity_weight: 0.1
```

### Distributed Optimization

For parallel optimization across multiple workers:

```yaml
models:
  optimization:
    distributed:
      enabled: false
      n_workers: 4
      storage: 'sqlite:///optuna.db'
```

### Monitoring and Checkpointing

Configure checkpointing and plot saving:

```yaml
models:
  optimization:
    monitoring:
      checkpoint_interval: 10
      checkpoint_dir: 'optimization_checkpoints'
      save_plots: true
      plot_dir: 'optimization_plots'
```

## Parameter Search Space

| Parameter | Search Range | Description | Impact |
|-----------|-------------|-------------|--------|
| max_depth | [3, 10] | Maximum tree depth | Controls model complexity |
| learning_rate | [0.001, 0.3] | Learning rate | Controls training speed and convergence |
| n_estimators | [50, 500] | Number of trees | More trees = better fit but slower |
| subsample | [0.6, 1.0] | Row sampling ratio | Prevents overfitting |
| colsample_bytree | [0.6, 1.0] | Column sampling ratio | Reduces overfitting |
| min_child_weight | [1, 10] | Minimum child weight | Controls tree splitting |
| gamma | [0, 5.0] | Minimum loss reduction | Regularization parameter |
| reg_alpha | [0, 10.0] | L1 regularization | Feature selection |
| reg_lambda | [0, 10.0] | L2 regularization | Prevents overfitting |

## Using the CLI

### Command-Line Arguments

```bash
python scripts/optimize_model.py [OPTIONS]

Required:
  --symbol SYMBOL          Stock symbol (e.g., AAPL)
  --start-date DATE        Start date (YYYY-MM-DD)
  --end-date DATE          End date (YYYY-MM-DD)

Optional:
  --n-trials N             Number of trials (default: 100)
  --objective OBJECTIVE    Objective: sharpe, sortino, returns, calmar
  --output FILE            Output file (default: best_params.json)
  --resume CHECKPOINT      Resume from checkpoint
  --distributed            Enable distributed optimization
```

### Example Commands

**Basic optimization:**
```bash
python scripts/optimize_model.py \
  --symbol AAPL \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --n-trials 100
```

**Optimize for Sortino ratio:**
```bash
python scripts/optimize_model.py \
  --symbol TSLA \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --n-trials 200 \
  --objective sortino
```

**Resume from checkpoint:**
```bash
python scripts/optimize_model.py \
  --symbol AAPL \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --resume optimization_checkpoints/checkpoint_trial_50.pkl
```

## Using the GUI

### Step-by-Step Guide

1. **Open Model Management Tab**: Navigate to the "Model Management" tab in the GUI
2. **Click Optimize Button**: Click "🔧 Optimize Hyperparameters"
3. **Configure Parameters**:
   - Enter stock symbol (e.g., AAPL)
   - Select start and end dates using date pickers
   - Set number of trials (10-1000)
   - Choose optimization objective
4. **Start Optimization**: Click "OK" to start
5. **Monitor Progress**: Watch progress in the status label
6. **Review Results**: After completion, review:
   - Best parameters found
   - Best objective value
   - Parameter importance insights
7. **Train with Best Params**: Optionally train a model with optimized parameters

### Interpreting Results

- **Best Value**: The best objective value achieved (e.g., Sharpe ratio)
- **Best Parameters**: The hyperparameters that achieved the best value
- **Parameter Importance**: Which parameters have the most impact
- **Optimal Regions**: Parameter ranges that consistently perform well

## Programmatic Usage

### Example 1: Using Trainer's Built-in Method

```python
from trading_bot.models import XGBoostTrainer
from trading_bot.config_loader import Config
from trading_bot.logger import setup_logger

# Initialize
config = Config()
config.load()
logger = setup_logger('optimizer', config)

# Create trainer
trainer = XGBoostTrainer(config, logger)

# Optimize hyperparameters
best_params = trainer.optimize_hyperparameters(
    X, y,
    n_trials=100,
    objective='sharpe',
    timeout=3600
)

# Update trainer with optimized params
trainer.set_hyperparameters(best_params)

# Train with optimized parameters
trainer.train(X_train, y_train, X_val, y_val)
```

### Example 2: Using Optimizer Directly

```python
from trading_bot.models import XGBoostOptimizer
from trading_bot.models.optimization_monitor import OptimizationMonitor

# Create optimizer
optimizer = XGBoostOptimizer(
    X_train, y_train, X_val, y_val,
    config, logger, objective='sharpe'
)

# Create monitor
monitor = OptimizationMonitor(config, logger)
callback = monitor.create_callback()

# Run optimization
best_params = optimizer.optimize(
    n_trials=100,
    timeout=3600,
    callbacks=[callback]
)

# Get best model
best_model = optimizer.get_best_model()

# Analyze results
from trading_bot.models import ParameterAnalyzer
analyzer = ParameterAnalyzer(optimizer.study, config, logger)
insights = analyzer.generate_insights()
print(insights)
```

### Example 3: Ensemble Optimization

```python
from trading_bot.models import EnsembleOptimizer

# After running optimization, create ensemble
ensemble = EnsembleOptimizer(optimizer.study, config, logger, n_models=5)

# Optimize ensemble weights
ensemble_info = ensemble.optimize_ensemble(X_train, y_train, X_val, y_val)

# Make predictions with ensemble
predictions = ensemble.predict_ensemble(X_test)

print(f"Ensemble diversity: {ensemble_info['diversity_score']:.4f}")
print(f"Model weights: {ensemble_info['weights']}")
```

## Ensemble Optimization

### When to Use Ensemble vs Single Model

- **Single Model**: Faster, simpler, easier to interpret
- **Ensemble**: Better performance, more robust, handles overfitting better

### Diversity Weighting

The ensemble optimizer balances:
- **Performance**: Weighted combination of predictions
- **Diversity**: Ensures models make different predictions

Higher diversity weight (e.g., 0.2) emphasizes model diversity.
Lower diversity weight (e.g., 0.05) focuses on performance.

## Monitoring and Checkpointing

### Progress Tracking

The optimizer automatically tracks:
- Number of completed trials
- Best value found
- Average trial duration
- Estimated time remaining

### Checkpoint/Resume

Checkpoints are saved every N trials (configurable). To resume:

```python
# In code
optimizer.load_study('optimization_checkpoints/checkpoint_trial_50.pkl')
optimizer.optimize(n_trials=100)  # Continues from checkpoint

# CLI
python scripts/optimize_model.py --resume checkpoint.pkl ...
```

### Notifications

Configure email or Slack notifications:

```yaml
models:
  optimization:
    notifications:
      enabled: true
      email:
        enabled: true
        smtp_server: 'smtp.gmail.com'
        smtp_port: 587
        from_address: 'your@email.com'
        to_address: 'notify@email.com'
      slack:
        enabled: true
        webhook_url: 'https://hooks.slack.com/services/...'
```

## Distributed Optimization

### Setup Instructions

1. **Install dependencies**: All workers need the same environment
2. **Configure database**: Use SQLite (single machine) or PostgreSQL (multi-machine)
3. **Start workers**: Run optimization script on each worker with `--distributed` flag

### Performance Scaling

- **Single machine**: 2-4x speedup with 4 workers
- **Multi-machine**: Linear scaling with number of workers
- **Database**: PostgreSQL recommended for >10 workers

## Interpreting Results

### Optimization History Plots

Shows:
- Objective value per trial
- Best value progression
- Convergence behavior

### Parameter Importance Analysis

Identifies:
- Most important parameters (high impact)
- Low sensitivity parameters (can be fixed)
- Optimal parameter ranges

### Generated Insights

The analyzer generates actionable recommendations:
- Focus on high-impact parameters
- Constrain search space to optimal regions
- Fix low-sensitivity parameters

## Best Practices

### Recommended n_trials

- **Quick test**: 20-50 trials
- **Standard optimization**: 100-200 trials
- **Thorough optimization**: 500+ trials
- **Production**: 200-500 trials with early stopping

### Timeout Settings

- **Quick test**: 600 seconds (10 minutes)
- **Standard**: 3600 seconds (1 hour)
- **Thorough**: 7200+ seconds (2+ hours)

### When to Use Pruning

- **Enable pruning**: For large search spaces, long trials
- **Disable pruning**: For small search spaces, fast trials

### Balancing Exploration vs Exploitation

- **More trials**: Better exploration, finds better solutions
- **Pruning**: Faster convergence, may miss better solutions
- **Early stopping**: Stops when no improvement, saves time

## Troubleshooting

### Common Issues

**"No completed trials" error:**
- Check data quality and feature engineering
- Verify objective function is valid
- Reduce parameter ranges if trials fail

**Slow optimization:**
- Reduce n_trials
- Enable pruning
- Use distributed optimization
- Reduce data size for testing

**Memory errors:**
- Reduce n_estimators range
- Use smaller dataset for optimization
- Reduce number of parallel workers

**Poor results:**
- Increase n_trials
- Adjust parameter ranges
- Try different objective function
- Check data quality

### Performance Optimization Tips

1. **Use pruning**: Automatically stops poor trials early
2. **Reduce data size**: Use subset for optimization, full set for final training
3. **Parallel processing**: Use distributed optimization for speedup
4. **Checkpointing**: Resume from checkpoints if interrupted

### Memory Management

- Monitor memory usage during optimization
- Reduce batch sizes if needed
- Use smaller datasets for optimization
- Consider using GPU if available

## API Reference

### XGBoostOptimizer

**Methods:**
- `optimize(n_trials, timeout, callbacks)` - Run optimization
- `get_best_params()` - Get best hyperparameters
- `get_best_model()` - Train and return best model
- `get_optimization_history()` - Get DataFrame of all trials
- `plot_optimization_history()` - Generate Plotly visualization
- `plot_param_importances()` - Generate importance plot
- `save_study(path)` - Save study to file
- `load_study(path)` - Load study from file

### EnsembleOptimizer

**Methods:**
- `optimize_ensemble(X_train, y_train, X_val, y_val)` - Optimize ensemble
- `predict_ensemble(X)` - Make ensemble predictions
- `get_ensemble_info()` - Get ensemble information
- `save_ensemble(path)` - Save ensemble to disk

### OptimizationMonitor

**Methods:**
- `create_callback()` - Create Optuna callback
- `get_progress_info()` - Get progress information
- `estimate_time_remaining()` - Calculate ETA
- `send_notification(message, level)` - Send notification

### ParameterAnalyzer

**Methods:**
- `analyze_importance()` - Calculate parameter importance
- `find_optimal_regions()` - Find optimal parameter ranges
- `generate_insights()` - Generate insights report
- `plot_param_relationships()` - Generate correlation heatmap
- `plot_parallel_coordinate()` - Generate parallel coordinate plot
- `plot_contour(param1, param2)` - Generate 2D contour plot

## Examples

### Complete End-to-End Example

```python
from trading_bot.models import XGBoostTrainer, XGBoostOptimizer, ParameterAnalyzer
from trading_bot.data import StockDataFetcher, FeatureEngineer
from trading_bot.config_loader import Config
from trading_bot.logger import setup_logger

# Initialize
config = Config()
config.load()
logger = setup_logger('optimizer', config)

# Fetch data
fetcher = StockDataFetcher(config, logger)
data = fetcher.fetch_historical_data('AAPL', '2022-01-01', '2024-12-31')

# Create features
feature_engineer = FeatureEngineer(config, logger)
features_df = feature_engineer.create_features(data)

# Prepare data
from trading_bot.data.preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split

preprocessor = DataPreprocessor(config, logger)
X, y = preprocessor.prepare_training_data(features_df, target_column='Close')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Optimize
optimizer = XGBoostOptimizer(X_train, y_train, X_val, y_val, config, logger)
best_params = optimizer.optimize(n_trials=100)

# Analyze
analyzer = ParameterAnalyzer(optimizer.study, config, logger)
insights = analyzer.generate_insights()
print(insights)

# Train with best params
trainer = XGBoostTrainer(config, logger)
trainer.set_hyperparameters(best_params)
trainer.train(X_train, y_train, X_val, y_val)
```

### Custom Objective Function Example

```python
def custom_objective(trial, X_train, y_train, X_val, y_val):
    # Suggest parameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        # ... other params
    }
    
    # Train model
    trainer = XGBoostTrainer(config, logger)
    trainer.set_hyperparameters(params)
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = trainer.evaluate(X_val, y_val)
    
    # Custom objective: combine multiple metrics
    score = metrics['r2'] * 0.7 + metrics['sharpe'] * 0.3
    
    return score
```

---

For more information, see:
- [XGBoost Integration Guide](XGBOOST_INTEGRATION.md)
- [Backtesting Guide](BACKTESTING_GUIDE.md)
- [Optuna Documentation](https://optuna.readthedocs.io/)



