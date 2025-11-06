# Walk-Forward Backtesting Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Walk-Forward Methodology](#walk-forward-methodology)
3. [Why Walk-Forward Prevents Overfitting](#why-walk-forward-prevents-overfitting)
4. [Getting Started](#getting-started)
5. [Configuration](#configuration)
6. [Running Backtests](#running-backtests)
7. [Interpreting Results](#interpreting-results)
8. [Validation Criteria](#validation-criteria)
9. [Transaction Costs](#transaction-costs)
10. [Robustness Testing](#robustness-testing)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## Introduction

The walk-forward backtesting framework validates trading strategies before deployment. It uses out-of-sample testing to prevent overfitting and provides realistic performance estimates.

Key capabilities:
- Validate trading strategies before deployment
- Prevent overfitting to historical data
- Estimate realistic performance expectations
- Identify strategy weaknesses

## Walk-Forward Methodology

Walk-forward analysis splits historical data into rolling train/test windows, ensuring no look-ahead bias.

### How It Works

1. **Training Window**: Train model on historical data (e.g., 252 days)
2. **Test Window**: Test predictions on next period (e.g., 21 days)
3. **Roll Forward**: Move both windows forward by step size (e.g., 21 days)
4. **Repeat**: Continue until all data is processed

```
Data: |----Train----|--Test--| Roll Forward
      |----Train----|--Test--|
      |----Train----|--Test--|
```

This ensures:
- No look-ahead bias (test data never seen during training)
- Realistic performance estimates
- Model retraining on fresh data

## Why Walk-Forward Prevents Overfitting

Traditional backtesting uses the same data for training and testing, leading to:
- **Overfitting**: Model memorizes training data
- **Look-ahead bias**: Using future information
- **Data snooping**: Optimizing on test data

Walk-forward prevents these by:
- Using separate train/test sets
- Testing on unseen data
- Rolling forward to simulate real trading

## Getting Started

### Using GUI

1. Open Backtesting tab in GUI
2. Select symbol and date range
3. Choose model
4. Configure walk-forward parameters
5. Click "Run Backtest"
6. View results in tabs

### Using CLI

```bash
python scripts/run_backtest.py \
    --symbol AAPL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --generate-report \
    --save-plots
```

### Using Python API

```python
from trading_bot.backtesting import WalkForwardBacktest
from trading_bot.data import StockDataFetcher, FeatureEngineer
from trading_bot.trading import SignalGenerator
from trading_bot.models import XGBoostTrainer

# Fetch data
fetcher = StockDataFetcher(config, logger)
data = fetcher.fetch_historical_data('AAPL', '2020-01-01', '2023-12-31')

# Initialize components
feature_engineer = FeatureEngineer(config, logger)
signal_generator = SignalGenerator(config, logger)

# Create backtest
backtest = WalkForwardBacktest(
    data=data,
    config=config,
    logger=logger,
    feature_engineer=feature_engineer,
    signal_generator=signal_generator,
    model_class=XGBoostTrainer
)

# Run
results = backtest.run()
print(f"Sharpe Ratio: {results.sharpe_ratio}")
results.plot_equity_curve()
```

## Configuration

Configure in `config/config.yaml`:

```yaml
backtesting:
  walk_forward:
    train_period_days: 252  # Training window (1 year)
    test_period_days: 21    # Testing window (1 month)
    step_size_days: 21      # Roll forward monthly
    min_data_points: 252    # Minimum data needed
  
  transaction_costs:
    commission_pct: 0.001   # 0.1% commission
    slippage_bps: 5         # 5 basis points
    spread_bps: 2           # 2 basis points
    market_impact_enabled: true
  
  validation:
    min_sharpe_ratio: 1.5   # Minimum Sharpe
    max_drawdown_pct: 15    # Maximum drawdown
    min_win_rate: 0.50      # Minimum win rate
    min_trades: 100         # Minimum trades
```

### Parameter Guidelines

- **train_period_days**: 180-252 days (longer = more data, slower)
- **test_period_days**: 10-30 days (shorter = more windows, faster)
- **step_size_days**: Match test_period for monthly retraining

## Running Backtests

### Step-by-Step

1. **Prepare Data**: Ensure sufficient historical data (3+ years recommended)
2. **Configure Parameters**: Set walk-forward windows and costs
3. **Run Backtest**: Execute via GUI, CLI, or API
4. **Monitor Progress**: Track window processing
5. **Review Results**: Analyze metrics and visualizations

### Progress Tracking

The framework provides progress updates:
- Current window being processed
- Total windows
- Estimated completion time

## Interpreting Results

### Key Metrics

**Sharpe Ratio**: Risk-adjusted return
- > 1.5: Good
- > 2.0: Excellent
- < 1.0: Poor

**Max Drawdown**: Worst peak-to-trough decline
- < 10%: Excellent
- < 15%: Good
- > 20%: Risky

**Win Rate**: Percentage of profitable trades
- > 55%: Good
- > 50%: Acceptable
- < 45%: Poor

**Profit Factor**: Gross profit / gross loss
- > 1.5: Good
- > 2.0: Excellent
- < 1.2: Poor

### Equity Curve

Should show:
- Smooth upward trend
- No steep drops
- Consistent growth

### Drawdown Chart

Look for:
- Quick recovery from drawdowns
- No extended underwater periods
- Manageable drawdown depth

## Validation Criteria

Before deploying, ensure:

- ✅ Sharpe ratio > 1.5
- ✅ Max drawdown < 15%
- ✅ Win rate > 50%
- ✅ At least 100 trades
- ✅ Positive returns in >60% of periods
- ✅ No parameter sensitivity (<20% change for ±10% param change)

## Transaction Costs

Realistic cost modeling includes:

- **Commission**: Fixed percentage per trade
- **Slippage**: Price movement during execution
- **Spread**: Bid-ask spread cost
- **Market Impact**: Price impact of large orders

These costs significantly impact performance - always include them.

## Robustness Testing

Test strategy stability through:

- **Parameter Sensitivity**: Vary parameters ±10%
- **Time Period Stability**: Test on different market regimes
- **Feature Stability**: Remove features randomly
- **Monte Carlo**: Simulate random outcomes

## Best Practices

1. Use sufficient data (3+ years)
2. Test on multiple symbols
3. Include transaction costs
4. Validate on out-of-sample data
5. Check for overfitting
6. Document assumptions
7. Regular re-validation

## Troubleshooting

### Insufficient Data

**Error**: "Insufficient data: need 252 rows"

**Solution**: Use longer date range or reduce min_data_points

### Model Training Failures

**Error**: "Training failed"

**Solution**: Check data quality, reduce feature complexity

### Poor Performance

**Symptoms**: Low Sharpe, high drawdown

**Solutions**:
- Check transaction costs
- Review signal generation logic
- Test different parameters
- Ensure no data leakage

### Slow Execution

**Cause**: Large datasets or many windows

**Solutions**:
- Reduce date range
- Increase step size
- Use fewer features
- Run during off-hours

For more help, see the main [README.md](../README.md) or open an issue.



