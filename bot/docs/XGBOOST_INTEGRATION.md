# XGBoost Integration Guide

## Overview

This guide explains the hybrid training-deployment architecture for XGBoost stock price prediction in the trading bot.

### Architecture Philosophy

The integration uses a **two-phase approach** that separates GPU-intensive training from real-time inference:

1. **Training Phase (Google Colab with GPU)**: Train models with free GPU acceleration
2. **Deployment Phase (Trading Bot - Local CPU)**: Run predictions on lightweight CPU for real-time trading

This approach leverages free Colab GPU resources while keeping the bot lightweight and cost-effective for deployment.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (Colab GPU)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │  Historical Data │
                   │   (yfinance)     │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │Feature Engineering│
                   │ • Lagged prices   │
                   │ • Tech indicators │
                   │ • Volatility      │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ XGBoost Training  │
                   │  (tree_method:    │
                   │   'gpu_hist')     │
                   └────────┬─────────┘
                            │
                            ▼
          ┌─────────────────┴─────────────────┐
          │       Export Model Files          │
          │  • model.json                     │
          │  • scaler.pkl                     │
          │  • metadata.json                  │
          └─────────────────┬─────────────────┘
                            │
                            ▼ (Download & Deploy)
┌─────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT PHASE (Bot CPU)                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Load Model Files    │
                │  (ModelManager)       │
                └───────────┬───────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │      Real-time Trading Loop          │
         └──────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐      ┌──────────┐      ┌──────────────┐
  │Fetch Data│      │ Engineer │      │Generate      │
  │(yfinance)│─────▶│ Features │─────▶│Predictions   │
  └──────────┘      └──────────┘      │(XGBoost CPU) │
                                       └──────┬───────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │Signal         │
                                      │Generation     │
                                      └───────┬───────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │Execute Trade  │
                                      │(if applicable)│
                                      └───────────────┘
```

---

## Training Workflow

### Step 1: Setup Google Colab

1. Open `notebooks/xgboost_training_colab.ipynb` in Google Colab
2. Enable GPU runtime:
   - Runtime → Change runtime type
   - Hardware accelerator → T4 GPU (or higher)
   - Click Save

3. Verify GPU availability:
   ```python
   !nvidia-smi
   ```

### Step 2: Configure Training Parameters

Edit the configuration cell in the notebook:

```python
SYMBOL = 'AAPL'  # Stock to train on
START_DATE = '2022-01-01'
MODEL_VERSION = 'v1'

# Hyperparameters
PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'tree_method': 'gpu_hist',  # GPU acceleration
    'device': 'cuda'
}
```

### Step 3: Run Training

Execute all notebook cells sequentially. The notebook will:

1. Download historical stock data (2+ years)
2. Engineer features (lagged prices, technical indicators)
3. Split data into train/test sets (time-series aware)
4. Fit StandardScaler on training data
5. Train XGBoost with GPU acceleration
6. Evaluate on test set (RMSE, MAE, R²)
7. Export model files

### Step 4: Download Trained Models

The notebook generates and downloads:

- `xgboost_AAPL_v1_20241030.json` - Trained model
- `scaler_AAPL_v1_20241030.pkl` - Feature scaler
- `xgboost_AAPL_v1_20241030_metadata.json` - Model metadata

### Step 5: Deploy to Bot

1. Place downloaded files in your bot's `models/` directory:
   ```
   bot/
   ├── models/
   │   ├── xgboost_AAPL_v1_20241030.json
   │   ├── xgboost_AAPL_v1_20241030_metadata.json
   │   └── scaler_AAPL_v1_20241030.pkl
   ```

2. Update `config/config.yaml`:
   ```yaml
   models:
     xgboost:
       enabled: true
       model_file: 'xgboost_AAPL_v1_20241030.json'
   ```

---

## Deployment Workflow

### Model Loading

The bot automatically loads models at startup:

```python
from trading_bot.models import ModelManager, XGBoostPredictor

# Initialize
manager = ModelManager(config, logger)
predictor = XGBoostPredictor(config, logger)

# Load model
model_info = manager.load_model('xgboost_AAPL_v1_20241030')
predictor.load_model(
    model_info['model_path'],
    model_info['metadata_path'],
    model_info['scaler_path']
)
```

### Real-Time Prediction Pipeline

```python
from trading_bot.data import StockDataFetcher, FeatureEngineer
from trading_bot.trading import SignalGenerator, XGBoostStrategy

# 1. Fetch latest market data
data_fetcher = StockDataFetcher(config, logger)
data = data_fetcher.fetch_latest_data('AAPL', period='60d')

# 2. Engineer features
feature_engineer = FeatureEngineer(config, logger)
features = feature_engineer.create_features(data)

# 3. Generate prediction
prediction = predictor.predict(features.tail(1))
confidence = predictor.get_confidence(prediction, features.tail(1))

# 4. Generate trading signal
signal_gen = SignalGenerator(config, logger)
signal = signal_gen.generate_signal(
    prediction[0], 
    confidence, 
    data['Close'].iloc[-1],
    'AAPL'
)

# 5. Execute trade (if signal warrants)
if signal['type'] in ['BUY', 'STRONG_BUY']:
    # Execute buy order
    pass
```

---

## Feature Engineering

### Features Created

The feature engineer creates the following features from raw OHLCV data:

#### 1. Price Change Features
- `return_1d` - Daily return (percentage change)
- `price_change_1d` - Absolute price change
- `intraday_change` - Open to close percentage change
- `high_low_range` - High-low range normalized by close price

#### 2. Lagged Features
Configurable lag periods (default: [1, 5, 10, 20]):
- `close_lag_N` - Close price N days ago
- `return_lag_N` - Return N days ago

#### 3. Technical Indicators
- `SMA_20`, `SMA_50` - Simple Moving Averages
- `RSI_14` - Relative Strength Index
- `MACD`, `MACD_signal`, `MACD_diff` - MACD indicators
- `BB_upper`, `BB_lower` - Bollinger Bands

#### 4. Volatility Features
- `volatility_20d` - Rolling 20-day volatility
- `price_volatility_20d` - Price standard deviation
- `ATR` - Average True Range

### Configuration

Customize features in `config/config.yaml`:

```yaml
models:
  features:
    lagged_prices: [1, 5, 10, 20]
    technical_indicators_list:
      - 'SMA_20'
      - 'SMA_50'
      - 'RSI_14'
      - 'MACD'
      - 'BB_upper'
      - 'BB_lower'
    volatility_window: 20
    scaling_method: 'standard'
```

### Feature Consistency

**Critical**: Feature engineering must be **identical** between training and inference to avoid training-serving skew.

The `feature_config.json` file exported during training ensures consistency:

```json
{
  "lagged_prices": [1, 5, 10, 20],
  "technical_indicators": ["SMA_20", "SMA_50", "RSI_14", "MACD"],
  "volatility_window": 20,
  "feature_names": ["return_1d", "close_lag_1", ...]
}
```

---

## Configuration Guide

### Complete XGBoost Configuration

```yaml
models:
  xgboost:
    enabled: true
    model_file: 'xgboost_stock_v1.json'
    device: 'cpu'  # Use CPU for inference (use 'gpu' or 'cuda' for GPU training)
    lookback_days: 60
    target_type: 'regression'  # or 'classification'
    retrain_interval_days: 7
    
    # Hyperparameters
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
    subsample: 0.8
    colsample_bytree: 0.8
  
  features:
    lagged_prices: [1, 5, 10, 20]
    technical_indicators_list:
      - 'SMA_20'
      - 'SMA_50'
      - 'RSI_14'
      - 'MACD'
      - 'BB_upper'
      - 'BB_lower'
    volatility_window: 20
    scaling_method: 'standard'
  
  prediction:
    buy_threshold: 0.6
    sell_threshold: 0.6
    min_confidence: 0.5

data:
  stock_data_source: 'yfinance'
  cache_historical_data: true
  historical_data_path: 'data/historical/'
```

### Tuning Thresholds

- **buy_threshold**: Minimum expected return (%) to generate buy signal
- **sell_threshold**: Minimum expected loss (%) to generate sell signal
- **min_confidence**: Minimum model confidence (0-1) to execute trades

---

## Model Management

### Model Versioning

Models follow this naming convention:
```
xgboost_{symbol}_{version}_{date}.json
```

Examples:
- `xgboost_AAPL_v1_20241030.json`
- `xgboost_TSLA_v2_20241105.json`

### Listing Available Models

```python
from trading_bot.models import ModelManager

manager = ModelManager(config, logger)
models = manager.list_available_models('xgboost')

for model in models:
    print(f"{model['filename']} - {model.get('date', 'unknown')}")
```

### Auto-Select Latest Model

```python
latest = manager.get_latest_model(symbol='AAPL')
if latest:
    print(f"Latest model: {latest['filename']}")
```

### Model Health Checks

Check if model needs retraining:

```python
predictor = XGBoostPredictor(config, logger)
# ... load model ...

retrain_interval = config.get('models.xgboost.retrain_interval_days', 7)
needs_retrain = predictor.check_model_age(retrain_interval)

if needs_retrain:
    logger.warning("Model is outdated, consider retraining")
```

### Cleanup Old Models

```python
manager.delete_old_models(keep_latest_n=3)
```

---

## Performance Considerations

### Prediction Latency

- **CPU inference**: ~5-20ms per prediction
- **Sufficient for real-time trading** (predictions needed every few seconds/minutes)
- No GPU needed in production

### Memory Usage

- **Model size**: ~100KB - 5MB (depending on n_estimators and max_depth)
- **Loaded model memory**: ~10-50MB
- **Very lightweight** for production deployment

### Scaling to Multiple Symbols

Strategies for handling multiple symbols:

1. **Single Model Per Symbol**: Train separate models for each symbol
   - Better accuracy (symbol-specific patterns)
   - More disk space and management overhead

2. **Universal Model**: Train one model on all symbols
   - Less management
   - May have lower accuracy

3. **Hybrid**: Universal model + symbol-specific fine-tuning

---

## Best Practices

### Retraining Frequency

- **Weekly**: Good balance for most stocks
- **Daily**: For highly volatile markets
- **Monthly**: For stable, long-term strategies

Set in config:
```yaml
models:
  xgboost:
    retrain_interval_days: 7
```

### Backtesting Before Deployment

Always backtest new models before live deployment:

```python
# In Colab notebook, add backtesting section
def backtest_strategy(model, data, threshold):
    # Simulate trades based on predictions
    # Calculate returns, Sharpe ratio, max drawdown
    pass
```

### Risk Management

Don't rely solely on model predictions:

1. Use **position sizing** based on confidence
2. Set **stop-loss** and **take-profit** levels
3. Respect **max_positions** limits
4. Monitor **model performance** continuously

### Model Monitoring

Track these metrics in production:

- **Prediction accuracy** over time
- **Trading signal distribution** (buy/sell/hold ratio)
- **Actual returns** vs predicted returns
- **Model confidence** trends

---

## Troubleshooting

### Common Issues

**Issue**: `Feature mismatch error`
```
ValidationError: Missing features: ['SMA_50', 'RSI_14']
```

**Solution**: Ensure feature engineering configuration matches training:
1. Check `feature_config.json` from training
2. Update `config.yaml` to match
3. Reload model

---

**Issue**: `Model loading fails`
```
ModelError: Failed to load model: No such file or directory
```

**Solution**: 
1. Verify model files are in `models/` directory
2. Check file paths in config
3. Ensure scaler file matches model

---

**Issue**: `yfinance data fetch fails`
```
DataError: No data retrieved for symbol AAPL
```

**Solution**:
1. Check internet connection
2. Verify symbol is valid
3. Try different date range
4. Check yfinance API status

---

**Issue**: `Predictions seem unrealistic`
```
Predicted price: $9999.99 (current: $150.00)
```

**Solution**:
1. Check if scaler was applied correctly
2. Verify feature values are in expected range
3. Retrain model with more recent data

---

## Future Enhancements

### Planned Features

1. **Ensemble Models**: Combine multiple models for better predictions
2. **Online Learning**: Update models with new data without full retraining
3. **Alternative Data**: Incorporate sentiment, news, fundamentals
4. **Advanced Features**: Fourier transforms, wavelets, autoencoders
5. **Multi-timeframe**: Predictions for different horizons (1-day, 1-week)

### Experimental Ideas

- **SHAP values** for prediction explainability
- **Uncertainty quantification** with conformal prediction
- **Reinforcement learning** for strategy optimization
- **Transfer learning** across similar stocks

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Technical Analysis Library (ta)](https://technical-analysis-library-in-python.readthedocs.io/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `logs/trading_bot_YYYY-MM-DD.log`
3. Consult the main README.md
4. Open an issue on GitHub (if applicable)

