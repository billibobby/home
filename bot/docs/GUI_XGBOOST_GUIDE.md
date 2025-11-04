# XGBoost GUI User Guide

## Overview

The Trading Bot GUI now includes comprehensive XGBoost model management and prediction tools, making it easy to train, deploy, and monitor machine learning models without touching code.

---

## Getting Started

### 1. Launch the GUI

```bash
# From command line
trading-bot-gui

# Or from Python
python -m trading_bot.gui_main
```

### 2. Initial Setup

The GUI will automatically initialize all components. You'll see:
- Bot status in the top panel
- Multiple tabs for different features
- Real-time log display

---

## Tab Overview

### 🤖 Models Tab

**Purpose:** Manage your XGBoost models

**Features:**
- **Model Browser**: View all available models in your `models/` directory
- **Model Selection**: Choose which model to load for predictions
- **Model Information**: View detailed metadata (accuracy, training date, features, etc.)
- **Load/Unload**: Easily switch between models
- **Upload**: Import new model files
- **Training Link**: Quick access to Colab training notebook
- **Cleanup**: Delete old model versions

**How to Use:**

1. **View Available Models**:
   - The table shows all XGBoost models found
   - Columns: Filename, Symbol, Version, Date, Type, Status
   - Green indicator (🟢) shows currently loaded model

2. **Load a Model**:
   - Select a model from the dropdown or table
   - Click "📥 Load Model"
   - Model info will display below
   - Status will update to "✅ Loaded"

3. **View Model Info**:
   - Select any model to see:
     - Symbol and version
     - Training date
     - Number of features
     - Performance metrics (RMSE, MAE, R²)

4. **Upload New Models**:
   - Click "📤 Upload Model Files"
   - Select the 3 required files:
     - `model.json` (trained model)
     - `metadata.json` (model info)
     - `scaler.pkl` (feature scaler)
   - Files will be copied to `models/` directory

5. **Open Training Notebook**:
   - Click "🚀 Open Training Notebook"
   - Follow instructions to:
     - Upload notebook to Google Colab
     - Enable GPU runtime
     - Train a new model
     - Download generated files

---

### 🔮 Predictions Tab

**Purpose:** Make real-time stock price predictions

**Features:**
- **Real-time Predictions**: Get instant predictions for any symbol
- **Signal Generation**: Automatic BUY/SELL/HOLD signals
- **Confidence Scores**: See how confident the model is
- **Expected Returns**: Calculate potential profit/loss
- **Auto-Predict**: Continuous predictions at regular intervals
- **History Tracking**: View past predictions

**How to Use:**

1. **Make a Prediction**:
   - Ensure a model is loaded (see Models tab)
   - Enter a stock symbol (e.g., "AAPL")
   - Click "🔮 Predict"
   - Wait for:
     - Data fetching
     - Feature engineering
     - Prediction generation

2. **Read the Results**:
   - **Current Price**: Latest market price
   - **Predicted Price**: Model's prediction for next day
   - **Expected Return**: Percentage gain/loss
     - Green = positive return
     - Red = negative return
   - **Signal**: Trading recommendation
     - 🟢 STRONG_BUY / BUY (Green)
     - ⚫ HOLD (Gray)
     - 🟠 SELL / 🔴 STRONG_SELL (Orange/Red)
   - **Confidence**: How sure the model is (0-100%)
   - **Strength**: Signal strength (0-100%)
   - **Reasoning**: Explanation of the signal

3. **Enable Auto-Predict**:
   - Click "▶️ Auto Predict"
   - Model will predict every 60 seconds
   - Button changes to "⏸️ Pause"
   - Click again to pause

4. **View History**:
   - Scroll down to see prediction history
   - Table shows last 100 predictions
   - Columns: Time, Symbol, Current Price, Predicted Price, Return %, Signal, Confidence

---

### 📊 Strategy Tab

**Purpose:** Monitor trading strategy and signals

**Features:**
- **Performance Metrics**: Track signal counts and positions
- **Active Positions**: See current open positions
- **Signal History**: Review all generated signals
- **Symbol Analysis**: Analyze any symbol on demand
- **Auto-Refresh**: Updates every 5 seconds

**How to Use:**

1. **View Performance Metrics**:
   - **Total Signals**: All signals generated
   - **Buy Signals**: Count of buy recommendations
   - **Sell Signals**: Count of sell recommendations
   - **Active Positions**: Number of open positions

2. **Monitor Active Positions**:
   - Table shows current positions:
     - Symbol
     - Entry Price
     - Position Size
     - Stop Loss level
     - Take Profit target

3. **Review Signal History**:
   - Table shows last 50 signals
   - Columns:
     - Time: When generated
     - Symbol: Stock ticker
     - Signal: Type (BUY/SELL/HOLD)
     - Confidence: Model confidence
     - Strength: Signal strength
     - Reasoning: Explanation

4. **Analyze a Symbol**:
   - Click "📊 Analyze Symbol"
   - Enter stock ticker
   - Strategy will:
     - Fetch latest data
     - Generate features
     - Make prediction
     - Generate signal
     - Calculate position size
     - Apply risk management
   - Results shown in popup

---

### 📋 Logs Tab

**Purpose:** View real-time application logs

**Features:**
- Color-coded log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Automatic scrolling
- Clear log button

---

### ⚙️ Configuration Tab

**Purpose:** Manage bot configuration (Coming Soon)

Will include:
- Trading parameters
- Risk management settings
- Model configuration
- Feature engineering settings

---

### 🔑 API Keys Tab

**Purpose:** Manage API keys securely (Coming Soon)

Will include:
- Exchange API key management
- Market data API keys
- Secure storage via OS keyring

---

## Workflow Example

### Training and Using a New Model

1. **Train in Colab**:
   - Go to Models tab
   - Click "🚀 Open Training Notebook"
   - Upload to Google Colab
   - Enable GPU runtime
   - Run all cells
   - Download 3 files

2. **Upload to Bot**:
   - Go to Models tab
   - Click "📤 Upload Model Files"
   - Select the 3 downloaded files
   - Or manually copy to `models/` folder

3. **Load Model**:
   - Click "🔄 Refresh" if needed
   - Select your new model
   - Click "📥 Load Model"
   - Verify "✅ Loaded" status

4. **Make Predictions**:
   - Go to Predictions tab
   - Enter symbol (e.g., "AAPL")
   - Click "🔮 Predict"
   - Review results

5. **Enable Auto-Predict**:
   - Click "▶️ Auto Predict"
   - Let it run continuously
   - Monitor in Strategy tab

6. **Monitor Performance**:
   - Go to Strategy tab
   - Watch metrics update
   - Review signal history
   - Track active positions

---

## Tips & Best Practices

### Model Management

✅ **DO**:
- Keep models organized with clear version numbers
- Test new models thoroughly before using in production
- Retrain models weekly or monthly with fresh data
- Keep the 3 most recent versions

❌ **DON'T**:
- Delete all model versions at once
- Use models older than the retrain interval
- Load models without checking metadata first

### Predictions

✅ **DO**:
- Start with well-known, liquid stocks (AAPL, GOOGL, MSFT)
- Check confidence scores before acting on signals
- Use auto-predict for continuous monitoring
- Review prediction history for patterns

❌ **DON'T**:
- Trade based on low-confidence predictions (<50%)
- Use predictions for highly volatile or illiquid stocks
- Ignore the reasoning field
- Make decisions on a single prediction

### Strategy

✅ **DO**:
- Monitor signal history regularly
- Track position performance
- Use proper position sizing
- Set stop-loss and take-profit levels

❌ **DON'T**:
- Execute every signal without review
- Exceed your risk tolerance
- Hold positions without stop-losses
- Ignore warning signals

---

## Troubleshooting

### "No Model" Error

**Problem**: Clicking Predict shows "Please load a model first"

**Solution**:
1. Go to Models tab
2. Select a model
3. Click "📥 Load Model"
4. Wait for "✅ Loaded" status

---

### "Prediction Failed" Error

**Possible Causes**:
1. **Invalid Symbol**: Check ticker is correct
2. **No Data**: Symbol might not have enough history
3. **Feature Mismatch**: Model features don't match current config
4. **Network Issue**: Can't fetch data from yfinance

**Solutions**:
- Verify symbol exists (try on Yahoo Finance)
- Check internet connection
- Try a different, well-known symbol
- Reload the model
- Check logs tab for detailed error

---

### Model Won't Load

**Possible Causes**:
1. Missing files (need all 3: model, metadata, scaler)
2. Corrupted files
3. Version mismatch

**Solutions**:
- Verify all 3 files are present
- Re-download from Colab
- Retrain the model
- Check file permissions

---

### Auto-Predict Stops Working

**Possible Causes**:
1. Network connection lost
2. Rate limit hit (too many requests)
3. Model became unloaded

**Solutions**:
- Check internet connection
- Pause auto-predict for a few minutes
- Reload the model
- Check logs for errors

---

## Keyboard Shortcuts

- `Ctrl+Q`: Quit application
- Tab selection: Use arrow keys or Ctrl+Tab

---

## Advanced Features

### Custom Symbols

You can predict any stock symbol that yfinance supports:
- US stocks: "AAPL", "GOOGL", "MSFT", "TSLA"
- International: "ASML", "TSM", etc.
- Just enter the correct ticker

### Multiple Models

You can keep multiple models for different:
- Symbols (AAPL model, GOOGL model)
- Strategies (short-term vs long-term)
- Market conditions (bull vs bear)

Switch between them easily in the Models tab.

---

## Performance Optimization

### For Best Results:

1. **Model Quality**:
   - Train on at least 2 years of data
   - Use 60+ day lookback window
   - Retrain regularly (weekly/monthly)

2. **Prediction Accuracy**:
   - Use liquid, high-volume stocks
   - Check confidence scores
   - Compare against multiple symbols

3. **GUI Performance**:
   - Close unused tabs
   - Clear prediction history regularly
   - Don't run too many auto-predictions simultaneously

---

## Getting Help

1. **Check Logs**: Always check the Logs tab first
2. **Review Documentation**: See `docs/XGBOOST_INTEGRATION.md`
3. **Model Issues**: Retrain or re-download model
4. **Bug Reports**: Include logs when reporting issues

---

## Future Enhancements

Planned features:
- 📈 Real-time price charts
- 📊 Performance analytics dashboard
- 🔔 Alert system for signals
- 💾 Export predictions to CSV
- 🎨 Customizable themes
- 📱 Mobile-friendly interface

---

**Version**: 0.1.0  
**Last Updated**: October 2024

