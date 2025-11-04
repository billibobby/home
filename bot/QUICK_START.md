# Quick Start Guide

## For End Users (GUI)

### Installation

1. Download the latest release:
   - Windows: `TradingBotGUI.exe`
   - Linux: `TradingBotGUI`
   - macOS: `TradingBotGUI.app`

2. Run the executable:
   - Windows: Double-click `TradingBotGUI.exe`
   - Linux: `chmod +x TradingBotGUI && ./TradingBotGUI`
   - macOS: Double-click `TradingBotGUI.app`

3. Configure API keys (future):
   - Go to "API Keys" tab
   - Enter your exchange credentials
   - Click "Save" (stored securely in OS keyring)

### First Run

- Logs are displayed in the GUI
- Bot status shown at top
- Use Start/Stop buttons to control
- Files stored in user directory (see below)

### User Data Locations

**Windows:**
```
C:\Users\<YourName>\AppData\Roaming\AITradingBot\
├── logs\
├── data\
└── models\
```

**Linux:**
```
/home/<username>/.local/share/ai-trading-bot/
├── logs/
├── data/
└── models/
```

**macOS:**
```
/Users/<username>/Library/Application Support/AITradingBot/
├── logs/
├── data/
└── models/
```

---

## For Developers (From Source)

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
cd bot

# Install core dependencies
pip install -e .

# Or install with GUI support
pip install -e .[gui]

# Or install everything (GUI + dev tools)
pip install -e .[all]
```

### Running

**CLI Mode:**
```bash
trading-bot
```

**GUI Mode:**
```bash
trading-bot-gui
```

**From Python:**
```bash
# CLI
python -m trading_bot.main

# GUI
python -m trading_bot.gui_main
```

### Configuration

1. Copy environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   ENVIRONMENT=development
   LOG_LEVEL=DEBUG
   BINANCE_API_KEY=your_key_here
   BINANCE_API_SECRET=your_secret_here
   ```

#### Environment Variable Overrides

You can override any configuration value from `config.yaml` using environment variables. The naming convention converts dots to underscores and uses uppercase:

**Examples:**
- `LOGGING_LEVEL=DEBUG` (overrides `logging.level`)
- `TRADING_MAX_POSITIONS=10` (overrides `trading.max_positions`)
- `MODELS_XGBOOST_DEVICE=gpu` (overrides `models.xgboost.device`)
- `MODELS_PREDICTION_BUY_THRESHOLD=0.7` (overrides `models.prediction.buy_threshold`)
- `DATA_CACHE_HISTORICAL_DATA=false` (overrides `data.cache_historical_data`)
- `DATA_HISTORICAL_DATA_PATH=/custom/path` (overrides `data.historical_data_path`)

**Precedence:** Environment variables > YAML config > Defaults

3. Or use keyring (recommended):
   ```python
   from trading_bot.utils.secrets_store import store_api_key
   
   store_api_key('binance', 'api_key', 'your_key')
   store_api_key('binance', 'api_secret', 'your_secret')
   ```

### Development

**Run Tests:**
```bash
pip install -e .[dev]
pytest
```

**Code Formatting:**
```bash
black src/ tests/
```

**Type Checking:**
```bash
mypy src/
```

### Building Executable

```bash
# Install build dependencies
pip install -e .[build,gui]

# Run build script
python scripts/build_gui.py

# Output: dist/TradingBotGUI[.exe]
```

---

## Model Setup (XGBoost)

### Option 1: Use Pre-trained Model (Recommended for Testing)

1. **Download a pre-trained model** from releases (if available)

2. **Place in models/ directory:**
   ```
   models/
   ├── xgboost_AAPL_v1.json
   ├── xgboost_AAPL_v1_metadata.json
   └── scaler_AAPL_v1.pkl
   ```

3. **Update config/config.yaml:**
   ```yaml
   models:
     xgboost:
       enabled: true
       model_file: 'xgboost_AAPL_v1.json'
   ```

### Option 2: Train Your Own Model

1. **Open the Colab notebook:**
   - Navigate to `notebooks/xgboost_training_colab.ipynb`
   - Upload to Google Colab
   - Enable GPU runtime (Runtime → Change runtime type → T4 GPU)

2. **Run all cells** to train model:
   - Downloads historical stock data
   - Engineers features
   - Trains XGBoost with GPU acceleration
   - Exports model files

3. **Download trained model files:**
   - `xgboost_SYMBOL_VERSION_DATE.json` (model)
   - `xgboost_SYMBOL_VERSION_DATE_metadata.json` (metadata)
   - `scaler_SYMBOL_VERSION_DATE.pkl` (scaler)

4. **Place in models/ directory** and update config as above

5. **Verify Model:**
   ```bash
   python -c "from trading_bot.models import ModelManager; mm = ModelManager(None, None); print(mm.list_available_models())"
   ```

For detailed instructions, see [docs/XGBOOST_INTEGRATION.md](docs/XGBOOST_INTEGRATION.md)

---

## API Key Setup

### Option 1: OS Keyring (Recommended)

**Advantages:**
- Secure (OS-level encryption)
- No plaintext files
- Best for GUI usage

**Python:**
```python
from trading_bot.utils.secrets_store import store_api_key

# Binance
store_api_key('binance', 'api_key', 'YOUR_API_KEY')
store_api_key('binance', 'api_secret', 'YOUR_API_SECRET')

# Coinbase
store_api_key('coinbase', 'api_key', 'YOUR_API_KEY')
store_api_key('coinbase', 'api_secret', 'YOUR_API_SECRET')

# Alpaca
store_api_key('alpaca', 'api_key', 'YOUR_API_KEY')
store_api_key('alpaca', 'api_secret', 'YOUR_API_SECRET')

# Alpha Vantage
store_api_key('alpha_vantage', 'api_key', 'YOUR_API_KEY')
```

### Option 2: Environment Variables

**Advantages:**
- Simple for CLI usage
- Good for headless/CI environments

**Setup:**
```bash
# .env file
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

**Or export directly:**
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### Getting API Keys

**Binance:**
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create new API key
3. Enable "Spot & Margin Trading" (if needed)
4. Save key and secret securely

**Coinbase:**
1. Go to https://www.coinbase.com/settings/api
2. Create new API key
3. Set permissions (view/trade)
4. Save key and secret

**Alpaca (Stock Trading):**
1. Go to https://app.alpaca.markets/paper/dashboard/overview
2. Generate API keys (use paper trading first!)
3. Save key and secret

**Alpha Vantage (Market Data):**
1. Go to https://www.alphavantage.co/support/#api-key
2. Get free API key
3. Save key

---

## Usage Examples

### CLI Mode

```bash
# Start bot
trading-bot

# With custom config
CONFIG_PATH=/path/to/config.yaml trading-bot

# With custom log level
LOG_LEVEL=DEBUG trading-bot
```

### GUI Mode

```bash
# Start GUI
trading-bot-gui

# Everything is controlled through the GUI
```

### Python API

```python
from trading_bot.app import BotApp

# Create bot instance
app = BotApp()

# Initialize
success = app.initialize(
    enable_console=True,  # False for GUI
    status_callback=print  # Optional progress callback
)

if success:
    # Start bot
    app.start(blocking=False)
    
    # Do other work...
    
    # Stop bot
    app.stop()
    
    # Cleanup
    app.shutdown()
```

---

## Configuration Files

### config/config.yaml

Main configuration file:

```yaml
# Environment
environment: development

# Logging
logging:
  level: INFO
  dir: logs
  console_colors: true
  file_rotation:
    max_bytes: 10485760  # 10 MB
    backup_count: 5

# Trading
trading:
  default_position_size: 100
  risk_percentage: 2
  max_positions: 5

# Models
models:
  default_model: random_forest
  prediction:
    buy_threshold: 0.6
    sell_threshold: 0.6

# Paths
database_path: data/trading_bot.db
model_path: models/
data_path: data/
```

---

## Troubleshooting

### "Module not found" Error

**Solution:**
```bash
# Reinstall with correct extras
pip install -e .[gui]
```

### "Cannot import PySide6"

**Solution:**
```bash
pip install PySide6
# or
pip install -e .[gui]
```

### "Keyring backend not available"

**Linux:**
```bash
# Install keyring backend
sudo apt install gnome-keyring  # GNOME
# or
sudo apt install kwallet         # KDE
```

**Windows/macOS:** Built-in, should work automatically

**Workaround:** Use environment variables instead

### Logs Not Found

Check platform-specific location:

```python
from trading_bot.utils.paths import get_writable_app_dir
print(get_writable_app_dir('logs'))
```

### Permission Errors

**Problem:** Can't write logs/data

**Solution:** 
- Bot now writes to user directory (no admin required)
- If using custom paths, ensure they're writable

---

## Next Steps

1. **Configure API Keys:** Use keyring or .env
2. **Test Connection:** Run bot to verify API keys work
3. **Review Logs:** Check user directory for log files
4. **Customize Config:** Edit config/config.yaml as needed
5. **Explore Features:** Check GUI tabs for options

---

## Getting Help

- **Documentation:** See `docs/` directory
- **Build Issues:** See `docs/GUI_PACKAGING_GUIDE.md`
- **Threading:** See `docs/GUI_THREADING_MODEL.md`
- **Implementation:** See `PACKAGING_IMPLEMENTATION.md`

---

## Safety Notice

⚠️ **IMPORTANT:**

- **Start with paper trading** to test strategies
- **Never risk more than you can afford to lose**
- **Thoroughly test** before using with real money
- **API keys** should have appropriate permissions only
- **Review all trades** before enabling live trading

This bot is currently in **foundation phase** and does not yet execute trades automatically. Trading functionality will be implemented in future versions.

---

## Resources

- **PyInstaller:** https://pyinstaller.org/
- **PySide6:** https://doc.qt.io/qtforpython/
- **Keyring:** https://github.com/jaraco/keyring
- **Binance API:** https://binance-docs.github.io/apidocs/
- **Alpaca API:** https://alpaca.markets/docs/

---

**Version:** 0.1.0  
**Status:** Foundation Phase  
**License:** See LICENSE file

