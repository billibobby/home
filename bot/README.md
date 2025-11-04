# AI Trading Bot

## Overview

A professional AI-powered trading bot for cryptocurrency and stock markets. This bot provides a scalable foundation for automated trading with machine learning capabilities, real-time data processing, and risk management.

## Features

- Multi-exchange support (Binance, Coinbase, Alpaca)
- XGBoost-based stock prediction with GPU training support
- Hybrid training-deployment architecture (Colab GPU + Local CPU)
- Technical indicator-based feature engineering
- Configurable trading strategies
- Machine learning model integration
- Comprehensive logging and monitoring
- Risk management controls
- Modular architecture for easy extension

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd bot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration Setup

#### Environment Variables

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` file and add your API credentials:
- Exchange API keys (Binance, Coinbase, Alpaca)
- Alpha Vantage API key for stock data
- Other configuration parameters

#### Configuration Files

The main configuration is in `config/config.yaml`. Adjust trading parameters, risk settings, and model configurations as needed.

## Usage

### Running the Bot

```bash
python main.py
```

### Validating Setup

Before running the bot, validate your configuration:

```bash
python scripts/validate_setup.py
```

### Running Tests

```bash
pytest tests/
```

## Project Structure

```
bot/
├── config/              # Configuration files
│   ├── config.yaml      # Main configuration
│   └── __init__.py
├── data/                # Historical data and cache
├── logs/                # Application logs
├── models/              # Trained ML models
├── scripts/             # Utility scripts
│   ├── setup_env.py     # Environment setup
│   └── validate_setup.py # Setup validation
├── src/                 # Main source code
│   ├── config_loader.py # Configuration management
│   ├── logger.py        # Logging setup
│   ├── data/            # Data collection modules
│   ├── models/          # ML model implementations
│   ├── trading/         # Trading logic and execution
│   └── utils/           # Utility functions
│       ├── helpers.py
│       └── exceptions.py
├── tests/               # Unit and integration tests
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── setup.py             # Package setup
└── .env.example         # Environment template
```

## Configuration

### Environment Variables

The bot supports environment variable overrides for any configuration key. Environment variables take precedence over YAML config and defaults.

#### Environment Variable Naming Convention

Configuration keys in YAML use dot notation (e.g., `logging.level`, `trading.max_positions`). To override via environment variables, convert dots to underscores and use uppercase:

**Conversion Rule:** `config.key.path` → `CONFIG_KEY_PATH`

**Examples:**
- `logging.level` → `LOGGING_LEVEL`
- `trading.max_positions` → `TRADING_MAX_POSITIONS`
- `models.xgboost.device` → `MODELS_XGBOOST_DEVICE`
- `models.prediction.buy_threshold` → `MODELS_PREDICTION_BUY_THRESHOLD`
- `data.cache_historical_data` → `DATA_CACHE_HISTORICAL_DATA`
- `data.historical_data_path` → `DATA_HISTORICAL_DATA_PATH`

**Precedence Order:**
1. Environment variables (highest priority)
2. YAML configuration file
3. Default values (lowest priority)

**Common Environment Variables:**
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENVIRONMENT`: Runtime environment (development, production)
- `LOGGING_LEVEL`: Override logging.level from config.yaml
- `TRADING_MAX_POSITIONS`: Override trading.max_positions
- `MODELS_XGBOOST_DEVICE`: Override models.xgboost.device (cpu, gpu, cuda)
- `DATA_CACHE_HISTORICAL_DATA`: Override data.cache_historical_data (true/false)
- Exchange API credentials
- Database and storage paths

### Trading Parameters

Configure in `config/config.yaml`:
- Position sizing
- Risk percentage per trade
- Maximum concurrent positions
- Stop-loss and take-profit levels
- XGBoost model settings and hyperparameters
- Feature engineering configuration
- Prediction thresholds

For detailed XGBoost integration documentation, see [docs/XGBOOST_INTEGRATION.md](docs/XGBOOST_INTEGRATION.md).

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure
- [x] Configuration management
- [x] Logging system
- [x] Utility functions

### Phase 2: Data Collection
- [x] Historical data fetching (yfinance)
- [x] Data storage and caching
- [x] Feature engineering pipeline
- [ ] Exchange API integrations
- [ ] Real-time data streaming

### Phase 3: ML Models
- [x] XGBoost model training (GPU)
- [x] Model prediction and inference
- [x] Feature engineering (lagged prices, technical indicators)
- [x] Model management and versioning
- [ ] Backtesting framework
- [ ] Model evaluation metrics

### Phase 4: Trading Logic
- [ ] Signal generation
- [ ] Order execution
- [ ] Risk management
- [ ] Portfolio management

### Phase 5: Production
- [ ] Live trading mode
- [ ] Performance monitoring
- [ ] Alert system
- [ ] Dashboard

## Safety and Disclaimer

**IMPORTANT**: This software is for educational purposes. Trading cryptocurrencies and stocks involves substantial risk of loss. Always:

- Test thoroughly in paper trading mode
- Start with small amounts
- Never invest more than you can afford to lose
- Understand the risks involved
- Comply with local regulations

## Contributing

Contributions are welcome! Please follow the existing code style and include tests for new features.

## License

[Add your license here]

## Documentation

- [XGBoost Integration Guide](docs/XGBOOST_INTEGRATION.md) - Comprehensive guide for ML model integration
- [XGBoost GUI User Guide](docs/GUI_XGBOOST_GUIDE.md) - Complete GUI walkthrough for model management
- [Quick Start Guide](QUICK_START.md) - Get started quickly
- [GUI Packaging Guide](docs/GUI_PACKAGING_GUIDE.md) - Building executables
- [GUI Threading Model](docs/GUI_THREADING_MODEL.md) - Thread safety patterns

## Support

For issues and questions, please open an issue on the repository.

