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
- Persistent SQLite database for trade history and performance tracking
- **Paper Trading Engine**: Simulated order execution with realistic slippage, commission, and portfolio tracking for risk-free strategy testing
- **Walk-forward backtesting framework**: Realistic transaction costs, comprehensive performance metrics, robustness testing, and visualization suite

## Database & Persistence

The trading bot uses SQLite for persistent storage of historical trades, positions, and performance metrics.

### Database Location

- **Development**: `data/trading_bot.db` (relative to project root)
- **Production**: Platform-specific writable directory (e.g., `%APPDATA%/AITradingBot/data/trading_bot.db` on Windows)

### Connection Management

The database manager uses **thread-local connections**, not a traditional connection pool. Each thread maintains its own SQLite connection that is reused across operations within that thread. The `connection_pool_size` configuration parameter is currently unused but reserved for future implementation.

**Important**: The `close()` method only closes the current thread's connection. In multi-threaded applications, ensure each thread calls `close()` when done, or let threads terminate cleanly to allow automatic connection cleanup.

### Database Schema

**Tables:**

- `trades` - Historical trade records with entry/exit prices, PnL, and metadata
- `positions` - Currently open positions with stop-loss/take-profit levels
- `portfolio_snapshots` - Periodic portfolio state for performance tracking
- `performance_metrics` - Calculated metrics (Sharpe, Sortino, MDD, etc.)

### Usage Example

```python
from trading_bot.data import DatabaseManager
from trading_bot.config_loader import Config
import logging

config = Config()
logger = logging.getLogger('trading_bot')
db = DatabaseManager(config, logger)

# Insert a trade
trade_id = db.insert_trade(
    symbol='AAPL',
    side='BUY',
    entry_price=150.0,
    exit_price=155.0,
    quantity=10,
    entry_time='2024-01-01T10:00:00',
    exit_time='2024-01-02T15:00:00',
    timeframe='1d',
    strategy='XGBoost'
)

# Get trade statistics
stats = db.get_trade_statistics(symbol='AAPL')
print(f"Win rate: {stats['win_rate']:.2%}")
```

### Backup & Maintenance

Automatic backups are created based on `database.backup_interval_hours` setting in `config.yaml`. Manual backup:

```python
db.create_backup()  # Creates timestamped backup in backups/ directory
```

Database maintenance:

```python
db.vacuum_database()  # Optimize database
db.check_integrity()  # Verify database health
db.cleanup_old_data(retention_days=365)  # Remove old records
```

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
│   ├── backtesting/     # Walk-forward backtesting framework
│   │   ├── walk_forward.py
│   │   ├── costs.py
│   │   ├── metrics.py
│   │   ├── results.py
│   │   ├── robustness.py
│   │   └── visualizations.py
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

## Paper Trading

### Overview

The paper trading engine allows you to test trading strategies with simulated money before risking real capital. It provides realistic simulation of:

- **Order execution** with configurable slippage and delays
- **Trading commissions** based on percentage of trade value
- **Portfolio tracking** with positions, cash balance, and equity
- **Stop-loss and take-profit** automatic position management
- **Performance metrics** including win rate, profit factor, Sharpe ratio, and drawdown

All trades and positions are persisted in the database for detailed analysis and backtesting.

### Configuration

Paper trading is configured in the `paper_trading` section of `config.yaml`:

```yaml
paper_trading:
  enabled: true  # Toggle paper trading mode
  initial_balance: 10000.0  # Starting cash balance (USD)
  commission: 0.1  # Commission as percentage (0.1 = 0.1%)
  slippage:
    enabled: true  # Toggle slippage simulation
    min_percentage: 0.01  # Minimum slippage (0.01 = 0.01%)
    max_percentage: 0.1  # Maximum slippage (0.1 = 0.1%)
    market_order_slippage: 0.2  # Higher slippage for market orders
  execution_delay_ms: 100  # Simulated execution delay (milliseconds)
  partial_fills: false  # Enable partial fill simulation
  reset_on_startup: false  # Reset portfolio on application startup
```

### Usage Example

```python
from trading_bot.trading import PaperTradingEngine
from trading_bot.data import DatabaseManager
from trading_bot.config_loader import Config
import logging

config = Config()
logger = logging.getLogger('trading_bot')
db = DatabaseManager(config, logger)

# Initialize paper trading engine
engine = PaperTradingEngine(config, logger, db)

# Execute a buy order
result = engine.execute_buy_order('AAPL', quantity=10.0, price=150.0)
if result['success']:
    print(f"Order executed at ${result['execution_price']:.2f}")
    print(f"Slippage: {result['slippage']:.2f}%")
    print(f"Commission: ${result['commission']:.2f}")

# Check portfolio status
summary = engine.get_account_summary()
print(f"Cash: ${summary['cash_balance']:.2f}")
print(f"Total Equity: ${summary['total_equity']:.2f}")
print(f"Positions: {summary['num_positions']}")

# View performance metrics
metrics = engine.calculate_performance_metrics()
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
```

### Integration with Strategy

The `XGBoostStrategy` can be connected to the `PaperTradingEngine` for automatic execution of trading decisions:

```python
from trading_bot.trading import XGBoostStrategy, PaperTradingEngine, SignalGenerator

# Initialize components
engine = PaperTradingEngine(config, logger, db)
strategy = XGBoostStrategy(
    config, logger, predictor, signal_generator,
    paper_trading_engine=engine
)

# Analyze and execute
decision = strategy.analyze('AAPL', account_balance=10000.0)
if decision:
    result = strategy.execute_decision(decision)
    print(f"Execution result: {result['success']}")
```

Signals from `SignalGenerator` are automatically converted to simulated orders when using the `execute_signal()` method. Position tracking is synchronized between the strategy and the paper trading engine.

### Testing and Validation

Comprehensive unit tests are available in `tests/test_paper_trading.py`. Run tests before using in production scenarios:

```bash
pytest tests/test_paper_trading.py -v
```

The test suite covers:
- Portfolio initialization and state management
- Buy and sell order execution
- Position management and updates
- Stop-loss and take-profit triggers
- Performance metrics calculation
- Signal integration
- Order validation and error handling

## Backtesting

### Overview

The walk-forward backtesting framework validates trading strategies before deployment using out-of-sample testing to prevent overfitting.

### Features

- **Walk-forward analysis**: Rolling train/test windows
- **Realistic costs**: Commission, slippage, spread, market impact
- **Comprehensive metrics**: Sharpe, Sortino, Calmar, drawdown, win rate
- **Robustness testing**: Parameter sensitivity, time period stability
- **Visualization**: Equity curves, drawdown plots, heatmaps
- **GUI and CLI**: Easy-to-use interfaces

### Quick Start

#### Using CLI:

```bash
python scripts/run_backtest.py \
    --symbol AAPL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --generate-report
```

#### Using Python API:

```python
from trading_bot.backtesting import WalkForwardBacktest
from trading_bot.data import StockDataFetcher, FeatureEngineer
from trading_bot.trading import SignalGenerator
from trading_bot.models import XGBoostTrainer

# Fetch data
fetcher = StockDataFetcher(config, logger)
data = fetcher.fetch_historical_data('AAPL', '2020-01-01', '2023-12-31')

# Create backtest
backtest = WalkForwardBacktest(
    data=data,
    config=config,
    logger=logger,
    feature_engineer=FeatureEngineer(config, logger),
    signal_generator=SignalGenerator(config, logger),
    model_class=XGBoostTrainer
)

# Run
results = backtest.run()
print(f"Sharpe Ratio: {results.sharpe_ratio}")
results.plot_equity_curve()
```

### Configuration

Configure in `config.yaml`:

```yaml
backtesting:
  walk_forward:
    train_period_days: 252
    test_period_days: 21
    step_size_days: 21
  transaction_costs:
    commission_pct: 0.001
    slippage_bps: 5
```

For detailed documentation, see [docs/BACKTESTING_GUIDE.md](docs/BACKTESTING_GUIDE.md).

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
- [x] Backtesting framework
- [x] Model evaluation metrics

### Phase 4: Trading Logic
- [x] Signal generation
- [x] Paper trading engine
- [x] Order execution simulation
- [x] Risk management
- [x] Portfolio management

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
- [Backtesting Guide](docs/BACKTESTING_GUIDE.md) - Walk-forward backtesting framework
- [Quick Start Guide](QUICK_START.md) - Get started quickly
- [GUI Packaging Guide](docs/GUI_PACKAGING_GUIDE.md) - Building executables
- [GUI Threading Model](docs/GUI_THREADING_MODEL.md) - Thread safety patterns

## Support

For issues and questions, please open an issue on the repository.

