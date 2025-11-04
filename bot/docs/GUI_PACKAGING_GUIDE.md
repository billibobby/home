# Trading Bot GUI Packaging Guide

## Overview

This guide provides comprehensive instructions for packaging and distributing the Trading Bot as a standalone GUI application.

## Implementation Summary

All 10 comments from the code review have been implemented:

### ✓ 1. Packaging-Aware Path Resolution
- **File**: `src/trading_bot/utils/paths.py`
- **Features**:
  - `resolve_resource_path()`: Handles frozen executable paths (PyInstaller)
  - `get_writable_app_dir()`: Platform-specific user directories
  - Auto-detects frozen state via `sys.frozen` and `sys._MEIPASS`
- **Updated**: `config_loader.py`, `logger.py` to use new path functions

### ✓ 2. GUI Entry Point
- **Files**: 
  - `src/trading_bot/gui/` package (main_window.py, __init__.py)
  - `src/trading_bot/gui_main.py` entry point
- **Console Scripts**:
  - `trading-bot`: CLI entry point
  - `trading-bot-gui`: GUI entry point

### ✓ 3. GUI-Friendly Logging
- **File**: `src/trading_bot/logger.py`
- **Features**:
  - `enable_console` flag (default True, set False for GUI)
  - `UILogHandler` class for thread-safe GUI log display
  - Queue-based log passing to avoid blocking

### ✓ 4. User-Writable Log Directory
- **Implementation**: Uses platform-specific paths
  - Windows: `%APPDATA%/AITradingBot/logs`
  - Linux: `~/.local/share/ai-trading-bot/logs`
  - macOS: `~/Library/Application Support/AITradingBot/logs`
- **Fallback**: Config can override with custom path

### ✓ 5. Keyring Integration
- **File**: `src/trading_bot/utils/secrets_store.py`
- **Features**:
  - OS keyring for secure credential storage
  - Fallback to environment variables
  - `store_api_key()`, `get_api_key()`, `delete_api_key()`
  - Migration helper from .env to keyring
- **Updated**: `helpers.py` to use keyring first

### ✓ 6. Service Layer (BotApp)
- **File**: `src/trading_bot/app.py`
- **Features**:
  - `BotApp` class with `initialize()`, `start()`, `stop()`, `status()`
  - Decouples bot logic from UI/CLI
  - Status enum for state tracking
  - Status callbacks for GUI splash screens
- **Updated**: `main.py` to use BotApp

### ✓ 7. Background Worker Model
- **File**: `docs/GUI_THREADING_MODEL.md`
- **Features**:
  - Threading best practices documentation
  - Stop event pattern for graceful shutdown
  - Thread-safe logging examples
  - Full GUI integration example

### ✓ 8. PyInstaller Packaging
- **Files**:
  - `build/gui.spec`: PyInstaller specification
  - `scripts/build_gui.py`: Automated build script
- **Features**:
  - Bundles config files and dependencies
  - Hidden imports for keyring backends
  - Excludes dev/test packages
  - Windowed mode (no console)

### ✓ 9. Logging Instead of Print
- **File**: `src/trading_bot/config_loader.py`
- **Changes**:
  - Replaced all `print()` with `logger.info/warning/error()`
  - Added optional `status_callback` for GUI integration
  - Fallback to logging if callback not provided

### ✓ 10. Optimized Dependencies
- **File**: `setup.py`
- **Changes**:
  - Separated core, GUI, dev, and build requirements
  - `extras_require` for optional features
  - Moved pytest to dev-only
  - Added keyring to core requirements

## Installation

### For Development

```bash
# Core + GUI dependencies
pip install -e .[gui]

# Include dev tools
pip install -e .[dev,gui]

# Everything (including PyInstaller)
pip install -e .[all]
```

### For End Users

```bash
# Core functionality only
pip install ai-trading-bot

# With GUI
pip install ai-trading-bot[gui]
```

## Running the Application

### CLI Mode
```bash
trading-bot
```

### GUI Mode
```bash
trading-bot-gui
```

### From Source
```bash
# CLI
python -m trading_bot.main

# GUI
python -m trading_bot.gui_main
```

## Building the Executable

### Prerequisites

1. Install build dependencies:
   ```bash
   pip install -e .[gui,build]
   ```

2. Ensure all tests pass:
   ```bash
   pytest
   ```

### Build Process

#### Option 1: Using Build Script (Recommended)
```bash
python scripts/build_gui.py
```

This script:
- Checks dependencies
- Cleans previous builds
- Runs PyInstaller
- Verifies output
- Reports executable size

#### Option 2: Manual PyInstaller
```bash
pyinstaller build/gui.spec --clean
```

### Output

Executable will be created in `dist/`:
- Windows: `dist/TradingBotGUI.exe`
- Linux: `dist/TradingBotGUI`
- macOS: `dist/TradingBotGUI.app`

## Directory Structure After Build

```
dist/
├── TradingBotGUI[.exe]     # Standalone executable
└── config/                  # Bundled config files
    └── config.yaml

User Data Locations:
Windows:  C:\Users\<user>\AppData\Roaming\AITradingBot\
Linux:    /home/<user>/.local/share/ai-trading-bot/
macOS:    /Users/<user>/Library/Application Support/AITradingBot/

Each contains:
├── logs/                    # Application logs
├── data/                    # Trading data
└── models/                  # ML models
```

## Configuration Management

### Bundled Config
The executable includes `config/config.yaml` as read-only template.

### User Config Override
Users can create custom configs at:
- Windows: `%APPDATA%/AITradingBot/config/config.yaml`
- Linux: `~/.local/share/ai-trading-bot/config/config.yaml`
- macOS: `~/Library/Application Support/AITradingBot/config/config.yaml`

The application checks user config first, then falls back to bundled config.

## API Key Management

### For End Users (GUI)

API keys are stored securely in OS keyring:
- Windows: Credential Manager
- macOS: Keychain
- Linux: Secret Service / KWallet / GNOME Keyring

In the GUI (future implementation):
1. Go to "API Keys" tab
2. Enter credentials
3. Click "Save" to store in keyring

### For Developers (CLI)

```python
from trading_bot.utils.secrets_store import store_api_key, get_api_key

# Store a key
store_api_key('binance', 'api_key', 'your_key_here')
store_api_key('binance', 'api_secret', 'your_secret_here')

# Retrieve a key
api_key = get_api_key('binance', 'api_key')
```

### Migration from .env

```python
from trading_bot.utils.secrets_store import migrate_from_env_to_keyring

# Migrate Binance keys
migrate_from_env_to_keyring('binance', {
    'api_key': 'BINANCE_API_KEY',
    'api_secret': 'BINANCE_API_SECRET'
})
```

### Fallback for Headless Environments

Keys are still read from environment variables if keyring is unavailable:
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

## Distribution

### Windows

1. Build executable: `python scripts/build_gui.py`
2. Test on clean Windows machine
3. Optional: Sign executable with `signtool`
4. Create installer (optional): Use Inno Setup or NSIS
5. Distribute `TradingBotGUI.exe`

### Linux

1. Build executable: `python scripts/build_gui.py`
2. Create `.desktop` file:
   ```ini
   [Desktop Entry]
   Name=AI Trading Bot
   Exec=/path/to/TradingBotGUI
   Icon=/path/to/icon.png
   Type=Application
   Categories=Office;Finance;
   ```
3. Optional: Create `.deb` or `.rpm` package
4. Distribute binary + desktop file

### macOS

1. Build executable: `python scripts/build_gui.py`
2. Optional: Code sign: `codesign -s "Developer ID" TradingBotGUI.app`
3. Optional: Notarize with Apple
4. Create DMG: `hdiutil create -volname "Trading Bot" -srcfolder dist/ -ov TradingBot.dmg`
5. Distribute DMG

## Troubleshooting

### Import Errors in Executable

**Problem**: Module not found errors when running executable

**Solutions**:
1. Add missing module to `hiddenimports` in `build/gui.spec`
2. Check for dynamic imports and make them explicit
3. Test in clean environment

### Large Executable Size

**Problem**: Executable is > 100 MB

**Solutions**:
1. Ensure dev dependencies are excluded (already done in spec)
2. Use UPX compression (enabled in spec)
3. Consider separate data files instead of bundling

### Keyring Not Working

**Problem**: Can't store/retrieve API keys

**Solutions**:
1. Ensure keyring backend is installed:
   - Windows: Built-in Credential Manager
   - Linux: `apt install gnome-keyring` or `apt install kwallet`
   - macOS: Built-in Keychain
2. Check keyring availability: `from trading_bot.utils.secrets_store import is_keyring_available`
3. Fall back to environment variables

### Logs Not Found

**Problem**: Can't find log files

**Solution**: Check platform-specific location:
```python
from trading_bot.utils.paths import get_writable_app_dir
print(get_writable_app_dir('logs'))
```

### Config Not Loading

**Problem**: Custom config not being read

**Solutions**:
1. Verify path: Use `resolve_resource_path('config/config.yaml')`
2. Check file permissions
3. Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

## Testing the Executable

### Manual Testing Checklist

- [ ] Executable starts without errors
- [ ] GUI window appears
- [ ] Logs display in GUI
- [ ] Config loads correctly
- [ ] API key validation works
- [ ] Bot can start/stop
- [ ] Logs are written to user directory
- [ ] Application closes cleanly
- [ ] No console window appears (Windows)

### Automated Testing

```bash
# Run tests before building
pytest tests/

# Test packaging
python scripts/build_gui.py

# Test executable (manual)
./dist/TradingBotGUI
```

## Future Enhancements

### GUI Features
- [ ] Settings panel for config editing
- [ ] API key management UI
- [ ] Real-time trading dashboard
- [ ] Performance charts
- [ ] Notification system

### Packaging
- [ ] Auto-update mechanism
- [ ] Crash reporting
- [ ] Analytics (opt-in)
- [ ] Multi-language support
- [ ] Custom icons and branding

### Distribution
- [ ] Windows Store package
- [ ] macOS App Store
- [ ] Snap/Flatpak for Linux
- [ ] Docker image

## Resources

- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [PySide6 Documentation](https://doc.qt.io/qtforpython/)
- [Python Keyring](https://github.com/jaraco/keyring)
- [Threading Model Guide](./GUI_THREADING_MODEL.md)

## Support

For issues related to:
- **Building**: Check `scripts/build_gui.py` output
- **Running**: Check log files in user directory
- **Configuration**: See `config/config.yaml` comments
- **API Keys**: See secrets_store.py documentation

## Changelog

### Version 0.1.0 (Current)
- Initial GUI implementation
- Keyring-based credential storage
- PyInstaller packaging support
- Platform-specific path resolution
- Thread-safe logging system
- Service layer architecture
- Complete documentation

