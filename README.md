# Home Launcher - Unified Application Manager

A GUI launcher for managing and launching AI Trading Bot and GPU Manager applications. This launcher provides a single entry point for multiple desktop applications, making it easy to access and manage your tools from one convenient interface.

## Project Structure

This repository contains three main components:

- **`home.py`** - Main launcher application with PySide6 GUI
- **`bot/`** - AI Trading Bot with XGBoost ML trading system (see [bot/README.md](bot/README.md) for detailed documentation)
- **`pc/`** - Novita.ai GPU Instance Manager (see [pc/README.md](pc/README.md) for detailed documentation)

Each sub-application has its own README with comprehensive documentation, setup instructions, and usage guidelines.

## Quick Start

### For Launcher Only

If you only want to use the launcher to access pre-built applications:

```bash
# Install launcher dependencies
pip install -r requirements.txt

# Run the launcher
python home.py
```

On Windows, you can also double-click `launch_home.bat` for a quick start.

### For Full Setup (Development Mode)

If you want to run the applications from source code:

```bash
# Install bot dependencies
cd bot
pip install -r requirements.txt
cd ..

# Install PC dependencies
cd pc
pip install -r requirements.txt
cd ..

# Return to root and launch
cd ..
python home.py
```

Or simply double-click `launch_home.bat` on Windows.

## Usage

The launcher window displays two prominent buttons:

1. **🤖 Launch AI Trading Bot** - Starts the trading bot GUI application (entry point: `bot/src/trading_bot/gui_main.py`)
2. **💻 Launch GPU Manager** - Starts the GPU instance manager application (entry point: `pc/main.py`)

The launcher automatically detects whether to use:
- **Production mode**: Standalone `.exe` files in `dist/` folders (if available)
- **Development mode**: Python scripts (`.py` files) when executables are not found

Status indicators provide real-time feedback:
- **Blue**: Ready to launch
- **Orange**: Launching...
- **Green**: Application started successfully
- **Red**: Error occurred (details shown in error dialog)

## Building Executables

Both applications can be packaged as standalone `.exe` files for distribution:

- **AI Trading Bot**: See [bot/docs/GUI_PACKAGING_GUIDE.md](bot/docs/GUI_PACKAGING_GUIDE.md) for detailed packaging instructions. The bot will produce `bot/dist/TradingBotGUI.exe` when packaged.
- **GPU Manager**: Packaging instructions will be covered in subsequent phases. The PC application will produce `pc/dist/GPUManager.exe` when packaged.

The launcher will automatically use `.exe` files if they are available in the respective `dist/` folders, falling back to Python scripts in development mode.

## Requirements

- **Python**: 3.8 or higher
- **GUI Framework**: PySide6 (installed via `requirements.txt`)
- **Operating System**: Windows, Linux, or macOS supported

## Project Links

- **GitHub Repository**: [https://github.com/billibobby/home](https://github.com/billibobby/home)
- **AI Trading Bot Documentation**: See [bot/README.md](bot/README.md)
- **GPU Manager Documentation**: See [pc/README.md](pc/README.md)

## Troubleshooting

### Application Not Found Error

If you see an "Application not found" error:
- Ensure sub-applications are installed or built
- Check that `bot/dist/TradingBotGUI.exe` or `pc/dist/GPUManager.exe` exist (for production mode)
- Verify that Python scripts are available in the respective directories (for development mode)

### Import Errors

If you encounter import errors:
- Run `pip install -r requirements.txt` in the root directory for launcher dependencies
- Run `pip install -r requirements.txt` in `bot/` directory for bot dependencies
- Run `pip install -r requirements.txt` in `pc/` directory for GPU Manager dependencies

### Python Not Found

If you get a "Python is not installed or not in PATH" error:
- Ensure Python 3.8+ is installed on your system
- Verify Python is added to your system PATH
- Test by running `python --version` in a terminal

## License

This is a personal project repository. Please refer to individual application directories for specific license information.

## Contributing

This is a personal project. For suggestions or issues, please refer to the individual application READMEs or create an issue in the repository.






