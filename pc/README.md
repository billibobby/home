# Novita.ai GPU Instance Manager

A desktop application for managing GPU instances on the Novita.ai platform. Provides a streamlined interface for creating, managing, and monitoring cloud GPU instances with ComfyUI integration.

## Features

### Current Features (Phase 1 + Phase 2 + Phase 3)
- ✅ Desktop GUI with menu bar, toolbar, and status bar
- ✅ Settings management with persistent storage
- ✅ Instance list view (table layout with real-time data)
- ✅ Activity logs panel
- ✅ API client for Novita.ai integration
- ✅ Configuration persistence (JSON storage)
- ✅ Cross-platform support (Windows, macOS, Linux)
- ✅ **Instance lifecycle management (create, start, stop, delete, refresh)**
- ✅ **Create instances with GPU product and cluster selection**
- ✅ **Real-time instance status monitoring with color-coded display**
- ✅ **Automatic instance refresh on startup**
- ✅ **Background API operations with non-blocking UI**

### Upcoming Features (Phase 4-6)
- 🚧 ComfyUI integration and model management
- 🚧 Real-time monitoring and browser integration
- 🚧 Snapshot and template system
- 🚧 Automatic model downloads and management
- 🚧 Multi-instance orchestration

## Requirements

- Python 3.8 or higher
- PyQt5
- Novita.ai API key ([Get one here](https://novita.ai/dashboard))

## Setup Instructions

### 1. Clone or Download the Project

```bash
# Clone from your repository
# git clone https://github.com/your-username/your-repo.git
# cd your-repo
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. API Key Setup

You can configure your Novita.ai API key in two ways:

**Option 1: Environment Variable (Recommended for development)**

Create a `.env` file in the project root:

```env
NOVITA_API_KEY=your_api_key_here
```

**Option 2: Settings Dialog (Recommended for production)**

Launch the application and configure your API key through the Settings dialog (File → Settings or Ctrl+S).

## Usage

### Running the Application

```bash
# Ensure virtual environment is activated
python main.py
```

The application will launch a desktop GUI window.

### First-Time Setup

1. Launch the application
2. Click **Settings** from the File menu or toolbar
3. Enter your **Novita.ai API key**
4. Configure default GPU preferences (optional)
5. Set your preferred Docker image (default: ComfyUI with CUDA support)
6. Click **Save**

### Basic Operations

- **Create Instance**: Click "Create Instance" button, select a GPU product, enter Docker image URL, and configure optional settings (ports, commands, environment variables). Click "Create" to launch the instance.
- **Start Instance**: Select a stopped instance from the table and click "Start" button or use the Instance menu
- **Stop Instance**: Select a running instance and click "Stop" button or use the Instance menu
- **Delete Instance**: Select any instance and click "Delete" button (confirmation required). This action cannot be undone.
- **Refresh**: Click "Refresh" to manually update the instance list from the server. The app also refreshes automatically on startup.
- **View Logs**: Activity logs appear in the bottom panel showing all operations and status updates

## Project Structure

```
gpu-manager/
├── main.py                 # Application entry point - launches PyQt5 GUI
├── novita_api.py          # Novita.ai API client for platform integration
├── config_manager.py      # Configuration persistence manager (JSON storage)
├── main_window.py         # Main PyQt5 application window with instance list, logs, and controls
├── settings_dialog.py     # Settings dialog for API key and preferences
├── requirements.txt       # Python package dependencies
├── .env                   # Environment variables (API key) - not tracked in git
└── README.md             # Project documentation
```

## Configuration

Application settings are stored in: `~/.novita_gpu_manager/config.json`

Configuration includes:
- API key
- Default GPU product preferences
- Default cluster selection
- Docker image URL
- Window geometry and state

## Security and Privacy

### API Key Storage

Your Novita.ai API key is stored locally in the configuration file at `~/.novita_gpu_manager/config.json`. This file contains sensitive information and should be protected.

### File Permissions

**Unix/Linux/macOS:**
- The application automatically sets restrictive file permissions (0600) on the config file
- Only your user account can read or write the configuration file
- This prevents other users on the same system from accessing your API key

**Windows:**
- File permission hardening is not applied on Windows due to different permission models
- The config file is protected by your Windows user account permissions
- For enhanced security, consider:
  - Enabling Windows file encryption (EFS) on the config directory
  - Using BitLocker to encrypt your system drive
  - Ensuring your Windows user account has a strong password
  - Restricting physical access to your computer

### Best Practices

1. **Never share your config file** - It contains your API key in plain text
2. **Keep your system secure** - Use strong passwords and keep your OS updated
3. **Monitor API usage** - Regularly check your Novita.ai dashboard for unexpected activity
4. **Rotate API keys** - If you suspect your key has been compromised, generate a new one immediately

## Troubleshooting

### Common Issues

**API key not working**
- Verify your API key is valid on the [Novita.ai dashboard](https://novita.ai/dashboard)
- Check that the key is properly entered in Settings (no extra spaces)
- API key should be at least 20 characters long

**Window doesn't open**
- Ensure Python 3.8+ is installed: `python --version`
- Verify PyQt5 is installed: `pip list | grep PyQt5`
- Check console output for error messages
- Try running with: `python main.py` from the project directory

**Config file location**
- Windows: `C:\Users\<username>\.novita_gpu_manager\config.json`
- macOS: `/Users/<username>/.novita_gpu_manager/config.json`
- Linux: `/home/<username>/.novita_gpu_manager/config.json`

**Missing dependencies**
```bash
pip install --upgrade -r requirements.txt
```

## Instance Management

### Creating Instances

1. Click "Create Instance" from the toolbar, menu, or control panel
2. Select a GPU product from the dropdown (displays GPU count, CPU, RAM, and price)
3. Choose a cluster (or use auto-select for automatic placement)
4. Enter or confirm the Docker image URL (defaults to ComfyUI with CUDA support)
5. Optionally configure:
   - Instance name (auto-generated if left empty)
   - Port mappings (e.g., "8080/http, 6006/tcp")
   - Container command override
   - Environment variables
6. Click "Create" to launch the instance

### Understanding Instance Status

The instance table displays status with color coding:
- **Green (Running)**: Instance is active and ready
- **Orange (Starting/Stopping/ToStart)**: Instance is transitioning states
- **Red (Stopped/Exited)**: Instance is not running
- **Gray (Unknown)**: Status is unavailable or unclear

### Managing Instances

- **Automatic Refresh**: Instance list refreshes automatically when the app starts
- **Manual Refresh**: Click "Refresh" to update the list at any time
- **Instance Selection**: Click on any instance in the table to select it
- **Available Actions**: Buttons and menu items enable/disable based on instance status
  - Start: Available for stopped instances
  - Stop: Available for running instances
  - Delete: Always available (requires confirmation)

### Instance Table Columns

1. **Name**: Instance name or "Unnamed" for auto-generated names
2. **Instance ID**: Unique identifier for the instance
3. **Status**: Current state (color-coded)
4. **GPU Type**: Product name or GPU configuration
5. **Cluster**: Cluster name or ID where instance is running
6. **Created**: Timestamp when instance was created

## Development

### Phase Roadmap

- **Phase 1**: ✅ API client infrastructure
- **Phase 2**: ✅ GUI scaffolding and settings
- **Phase 3**: ✅ Instance lifecycle management
- **Phase 4**: 🚧 ComfyUI integration
- **Phase 5**: 🚧 Real-time monitoring
- **Phase 6**: 🚧 Snapshots and templates

### Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Novita.ai GPU Manager Contributors. All rights reserved.

## Credits

Built with:
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [Novita.ai](https://novita.ai) - GPU cloud platform
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Stable Diffusion interface

## Screenshots

*Screenshots coming soon*

### Main Window
The main window features:
- **Menu Bar**: File, Instance, Tools, and Help menus
- **Toolbar**: Quick access to common actions
- **Instance List**: Table view of all GPU instances with status
- **Control Panel**: Buttons for instance management
- **Activity Logs**: Real-time logging of all operations
- **Status Bar**: API connection status and credit balance

### Settings Dialog
Configure:
- Novita.ai API key
- Default GPU product preferences
- Default cluster selection
- Docker image URL for ComfyUI

## Support

For issues and questions:
- Project Issues: Check your project repository for issue tracking
- Novita.ai Platform Support: [Contact Novita.ai](https://novita.ai)

---

**Version**: 1.0.0  
**Status**: Phase 3 Complete - Instance Management Ready

