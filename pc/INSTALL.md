# Installation Guide - Novita.ai GPU Manager

## Quick Start (5 minutes)

### Step 1: Prerequisites
- Python 3.8 or higher installed
- Novita.ai account with API key ([Sign up here](https://novita.ai))

### Step 2: Setup

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Step 3: Configure API Key

When the application launches:
1. Go to **File → Settings** (or press **Ctrl+S**)
2. Enter your Novita.ai API key
3. Click **Save**

That's it! You're ready to manage GPU instances.

---

## Detailed Installation

### 1. Verify Python Installation

```bash
python --version
# Should show Python 3.8 or higher
```

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Use Homebrew: `brew install python3`
- **Linux**: Use package manager: `sudo apt install python3 python3-venv`

### 2. Clone/Download Project

```bash
# Clone from your repository
# git clone https://github.com/your-username/your-repo.git
# cd your-repo
```

Or download and extract the project files.

### 3. Create Virtual Environment

A virtual environment keeps dependencies isolated:

```bash
# Windows
python -m venv venv

# macOS/Linux
python3 -m venv venv
```

### 4. Activate Virtual Environment

```bash
# Windows Command Prompt
venv\Scripts\activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyQt5 (GUI framework)
- requests (HTTP client)
- python-dotenv (environment variables)

### 6. Configure API Key (Optional)

**Option A: Environment Variable** (Development)

Create `.env` file in project root:
```env
NOVITA_API_KEY=your_actual_api_key_here
```

**Option B: Settings Dialog** (Recommended)

Configure after launching the application through the Settings dialog.

### 7. Launch Application

```bash
python main.py
```

The desktop application window should open.

---

## Troubleshooting Installation

### Python not found
```bash
# Try python3 instead
python3 --version
python3 -m venv venv
```

### pip not found
```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

### PyQt5 installation fails

**Windows:**
- Ensure Visual C++ Redistributable is installed
- Download from Microsoft website

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3-pyqt5
pip install -r requirements.txt
```

### Permission errors
```bash
# Use pip with --user flag
pip install --user -r requirements.txt
```

### Virtual environment activation issues

**Windows PowerShell** may block scripts:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Verifying Installation

After launching, you should see:
1. Main window with "Novita.ai GPU Instance Manager" title
2. Empty instance table
3. Activity logs showing "Novita.ai GPU Manager started"
4. Status bar showing "API: Disconnected" (until you configure API key)

---

## Next Steps

1. **Configure API Key**: File → Settings
2. **Read Documentation**: See [README.md](README.md)
3. **Wait for Phase 3**: Instance management coming soon!

---

## Uninstallation

1. Delete the project folder
2. Delete config directory:
   - Windows: `C:\Users\<username>\.novita_gpu_manager\`
   - macOS: `/Users/<username>/.novita_gpu_manager/`
   - Linux: `/home/<username>/.novita_gpu_manager/`

---

## Getting Help

- **Documentation**: [README.md](README.md)
- **Project Support**: Check your project repository for support resources
- **Novita.ai Platform**: [novita.ai](https://novita.ai)

