"""
Environment Setup Script

Automates the initial setup process for the trading bot:
- Checks Python version
- Creates virtual environment
- Installs dependencies
- Creates .env file from template
- Validates directory structure
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"✗ {text}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


def check_python_version():
    """Check if Python version is 3.9 or higher."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print_info(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error("Python 3.9 or higher is required")
        return False
    
    print_success("Python version check passed")
    return True


def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    print_header("Setting Up Virtual Environment")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_info("Virtual environment already exists")
        return True
    
    print_info("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print_success("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def get_pip_executable():
    """Get the path to pip executable in virtual environment."""
    if sys.platform == "win32":
        return Path("venv") / "Scripts" / "pip.exe"
    else:
        return Path("venv") / "bin" / "pip"


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print_header("Installing Dependencies")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False
    
    pip_executable = get_pip_executable()
    
    if not pip_executable.exists():
        print_error("Virtual environment pip not found")
        print_info("Please activate the virtual environment and run: pip install -r requirements.txt")
        return False
    
    print_info("Installing packages from requirements.txt...")
    try:
        subprocess.run([str(pip_executable), "install", "-r", "requirements.txt"], check=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    print_header("Setting Up Environment Variables")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_info(".env file already exists")
        return True
    
    if not env_example.exists():
        print_error(".env.example not found")
        return False
    
    print_info("Creating .env file from template...")
    try:
        shutil.copy(env_example, env_file)
        print_success(".env file created successfully")
        print_info("Please edit .env file and add your API keys")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False


def validate_directory_structure():
    """Validate and create required directories."""
    print_header("Validating Directory Structure")
    
    required_dirs = [
        "config",
        "data",
        "models",
        "logs",
        "src",
        "src/data",
        "src/models",
        "src/trading",
        "src/utils",
        "tests",
        "scripts"
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            print_info(f"✓ {directory}/")
        else:
            print_info(f"✗ {directory}/ (creating...)")
            dir_path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    if all_exist:
        print_success("All directories exist")
    else:
        print_success("Created missing directories")
    
    return True


def display_next_steps():
    """Display next steps for the user."""
    print_header("Setup Complete!")
    
    print("Next steps:")
    print()
    print("1. Activate the virtual environment:")
    
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print()
    print("2. Edit the .env file and add your API keys:")
    print("   - Binance API credentials")
    print("   - Coinbase API credentials")
    print("   - Alpaca API credentials")
    print("   - Alpha Vantage API key")
    print()
    print("3. Review and adjust config/config.yaml if needed")
    print()
    print("4. Validate your setup:")
    print("   python scripts/validate_setup.py")
    print()
    print("5. Run the trading bot:")
    print("   python main.py")
    print()
    print("6. Run tests:")
    print("   pytest tests/")
    print()


def main():
    """Main setup function."""
    print_header("AI Trading Bot - Environment Setup")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating .env file", create_env_file),
        ("Validating directory structure", validate_directory_structure),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print_error(f"Setup failed at: {step_name}")
            sys.exit(1)
    
    display_next_steps()


if __name__ == '__main__':
    main()

