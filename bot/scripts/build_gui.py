#!/usr/bin/env python3
"""
Build script for creating the Trading Bot GUI executable using PyInstaller.

This script automates the process of building a standalone executable
for the Trading Bot GUI application.

Requirements:
- PyInstaller must be installed (pip install pyinstaller)
- All dependencies must be installed (pip install -e .[gui,build])

Usage:
    python scripts/build_gui.py

Output:
    Standalone executable will be created in dist/TradingBotGUI[.exe]
"""

import sys
import subprocess
import shutil
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("✗ PyInstaller not found")
        print("\nInstall PyInstaller with:")
        print("  pip install pyinstaller")
        print("  or")
        print("  pip install -e .[build]")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['PySide6', 'colorlog', 'keyring', 'yaml', 'dotenv']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.lower() if pkg != 'yaml' else 'yaml')
            print(f"✓ {pkg} found")
        except ImportError:
            print(f"✗ {pkg} not found")
            missing.append(pkg)
    
    if missing:
        print("\nMissing dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print("  pip install -e .[gui]")
        return False
    
    return True


def clean_build_artifacts():
    """Clean previous build artifacts."""
    dirs_to_clean = ['build', 'dist']
    files_to_clean = ['*.spec']
    
    project_root = Path(__file__).parent.parent
    
    print("\nCleaning previous build artifacts...")
    
    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            # Don't delete the build dir itself, just its contents
            if dir_name == 'build':
                for item in dir_path.iterdir():
                    if item.name != 'gui.spec':
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                print(f"  Cleaned {dir_name}/ contents")
            else:
                shutil.rmtree(dir_path)
                print(f"  Removed {dir_name}/")


def build_executable():
    """Build the executable using PyInstaller."""
    project_root = Path(__file__).parent.parent
    spec_file = project_root / 'build' / 'gui.spec'
    
    if not spec_file.exists():
        print(f"\n✗ Spec file not found: {spec_file}")
        return False
    
    print(f"\nBuilding executable using: {spec_file}")
    print("This may take several minutes...\n")
    
    try:
        # Run PyInstaller with the spec file
        result = subprocess.run(
            [sys.executable, '-m', 'PyInstaller', str(spec_file), '--clean'],
            cwd=str(project_root),
            check=True,
            capture_output=False
        )
        
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Build failed: {e}")
        return False


def verify_output():
    """Verify the output executable exists."""
    project_root = Path(__file__).parent.parent
    dist_dir = project_root / 'dist'
    
    if not dist_dir.exists():
        print("\n✗ dist/ directory not found")
        return False
    
    # Check for executable (with or without .exe extension)
    exe_name = 'TradingBotGUI.exe' if sys.platform == 'win32' else 'TradingBotGUI'
    exe_path = dist_dir / exe_name
    
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Executable created: {exe_path}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"\n✗ Executable not found: {exe_path}")
        return False


def main():
    """Main build process."""
    print("=" * 70)
    print("Trading Bot GUI - Build Script")
    print("=" * 70)
    
    print("\n1. Checking dependencies...")
    if not check_pyinstaller():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n2. Cleaning previous build artifacts...")
    clean_build_artifacts()
    
    print("\n3. Building executable...")
    if not build_executable():
        print("\n✗ Build failed!")
        sys.exit(1)
    
    print("\n4. Verifying output...")
    if not verify_output():
        print("\n✗ Verification failed!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ Build completed successfully!")
    print("=" * 70)
    
    print("\nThe executable is located in the dist/ directory.")
    print("\nNotes:")
    print("  - First run may be slower as it extracts bundled files")
    print("  - Logs are written to user-writable directory:")
    print("    Windows: %APPDATA%/AITradingBot/logs")
    print("    Linux:   ~/.local/share/ai-trading-bot/logs")
    print("    macOS:   ~/Library/Application Support/AITradingBot/logs")
    print("\nYou can now distribute the executable to users!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBuild cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

