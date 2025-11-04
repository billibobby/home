@echo off
REM Home Launcher - Quick Start Script
REM This script launches the Home Launcher application

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ============================================
    echo ERROR: Python is not installed or not in PATH
    echo ============================================
    echo.
    echo Please ensure Python 3.8+ is installed and added to your system PATH.
    echo.
    echo ============================================
    pause
    exit /b 1
)

REM Launch the Home Launcher
python home.py

REM Check if launch failed
if errorlevel 1 (
    echo.
    echo ============================================
    echo Failed to launch Home Launcher. Check error above.
    echo ============================================
    echo.
    echo If you see import errors, try running:
    echo   pip install -r requirements.txt
    echo.
    echo ============================================
    pause
)

