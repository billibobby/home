@echo off
REM Novita.ai GPU Manager Launcher
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ============================================
    echo ERROR: Virtual environment not found!
    echo ============================================
    echo.
    echo The virtual environment has not been set up yet.
    echo Please follow these steps to set up the application:
    echo.
    echo 1. Open Command Prompt or PowerShell in this directory
    echo 2. Run: python -m venv venv
    echo 3. Run: venv\Scripts\activate
    echo 4. Run: pip install -r requirements.txt
    echo.
    echo Alternatively, run the following commands:
    echo    python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    echo.
    echo ============================================
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

call venv\Scripts\activate.bat
python main.py
pause

