@echo off
REM AI Trading Bot GUI Launcher
REM This script launches the trading bot GUI

cd /d "%~dp0"
python -m trading_bot.gui_main

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Error launching GUI. Press any key to exit...
    pause > nul
)

