@echo off
echo ============================================
echo   AutoLabel Pro - Unified Server Launcher
echo ============================================
echo.

REM Activate conda environment
call conda activate autolabel-pro

REM Check if activation was successful
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'autolabel-pro'
    echo.
    echo Please create the environment first:
    echo   conda create -n autolabel-pro python=3.10 -y
    echo   conda activate autolabel-pro
    echo   pip install fastapi uvicorn python-multipart pillow ultralytics transformers torch torchvision
    echo.
    pause
    exit /b 1
)

echo [OK] Conda environment activated
echo.

REM Start the server
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop
echo.

python unified_server.py

pause
