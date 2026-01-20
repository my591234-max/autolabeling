@echo off
echo ============================================
echo   AutoLabel Pro - ngrok Tunnel Launcher
echo ============================================
echo.
echo Make sure the server is running on port 8000 first!
echo.
echo Starting ngrok tunnel...
echo.

.\ngrok.exe http 8000
pause
