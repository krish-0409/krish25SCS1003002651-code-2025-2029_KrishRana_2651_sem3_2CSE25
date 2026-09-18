@echo off
title AI Image Recognition Server
cd /d "%~dp0"

echo ============================================
echo   Starting AI Image Recognition App...
echo ============================================
echo.

call venv\Scripts\activate.bat

python app.py

echo.
echo Server stopped. Press any key to close this window.
pause >nul
