@echo off
setlocal
cd /d "%~dp0"
set "VRC_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "VRC_PYTHON=%~dp0.venv\Scripts\python.exe"
"%VRC_PYTHON%" server.py --open
if errorlevel 1 (
  echo.
  echo Start failed. Install dependencies first. See README.md.
  pause
  exit /b 1
)
endlocal
