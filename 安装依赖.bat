@echo off
setlocal
cd /d "%~dp0"
python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)"
if errorlevel 1 (
  echo Python 3.11-3.13 64-bit is required. Install Python and add it to PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Install failed. Check your network and Python version.
  pause
  exit /b 1
)
echo Installation complete. Run the GUI launcher.
pause
endlocal
