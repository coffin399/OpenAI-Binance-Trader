@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%venv"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%binance_auto_trader\requirements.txt"

rem --- Check Python version (require 3.10.x) ---
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
)
if not "%MAJOR%"=="3" (
    echo [ERROR] Python 3 is required. Detected: %PYTHON_VERSION%
    pause
    exit /b 1
)
if not "%MINOR%"=="10" (
    echo [ERROR] Python 3.10.x is required. Detected: %PYTHON_VERSION%
    pause
    exit /b 1
)
echo [INFO] Python %MAJOR%.%MINOR% detected.

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip. Please verify your Python installation.
    pause
    exit /b 1
)
python -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo TA-Lib などのネイティブ拡展が原因の場合は、Windows 用の事前ビルド済みホイールを手動でインストールしてください。
    echo 例: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
    pause
    exit /b 1
)

python -m binance_auto_trader.main %*

endlocal
