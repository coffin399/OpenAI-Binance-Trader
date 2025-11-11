@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%venv"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%binance_auto_trader\requirements.txt"

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
    echo TA-Lib などのネイティブ拡張が原因の場合は、Windows 用の事前ビルド済みホイールを手動でインストールしてください。
    echo 例: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
    pause
    exit /b 1
)

python -m binance_auto_trader.main %*

endlocal
