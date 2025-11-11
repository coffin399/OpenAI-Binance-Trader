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
python -m pip install -r "%REQUIREMENTS_FILE%"

python -m binance_auto_trader.main %*

endlocal
