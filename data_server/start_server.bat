@echo off
echo ========================================
echo   Data Provider Server - Starting...
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.

REM Start the server
echo ========================================
echo   Server starting on http://localhost:5001
echo   Dashboard: http://localhost:5001
echo   API Info: http://localhost:5001/api
echo ========================================
echo.

python server.py

pause
