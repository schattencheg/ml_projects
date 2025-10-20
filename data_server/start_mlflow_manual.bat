@echo off
REM Manual MLflow Server Startup Script
REM Use this if you want to start MLflow separately from the data server

echo ============================================================
echo Starting MLflow Server Manually
echo ============================================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

echo.
echo Starting MLflow server on http://127.0.0.1:5000
echo Backend store: file:///mlruns
echo Artifact root: file:///mlartifacts
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

REM Get current directory and convert to file:// URI format
set CURRENT_DIR=%CD%
set BACKEND_URI=file:///%CURRENT_DIR:\=/%/mlruns
set ARTIFACT_URI=file:///%CURRENT_DIR:\=/%/mlartifacts

mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "%BACKEND_URI%" --default-artifact-root "%ARTIFACT_URI%"

pause
