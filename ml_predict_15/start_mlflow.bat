@echo off
echo ================================================================================
echo Starting MLflow Tracking Server
echo ================================================================================
echo.
echo Server will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

mlflow server --host 127.0.0.1 --port 5000
