@echo OFF
echo Starting MLflow Tracking Server...

REM Activate the virtual environment
call venv\Scripts\activate

REM Set the tracking URI so that scripts can connect to this server
set MLFLOW_TRACKING_URI=http://127.0.0.1:8080

echo MLflow Tracking URI set to %MLFLOW_TRACKING_URI%

REM Start the MLflow server
REM --backend-store-uri: where to store experiment and run metadata
REM --default-artifact-root: where to store artifacts (models, plots, etc.)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlflow_artifacts --host 127.0.0.1 --port 8080

echo MLflow server stopped.
