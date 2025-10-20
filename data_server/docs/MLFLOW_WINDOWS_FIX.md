# MLflow Windows Path Fix

## Issue

MLflow on Windows requires paths to be in `file://` URI format, not standard Windows paths.

**Error:**
```
Error initializing backend store
Model registry functionality is unavailable; got unsupported URI 'D:\Work\Repos\ml_projects\data_server\mlruns'
```

## Solution

Convert Windows paths to `file://` URI format.

### Before (Incorrect)
```python
MLFLOW_BACKEND_STORE_URI = 'D:\\Work\\Repos\\ml_projects\\data_server\\mlruns'
```

### After (Correct)
```python
MLFLOW_BACKEND_STORE_URI = 'file:///D:/Work/Repos/ml_projects/data_server/mlruns'
```

## Implementation

The fix is implemented in `config.py`:

```python
# Convert paths to file:// URIs for MLflow compatibility
_mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlruns'))
_mlartifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlartifacts'))

# Use file:// URI format (required by MLflow)
MLFLOW_BACKEND_STORE_URI = f"file:///{_mlruns_path.replace(os.sep, '/')}"
MLFLOW_DEFAULT_ARTIFACT_ROOT = f"file:///{_mlartifacts_path.replace(os.sep, '/')}"
```

## Verification

Check that URIs are correctly formatted:

```bash
python -c "import config; print(config.MLFLOW_BACKEND_STORE_URI)"
```

Expected output:
```
file:///D:/Work/Repos/ml_projects/data_server/mlruns
```

## Testing

Start the server:
```bash
python server.py
```

Expected output:
```
============================================================
Data Provider Server with MLflow Integration
============================================================

Starting MLflow server on http://127.0.0.1:5000...
Waiting for MLflow server to start..........
✓ MLflow server started successfully on http://127.0.0.1:5000

============================================================
Data Provider Server: http://0.0.0.0:5001
MLflow Server: http://127.0.0.1:5000
============================================================
```

## Manual MLflow Startup

The `start_mlflow_manual.bat` file has also been updated to use proper URI format:

```batch
set CURRENT_DIR=%CD%
set BACKEND_URI=file:///%CURRENT_DIR:\=/%/mlruns
set ARTIFACT_URI=file:///%CURRENT_DIR:\=/%/mlartifacts

mlflow server --host 127.0.0.1 --port 5000 ^
  --backend-store-uri "%BACKEND_URI%" ^
  --default-artifact-root "%ARTIFACT_URI%"
```

## Key Points

1. **Use `file:///` prefix** (three slashes)
2. **Use forward slashes** (`/`) instead of backslashes (`\`)
3. **Absolute paths** work best
4. **No spaces** in the URI (or properly escape them)

## Common Mistakes

❌ **Wrong:**
```python
# Missing file:// prefix
MLFLOW_BACKEND_STORE_URI = 'D:/Work/Repos/ml_projects/data_server/mlruns'

# Using backslashes
MLFLOW_BACKEND_STORE_URI = 'file:///D:\\Work\\Repos\\ml_projects\\data_server\\mlruns'

# Only two slashes
MLFLOW_BACKEND_STORE_URI = 'file://D:/Work/Repos/ml_projects/data_server/mlruns'
```

✅ **Correct:**
```python
MLFLOW_BACKEND_STORE_URI = 'file:///D:/Work/Repos/ml_projects/data_server/mlruns'
```

## Additional Notes

- This fix is specific to Windows
- Linux/Mac paths work differently but the `file://` format is still recommended
- The server code automatically extracts the actual path from the URI when creating directories

## Related Files

- `config.py` - MLflow configuration with URI conversion
- `server.py` - Server startup with directory creation
- `start_mlflow_manual.bat` - Manual startup script
- `MLFLOW_TROUBLESHOOTING.md` - Comprehensive troubleshooting guide

## References

- [MLflow Tracking URI](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded)
- [File URI Scheme](https://en.wikipedia.org/wiki/File_URI_scheme)
