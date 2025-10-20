# MLflow Troubleshooting Guide

This guide helps you diagnose and fix MLflow server connectivity issues.

## Quick Diagnosis

Run the test script to check the status:
```bash
python test_mlflow_connection.py
```

## Common Issues and Solutions

### Issue 1: "MLflow is not installed"

**Symptoms:**
- Error message: `✗ MLflow is not installed`
- Server logs show: `FileNotFoundError: mlflow`

**Solution:**
```bash
pip install mlflow==2.9.2
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
mlflow --version
```

---

### Issue 2: "MLflow server is not responding"

**Symptoms:**
- Error message: `MLflow server is not responding. It may still be starting up or not running.`
- Cannot access http://localhost:5000

**Solutions:**

#### Step 1: Check if MLflow is enabled
Edit `config.py` and verify:
```python
MLFLOW_ENABLED = True
```

#### Step 2: Check if port 5000 is available
```bash
# Windows
netstat -ano | findstr :5000

# If something is using port 5000, you can:
# 1. Stop that process
# 2. Change MLFLOW_PORT in config.py to a different port (e.g., 5002)
```

#### Step 3: Check server logs
Look at `data_server.log` for error messages:
```bash
# Windows PowerShell
Get-Content data_server.log -Tail 50

# Or open in text editor
notepad data_server.log
```

#### Step 4: Try starting MLflow manually
```bash
# Windows
start_mlflow_manual.bat

# Or manually:
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri mlruns --default-artifact-root mlartifacts
```

If MLflow starts successfully manually, the issue is with automatic startup.

---

### Issue 3: "Port 5000 is in use"

**Symptoms:**
- Error message: `⚠ Port 5000 is in use but not responding as MLflow server`
- Another application is using port 5000

**Solution 1: Change MLflow port**

Edit `config.py`:
```python
MLFLOW_PORT = 5002  # Or any available port
```

Restart the data server.

**Solution 2: Stop the conflicting process**

Find what's using port 5000:
```bash
# Windows
netstat -ano | findstr :5000
# Note the PID (last column)

# Stop the process (replace PID with actual number)
taskkill /PID <PID> /F
```

---

### Issue 4: MLflow starts but times out

**Symptoms:**
- Message: `⚠ MLflow server started but not responding after 10 seconds`
- Process is running but not accessible

**Solutions:**

#### Solution 1: Wait longer
MLflow may take longer to initialize on slower systems. Wait 30-60 seconds and test again:
```bash
python test_mlflow_connection.py
```

#### Solution 2: Increase timeout
Edit `server.py` and increase `max_retries`:
```python
# In start_mlflow_server() function
max_retries = 20  # Increase from 10 to 20
```

#### Solution 3: Check firewall
Windows Firewall may be blocking MLflow. Add an exception:
1. Open Windows Defender Firewall
2. Click "Allow an app through firewall"
3. Add Python or MLflow to the allowed list

---

### Issue 5: MLflow process crashes immediately

**Symptoms:**
- Error: `✗ MLflow server failed to start`
- Process terminates right after starting

**Solutions:**

#### Check Python version
MLflow requires Python 3.8+:
```bash
python --version
```

#### Check dependencies
Some MLflow dependencies may be missing:
```bash
pip install --upgrade mlflow
pip install --upgrade sqlalchemy alembic
```

#### Check file permissions
Ensure you have write permissions for:
- `mlruns/` directory
- `mlartifacts/` directory
- Current working directory

#### View detailed error
Check the error message in the console or `data_server.log`.

---

### Issue 6: "Cannot connect to MLflow API"

**Symptoms:**
- MLflow UI loads but API doesn't work
- Error accessing `/api/2.0/mlflow/experiments/list`

**Solution:**

This usually means MLflow is still starting. Wait a few seconds and try again.

If the issue persists:
```bash
# Restart MLflow
# 1. Stop the data server (Ctrl+C)
# 2. Start it again
python server.py
```

---

## Manual MLflow Management

### Start MLflow Manually (Windows)

Use the provided batch file:
```bash
start_mlflow_manual.bat
```

Or run directly:
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri mlruns --default-artifact-root mlartifacts
```

### Start MLflow Manually (Linux/Mac)

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri mlruns --default-artifact-root mlartifacts
```

### Disable MLflow Auto-Start

If you prefer to manage MLflow separately, disable auto-start in `config.py`:
```python
MLFLOW_ENABLED = False
```

Then start MLflow manually when needed.

---

## Verification Steps

### 1. Test Data Server
```bash
curl http://localhost:5001/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-10-20T12:00:00"
}
```

### 2. Test MLflow Status API
```bash
curl http://localhost:5001/api/mlflow/status
```

Expected response (if working):
```json
{
  "enabled": true,
  "status": "running",
  "url": "http://127.0.0.1:5000",
  "host": "127.0.0.1",
  "port": 5000,
  "message": "MLflow server is running and accessible"
}
```

### 3. Test MLflow Directly
```bash
curl http://localhost:5000/health
```

Expected response:
```
OK
```

### 4. Test MLflow UI
Open in browser:
```
http://localhost:5000
```

You should see the MLflow UI.

### 5. Test MLflow Python Client
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")
```

---

## Advanced Debugging

### Enable Debug Logging

Edit `config.py`:
```python
LOG_LEVEL = 'DEBUG'
```

Restart the server and check `data_server.log` for detailed information.

### Check MLflow Logs

MLflow logs are usually in the console output. If running as a background process, check:
- Console output when starting the data server
- `data_server.log` file

### Test Network Connectivity

```bash
# Test if localhost resolves correctly
ping 127.0.0.1

# Test if port is reachable
telnet 127.0.0.1 5000
```

### Check Process Status

```bash
# Windows - Check if MLflow process is running
tasklist | findstr python

# Windows - Check listening ports
netstat -ano | findstr LISTENING
```

---

## Getting Help

If none of these solutions work:

1. **Collect diagnostic information:**
   ```bash
   python test_mlflow_connection.py > mlflow_test.txt 2>&1
   ```

2. **Check logs:**
   - `data_server.log`
   - Console output from `python server.py`

3. **Provide information:**
   - Python version: `python --version`
   - MLflow version: `mlflow --version`
   - Operating system
   - Error messages from logs
   - Output from test script

4. **Try minimal setup:**
   ```bash
   # Just start MLflow manually
   mlflow server --host 127.0.0.1 --port 5000
   
   # In another terminal, test it
   curl http://localhost:5000/health
   ```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python server.py` | Start data server with MLflow |
| `python test_mlflow_connection.py` | Test connectivity |
| `start_mlflow_manual.bat` | Start MLflow manually (Windows) |
| `curl http://localhost:5001/api/mlflow/status` | Check MLflow status |
| `curl http://localhost:5000/health` | Test MLflow directly |
| `netstat -ano \| findstr :5000` | Check if port 5000 is in use |
| `mlflow --version` | Check MLflow installation |

---

## Prevention

To avoid issues in the future:

1. **Keep dependencies updated:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use virtual environment:**
   Always activate the venv before running the server

3. **Check port availability:**
   Before starting, ensure port 5000 is free

4. **Monitor logs:**
   Regularly check `data_server.log` for warnings

5. **Test after changes:**
   Run `python test_mlflow_connection.py` after configuration changes
