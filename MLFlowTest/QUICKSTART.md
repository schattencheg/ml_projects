# 🚀 OHLC Price Prediction - Quick Start Guide

Get up and running with OHLC price prediction in minutes!

## Prerequisites

- Python 3.8+ 
- Git
- 8GB+ RAM recommended

## 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MLFlowTest

# Install dependencies
pip install -r requirements.txt

# Or use make (if available)
make install
```

## 2. Setup MLflow

```bash
# Setup MLflow tracking server
python scripts/setup_mlflow.py

# Or use make
make setup-mlflow
```

This will:
- Start MLflow server at http://127.0.0.1:5000
- Create necessary directories
- Set up a sample experiment

## 3. Run Complete Example

```bash
# Run the full pipeline example
python scripts/run_example.py

# Or use make
make example
```

This will:
- Fetch AAPL stock data
- Train Random Forest and XGBoost models
- Evaluate model performance
- Generate plots and reports
- Log everything to MLflow

## 4. View Results

### MLflow UI
Open http://127.0.0.1:5000 in your browser to:
- Compare model performance
- View training metrics
- Explore model artifacts

### Generated Files
- `evaluation_plots/` - Model performance visualizations
- `evaluation_report.md` - Detailed evaluation report
- `models/` - Trained model files
- `data/` - Downloaded market data

## 5. Train Custom Models

### Single Model Training
```bash
# Train LSTM model for Apple
python src/train.py --symbol AAPL --model lstm

# Train Random Forest for Google
python src/train.py --symbol GOOGL --model rf

# Train XGBoost for Tesla
python src/train.py --symbol TSLA --model xgb

# Or use make
make train SYMBOL=AAPL MODEL=lstm
```

### Evaluate Models
```bash
# Evaluate trained model
python src/evaluate.py --model-path models/AAPL_lstm_model.pkl --symbol AAPL

# Or use make
make evaluate SYMBOL=AAPL MODEL=lstm
```

## 6. Start Prediction API

```bash
# Start the FastAPI server
python api/app.py

# Or use make
make api
```

API will be available at:
- **API**: http://127.0.0.1:8000
- **Docs**: http://127.0.0.1:8000/docs

### API Usage Examples

```bash
# Health check
curl http://127.0.0.1:8000/health

# Make predictions
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "model_type": "rf",
    "days": 5,
    "use_latest_data": true
  }'

# List available models
curl http://127.0.0.1:8000/models

# Get available symbols
curl http://127.0.0.1:8000/symbols
```

## 7. Explore with Jupyter

```bash
# Start Jupyter notebook
make notebook

# Or manually
jupyter notebook notebooks/
```

Open `01_data_exploration.ipynb` to explore the data and features.

## 8. Docker Deployment (Optional)

```bash
# Start all services with Docker Compose
docker-compose up -d

# Services will be available at:
# - MLflow: http://localhost:5000
# - API: http://localhost:8000  
# - Jupyter: http://localhost:8888
```

## Common Commands

```bash
# Development setup
make dev-setup

# Complete quickstart
make quickstart

# Clean generated files
make clean

# Run tests
make test

# View all available commands
make help
```

## Configuration

Edit `config/config.yaml` to customize:
- Data sources and symbols
- Model parameters  
- Feature engineering settings
- MLflow configuration

Set environment variables in `.env`:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Troubleshooting

### MLflow Server Issues
```bash
# Check if server is running
curl http://127.0.0.1:5000/health

# Restart MLflow server
pkill -f "mlflow server"
python scripts/setup_mlflow.py
```

### Data Fetching Issues
- Ensure internet connection for Yahoo Finance
- Add Alpha Vantage API key to `.env` for additional data sources

### Model Training Issues
- Ensure sufficient RAM (8GB+ recommended)
- Reduce model complexity for faster training
- Check data quality and missing values

### API Issues
```bash
# Check if models exist
ls models/

# Train a model first
make train SYMBOL=AAPL MODEL=rf
```

## Next Steps

1. **Experiment with different symbols**: Try GOOGL, MSFT, TSLA, etc.
2. **Tune hyperparameters**: Modify `config/config.yaml`
3. **Add new features**: Extend `src/features/feature_engineer.py`
4. **Deploy to production**: Use Docker or cloud platforms
5. **Monitor performance**: Use MLflow for experiment tracking

## Support

- Check the main `README.md` for detailed documentation
- Review code comments and docstrings
- Explore Jupyter notebooks for examples
- Check MLflow UI for experiment details

Happy predicting! 📈
