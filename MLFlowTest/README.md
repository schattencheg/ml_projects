# OHLC Price Prediction with MLflow

A comprehensive machine learning project for predicting market OHLC (Open, High, Low, Close) prices using various ML models with MLflow experiment tracking.

## Features

- **Data Acquisition**: Fetch market data from Yahoo Finance and Alpha Vantage
- **Feature Engineering**: Technical indicators, moving averages, and custom features
- **Multiple Models**: LSTM, Random Forest, XGBoost, and ensemble methods
- **MLflow Integration**: Experiment tracking, model versioning, and deployment
- **Real-time Predictions**: API endpoints for live price predictions

## Project Structure

```
├── config/                 # Configuration files
├── data/                  # Raw and processed data
├── src/                   # Source code
│   ├── data/             # Data acquisition and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # ML model implementations
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for exploration
├── experiments/          # MLflow experiments and runs
├── models/              # Saved model artifacts
└── api/                 # Model serving API
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start MLflow server**:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

4. **Run data acquisition**:
   ```bash
   python src/data/fetch_data.py --symbol AAPL --period 2y
   ```

5. **Train models**:
   ```bash
   python src/train.py --model lstm --symbol AAPL
   ```

6. **View experiments**:
   Open http://127.0.0.1:5000 in your browser

## Configuration

Edit `config/config.yaml` to customize:
- Data sources and symbols
- Model parameters
- Feature engineering settings
- MLflow tracking URI

## API Usage

Start the prediction API:
```bash
python api/app.py
```

Make predictions:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 5}'
```

## Models Supported

- **LSTM**: Long Short-Term Memory networks for time series
- **Random Forest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high performance
- **Ensemble**: Combination of multiple models

## License

MIT License
