# ML CNN Trend Prediction

This project predicts trend direction for cryptocurrency data (EC and BTC) using CNN networks with profit threshold optimization.

## Features

- Loads EC (Ethereum Classic) and BTC (Bitcoin) data from Yahoo Finance
- Generates features from OHLC data using a customizable FeaturesGenerator
- Implements multiple CNN architectures for trend prediction
- Uses Optuna for hyperparameter optimization
- Includes profit threshold for filtering profitable trends

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Structure

- `main.py` - Entry point to run the complete pipeline
- `requirements.txt` - Project dependencies
- `README.md` - Project documentation
- `src/` - Source code directory containing:
  - `data_loader.py` - Loads EC and BTC data from Yahoo Finance
  - `features_generator.py` - Class to generate features from OHLC data
  - `cnn_architectures.py` - Multiple CNN architectures for trend prediction
  - `hyperparameter_optimizer.py` - Optuna-based hyperparameter optimization
  - `trend_predictor.py` - Main trend prediction model with profit threshold
  - `__init__.py` - Makes src a Python package

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py
```

## Customization

- Modify the `FeaturesGenerator.generate_features()` method to implement custom technical indicators
- Adjust the profit threshold in `TrendPredictor` to change the minimum acceptable profit
- Experiment with different CNN architectures in `cnn_architectures.py`
- Tune hyperparameter search space in `hyperparameter_optimizer.py`

## Models

The project includes three different CNN architectures:
1. Simple CNN: Basic convolutional layers
2. Deeper CNN: More convolutional layers with batch normalization
3. CNN-LSTM: Combines CNN for feature extraction with LSTM for sequence modeling