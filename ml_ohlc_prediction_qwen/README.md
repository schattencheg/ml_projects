# ML OHLC Price Prediction

This project compares different machine learning models for predicting OHLC (Open, High, Low, Close) prices of financial instruments. The models are retrained each year to evaluate their performance over time.

## Structure
- `data/` - Raw and processed data
- `src/` - Source code for data processing, models, and training
- `notebooks/` - Jupyter notebooks for exploratory analysis
- `results/` - Model outputs, metrics, and visualizations
- `tests/` - Unit tests for the code

## Features
- Yearly retraining of models
- Comparison of different ML algorithms
- Performance metrics tracking over time
- Visualization of results

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your OHLC data in the `data/raw/` directory
3. Run the main script: `python main.py`

## Models
- LSTM Neural Network
- Random Forest
- Linear Regression