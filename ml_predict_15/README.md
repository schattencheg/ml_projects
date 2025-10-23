# ML Price Prediction with Backtesting

A modular machine learning framework for cryptocurrency price prediction with integrated backtesting and trailing stop loss functionality.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Save ML Models (First Time Only)

```bash
python train_and_save_models.py
```

This will train 13 different ML models and save them to the `models/` folder.

### 3. Run Backtesting

```bash
# Quick start example
python backtest_quick_start.py

# Advanced examples
python backtest_example.py
```

## 📁 Project Structure

```
ml_predict_15/
├── src/                          # Source code modules
│   ├── data_preparation.py       # Data preparation functions
│   ├── model_training.py         # Model training and evaluation
│   ├── neural_models.py          # Neural network architectures
│   ├── visualization.py          # Plotting and visualization
│   ├── MLBacktester.py           # Backtesting engine
│   ├── model_loader.py           # Model loading/saving utilities
│   └── FeaturesGenerator.py      # Technical indicators
├── docs/                         # Documentation
│   ├── README.md                 # Detailed documentation
│   ├── BACKTEST_GUIDE.md         # Backtesting guide
│   ├── README_MODELS.md          # Model descriptions
│   └── CHANGES.md                # Change log
├── data/                         # Data files
│   ├── btc_2022.csv             # Training data
│   └── btc_2023.csv             # Test data
├── models/                       # Saved models
├── plots/                        # Generated visualizations
├── run_me.py                     # Main training script
├── train_and_save_models.py     # Train and save all models
├── backtest_quick_start.py      # Quick backtesting example
└── backtest_example.py          # Advanced backtesting examples
```

## 🎯 Features

- **13 ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, LSTM, CNN, and more
- **Modular Design**: Clean separation of concerns with dedicated modules
- **Trailing Stop Loss**: Dynamic risk management for backtesting
- **Model Persistence**: Train once, backtest many times
- **Comprehensive Metrics**: Returns, win rate, Sharpe ratio, max drawdown
- **Beautiful Visualizations**: Charts for analysis

## 📊 Available Models

### Traditional ML
- Logistic Regression
- Ridge Classifier
- Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine

### Advanced ML
- XGBoost
- LightGBM

### Neural Networks
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- LSTM-CNN Hybrid

## 🎓 Workflow

1. **Train Models**: Run `train_and_save_models.py` (once)
2. **Load Models**: Backtest scripts automatically load saved models
3. **Backtest**: Test strategies with trailing stop loss
4. **Compare**: Evaluate all models and parameters
5. **Optimize**: Find best performing configurations

## 📚 Documentation

- **[docs/README.md](docs/README.md)** - Complete documentation
- **[docs/BACKTEST_GUIDE.md](docs/BACKTEST_GUIDE.md)** - Backtesting guide
- **[docs/README_MODELS.md](docs/README_MODELS.md)** - Model descriptions
- **[docs/CHANGES.md](docs/CHANGES.md)** - Recent changes

## 💡 Example Usage

```python
from src.model_loader import load_all_models, load_scaler
from src.MLBacktester import MLBacktester

# Load pre-trained models
models = load_all_models()
scaler = load_scaler()

# Initialize backtester
backtester = MLBacktester(
    initial_capital=10000.0,
    trailing_stop_pct=2.0,
    take_profit_pct=5.0,
    probability_threshold=0.6
)

# Run backtest
results = backtester.run_backtest(
    df=df_test,
    model=models['random_forest'],
    scaler=scaler,
    X_columns=feature_columns
)

# Display results
backtester.print_results(results)
```

## 🔧 Requirements

```
pandas
numpy
scikit-learn
matplotlib
tensorflow>=2.10.0
xgboost>=1.6.0
lightgbm>=3.3.0
joblib
```

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results.

---

**For detailed documentation, see [docs/README.md](docs/README.md)**
