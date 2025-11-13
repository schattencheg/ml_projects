# ML Framework

A simple and functional machine learning framework for financial data analysis and backtesting.

## Project Structure

```
ml_framework/
├── src/
│   ├── __init__.py
│   ├── data_provider.py      # Data loading and management
│   ├── features_generator.py # Feature engineering
│   ├── model_manager.py      # Model configuration and management
│   ├── ml_trainer.py         # Model training
│   ├── ml_tester.py          # Model testing and evaluation
│   └── backtester.py         # Strategy backtesting
├── examples/
│   └── basic_workflow.py     # Example usage
├── requirements.txt
└── README.md
```

## Core Classes

### DataProvider
Handles data loading, validation, and basic preprocessing.

### FeaturesGenerator
Generates technical indicators and features from raw OHLC data.

### ModelManager
Manages model configurations, saving, and loading.

### ML_Trainer
Trains machine learning models on prepared data.

### ML_Tester
Evaluates trained models and generates performance metrics.

### Backtester
Backtests trading strategies using ML predictions.

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.model_manager import ModelManager
from src.ml_trainer import ML_Trainer
from src.ml_tester import ML_Tester
from src.backtester import Backtester

# 1. Load data
data_provider = DataProvider()
df = data_provider.load_data('data/sample.csv')

# 2. Generate features
features_gen = FeaturesGenerator()
df_features = features_gen.generate_features(df)

# 3. Setup models
model_manager = ModelManager()
models = model_manager.get_models()

# 4. Train models
trainer = ML_Trainer()
trained_models = trainer.train(df_features, models)

# 5. Test models
tester = ML_Tester()
results = tester.evaluate(df_features, trained_models)

# 6. Backtest strategy
backtester = Backtester()
backtest_results = backtester.run(df, trained_models['best_model'])
```

## Features

- ✅ Simple and intuitive API
- ✅ Modular design for easy customization
- ✅ Built-in feature engineering
- ✅ Multiple ML model support
- ✅ Comprehensive backtesting
- ✅ Performance metrics and visualization

## License

MIT
