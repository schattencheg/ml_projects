# Enhanced Framework Features

This document describes the enhanced features added to the ML framework.

## Table of Contents

1. [Smart Data Caching](#smart-data-caching)
2. [Model Classes](#model-classes)
3. [Usage Examples](#usage-examples)
4. [Benefits](#benefits)

---

## Smart Data Caching

### Overview

The `DataProvider` class now includes intelligent data caching that:
- **Saves downloaded data locally** to avoid redundant API calls
- **Loads existing data** from cache when available
- **Downloads only missing data** when extending date ranges
- **Merges cached and new data** seamlessly

### How It Works

```python
from src.data_provider import DataProvider

data_provider = DataProvider(data_dir='data')

# First call: Downloads and caches data
df = data_provider.load_yahoo(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2024-11-24',
    interval='1d',
    use_cache=True  # Enable caching (default)
)
```

**First Run Output:**
```
Downloading BTC-USD data from Yahoo Finance...
Period: 2020-01-01 to 2024-11-24, Interval: 1d
✓ Downloaded 1789 rows
✓ Data cached to data/BTC_USD_1d.csv
```

**Second Run Output:**
```
✓ Found cached data: 1789 rows
  Cached range: 2020-01-01 to 2024-11-24
✓ Cache covers requested range, no download needed
✓ Loaded 1789 rows from cache
```

### Partial Data Download

When you extend the date range, only missing data is downloaded:

```python
# Extend date range to include more recent data
df = data_provider.load_yahoo(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2024-12-31',  # Extended range
    interval='1d',
    use_cache=True
)
```

**Output:**
```
✓ Found cached data: 1789 rows
  Cached range: 2020-01-01 to 2024-11-24
Downloading data after 2024-11-24...
✓ Downloaded 37 rows (after cache)
✓ Data cached to data/BTC_USD_1d.csv
✓ Total data: 1826 rows
```

### Cache File Structure

Cache files are stored in the `data/` directory with the naming convention:
```
data/
├── BTC_USD_1d.csv      # BTC-USD daily data
├── BTC_USD_1h.csv      # BTC-USD hourly data
├── ETH_USD_1d.csv      # ETH-USD daily data
└── ...
```

### API Reference

#### `load_yahoo()`

```python
def load_yahoo(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = '1d',
    use_cache: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `ticker`: Stock/crypto ticker symbol (e.g., 'BTC-USD', 'AAPL')
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `interval`: Data interval (1d, 1h, 5m, etc.)
- `use_cache`: Whether to use cached data (default: True)

**Returns:**
- DataFrame with OHLCV data

**Features:**
1. Checks if data exists in cache
2. Loads cached data if available
3. Downloads only missing data if needed
4. Merges cached and new data
5. Saves updated data to cache

---

## Model Classes

### Overview

The framework provides ready-to-use model classes that wrap scikit-learn models with additional functionality:

1. **LogisticRegressionModel** - Logistic regression for classification
2. **RandomForestModel** - Random forest classifier

### LogisticRegressionModel

Logistic regression model with automatic configuration and coefficient access.

```python
from src.models_lib.linear_model import LogisticRegressionModel

# Create model
model = LogisticRegressionModel(
    name='LogisticRegression',
    max_iter=1000,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get coefficients
coefficients = model.get_coefficients()
print(f"Coefficients shape: {coefficients.shape}")
```

**Features:**
- Multi-core CPU support (`n_jobs=-1`)
- Automatic target conversion
- Probability predictions
- Coefficient access for interpretability

**Default Parameters:**
```python
{
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}
```

### RandomForestModel

Random forest classifier with feature importance.

```python
from src.models_lib.linear_model import RandomForestModel

# Create model
model = RandomForestModel(
    name='RandomForest',
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get feature importance
importance = model.get_feature_importance()

# Show top features
top_indices = np.argsort(importance)[-10:][::-1]
for idx in top_indices:
    print(f"{feature_names[idx]}: {importance[idx]:.4f}")
```

**Features:**
- Multi-core CPU support (`n_jobs=-1`)
- Automatic target conversion
- Probability predictions
- Feature importance for interpretability

**Default Parameters:**
```python
{
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42,
    'n_jobs': -1
}
```

### Base Model Interface

All model classes inherit from `BaseModel` and provide a consistent interface:

```python
# Common methods for all models
model.fit(X, y)                    # Train the model
model.predict(X)                   # Make predictions
model.predict_proba(X)             # Get probabilities (if supported)
model.save(filepath)               # Save model to disk
model.load(filepath)               # Load model from disk
model.get_params()                 # Get model parameters
model.set_params(**params)         # Set model parameters
```

---

## Usage Examples

### Example 1: Basic Workflow with Caching

```python
from src.data_provider import DataProvider
from src.features_generator import FeaturesGenerator
from src.models_lib.linear_model import LogisticRegressionModel

# Load data (with caching)
data_provider = DataProvider(data_dir='data')
df = data_provider.load_yahoo(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2024-11-24',
    interval='1d',
    use_cache=True
)

# Generate features
features_gen = FeaturesGenerator()
df_features = features_gen.generate_features(df, feature_set='advanced')
df_features = features_gen.create_target(
    df_features,
    target_type='classification',
    future_bars=5,
    threshold=0.02
)

# Prepare data
df_features = df_features.dropna()
feature_cols = features_gen.get_feature_names()

# Split data
train_df, val_df, test_df = data_provider.split_data(df_features)

X_train = train_df[feature_cols].values
y_train = train_df['target'].values
X_test = test_df[feature_cols].values
y_test = test_df['target'].values

# Train model
model = LogisticRegressionModel(n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

### Example 2: Comparing Multiple Models

```python
from src.models_lib.linear_model import LogisticRegressionModel, RandomForestModel
from sklearn.metrics import f1_score

# Create models
models = {
    'LogisticRegression': LogisticRegressionModel(n_jobs=-1),
    'RandomForest': RandomForestModel(n_estimators=100, n_jobs=-1)
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    results[name] = f1
    print(f"{name}: F1 = {f1:.4f}")

# Find best model
best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name}")
```

### Example 3: Feature Importance Analysis

```python
from src.models_lib.linear_model import RandomForestModel
import numpy as np

# Train Random Forest
model = RandomForestModel(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)

# Get feature importance
importance = model.get_feature_importance()

# Sort features by importance
indices = np.argsort(importance)[::-1]

# Print top 10 features
print("Top 10 Most Important Features:")
for i in range(10):
    idx = indices[i]
    print(f"{i+1}. {feature_cols[idx]}: {importance[idx]:.4f}")
```

### Example 4: Model Persistence

```python
import joblib
from pathlib import Path
from datetime import datetime

# Train model
model = LogisticRegressionModel(n_jobs=-1)
model.fit(X_train, y_train)

# Create timestamped save directory
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = Path('models') / timestamp
save_dir.mkdir(parents=True, exist_ok=True)

# Save model
model_path = save_dir / 'logistic_regression.joblib'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Load model later
loaded_model = joblib.load(model_path)
predictions = loaded_model.predict(X_test)
```

---

## Benefits

### 1. Faster Development

**Smart Caching:**
- No need to re-download data every time
- Instant data loading from cache
- Automatic handling of date range extensions

**Ready-to-Use Models:**
- Pre-configured with sensible defaults
- Consistent interface across all models
- No need to remember scikit-learn parameters

### 2. Reduced API Calls

**Before (without caching):**
```
Run 1: Download 1789 rows (10 seconds)
Run 2: Download 1789 rows (10 seconds)
Run 3: Download 1789 rows (10 seconds)
Total: 30 seconds, 3 API calls
```

**After (with caching):**
```
Run 1: Download 1789 rows (10 seconds)
Run 2: Load from cache (0.5 seconds)
Run 3: Load from cache (0.5 seconds)
Total: 11 seconds, 1 API call
```

### 3. Better Performance

**Multi-Core CPU Support:**
- All models use `n_jobs=-1` by default
- Automatic parallelization
- 3-8x faster training on multi-core systems

### 4. Enhanced Interpretability

**LogisticRegression:**
- Access coefficients to understand feature impact
- Identify most influential features

**RandomForest:**
- Feature importance scores
- Understand which features drive predictions
- Feature selection based on importance

### 5. Consistent Interface

All models follow the same pattern:
```python
# Create
model = ModelClass(name='...', **params)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Analyze
info = model.get_coefficients()  # or get_feature_importance()
```

### 6. Production Ready

**Model Persistence:**
- Save trained models with joblib
- Load models for inference
- Timestamped saves prevent overwriting

**Metadata Tracking:**
- Save training configuration
- Track model performance
- Reproducible experiments

---

## Complete Example Script

See `btcusdt_framework_example.py` for a complete working example that demonstrates:

1. ✅ Smart data loading with caching
2. ✅ Feature generation
3. ✅ Training LogisticRegressionModel and RandomForestModel
4. ✅ Model evaluation and comparison
5. ✅ Feature importance analysis
6. ✅ Model persistence with timestamps
7. ✅ Loading saved models

**Run the example:**
```bash
python btcusdt_framework_example.py
```

**Run it again to see caching in action:**
```bash
python btcusdt_framework_example.py
# Notice: Data loads instantly from cache!
```

---

## Summary

### What's New

1. **Smart Data Caching** in `DataProvider`
   - Automatic caching to disk
   - Partial data downloads
   - Seamless cache management

2. **Enhanced Model Classes**
   - `LogisticRegressionModel` with coefficients
   - `RandomForestModel` with feature importance
   - Multi-core CPU support
   - Consistent interface

3. **Better Developer Experience**
   - Faster iterations
   - Less boilerplate code
   - More interpretable models

### Next Steps

1. Run `btcusdt_framework_example.py` to see the features in action
2. Check the `data/` folder to see cached data files
3. Check the `models/` folder to see timestamped model saves
4. Modify the example to work with your own data and models

---

## API Reference Summary

### DataProvider

```python
DataProvider(data_dir='data')
load_yahoo(ticker, start_date, end_date, interval='1d', use_cache=True)
load_csv(filepath, **kwargs)
validate_data(df)
clean_data(df, method='drop')
split_data(df, train_ratio=0.7, val_ratio=0.15)
```

### LogisticRegressionModel

```python
LogisticRegressionModel(name='LogisticRegression', **params)
fit(X, y)
predict(X)
predict_proba(X)
get_coefficients()
```

### RandomForestModel

```python
RandomForestModel(name='RandomForest', **params)
fit(X, y)
predict(X)
predict_proba(X)
get_feature_importance()
```

---

**Last Updated:** 2024-11-24
**Framework Version:** 1.1.0
